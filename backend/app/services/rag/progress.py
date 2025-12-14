"""Learning progress tracking service - Quizlet-like functionality."""
from typing import Dict, List, Optional, Any
from uuid import UUID
from sqlalchemy import select, func, and_, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from app.models.progress import ChunkInteraction, DocumentProgress
from app.models.chunk import DocumentChunk
from app.models.document import Document


class ProgressService:
    """Service for tracking and calculating learning progress."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def record_chunk_interaction(
        self,
        user_id: UUID,
        chunk_ids: List[str],
        interaction_type: str,  # 'chat', 'quiz', 'flashcard'
        was_successful: bool = False
    ) -> None:
        """
        Record that a user interacted with specific chunks.
        Updates both ChunkInteraction and DocumentProgress tables.
        """
        if not chunk_ids:
            return
        
        for chunk_id in chunk_ids:
            # Upsert the chunk interaction
            stmt = insert(ChunkInteraction).values(
                user_id=user_id,
                chunk_id=chunk_id,
                interaction_type=interaction_type,
                was_successful=was_successful,
                interaction_count=1
            )
            stmt = stmt.on_conflict_do_update(
                constraint='uix_user_chunk_interaction',
                set_={
                    'interaction_count': ChunkInteraction.interaction_count + 1,
                    'was_successful': was_successful if was_successful else ChunkInteraction.was_successful,
                    'last_interaction': func.now()
                }
            )
            await self.session.execute(stmt)
        
        await self.session.commit()
        
        # Update document progress for affected documents
        await self._update_document_progress(user_id, chunk_ids, interaction_type)
    
    async def _update_document_progress(
        self,
        user_id: UUID,
        chunk_ids: List[str],
        interaction_type: str
    ) -> None:
        """Update DocumentProgress table after chunk interactions."""
        # Get document IDs from the chunks
        chunk_result = await self.session.execute(
            select(DocumentChunk.document_id).where(DocumentChunk.id.in_(chunk_ids)).distinct()
        )
        document_ids = [row[0] for row in chunk_result.fetchall()]
        
        for doc_id in document_ids:
            await self._recalculate_document_progress(user_id, doc_id)
    
    async def _recalculate_document_progress(self, user_id: UUID, document_id: int) -> None:
        """Recalculate and update progress for a specific user-document pair."""
        # Get total chunks in document
        total_result = await self.session.execute(
            select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == document_id)
        )
        total_chunks = total_result.scalar() or 0
        
        # Get chunks from this document
        chunk_ids_result = await self.session.execute(
            select(DocumentChunk.id).where(DocumentChunk.document_id == document_id)
        )
        doc_chunk_ids = [row[0] for row in chunk_ids_result.fetchall()]
        
        if not doc_chunk_ids:
            return
        
        # Count interactions by type
        stats = {}
        for interaction_type in ['chat', 'quiz', 'flashcard']:
            count_result = await self.session.execute(
                select(func.count(ChunkInteraction.id.distinct())).where(
                    and_(
                        ChunkInteraction.user_id == user_id,
                        ChunkInteraction.chunk_id.in_(doc_chunk_ids),
                        ChunkInteraction.interaction_type == interaction_type
                    )
                )
            )
            stats[interaction_type] = count_result.scalar() or 0
        
        # Count mastered chunks (successful quiz interactions)
        mastered_result = await self.session.execute(
            select(func.count(ChunkInteraction.id.distinct())).where(
                and_(
                    ChunkInteraction.user_id == user_id,
                    ChunkInteraction.chunk_id.in_(doc_chunk_ids),
                    ChunkInteraction.interaction_type == 'quiz',
                    ChunkInteraction.was_successful == True
                )
            )
        )
        chunks_mastered = mastered_result.scalar() or 0
        
        # Calculate overall progress (weighted average)
        # Chat: 20%, Quiz: 50%, Flashcard: 30%
        if total_chunks > 0:
            chat_progress = (stats['chat'] / total_chunks) * 0.2
            quiz_progress = (stats['quiz'] / total_chunks) * 0.5
            flash_progress = (stats['flashcard'] / total_chunks) * 0.3
            overall_progress = min(1.0, chat_progress + quiz_progress + flash_progress)
        else:
            overall_progress = 0.0
        
        # Upsert DocumentProgress
        stmt = insert(DocumentProgress).values(
            user_id=user_id,
            document_id=document_id,
            total_chunks=total_chunks,
            chunks_studied=stats['chat'],
            chunks_quizzed=stats['quiz'],
            chunks_flashcarded=stats['flashcard'],
            chunks_mastered=chunks_mastered,
            overall_progress=overall_progress
        )
        stmt = stmt.on_conflict_do_update(
            constraint='uix_user_document_progress',
            set_={
                'total_chunks': total_chunks,
                'chunks_studied': stats['chat'],
                'chunks_quizzed': stats['quiz'],
                'chunks_flashcarded': stats['flashcard'],
                'chunks_mastered': chunks_mastered,
                'overall_progress': overall_progress,
                'last_activity': func.now()
            }
        )
        await self.session.execute(stmt)
        await self.session.commit()
    
    async def get_user_progress(self, user_id: UUID) -> List[Dict[str, Any]]:
        """Get learning progress for all documents for a user."""
        result = await self.session.execute(
            select(DocumentProgress, Document.title, Document.num_pages)
            .join(Document, DocumentProgress.document_id == Document.id)
            .where(DocumentProgress.user_id == user_id)
            .order_by(DocumentProgress.last_activity.desc())
        )
        
        progress_list = []
        for row in result.fetchall():
            progress, title, num_pages = row
            progress_list.append({
                'document_id': progress.document_id,
                'document_title': title,
                'num_pages': num_pages,
                'total_chunks': progress.total_chunks,
                'chunks_studied': progress.chunks_studied,
                'chunks_quizzed': progress.chunks_quizzed,
                'chunks_flashcarded': progress.chunks_flashcarded,
                'chunks_mastered': progress.chunks_mastered,
                'overall_progress': progress.overall_progress,
                'last_activity': progress.last_activity.isoformat() if progress.last_activity else None
            })
        
        return progress_list
    
    async def get_document_progress(self, user_id: UUID, document_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed progress for a specific document."""
        result = await self.session.execute(
            select(DocumentProgress)
            .where(
                and_(
                    DocumentProgress.user_id == user_id,
                    DocumentProgress.document_id == document_id
                )
            )
        )
        progress = result.scalar_one_or_none()
        
        if not progress:
            # Calculate on the fly if not exists
            await self._recalculate_document_progress(user_id, document_id)
            result = await self.session.execute(
                select(DocumentProgress)
                .where(
                    and_(
                        DocumentProgress.user_id == user_id,
                        DocumentProgress.document_id == document_id
                    )
                )
            )
            progress = result.scalar_one_or_none()
        
        if not progress:
            return None
        
        return {
            'document_id': progress.document_id,
            'total_chunks': progress.total_chunks,
            'chunks_studied': progress.chunks_studied,
            'chunks_quizzed': progress.chunks_quizzed,
            'chunks_flashcarded': progress.chunks_flashcarded,
            'chunks_mastered': progress.chunks_mastered,
            'overall_progress': progress.overall_progress,
            'study_progress': progress.chunks_studied / progress.total_chunks if progress.total_chunks > 0 else 0,
            'quiz_progress': progress.chunks_quizzed / progress.total_chunks if progress.total_chunks > 0 else 0,
            'flashcard_progress': progress.chunks_flashcarded / progress.total_chunks if progress.total_chunks > 0 else 0,
            'mastery_rate': progress.chunks_mastered / progress.total_chunks if progress.total_chunks > 0 else 0,
        }
    
    async def get_unstudied_chunks(self, user_id: UUID, document_id: int, limit: int = 10) -> List[str]:
        """Get chunk IDs that haven't been studied yet (for smart recommendations)."""
        # Get all chunks for the document
        all_chunks_result = await self.session.execute(
            select(DocumentChunk.id).where(DocumentChunk.document_id == document_id)
        )
        all_chunk_ids = {row[0] for row in all_chunks_result.fetchall()}
        
        # Get studied chunks
        studied_result = await self.session.execute(
            select(ChunkInteraction.chunk_id).where(
                and_(
                    ChunkInteraction.user_id == user_id,
                    ChunkInteraction.chunk_id.in_(all_chunk_ids)
                )
            ).distinct()
        )
        studied_chunk_ids = {row[0] for row in studied_result.fetchall()}
        
        # Return unstudied chunks
        unstudied = list(all_chunk_ids - studied_chunk_ids)
        return unstudied[:limit]
