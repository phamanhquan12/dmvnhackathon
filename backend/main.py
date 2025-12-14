import asyncio
import os
import re
import tempfile
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
from difflib import SequenceMatcher
from pathlib import Path

import streamlit as st
from sqlalchemy import select

# Import app modules
from app.core.database import get_async_session
from app.models.user import User
from app.models.document import FileTypeEnum
from app.services.rag.ingestion import IngestionService
from app.services.rag.chat import ChatService
from app.services.rag.progress import ProgressService
from app.services.rag.content_generator import ContentGenerationService, ProgressTrackingService

# Stage B - SOP Video Assessment imports (conditional to avoid import errors if not installed)
try:
    from app.services.sop import VisionEngine, SOPEngine, InteractionEngine
    from app.services.sop.assessment_service import (
        save_assessment_result, 
        get_user_assessment_history, 
        get_assessment_details,
        generate_ai_feedback,
        get_all_assessments,
        delete_user_assessments,
        delete_all_assessments
    )
    SOP_AVAILABLE = True
except ImportError:
    SOP_AVAILABLE = False


def find_best_matching_text(page_content: str, source_chunk: str, answer_text: str) -> List[str]:
    """
    Find the best matching text segments in page content.
    Returns a list of text segments to highlight.
    """
    matches_to_highlight = []
    
    # 1. First try: Find sentences from source chunk that appear in page content
    if source_chunk:
        # Split source chunk into sentences
        sentences = re.split(r'(?<=[.!?])\s+', source_chunk)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Only meaningful sentences
                # Check if sentence appears in page content
                if sentence.lower() in page_content.lower():
                    matches_to_highlight.append(sentence)
    
    # 2. Second try: Find longest common phrases between answer and content
    if answer_text and len(matches_to_highlight) == 0:
        # Clean up answer text - remove common filler words at start
        clean_answer = answer_text.strip()
        
        # Split answer into phrases/sentences
        phrases = re.split(r'[.!?;\n]', clean_answer)
        for phrase in phrases:
            phrase = phrase.strip()
            if len(phrase) > 15:  # Only meaningful phrases
                # Try to find this phrase in content
                if phrase.lower() in page_content.lower():
                    matches_to_highlight.append(phrase)
    
    # 3. Third try: Use sequence matching to find longest common substring
    if len(matches_to_highlight) == 0 and (source_chunk or answer_text):
        search_text = source_chunk if source_chunk else answer_text
        # Find longest common substring
        matcher = SequenceMatcher(None, page_content.lower(), search_text.lower())
        match = matcher.find_longest_match(0, len(page_content), 0, len(search_text))
        if match.size > 30:  # At least 30 chars for meaningful match
            matched_text = page_content[match.a:match.a + match.size]
            matches_to_highlight.append(matched_text.strip())
    
    return matches_to_highlight


def highlight_text_in_content(content: str, highlights: List[str]) -> str:
    """
    Apply highlighting to content by wrapping matched text in HTML span with background.
    Falls back to markdown bold if HTML not possible.
    """
    if not highlights:
        return content
    
    result = content
    for text in highlights:
        if len(text) > 10:  # Only highlight meaningful text
            try:
                # Use case-insensitive replacement while preserving original case
                pattern = re.compile(re.escape(text), re.IGNORECASE)
                # Use markdown bold + italic for emphasis
                result = pattern.sub(lambda m: f"**__{m.group(0)}__**", result, count=1)
            except:
                pass
    
    return result


def run_async(coro):
    """Helper to run async code in Streamlit's sync context."""
    # Set Windows-compatible event loop policy
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Create a new event loop for each call to avoid loop conflicts
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def get_or_create_user(employee_id: str, full_name: str) -> User:
    """Get existing user or create a new one in the database."""
    session_factory = get_async_session()
    async with session_factory() as session:
        # Check if user already exists
        result = await session.execute(
            select(User).where(User.employee_id == employee_id)
        )
        user = result.scalar_one_or_none()
        
        if user:
            # Update name if changed
            if user.full_name != full_name:
                user.full_name = full_name
                await session.commit()
            return user
        
        # Create new user with default role
        from app.models.user import UserRole
        user = User(employee_id=employee_id, full_name=full_name, role=UserRole.USER)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


async def authenticate_admin(employee_id: str, password: str) -> User | None:
    """Authenticate an admin user with password."""
    session_factory = get_async_session()
    async with session_factory() as session:
        result = await session.execute(
            select(User).where(User.employee_id == employee_id)
        )
        user = result.scalar_one_or_none()
        
        if user and user.role and user.role.value == "admin":
            if user.check_password(password):
                return user
        return None


async def get_all_users():
    """Get all users from the database (admin function)."""
    session_factory = get_async_session()
    async with session_factory() as session:
        result = await session.execute(select(User).order_by(User.created_at.desc()))
        return result.scalars().all()


async def update_user_role(user_id: str, role: str):
    """Update user role (admin function)."""
    from app.models.user import UserRole
    from uuid import UUID
    session_factory = get_async_session()
    async with session_factory() as session:
        result = await session.execute(
            select(User).where(User.id == UUID(user_id))
        )
        user = result.scalar_one_or_none()
        if user:
            user.role = UserRole(role)
            await session.commit()
            return True
        return False


async def set_admin_password(user_id: str, password: str):
    """Set password for admin user."""
    from uuid import UUID
    session_factory = get_async_session()
    async with session_factory() as session:
        result = await session.execute(
            select(User).where(User.id == UUID(user_id))
        )
        user = result.scalar_one_or_none()
        if user:
            user.set_password(password)
            await session.commit()
            return True
        return False


async def create_user(employee_id: str, full_name: str, role: str = "user", password: str = None) -> Dict[str, Any]:
    """Create a new user."""
    from app.models.user import UserRole
    session_factory = get_async_session()
    async with session_factory() as session:
        # Check if user already exists
        result = await session.execute(
            select(User).where(User.employee_id == employee_id)
        )
        existing_user = result.scalar_one_or_none()
        if existing_user:
            return {"success": False, "message": f"User with employee ID {employee_id} already exists"}
        
        # Create new user
        user = User(
            employee_id=employee_id,
            full_name=full_name,
            role=UserRole(role)
        )
        if password:
            user.set_password(password)
        
        session.add(user)
        await session.commit()
        return {"success": True, "message": f"User {full_name} created successfully", "user_id": str(user.id)}


async def delete_user(user_id: str) -> Dict[str, Any]:
    """Delete a user and all their related data."""
    from uuid import UUID
    from app.models.session import TheorySession, PracticalSession
    from app.models.progress import DocumentProgress, ChunkInteraction
    
    session_factory = get_async_session()
    async with session_factory() as session:
        user_uuid = UUID(user_id)
        
        # Check if user exists
        result = await session.execute(select(User).where(User.id == user_uuid))
        user = result.scalar_one_or_none()
        if not user:
            return {"success": False, "message": "User not found"}
        
        user_name = user.full_name
        
        # Delete related data
        # 1. Delete theory sessions
        await session.execute(
            TheorySession.__table__.delete().where(TheorySession.user_id == user_uuid)
        )
        
        # 2. Delete practical sessions
        await session.execute(
            PracticalSession.__table__.delete().where(PracticalSession.user_id == user_uuid)
        )
        
        # 3. Delete document progress
        await session.execute(
            DocumentProgress.__table__.delete().where(DocumentProgress.user_id == user_uuid)
        )
        
        # 4. Delete chunk interactions
        await session.execute(
            ChunkInteraction.__table__.delete().where(ChunkInteraction.user_id == user_uuid)
        )
        
        # 4. Delete the user
        await session.delete(user)
        await session.commit()
        
        return {"success": True, "message": f"User {user_name} and all their data deleted successfully"}


async def get_user_learning_progress_admin(employee_id: str) -> Dict[str, Any]:
    """Get comprehensive learning progress for a user (admin view)."""
    from app.models.session import TheorySession, PracticalSession
    from app.models.progress import DocumentProgress
    from uuid import UUID
    
    session_factory = get_async_session()
    async with session_factory() as session:
        # Get user
        result = await session.execute(select(User).where(User.employee_id == employee_id))
        user = result.scalar_one_or_none()
        if not user:
            return {"error": "User not found"}
        
        # Get theory sessions
        theory_result = await session.execute(
            select(TheorySession).where(TheorySession.user_id == user.id).order_by(TheorySession.created_at.desc())
        )
        theory_sessions = theory_result.scalars().all()
        
        # Get practical sessions (assessments)
        practical_result = await session.execute(
            select(PracticalSession).where(PracticalSession.user_id == user.id).order_by(PracticalSession.created_at.desc())
        )
        practical_sessions = practical_result.scalars().all()
        
        # Get document progress
        progress_result = await session.execute(
            select(DocumentProgress).where(DocumentProgress.user_id == user.id)
        )
        progress_records = progress_result.scalars().all()
        
        # Calculate statistics
        theory_passed = sum(1 for s in theory_sessions if s.status == "PASSED")
        theory_failed = sum(1 for s in theory_sessions if s.status == "FAILED")
        theory_total_score = sum(s.score or 0 for s in theory_sessions)
        theory_avg_score = theory_total_score / len(theory_sessions) if theory_sessions else 0
        
        practical_passed = sum(1 for s in practical_sessions if s.status == "PASSED")
        practical_failed = sum(1 for s in practical_sessions if s.status == "FAILED")
        practical_total_score = sum(s.score or 0 for s in practical_sessions)
        practical_avg_score = practical_total_score / len(practical_sessions) if practical_sessions else 0
        
        return {
            "user": {
                "id": str(user.id),
                "employee_id": user.employee_id,
                "full_name": user.full_name,
                "role": user.role.value if user.role else "user"
            },
            "theory": {
                "total_sessions": len(theory_sessions),
                "passed": theory_passed,
                "failed": theory_failed,
                "average_score": round(theory_avg_score, 1),
                "sessions": [
                    {
                        "id": str(s.id),
                        "score": s.score,
                        "status": s.status,
                        "created_at": s.created_at.strftime('%Y-%m-%d %H:%M') if s.created_at else None,
                        "details": s.details
                    } for s in theory_sessions[:10]  # Last 10 sessions
                ]
            },
            "practical": {
                "total_sessions": len(practical_sessions),
                "passed": practical_passed,
                "failed": practical_failed,
                "average_score": round(practical_avg_score, 1),
                "sessions": [
                    {
                        "id": str(s.id),
                        "session_code": s.session_code,
                        "process_name": s.process_name,
                        "score": s.score,
                        "status": s.status,
                        "completed_steps": s.completed_steps,
                        "total_steps": s.total_steps,
                        "total_duration": s.total_duration,
                        "created_at": s.created_at.strftime('%Y-%m-%d %H:%M') if s.created_at else None
                    } for s in practical_sessions[:10]  # Last 10 sessions
                ]
            },
            "documents_progress": [
                {
                    "document_id": p.document_id,
                    "total_chunks": p.total_chunks,
                    "chunks_studied": p.chunks_studied,
                    "chunks_quizzed": p.chunks_quizzed,
                    "chunks_flashcarded": p.chunks_flashcarded,
                    "chunks_mastered": p.chunks_mastered,
                    "overall_progress": round(p.overall_progress * 100, 1) if p.overall_progress else 0,
                    "last_activity": p.last_activity.strftime('%Y-%m-%d %H:%M') if p.last_activity else None
                } for p in progress_records
            ]
        }


async def ingest_pdf_file(file_path: str, filename: str) -> Dict[str, Any]:
    """Process and ingest a PDF file into the database."""
    session_factory = get_async_session()
    async with session_factory() as session:
        service = IngestionService(session)
        result = await service.process_document(file_path, filename, FileTypeEnum.PDF)
        return result


async def ingest_video_file(file_path: str, filename: str, subtitle_path: str = None) -> Dict[str, Any]:
    """Process and ingest a video file into the database."""
    session_factory = get_async_session()
    async with session_factory() as session:
        service = IngestionService(session)
        result = await service.process_document(
            file_path, 
            filename, 
            FileTypeEnum.VIDEO,
            subtitle_path=subtitle_path
        )
        return result


async def get_all_documents() -> List[Dict[str, Any]]:
    """Fetch all documents from the database for UI selection."""
    session_factory = get_async_session()
    async with session_factory() as session:
        chat_service = ChatService(session)
        documents = await chat_service.get_all_documents()
        return [{
            "id": doc.id, 
            "title": doc.title, 
            "file_path": doc.file_path, 
            "num_pages": doc.num_pages,
            "file_type": doc.file_type,
            "duration_seconds": doc.duration_seconds,
            "flashcards_generated": doc.flashcards_generated,
            "quizzes_generated": doc.quizzes_generated
        } for doc in documents]


async def delete_document(document_id: int) -> Dict[str, Any]:
    """Delete a document and all its related data (chunks, flashcards, quiz sets, progress)."""
    from app.models.document import Document
    session_factory = get_async_session()
    async with session_factory() as session:
        # Get the document
        result = await session.execute(select(Document).where(Document.id == document_id))
        document = result.scalar_one_or_none()
        
        if not document:
            return {"success": False, "error": "Document not found"}
        
        # Store info for response
        title = document.title
        file_path = document.file_path
        
        # Delete the document (cascades to chunks, flashcards, quiz_sets, progress)
        await session.delete(document)
        await session.commit()
        
        # Optionally delete the physical file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                # Log but don't fail if file deletion fails
                pass
        
        return {"success": True, "title": title, "message": f"Deleted document '{title}' and all related data"}


async def delete_flashcards_for_document(document_id: int) -> Dict[str, Any]:
    """Delete all flashcards for a specific document."""
    from app.models.document import Document
    from app.models.learning_content import Flashcard
    session_factory = get_async_session()
    async with session_factory() as session:
        # Delete all flashcards for this document
        result = await session.execute(select(Flashcard).where(Flashcard.document_id == document_id))
        flashcards = result.scalars().all()
        count = len(flashcards)
        
        for fc in flashcards:
            await session.delete(fc)
        
        # Update document flag
        doc_result = await session.execute(select(Document).where(Document.id == document_id))
        document = doc_result.scalar_one_or_none()
        if document:
            document.flashcards_generated = False
        
        await session.commit()
        return {"success": True, "deleted_count": count}


async def delete_quiz_sets_for_document(document_id: int) -> Dict[str, Any]:
    """Delete all quiz sets for a specific document."""
    from app.models.document import Document
    from app.models.learning_content import QuizSet
    session_factory = get_async_session()
    async with session_factory() as session:
        # Delete all quiz sets for this document
        result = await session.execute(select(QuizSet).where(QuizSet.document_id == document_id))
        quiz_sets = result.scalars().all()
        count = len(quiz_sets)
        
        for qs in quiz_sets:
            await session.delete(qs)
        
        # Update document flag
        doc_result = await session.execute(select(Document).where(Document.id == document_id))
        document = doc_result.scalar_one_or_none()
        if document:
            document.quizzes_generated = False
        
        await session.commit()
        return {"success": True, "deleted_count": count}


async def rag_chat(
    query: str, 
    document_ids: Optional[List[int]] = None, 
    user_id: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> Tuple[str, Dict[int, Dict[str, Any]]]:
    """
    Perform enhanced RAG-based chat with query transformation and citations.
    Returns response text and citation metadata.
    """
    session_factory = get_async_session()
    async with session_factory() as session:
        chat_service = ChatService(session)
        response, chunk_ids, citations = await chat_service.chat(
            query, 
            document_ids=document_ids,
            chat_history=chat_history
        )
        
        # Track progress if user is logged in
        if user_id and chunk_ids:
            try:
                progress_service = ProgressService(session)
                await progress_service.record_chunk_interaction(
                    user_id=UUID(user_id),
                    chunk_ids=chunk_ids,
                    interaction_type='chat',
                    was_successful=True
                )
            except Exception as e:
                # Don't fail the chat if progress tracking fails
                pass
        
        return response, citations


async def rag_chat_stream(
    query: str, 
    document_ids: Optional[List[int]] = None, 
    user_id: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None
):
    """
    Streaming version of RAG chat - yields response chunks.
    """
    session_factory = get_async_session()
    async with session_factory() as session:
        chat_service = ChatService(session)
        chunk_ids_for_tracking = []
        
        async for item in chat_service.chat_stream(
            query, 
            document_ids=document_ids,
            chat_history=chat_history
        ):
            if item["type"] == "citations":
                chunk_ids_for_tracking = item.get("chunk_ids", [])
            yield item
        
        # Track progress if user is logged in
        if user_id and chunk_ids_for_tracking:
            try:
                progress_service = ProgressService(session)
                await progress_service.record_chunk_interaction(
                    user_id=UUID(user_id),
                    chunk_ids=chunk_ids_for_tracking,
                    interaction_type='chat',
                    was_successful=True
                )
            except Exception as e:
                pass


async def generate_flashcards(document_id: Optional[int] = None, title: Optional[str] = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Generate flashcards from selected document using the ChatService with progress tracking."""
    session_factory = get_async_session()
    async with session_factory() as session:
        chat_service = ChatService(session)
        flashcards, chunk_ids = await chat_service.flash_cards(id=document_id, title=title)
        
        # Track progress if user is logged in
        if user_id and chunk_ids:
            try:
                progress_service = ProgressService(session)
                await progress_service.record_chunk_interaction(
                    user_id=UUID(user_id),
                    chunk_ids=chunk_ids,
                    interaction_type='flashcard',
                    was_successful=False  # Will be updated when user reviews
                )
            except Exception as e:
                pass
        
        return flashcards


async def generate_quiz(document_ids: Optional[List[int]] = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Generate quiz questions using the ChatService with optional document filtering and progress tracking."""
    session_factory = get_async_session()
    async with session_factory() as session:
        chat_service = ChatService(session)
        quiz_data, chunk_ids = await chat_service.quiz(document_ids=document_ids)
        
        # Store chunk IDs in session state for grading
        st.session_state.quiz_chunk_ids = chunk_ids
        
        return quiz_data


async def get_user_progress(user_id: str) -> List[Dict[str, Any]]:
    """Get learning progress for all documents for a user."""
    session_factory = get_async_session()
    async with session_factory() as session:
        progress_service = ProgressService(session)
        return await progress_service.get_user_progress(UUID(user_id))


async def get_document_progress(user_id: str, document_id: int) -> Optional[Dict[str, Any]]:
    """Get detailed progress for a specific document."""
    session_factory = get_async_session()
    async with session_factory() as session:
        progress_service = ProgressService(session)
        return await progress_service.get_document_progress(UUID(user_id), document_id)


async def grade_quiz(quiz_response: List[Dict], user_answers: Dict[int, str], user_id: str) -> Dict[str, Any]:
    """Grade the quiz and save results to TheorySession with progress tracking."""
    session_factory = get_async_session()
    async with session_factory() as session:
        possible_choices = ['A', 'B', 'C', 'D']
        score = 0
        wrong_questions_answers = {'question': [], 'user_answer': [], 'correct_answer': [], 'explanation': []}
        
        for i, q in enumerate(quiz_response):
            question = q.get("question")
            options = q.get("options", [])
            correct_answer = q.get("answer")
            explanation = q.get("explanation", "")
            
            user_answer = user_answers.get(i)
            if user_answer and user_answer in possible_choices:
                user_answer_index = possible_choices.index(user_answer)
                user_answer_text = options[user_answer_index] if user_answer_index < len(options) else ""
                
                if user_answer_text == correct_answer:
                    score += 1
                else:
                    wrong_questions_answers['question'].append(question)
                    wrong_questions_answers['user_answer'].append(user_answer_text)
                    wrong_questions_answers['correct_answer'].append(correct_answer)
                    wrong_questions_answers['explanation'].append(explanation)
        
        total = len(quiz_response)
        status = "PASSED" if total > 0 and (score / total) >= 0.6 else "FAILED"
        
        # Get user UUID from employee_id
        result = await session.execute(select(User).where(User.employee_id == user_id))
        user = result.scalar_one_or_none()
        
        if user:
            from app.models.session import TheorySession
            session_obj = TheorySession(
                user_id=user.id,
                score=score,
                status=status,
                details=wrong_questions_answers
            )
            session.add(session_obj)
            await session.commit()
            
            # Track quiz progress
            chunk_ids = st.session_state.get('quiz_chunk_ids', [])
            if chunk_ids:
                try:
                    progress_service = ProgressService(session)
                    await progress_service.record_chunk_interaction(
                        user_id=user.id,
                        chunk_ids=chunk_ids,
                        interaction_type='quiz',
                        was_successful=(status == "PASSED")
                    )
                except Exception as e:
                    pass
        
        return {
            "score": score,
            "total": total,
            "status": status,
            "details": wrong_questions_answers
        }


# ============== NEW PRE-GENERATED CONTENT FUNCTIONS ==============

async def generate_content_for_document(document_id: int) -> Dict[str, Any]:
    """Generate flashcards and quiz sets for a document (called after ingestion)."""
    session_factory = get_async_session()
    async with session_factory() as session:
        content_service = ContentGenerationService(session)
        
        # Generate flashcards
        flashcards = await content_service.generate_all_flashcards(document_id)
        
        # Generate 3 quiz sets
        quiz_sets = await content_service.generate_quiz_sets(document_id, num_sets=3)
        
        return {
            "flashcards_count": len(flashcards),
            "quiz_sets_count": len(quiz_sets)
        }


async def get_document_flashcards(document_id: int) -> List[Dict[str, Any]]:
    """Get pre-generated flashcards for a document."""
    session_factory = get_async_session()
    async with session_factory() as session:
        content_service = ContentGenerationService(session)
        flashcards = await content_service.get_document_flashcards(document_id)
        return [
            {
                "id": str(fc.id), 
                "front": fc.front, 
                "back": fc.back, 
                "order": fc.order_index,
                "chunk_ids": fc.chunk_ids or []
            }
            for fc in flashcards
        ]


async def get_document_quiz_sets(document_id: int) -> List[Dict[str, Any]]:
    """Get pre-generated quiz sets for a document."""
    session_factory = get_async_session()
    async with session_factory() as session:
        content_service = ContentGenerationService(session)
        return await content_service.get_document_quiz_sets(document_id)


async def get_document_content_for_viewer(document_id: int) -> Dict[str, Any]:
    """Get document content for the document viewer."""
    from app.models.document import Document
    from app.models.chunk import DocumentChunk
    session_factory = get_async_session()
    async with session_factory() as session:
        # Get document info
        doc_result = await session.execute(select(Document).where(Document.id == document_id))
        doc = doc_result.scalar_one_or_none()
        if not doc:
            return {"error": "Document not found"}
        
        # Get all chunks for text content
        chunks_result = await session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.page_number)
        )
        chunks = chunks_result.scalars().all()
        
        # Try to find the actual file path
        file_path = doc.file_path
        
        # If stored path doesn't exist, try to find in uploads folder
        if not file_path or not os.path.exists(file_path):
            # Define uploads directory
            uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
            
            # Try common extensions
            extensions = ['.pdf', '.mp4', '.mov', '.avi', '.mkv', '.webm']
            title = doc.title
            
            for ext in extensions:
                potential_path = os.path.join(uploads_dir, f"{title}{ext}")
                if os.path.exists(potential_path):
                    file_path = potential_path
                    break
                # Also try without cleaning the extension
                potential_path = os.path.join(uploads_dir, title)
                if os.path.exists(potential_path):
                    file_path = potential_path
                    break
        
        return {
            "id": doc.id,
            "title": doc.title,
            "file_type": doc.file_type,
            "file_path": file_path,
            "num_pages": doc.num_pages,
            "chunks": [
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "page_number": chunk.page_number
                }
                for chunk in chunks
            ]
        }


async def get_chunk_content(chunk_ids: List[str]) -> List[Dict[str, Any]]:
    """Get content of specific chunks by their IDs."""
    from app.models.chunk import DocumentChunk
    session_factory = get_async_session()
    async with session_factory() as session:
        if not chunk_ids:
            return []
        result = await session.execute(
            select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids))
        )
        chunks = result.scalars().all()
        return [
            {
                "id": chunk.id,
                "content": chunk.content,
                "page_number": chunk.page_number
            }
            for chunk in chunks
        ]


async def mark_flashcard_complete(user_id: str, flashcard_id: str, ease: int = 2) -> None:
    """Mark a flashcard as completed/known."""
    session_factory = get_async_session()
    async with session_factory() as session:
        # Get user UUID
        result = await session.execute(select(User).where(User.employee_id == user_id))
        user = result.scalar_one_or_none()
        if user:
            progress_service = ProgressTrackingService(session)
            await progress_service.mark_flashcard_completed(
                user_id=user.id,
                flashcard_id=UUID(flashcard_id),
                ease_factor=ease
            )


async def mark_flashcard_reviewed(user_id: str, flashcard_id: str) -> None:
    """Mark a flashcard as reviewed (seen but not mastered)."""
    session_factory = get_async_session()
    async with session_factory() as session:
        result = await session.execute(select(User).where(User.employee_id == user_id))
        user = result.scalar_one_or_none()
        if user:
            progress_service = ProgressTrackingService(session)
            await progress_service.mark_flashcard_reviewed(
                user_id=user.id,
                flashcard_id=UUID(flashcard_id)
            )


async def submit_quiz_attempt(
    user_id: str, 
    quiz_set_id: str, 
    answers: Dict[str, str],
    score: int,
    total: int
) -> Dict[str, Any]:
    """Submit a quiz attempt and record progress."""
    session_factory = get_async_session()
    async with session_factory() as session:
        result = await session.execute(select(User).where(User.employee_id == user_id))
        user = result.scalar_one_or_none()
        
        if user:
            progress_service = ProgressTrackingService(session)
            attempt = await progress_service.record_quiz_attempt(
                user_id=user.id,
                quiz_set_id=UUID(quiz_set_id),
                answers=answers,
                score=score,
                total=total
            )
            
            return {
                "score": score,
                "total": total,
                "is_passed": attempt.is_passed,
                "status": "PASSED" if attempt.is_passed else "FAILED"
            }
        
        return {"error": "User not found"}


async def get_document_progress_new(user_id: str, document_id: int) -> Dict[str, Any]:
    """Get actual progress for a document based on completed content."""
    session_factory = get_async_session()
    async with session_factory() as session:
        result = await session.execute(select(User).where(User.employee_id == user_id))
        user = result.scalar_one_or_none()
        
        if user:
            progress_service = ProgressTrackingService(session)
            return await progress_service.get_overall_document_progress(user.id, document_id)
        
        return {
            "document_id": document_id,
            "overall_progress": 0,
            "flashcard_progress": {"total": 0, "completed": 0, "progress_percentage": 0},
            "quiz_progress": {"total_sets": 0, "passed_sets": 0, "progress_percentage": 0}
        }


def init_state() -> None:
    # Core state only - removed unused mock/config options
    st.session_state.setdefault("events", [])
    st.session_state.setdefault("user_name", "")
    st.session_state.setdefault("employee_id", "")
    st.session_state.setdefault("user_db_id", None)  # Store user UUID from DB
    st.session_state.setdefault("user_role", "user")  # user or admin
    st.session_state.setdefault("mode", "Practice")
    
    # Chat state
    st.session_state.setdefault(
        "chat_history",
        [{"role": "system", "content": "You are a concise training assistant for factory SOPs."}],
    )
    st.session_state.setdefault("chat_reply", "")
    st.session_state.setdefault("chat_citations", {})  # Citation metadata for last response
    st.session_state.setdefault("chat_doc_viewer", None)  # Document viewer state for chat citations
    st.session_state.setdefault("chat_pending_query", None)  # For streaming chat
    
    # RAG Quiz state (pre-generated)
    st.session_state.setdefault("quiz_sets", [])  # Pre-generated quiz sets for selected doc
    st.session_state.setdefault("selected_quiz_set", None)  # Currently selected quiz set
    st.session_state.setdefault("quiz_answers", {})  # User's answers
    st.session_state.setdefault("quiz_result", None)  # Quiz result
    
    # Documents and flashcards (pre-generated)
    st.session_state.setdefault("ingestion_status", None)
    st.session_state.setdefault("available_documents", [])
    st.session_state.setdefault("selected_doc_ids", [])
    st.session_state.setdefault("document_flashcards", [])  # Pre-generated flashcards
    st.session_state.setdefault("flashcard_index", 0)
    st.session_state.setdefault("flashcard_revealed", False)
    
    # Document viewer state
    st.session_state.setdefault("doc_viewer_content", None)  # Content for document viewer
    st.session_state.setdefault("doc_viewer_highlight", None)  # Text to highlight
    st.session_state.setdefault("doc_viewer_source_chunk", None)  # Source chunk content for exact matching
    st.session_state.setdefault("doc_viewer_page", None)  # Page to scroll to
    st.session_state.setdefault("show_doc_viewer", False)  # Whether to show viewer
    
    # Learning progress tracking (based on actual completion)
    st.session_state.setdefault("document_progress", {})
    
    # Stage B - Video Assessment state
    st.session_state.setdefault("sop_rules", None)  # Loaded SOP rules
    st.session_state.setdefault("assessment_running", False)  # Assessment in progress
    st.session_state.setdefault("assessment_result", None)  # Final assessment result
    st.session_state.setdefault("assessment_progress", {})  # Current progress during assessment
    st.session_state.setdefault("assessment_video_path", None)  # Path to uploaded video
    st.session_state.setdefault("assessment_video_name", None)  # Original video filename
    st.session_state.setdefault("assessment_history", None)  # Past assessment history
    st.session_state.setdefault("ai_feedback", None)  # AI-generated feedback from RAG


def add_event(evt_type: str, message: str) -> None:
    st.session_state.events.insert(0, {"type": evt_type, "message": message, "at": datetime.now()})
    st.session_state.events = st.session_state.events[:30]


def render_events() -> None:
    """Render event log in a compact format."""
    with st.expander("📋 Activity Log", expanded=False):
        if not st.session_state.events:
            st.info("No events yet.")
            return
        for evt in st.session_state.events[:10]:
            st.caption(f"• {evt['message']} - {evt['at'].strftime('%H:%M:%S')}")


def layout() -> None:
    st.set_page_config(page_title="DENSO-MIND Training Platform", layout="wide", page_icon="🏭")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        /* Main header styling */
        .main-header {
            background: linear-gradient(90deg, #1e3a5f 0%, #2d5a87 100%);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .main-header h1 {
            color: white;
            margin: 0;
            font-size: 2rem;
        }
        .main-header p {
            color: #b8d4e8;
            margin: 0.5rem 0 0 0;
        }
        
        /* Button styling - prevent text overflow */
        .stButton > button {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-size: 0.85rem;
            padding: 0.5rem 1rem;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            font-size: 0.9rem;
        }
        
        /* Progress indicators */
        .progress-stat {
            text-align: center;
            padding: 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            color: white;
        }
        
        /* Flashcard styling */
        .flashcard {
            border-radius: 12px;
            padding: 2rem;
            min-height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        .flashcard-front {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .flashcard-back {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
        }
        
        /* Progress bar colors */
        .stProgress > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Hide default Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏭 DENSO-MIND</h1>
        <p>AI-Powered Training & Assessment Platform</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 📍 Navigation")
        
        # Different navigation based on role
        if st.session_state.user_role == "admin":
            # Admin sees all options
            col_nav1, col_nav2 = st.columns(2)
            if col_nav1.button("📚 Learn", use_container_width=True):
                st.session_state.mode = "Practice"
            if col_nav2.button("📝 Test", use_container_width=True):
                st.session_state.mode = "Testing"
            if st.button("⚙️ Admin Panel", use_container_width=True):
                st.session_state.mode = "Admin"
        else:
            # User sees only Learn and My Results
            if st.button("📚 Learn", use_container_width=True):
                st.session_state.mode = "Practice"
            if st.button("📊 Results", use_container_width=True):
                st.session_state.mode = "MyResults"
        
        st.markdown("---")
        
        # Profile section
        st.markdown("### 👤 Profile")
        if st.session_state.user_name:
            st.write(f"**{st.session_state.user_name}**")
            st.caption(f"ID: {st.session_state.employee_id}")
            # Show role badge
            if st.session_state.user_role == "admin":
                st.markdown("🔐 **Admin**")
            else:
                st.markdown("👤 User")
        else:
            st.caption("Not signed in")
        
        if st.button("🔄 Switch Account", use_container_width=True):
            st.session_state.user_name = ""
            st.session_state.employee_id = ""
            st.session_state.user_role = "user"
            st.rerun()
        
        st.markdown("---")
        
        # Learning Progress Summary (using new system)
        st.markdown("### 📊 Learning Progress")
        if st.session_state.user_db_id and st.session_state.employee_id:
            # Show progress for each document
            if st.session_state.available_documents:
                for doc in st.session_state.available_documents[:3]:  # Show top 3
                    try:
                        progress = run_async(get_document_progress_new(st.session_state.employee_id, doc['id']))
                        overall = progress.get('overall_progress', 0)
                        fc_prog = progress.get('flashcard_progress', {})
                        quiz_prog = progress.get('quiz_progress', {})
                        
                        st.caption(f"📄 {doc['title'][:20]}...")
                        st.progress(overall / 100 if overall > 0 else 0)
                        col1, col2 = st.columns(2)
                        col1.caption(f"🃏 {fc_prog.get('completed', 0)}/{fc_prog.get('total', 0)}")
                        col2.caption(f"📝 {quiz_prog.get('passed_sets', 0)}/{quiz_prog.get('total_sets', 0)}")
                    except:
                        pass
                
                if len(st.session_state.available_documents) > 3:
                    st.caption(f"...and {len(st.session_state.available_documents) - 3} more")
            else:
                st.caption("No documents loaded. Click Refresh.")
        else:
            st.caption("Sign in to see progress")

    # Sign-in screen
    if not st.session_state.user_name or not st.session_state.employee_id:
        col_signin = st.columns([1, 2, 1])
        with col_signin[1]:
            st.markdown("""
            <div style="text-align: center; padding: 2rem 0;">
                <h2>👋 Welcome to DENSO-MIND</h2>
                <p style="color: #666;">Please sign in to start your training</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Dual login tabs: User and Admin
            login_tab1, login_tab2 = st.tabs(["👤 User Sign In", "🔐 Admin Sign In"])
            
            with login_tab1:
                with st.form("user_login_form"):
                    name = st.text_input("Full Name", placeholder="Enter your name")
                    emp_id = st.text_input("Employee ID", placeholder="Enter your employee ID")
                    user_saved = st.form_submit_button("🚀 Sign In as User", use_container_width=True)
                
                if user_saved and name.strip() and emp_id.strip():
                    st.session_state.user_name = name.strip()
                    st.session_state.employee_id = emp_id.strip()
                    st.session_state.user_role = "user"
                    # Save user to database
                    try:
                        user = run_async(get_or_create_user(emp_id.strip(), name.strip()))
                        st.session_state.user_db_id = str(user.id)
                        # Check if user has a role assigned
                        if hasattr(user, 'role') and user.role:
                            st.session_state.user_role = user.role.value
                        add_event("info", f"User '{name.strip()}' signed in.")
                    except Exception as e:
                        add_event("info", f"Profile saved (DB error: {e}).")
                    st.rerun()
                elif user_saved:
                    st.warning("Please fill in both Name and Employee ID.")
            
            with login_tab2:
                st.caption("🔒 Admin access requires password authentication")
                with st.form("admin_login_form"):
                    admin_emp_id = st.text_input("Admin Employee ID", placeholder="Enter admin employee ID")
                    admin_password = st.text_input("Password", type="password", placeholder="Enter password")
                    admin_saved = st.form_submit_button("🔐 Sign In as Admin", use_container_width=True)
                
                if admin_saved and admin_emp_id.strip() and admin_password:
                    try:
                        admin_user = run_async(authenticate_admin(admin_emp_id.strip(), admin_password))
                        if admin_user:
                            st.session_state.user_name = admin_user.full_name
                            st.session_state.employee_id = admin_user.employee_id
                            st.session_state.user_db_id = str(admin_user.id)
                            st.session_state.user_role = "admin"
                            add_event("info", f"Admin '{admin_user.full_name}' signed in.")
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials or insufficient permissions")
                    except Exception as e:
                        st.error(f"Authentication error: {e}")
                elif admin_saved:
                    st.warning("Please fill in Employee ID and Password.")
        st.stop()

    # Main content area after login
    if st.session_state.mode == "Practice":
        # Different tabs based on role
        if st.session_state.user_role == "admin":
            practice_tabs = st.tabs(["💬 Chat", "🃏 Flashcards", "📝 Quiz", "📤 Upload", "⚙️ Manage"])
        else:
            practice_tabs = st.tabs(["💬 Chat", "🃏 Flashcards", "📝 Quiz"])
        
        with practice_tabs[0]:
            st.markdown("### 💬 Chat with Documents")
            st.caption("Ask questions about your training materials")
            
            # Document filter
            col_filter, col_refresh = st.columns([4, 1])
            with col_refresh:
                if st.button("🔄", key="refresh_docs_chat", help="Refresh documents"):
                    try:
                        docs = run_async(get_all_documents())
                        st.session_state.available_documents = docs
                        add_event("info", f"Loaded {len(docs)} documents.")
                    except Exception as e:
                        st.error(f"Error loading documents: {e}")
            
            with col_filter:
                if st.session_state.available_documents:
                    doc_options = {f"{doc['title']} ({doc['num_pages']}p)": doc['id'] 
                                  for doc in st.session_state.available_documents}
                    selected_docs = st.multiselect(
                        "📚 Filter by document (optional)",
                        options=list(doc_options.keys()),
                        default=[],
                        key="doc_filter_chat",
                        placeholder="Search all documents"
                    )
                    st.session_state.selected_doc_ids = [doc_options[d] for d in selected_docs] if selected_docs else []
            
            st.markdown("---")
            
            # Display chat history
            if st.session_state.chat_history:
                for idx, msg in enumerate(st.session_state.chat_history):
                    if msg["role"] == "system":
                        continue
                    
                    if msg["role"] == "user":
                        st.markdown(f"**🧑 You:** {msg['content']}")
                    else:
                        st.markdown(f"**🤖 Assistant:** {msg['content']}")
                
                # Show citations for last response
                citations = st.session_state.get('chat_citations', {})
                if citations:
                    with st.expander(f"📚 {len(citations)} sources referenced", expanded=False):
                        docs_dict = {}
                        doc_info = {}
                        for idx, cite in citations.items():
                            doc_id = cite.get('document_id')
                            page = cite.get('page_number', 0)
                            doc_title = cite.get('document_title', f'Document {doc_id}')
                            file_type = cite.get('file_type', 'PDF')
                            doc_info[doc_id] = {'title': doc_title, 'file_type': file_type}
                            if doc_id not in docs_dict:
                                docs_dict[doc_id] = []
                            docs_dict[doc_id].append(page)
                        
                        for doc_id, pages in docs_dict.items():
                            info = doc_info.get(doc_id, {})
                            doc_title = info.get('title', f'Document {doc_id}')
                            file_type = info.get('file_type', 'PDF')
                            icon = "🎬" if file_type == 'VIDEO' else "📄"
                            unique_pages = sorted(set(pages))
                            st.markdown(f"**{icon} {doc_title}** - Pages: {', '.join(map(str, unique_pages))}")
                
                st.markdown("---")
            
            # Chat input
            user_query = st.text_input("Ask a question:", key="chat_input_simple", placeholder="Type your question here...")
            
            col_send, col_clear = st.columns([3, 1])
            with col_send:
                if st.button("📤 Send", type="primary", use_container_width=True, key="send_chat"):
                    if user_query and user_query.strip():
                        st.session_state.chat_history.append({"role": "user", "content": user_query.strip()})
                        
                        with st.spinner("Thinking..."):
                            try:
                                doc_ids = st.session_state.selected_doc_ids if st.session_state.selected_doc_ids else None
                                user_id = st.session_state.user_db_id
                                chat_hist = st.session_state.chat_history[:-1] if len(st.session_state.chat_history) > 1 else None
                                
                                # Run chat
                                async def run_chat():
                                    session_factory = get_async_session()
                                    async with session_factory() as session:
                                        chat_service = ChatService(session)
                                        full_response = ""
                                        citations = {}
                                        chunk_ids_for_tracking = []
                                        
                                        async for item in chat_service.chat_stream(
                                            user_query.strip(), 
                                            document_ids=doc_ids,
                                            chat_history=chat_hist
                                        ):
                                            if item["type"] == "text":
                                                full_response += item["content"]
                                            elif item["type"] == "citations":
                                                citations = item.get("data", {})
                                                chunk_ids_for_tracking = item.get("chunk_ids", [])
                                        
                                        # Track progress
                                        if user_id and chunk_ids_for_tracking:
                                            try:
                                                progress_service = ProgressService(session)
                                                await progress_service.record_chunk_interaction(
                                                    user_id=UUID(user_id),
                                                    chunk_ids=chunk_ids_for_tracking,
                                                    interaction_type='chat',
                                                    was_successful=True
                                                )
                                            except:
                                                pass
                                        
                                        return full_response, citations
                                
                                response, citations = run_async(run_chat())
                                st.session_state.chat_history.append({"role": "assistant", "content": response})
                                st.session_state.chat_citations = citations
                                add_event("chat", "RAG Chatbot responded.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                                st.session_state.chat_history.append({"role": "assistant", "content": f"❌ Error: {e}"})
            
            with col_clear:
                if st.button("🗑️ Clear", use_container_width=True, key="clear_chat"):
                    st.session_state.chat_history = []
                    st.session_state.chat_reply = ""
                    st.session_state.chat_citations = {}
                    st.rerun()

        with practice_tabs[1]:
            st.markdown("### 🃏 Study Flashcards")
            st.caption("Study with pre-generated flashcards. Mark cards as 'Known' to track your progress.")
            
            # Document selection
            col_doc, col_btn = st.columns([3, 1])
            with col_btn:
                if st.button("🔄 Refresh", key="refresh_docs_flash"):
                    try:
                        docs = run_async(get_all_documents())
                        st.session_state.available_documents = docs
                        add_event("info", f"Loaded {len(docs)} documents.")
                    except Exception as e:
                        st.error(f"Error loading documents: {e}")
            
            with col_doc:
                if st.session_state.available_documents:
                    doc_options_flash = {f"{doc['title']}": doc['id'] 
                                        for doc in st.session_state.available_documents}
                    selected_doc_flash = st.selectbox(
                        "Choose a document:",
                        options=["-- Select --"] + list(doc_options_flash.keys()),
                        key="doc_select_flash"
                    )
                else:
                    st.info("Click 'Refresh' to load documents.")
                    selected_doc_flash = None
            
            # Load flashcards when document is selected
            if st.session_state.available_documents and selected_doc_flash and selected_doc_flash != "-- Select --":
                doc_id = doc_options_flash[selected_doc_flash]
                
                col_gen, col_load = st.columns(2)
                with col_gen:
                    if st.button("✨ Generate", use_container_width=True, help="Generate new flashcards for this document"):
                        with st.spinner("Generating flashcards (this may take a minute)..."):
                            try:
                                result = run_async(generate_content_for_document(doc_id))
                                fc_count = result.get('flashcards_count', 0)
                                quiz_count = result.get('quiz_sets_count', 0)
                                add_event("flash", f"Generated {fc_count} flashcards & {quiz_count} quiz sets")
                                st.success(f"✅ Generated {fc_count} flashcards!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Generation error: {e}")
                
                with col_load:
                    if st.button("📚 Load", type="primary", use_container_width=True):
                        with st.spinner("Loading flashcards..."):
                            try:
                                flashcards = run_async(get_document_flashcards(doc_id))
                                if flashcards:
                                    st.session_state.document_flashcards = flashcards
                                    st.session_state.flashcard_index = 0
                                    st.session_state.flashcard_revealed = False
                                    st.session_state.current_flash_doc_id = doc_id
                                    add_event("flash", f"Loaded {len(flashcards)} flashcards.")
                                    st.rerun()
                                else:
                                    st.warning("No flashcards yet. Click 'Generate' first.")
                            except Exception as e:
                                st.error(f"Error loading flashcards: {e}")
                
                # Show progress for this document
                if st.session_state.employee_id and st.session_state.user_db_id:
                    try:
                        progress = run_async(get_document_progress_new(st.session_state.employee_id, doc_id))
                        fc_progress = progress.get('flashcard_progress', {})
                        completed = fc_progress.get('completed', 0)
                        total = fc_progress.get('total', 0)
                        percentage = fc_progress.get('progress_percentage', 0)
                        
                        st.markdown("#### 📊 Your Progress")
                        col_p1, col_p2 = st.columns([3, 1])
                        with col_p1:
                            st.progress(percentage / 100 if total > 0 else 0)
                        with col_p2:
                            st.caption(f"{completed}/{total} mastered")
                    except:
                        pass
            
            st.markdown("---")
            
            # Display flashcards with document viewer
            if st.session_state.document_flashcards:
                cards = st.session_state.document_flashcards
                idx = st.session_state.flashcard_index
                card = cards[idx]
                
                # Create two columns: flashcard and document viewer
                if st.session_state.show_doc_viewer:
                    flash_col, viewer_col = st.columns([1, 1])
                else:
                    flash_col = st.container()
                    viewer_col = None
                
                with flash_col:
                    # Card progress
                    st.caption(f"Card {idx + 1} of {len(cards)}")
                    
                    # Flashcard display
                    if not st.session_state.flashcard_revealed:
                        st.markdown(f"""
                        <div class="flashcard flashcard-front" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    color: white; padding: 2rem; border-radius: 15px; 
                                    min-height: 180px; margin: 1rem 0;">
                            <h3 style="margin: 0; color: white;">❓ {card.get('front', 'N/A')}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="flashcard flashcard-back" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                                    color: white; padding: 2rem; border-radius: 15px; 
                                    min-height: 180px; margin: 1rem 0;">
                            <h3 style="margin: 0; color: white;">✅ {card.get('back', 'N/A')}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Navigation and action buttons
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if st.button("⬅️ Prev", key="flash_prev", use_container_width=True):
                            st.session_state.flashcard_index = (idx - 1) % len(cards)
                            st.session_state.flashcard_revealed = False
                            st.session_state.doc_viewer_highlight = None
                            st.session_state.doc_viewer_source_chunk = None
                            st.rerun()
                    with col2:
                        btn_text = "🙈 Hide" if st.session_state.flashcard_revealed else "👁️ Show"
                        if st.button(btn_text, key="flash_reveal", use_container_width=True):
                            st.session_state.flashcard_revealed = not st.session_state.flashcard_revealed
                            # Mark as reviewed when revealed
                            if st.session_state.flashcard_revealed and st.session_state.employee_id:
                                try:
                                    run_async(mark_flashcard_reviewed(st.session_state.employee_id, card['id']))
                                except:
                                    pass
                            st.rerun()
                    with col3:
                        if st.button("✅ Know", key="flash_known", use_container_width=True, type="primary"):
                            # Mark as completed
                            if st.session_state.employee_id:
                                try:
                                    run_async(mark_flashcard_complete(st.session_state.employee_id, card['id'], ease=3))
                                    add_event("flash", f"Marked card {idx+1} as known")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                            # Move to next
                            st.session_state.flashcard_index = (idx + 1) % len(cards)
                            st.session_state.flashcard_revealed = False
                            st.session_state.doc_viewer_highlight = None
                            st.session_state.doc_viewer_source_chunk = None
                            st.rerun()
                    with col4:
                        if st.button("Next ➡️", key="flash_next", use_container_width=True):
                            st.session_state.flashcard_index = (idx + 1) % len(cards)
                            st.session_state.flashcard_revealed = False
                            st.session_state.doc_viewer_highlight = None
                            st.session_state.doc_viewer_source_chunk = None
                            st.rerun()
                    
                    # Find in Document button
                    col_find, col_view = st.columns(2)
                    with col_find:
                        if st.button("📍 Find in Document", key="find_in_doc", use_container_width=True):
                            chunk_ids = card.get('chunk_ids', [])
                            if chunk_ids:
                                try:
                                    chunks = run_async(get_chunk_content(chunk_ids))
                                    if chunks:
                                        # Load document content if not loaded
                                        if not st.session_state.doc_viewer_content or st.session_state.doc_viewer_content.get('id') != st.session_state.current_flash_doc_id:
                                            doc_content = run_async(get_document_content_for_viewer(st.session_state.current_flash_doc_id))
                                            st.session_state.doc_viewer_content = doc_content
                                        
                                        # Store the source chunk content for exact highlighting
                                        source_chunk_content = chunks[0].get('content', '')
                                        st.session_state.doc_viewer_highlight = card.get('back', '')
                                        st.session_state.doc_viewer_source_chunk = source_chunk_content
                                        st.session_state.doc_viewer_page = chunks[0].get('page_number', 1)
                                        st.session_state.show_doc_viewer = True
                                        add_event("flash", f"Found source in page {st.session_state.doc_viewer_page}")
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Error finding source: {e}")
                            else:
                                st.warning("No source location available for this card.")
                    
                    with col_view:
                        if st.session_state.show_doc_viewer:
                            if st.button("🔽 Hide Document", key="hide_doc", use_container_width=True):
                                st.session_state.show_doc_viewer = False
                                st.session_state.doc_viewer_highlight = None
                                st.session_state.doc_viewer_source_chunk = None
                                st.rerun()
                        else:
                            if st.button("📄 View Document", key="show_doc", use_container_width=True):
                                try:
                                    if not st.session_state.doc_viewer_content or st.session_state.doc_viewer_content.get('id') != st.session_state.current_flash_doc_id:
                                        doc_content = run_async(get_document_content_for_viewer(st.session_state.current_flash_doc_id))
                                        st.session_state.doc_viewer_content = doc_content
                                    # Don't set highlight when just viewing
                                    st.session_state.doc_viewer_highlight = None
                                    st.session_state.doc_viewer_source_chunk = None
                                    st.session_state.show_doc_viewer = True
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error loading document: {e}")
                
                # Document viewer panel
                if st.session_state.show_doc_viewer and viewer_col:
                    with viewer_col:
                        doc = st.session_state.doc_viewer_content
                        if doc and not doc.get('error'):
                            st.markdown(f"#### 📄 {doc.get('title', 'Document')}")
                            
                            file_type = doc.get('file_type', 'PDF')
                            file_path = doc.get('file_path')
                            highlight_text = st.session_state.doc_viewer_highlight
                            source_chunk = st.session_state.get('doc_viewer_source_chunk', '')
                            target_page = st.session_state.doc_viewer_page
                            
                            # Debug: show file path status
                            file_exists = file_path and os.path.exists(file_path)
                            
                            if file_type == 'PDF' or file_type is None:
                                if file_exists:
                                    # Tab for PDF view vs Text view
                                    view_mode = st.radio("View mode:", ["📄 PDF", "📝 Text"], horizontal=True, key="pdf_view_mode", label_visibility="collapsed")
                                
                                if view_mode == "📄 PDF":
                                    # Display actual PDF using iframe/embed
                                    try:
                                        import base64
                                        with open(file_path, "rb") as f:
                                            pdf_data = f.read()
                                        b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
                                        
                                        # Show page navigation info
                                        if target_page:
                                            st.info(f"📍 Source found on **Page {target_page}**")
                                        
                                        # Embed PDF with specific page
                                        page_param = f"#page={target_page}" if target_page else ""
                                        pdf_display = f'''
                                        <iframe 
                                            src="data:application/pdf;base64,{b64_pdf}{page_param}" 
                                            width="100%" 
                                            height="600" 
                                            type="application/pdf"
                                            style="border: 1px solid #444; border-radius: 8px;">
                                        </iframe>
                                        '''
                                        st.markdown(pdf_display, unsafe_allow_html=True)
                                        
                                        # Show highlight info below PDF
                                        if highlight_text:
                                            st.markdown("---")
                                            st.markdown("**🔍 Looking for:**")
                                            st.caption(highlight_text[:200] + ('...' if len(highlight_text) > 200 else ''))
                                    except Exception as e:
                                        st.error(f"Could not display PDF: {e}")
                                        # Fallback to text view
                                        view_mode = "📝 Text"
                                
                                if view_mode == "📝 Text":
                                    # Text content view with highlighting
                                    chunks = doc.get('chunks', [])
                                    
                                    # Group by page
                                    pages = {}
                                    for chunk in chunks:
                                        page_num = chunk.get('page_number', 1)
                                        if page_num not in pages:
                                            pages[page_num] = []
                                        pages[page_num].append(chunk.get('content', ''))
                                    
                                    # Page selector
                                    page_nums = sorted(pages.keys())
                                    if target_page and target_page in page_nums:
                                        default_idx = page_nums.index(target_page)
                                    else:
                                        default_idx = 0
                                    
                                    selected_page = st.selectbox(
                                        "Go to page:",
                                        page_nums,
                                        index=default_idx,
                                        key="text_view_page"
                                    )
                                    
                                    # Display selected page content
                                    page_content = "\n\n".join(pages.get(selected_page, []))
                                    
                                    # Apply highlighting if on target page
                                    if selected_page == target_page and (highlight_text or source_chunk):
                                        st.success(f"📍 **Source found on this page!**")
                                        
                                        # Find best matching text segments
                                        matches = find_best_matching_text(page_content, source_chunk, highlight_text)
                                        
                                        if matches:
                                            # Apply highlighting
                                            display_content = highlight_text_in_content(page_content, matches)
                                        else:
                                            # Fallback to keyword highlighting
                                            display_content = page_content
                                            keywords = [w for w in (highlight_text or '').split() if len(w) > 4][:6]
                                            for kw in keywords:
                                                try:
                                                    pattern = re.compile(f'({re.escape(kw)})', re.IGNORECASE)
                                                    display_content = pattern.sub(r'**\1**', display_content)
                                                except:
                                                    pass
                                        
                                        # Use native Streamlit container
                                        text_container = st.container(height=450)
                                        with text_container:
                                            st.markdown(display_content)
                                    else:
                                        # Use native Streamlit container
                                        text_container = st.container(height=450)
                                        with text_container:
                                            st.markdown(page_content)
                                else:
                                    # PDF file not found - show text from chunks
                                    st.caption(f"📝 Extracted Text (file not found: `{file_path}`)")
                                    chunks = doc.get('chunks', [])
                                    if chunks:
                                        pages = {}
                                        for chunk in chunks:
                                            page_num = chunk.get('page_number', 1)
                                            if page_num not in pages:
                                                pages[page_num] = []
                                            pages[page_num].append(chunk.get('content', ''))
                                        
                                        page_nums = sorted(pages.keys())
                                        default_idx = page_nums.index(target_page) if target_page in page_nums else 0
                                        selected_page = st.selectbox("Page:", page_nums, index=default_idx, key="fallback_page_flash")
                                        page_content = "\n\n".join(pages.get(selected_page, []))
                                        
                                        if selected_page == target_page and (highlight_text or source_chunk):
                                            st.success(f"📍 **Source found on this page!**")
                                            # Apply highlighting
                                            matches = find_best_matching_text(page_content, source_chunk, highlight_text)
                                            if matches:
                                                page_content = highlight_text_in_content(page_content, matches)
                                        
                                        text_container = st.container(height=450)
                                        with text_container:
                                            st.markdown(page_content)
                                    else:
                                        st.error("No content available.")
                            
                            elif file_type == 'VIDEO':
                                # Video transcript viewer
                                st.caption("🎬 Video Transcript")
                                
                                chunks = doc.get('chunks', [])
                                
                                # Create timeline view
                                if target_page is not None:
                                    mins = target_page // 60
                                    secs = target_page % 60
                                    st.info(f"📍 Source found at **[{mins}:{secs:02d}]**")
                                
                                # Scrollable transcript using native container
                                transcript_container = st.container(height=500)
                                with transcript_container:
                                    for chunk in chunks:
                                        page_num = chunk.get('page_number', 0)
                                        content = chunk.get('content', '')
                                        
                                        mins = page_num // 60
                                        secs = page_num % 60
                                        time_label = f"[{mins}:{secs:02d}]"
                                        is_target = (page_num == target_page)
                                        
                                        if is_target:
                                            st.success(f"**{time_label} 📍 Source**")
                                            st.markdown(content)
                                        else:
                                            with st.expander(time_label):
                                                st.markdown(content)
                        else:
                            st.error("Could not load document content.")
            else:
                st.info("Select a document and click 'Load Flashcards' to start studying.")
        
        # Quiz tab (index 2) - available to all users
        with practice_tabs[2]:
            st.markdown("### 📝 Quiz Assessment")
            st.caption("Take pre-generated quiz sets to test your knowledge.")
            
            # Document selection for quiz
            col_quiz_doc, col_quiz_refresh = st.columns([4, 1])
            with col_quiz_refresh:
                if st.button("🔄 Refresh", key="refresh_docs_quiz_practice"):
                    try:
                        docs = run_async(get_all_documents())
                        st.session_state.available_documents = docs
                        add_event("info", f"Loaded {len(docs)} documents.")
                    except Exception as e:
                        st.error(f"Error loading documents: {e}")
            
            with col_quiz_doc:
                if st.session_state.available_documents:
                    doc_options_quiz = {f"{doc['title']}": doc['id'] 
                                       for doc in st.session_state.available_documents}
                    selected_doc_quiz = st.selectbox(
                        "Select a document:",
                        options=["-- Select --"] + list(doc_options_quiz.keys()),
                        key="doc_select_quiz_practice"
                    )
                else:
                    st.info("Click 'Refresh' to load documents.")
                    selected_doc_quiz = None
            
            # Load quiz sets
            if st.session_state.available_documents and selected_doc_quiz and selected_doc_quiz != "-- Select --":
                doc_id = doc_options_quiz[selected_doc_quiz]
                
                if st.button("📥 Load Quiz Sets", key="load_quiz_practice", use_container_width=True):
                    try:
                        quiz_sets = run_async(get_document_quiz_sets(doc_id))
                        st.session_state.quiz_sets = quiz_sets
                        st.session_state.selected_quiz_set = None
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_result = None
                        if quiz_sets:
                            add_event("quiz", f"Loaded {len(quiz_sets)} quiz sets")
                        else:
                            st.info("No quiz sets available. Admin needs to generate them first.")
                    except Exception as e:
                        st.error(f"Error loading quiz sets: {e}")
            
            # Quiz set selection or active quiz
            if st.session_state.quiz_sets and not st.session_state.selected_quiz_set:
                st.markdown("---")
                st.markdown("#### Select a Quiz Set:")
                
                for idx, qs in enumerate(st.session_state.quiz_sets):
                    set_id = qs.get('id')
                    set_name = qs.get('title', f"Quiz Set {idx + 1}")
                    questions = qs.get('questions', [])
                    attempt_key = f"quiz_attempt_{set_id}"
                    is_passed = st.session_state.get(attempt_key, {}).get('passed', False)
                    
                    col_set, col_btn = st.columns([3, 1])
                    with col_set:
                        if is_passed:
                            st.markdown(f"✅ **{set_name}** ({len(questions)} questions) - Passed!")
                        else:
                            st.markdown(f"📋 **{set_name}** ({len(questions)} questions)")
                    with col_btn:
                        btn_label = "✅ Passed" if is_passed else "Start"
                        btn_type = "secondary" if is_passed else "primary"
                        if st.button(btn_label, key=f"start_set_{set_id}", use_container_width=True, type=btn_type):
                            st.session_state.selected_quiz_set = qs
                            st.session_state.quiz_answers = {}
                            st.session_state.quiz_result = None
                            st.rerun()
            
            # Active quiz
            elif st.session_state.selected_quiz_set:
                qs = st.session_state.selected_quiz_set
                questions = qs.get('questions', [])
                set_id = qs.get('id')
                
                st.markdown(f"### {qs.get('title', 'Quiz')}")
                
                # Back button
                if st.button("⬅️ Back to Set Selection"):
                    st.session_state.selected_quiz_set = None
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_result = None
                    st.rerun()
                
                st.markdown("---")
                
                if not st.session_state.quiz_result:
                    # Show questions one by one style but all visible
                    for i, q in enumerate(questions):
                        st.markdown(f"**Question {i + 1}:** {q.get('question', 'N/A')}")
                        
                        options = q.get('options', [])
                        q_id = q.get('id', i)
                        
                        # Use index=None to prevent auto-selection
                        selected = st.radio(
                            f"Select answer for Q{i+1}:",
                            options=options,
                            key=f"quiz_q_{set_id}_{i}",
                            index=None,
                            label_visibility="collapsed"
                        )
                        if selected:
                            st.session_state.quiz_answers[q_id] = selected
                        
                        st.markdown("---")
                    
                    # Progress and submit
                    answered = len(st.session_state.quiz_answers)
                    total = len(questions)
                    st.progress(answered / total if total > 0 else 0)
                    st.caption(f"Answered: {answered}/{total}")
                    
                    if st.button("📊 Submit Quiz", type="primary", use_container_width=True):
                        if answered < total:
                            st.warning("Please answer all questions before submitting.")
                        else:
                            # Grade the quiz
                            score = 0
                            for q in questions:
                                q_id = q.get('id', questions.index(q))
                                user_ans = st.session_state.quiz_answers.get(q_id)
                                correct_ans = q.get('answer', q.get('correct_answer', ''))
                                if user_ans == correct_ans:
                                    score += 1
                            
                            # Submit attempt
                            try:
                                run_async(submit_quiz_attempt(
                                    st.session_state.employee_id,
                                    set_id,
                                    st.session_state.quiz_answers,
                                    score,
                                    total
                                ))
                            except:
                                pass
                            
                            passed = (score / total * 100) >= 60 if total > 0 else False
                            st.session_state.quiz_result = {
                                "score": score,
                                "total": total,
                                "passed": passed,
                                "questions": questions
                            }
                            st.session_state[f"quiz_attempt_{set_id}"] = {
                                'score': int(score / total * 100) if total > 0 else 0,
                                'passed': passed
                            }
                            add_event("quiz", f"Quiz: {score}/{total} - {'PASSED' if passed else 'FAILED'}")
                            st.rerun()
                
                else:
                    # Show results
                    result = st.session_state.quiz_result
                    if result["passed"]:
                        st.balloons()
                        st.success(f"🎉 **PASSED!** {result['score']}/{result['total']} correct ({int(result['score']/result['total']*100)}%)")
                    else:
                        st.error(f"❌ **FAILED:** {result['score']}/{result['total']} correct ({int(result['score']/result['total']*100)}%) - 60% needed")
                    
                    # Show review
                    st.markdown("#### Review Answers:")
                    for i, q in enumerate(result.get('questions', [])):
                        q_id = q.get('id', i)
                        user_ans = st.session_state.quiz_answers.get(q_id, "N/A")
                        correct_ans = q.get('answer', q.get('correct_answer', ''))
                        is_correct = user_ans == correct_ans
                        
                        icon = "✅" if is_correct else "❌"
                        with st.expander(f"{icon} Q{i+1}: {q.get('question', '')[:50]}..."):
                            st.write(f"**Your answer:** {user_ans}")
                            st.write(f"**Correct answer:** {correct_ans}")
                            if q.get('explanation'):
                                st.info(f"💡 {q.get('explanation')}")
                    
                    # Retry or continue
                    col_retry, col_next = st.columns(2)
                    with col_retry:
                        if st.button("🔄 Retry This Set", use_container_width=True):
                            st.session_state.quiz_answers = {}
                            st.session_state.quiz_result = None
                            st.rerun()
                    with col_next:
                        if st.button("➡️ Next Set", use_container_width=True):
                            st.session_state.selected_quiz_set = None
                            st.session_state.quiz_answers = {}
                            st.session_state.quiz_result = None
                            st.rerun()
        
        # Upload tab (index 3) - admin only
        if st.session_state.user_role == "admin":
            with practice_tabs[3]:
                st.markdown("### Upload Training Documents")
                st.caption("Upload PDF or video files to add them to the knowledge base.")
                
                # File type selection
                file_type = st.radio("File type:", ["PDF", "Video"], horizontal=True, key="upload_file_type")
                
                if file_type == "PDF":
                    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_uploader")
                    
                    if uploaded_file is not None:
                        col_info, col_action = st.columns([3, 1])
                        with col_info:
                            st.info(f"📄 **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
                        with col_action:
                            if st.button("📤 Upload", type="primary", use_container_width=True):
                                progress_bar = st.progress(0, text="Starting ingestion...")
                                try:
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                        tmp_file.write(uploaded_file.getvalue())
                                        tmp_path = tmp_file.name
                                    
                                    progress_bar.progress(20, text="Processing PDF...")
                                    result = run_async(ingest_pdf_file(tmp_path, uploaded_file.name))
                                    os.unlink(tmp_path)
                                    
                                    if result.get('status') == 'SUCCESS' and result.get('document_id'):
                                        # Auto-generate learning content
                                        progress_bar.progress(50, text="Generating flashcards...")
                                        try:
                                            content_result = run_async(generate_content_for_document(result['document_id']))
                                            progress_bar.progress(100, text="Complete!")
                                            result['flashcards_generated'] = content_result.get('flashcards_count', 0)
                                            result['quiz_sets_generated'] = content_result.get('quiz_sets_count', 0)
                                            add_event("info", f"Generated {content_result.get('flashcards_count', 0)} flashcards & {content_result.get('quiz_sets_count', 0)} quiz sets")
                                        except Exception as e:
                                            st.warning(f"Document ingested but content generation failed: {e}")
                                    
                                    st.session_state.ingestion_status = result
                                    add_event("info", f"Ingested: {uploaded_file.name} ({result['num_chunks']} chunks)")
                                    st.success(f"✅ Document '{result['title']}' ingested! Created {result['num_chunks']} chunks.")
                                except Exception as e:
                                    st.error(f"Ingestion error: {e}")
                else:
                    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv", "webm"], key="video_uploader")
                    uploaded_subtitle = st.file_uploader("(Optional) Upload subtitle file", type=["srt", "vtt"], key="subtitle_uploader")
                    
                    if uploaded_video:
                        col_info, col_action = st.columns([3, 1])
                        with col_info:
                            st.info(f"🎬 **{uploaded_video.name}** ({uploaded_video.size / (1024*1024):.1f} MB)")
                            if uploaded_subtitle:
                                st.caption(f"📝 Subtitle: {uploaded_subtitle.name}")
                        with col_action:
                            if st.button("📤 Upload", type="primary", use_container_width=True, key="upload_video_btn"):
                                with st.spinner("Processing video and extracting transcript..."):
                                    try:
                                        # Save video file
                                        video_suffix = os.path.splitext(uploaded_video.name)[1]
                                        with tempfile.NamedTemporaryFile(delete=False, suffix=video_suffix) as tmp_video:
                                            tmp_video.write(uploaded_video.getvalue())
                                            video_path = tmp_video.name
                                        
                                        # Save subtitle file if provided
                                        subtitle_path = None
                                        if uploaded_subtitle:
                                            sub_suffix = os.path.splitext(uploaded_subtitle.name)[1]
                                            with tempfile.NamedTemporaryFile(delete=False, suffix=sub_suffix) as tmp_sub:
                                                tmp_sub.write(uploaded_subtitle.getvalue())
                                                subtitle_path = tmp_sub.name
                                        
                                        result = run_async(ingest_video_file(video_path, uploaded_video.name, subtitle_path))
                                        
                                        # Cleanup temp files
                                        os.unlink(video_path)
                                        if subtitle_path:
                                            os.unlink(subtitle_path)
                                        
                                        st.session_state.ingestion_status = result
                                        
                                        if result['status'] == 'SUCCESS':
                                            duration_str = ""
                                            if result.get('duration_seconds'):
                                                mins = result['duration_seconds'] // 60
                                                secs = result['duration_seconds'] % 60
                                                duration_str = f" ({mins}:{secs:02d})"
                                            add_event("info", f"Ingested video: {uploaded_video.name}{duration_str} ({result['num_chunks']} chunks)")
                                            st.success(f"✅ Video '{result['title']}' processed! Created {result['num_chunks']} transcript chunks.")
                                        else:
                                            st.error(f"❌ Failed to process video: {result.get('error', 'Unknown error')}")
                                    except Exception as e:
                                        st.error(f"Video ingestion error: {e}")
                
                if st.session_state.ingestion_status:
                    with st.expander("Last ingestion result", expanded=False):
                        st.json(st.session_state.ingestion_status)
            
            # Manage tab (index 4) - admin only
            with practice_tabs[4]:
                st.markdown("### 📚 Document Management")
                st.caption("View, manage, and delete documents and their learning content.")
                
                col_header, col_refresh = st.columns([4, 1])
                with col_refresh:
                    if st.button("🔄 Refresh", key="refresh_docs_view"):
                        try:
                            docs = run_async(get_all_documents())
                            st.session_state.available_documents = docs
                            add_event("info", f"Loaded {len(docs)} documents.")
                        except Exception as e:
                            st.error(f"Error loading documents: {e}")
                
                if st.session_state.available_documents:
                    for doc in st.session_state.available_documents:
                        # Format display based on file type
                        is_video = doc.get('file_type') == 'VIDEO'
                        if is_video:
                            duration = doc.get('duration_seconds', 0)
                            if duration:
                                mins = duration // 60
                                secs = duration % 60
                                doc_label = f"🎬 {doc['title']} ({mins}:{secs:02d})"
                            else:
                                doc_label = f"🎬 {doc['title']} ({doc['num_pages']} segments)"
                        else:
                            doc_label = f"📄 {doc['title']} ({doc['num_pages']} pages)"
                        
                        # Add status indicators
                        fc_status = "✅" if doc.get('flashcards_generated') else "❌"
                        quiz_status = "✅" if doc.get('quizzes_generated') else "❌"
                        doc_label = f"{doc_label} | FC:{fc_status} Quiz:{quiz_status}"
                        
                        with st.expander(doc_label):
                            col_info, col_actions = st.columns([2, 1])
                            
                            with col_info:
                                st.write(f"**ID:** {doc['id']}")
                                st.write(f"**Type:** {doc.get('file_type', 'PDF')}")
                                st.write(f"**Flashcards:** {'Generated' if doc.get('flashcards_generated') else 'Not generated'}")
                                st.write(f"**Quiz Sets:** {'Generated' if doc.get('quizzes_generated') else 'Not generated'}")
                            
                            with col_actions:
                                if st.button("🗑️ Delete", key=f"del_doc_{doc['id']}", type="primary", use_container_width=True):
                                    st.session_state[f"confirm_del_doc_{doc['id']}"] = True
                                
                                if st.session_state.get(f"confirm_del_doc_{doc['id']}", False):
                                    st.error("⚠️ DELETE?")
                                    col_yes, col_no = st.columns(2)
                                    with col_yes:
                                        if st.button("✅ Yes", key=f"confirm_del_doc_yes_{doc['id']}"):
                                            try:
                                                result = run_async(delete_document(doc['id']))
                                                if result['success']:
                                                    add_event("info", f"Deleted: {result['title']}")
                                                    st.session_state[f"confirm_del_doc_{doc['id']}"] = False
                                                    st.session_state.available_documents = run_async(get_all_documents())
                                                    st.rerun()
                                            except Exception as e:
                                                st.error(f"Error: {e}")
                                    with col_no:
                                        if st.button("❌ No", key=f"confirm_del_doc_no_{doc['id']}"):
                                            st.session_state[f"confirm_del_doc_{doc['id']}"] = False
                                            st.rerun()
                else:
                    st.info("No documents yet. Upload documents in the Upload tab.")

    # My Results mode - For users to view their learning and assessment progress
    if st.session_state.mode == "MyResults":
        st.markdown("### 📊 My Results")
        st.caption("View your learning progress and assessment results")
        
        results_tabs = st.tabs(["📚 Learning Progress", "🎥 Assessment History"])
        
        with results_tabs[0]:
            st.markdown("#### 📚 Learning Progress")
            
            # Refresh documents button
            if st.button("🔄 Refresh", key="refresh_results_docs"):
                try:
                    docs = run_async(get_all_documents())
                    st.session_state.available_documents = docs
                except Exception as e:
                    st.error(f"Error loading documents: {e}")
            
            if st.session_state.available_documents:
                for doc in st.session_state.available_documents:
                    doc_id = doc['id']
                    try:
                        progress = run_async(get_document_progress_new(st.session_state.employee_id, doc_id))
                        
                        # Document progress card
                        with st.expander(f"📄 {doc['title']}", expanded=False):
                            # Flashcard progress
                            fc_prog = progress.get('flashcard_progress', {})
                            fc_reviewed = fc_prog.get('reviewed_count', 0)
                            fc_total = fc_prog.get('total_count', 0)
                            fc_pct = fc_prog.get('progress_percentage', 0)
                            
                            st.markdown("**🃏 Flashcards**")
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.progress(fc_pct / 100 if fc_total > 0 else 0)
                            with col2:
                                st.caption(f"{fc_reviewed}/{fc_total}")
                            
                            # Quiz progress
                            quiz_prog = progress.get('quiz_progress', {})
                            quiz_passed = quiz_prog.get('passed_sets', 0)
                            quiz_total = quiz_prog.get('total_sets', 0)
                            quiz_pct = quiz_prog.get('progress_percentage', 0)
                            
                            st.markdown("**📝 Quizzes**")
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.progress(quiz_pct / 100 if quiz_total > 0 else 0)
                            with col2:
                                st.caption(f"{quiz_passed}/{quiz_total} passed")
                            
                            # Quiz set details
                            if quiz_prog.get('sets_detail'):
                                st.markdown("**Quiz Sets:**")
                                for detail in quiz_prog['sets_detail']:
                                    status_icon = "✅" if detail.get('passed') else "❌"
                                    attempts = detail.get('attempts', 0)
                                    best_score = detail.get('best_score', 0)
                                    st.caption(f"{status_icon} Set {detail.get('set_number', '?')}: Best {best_score}% ({attempts} attempts)")
                    except Exception as e:
                        with st.expander(f"📄 {doc['title']}", expanded=False):
                            st.caption("No progress data available")
            else:
                st.info("No documents available. Click 'Refresh' to load.")
        
        with results_tabs[1]:
            st.markdown("#### 🎥 Assessment History")
            st.caption("Your SOP video assessment results")
            
            if st.button("🔄 Load History", key="load_my_assessment_history"):
                try:
                    history = run_async(get_user_assessment_history(
                        st.session_state.employee_id, 
                        limit=20
                    ))
                    st.session_state.my_assessment_history = history
                except Exception as e:
                    st.error(f"Failed to load history: {e}")
            
            if st.session_state.get('my_assessment_history'):
                for assess in st.session_state.my_assessment_history:
                    status_icon = "✅" if assess['status'] == 'PASSED' else "❌"
                    score_pct = assess.get('score', 0)
                    
                    with st.expander(
                        f"{status_icon} {assess.get('process_name', 'Assessment')} - {score_pct:.0f}% | {assess.get('created_at', 'N/A')[:10] if assess.get('created_at') else 'N/A'}",
                        expanded=False
                    ):
                        col_info, col_action = st.columns([3, 1])
                        with col_info:
                            st.markdown(f"**Session:** `{assess.get('session_code', 'N/A')}`")
                            st.markdown(f"**Steps:** {assess.get('completed_steps', 0)}/{assess.get('total_steps', 0)}")
                            st.markdown(f"**Duration:** {assess.get('total_duration', 0):.1f}s")
                            st.markdown(f"**Video:** {assess.get('video_filename', 'N/A')}")
                        
                        with col_action:
                            if st.button("👁️ Details", key=f"view_my_assess_{assess['id']}", use_container_width=True):
                                try:
                                    details = run_async(get_assessment_details(assess['id']))
                                    st.session_state[f"my_assess_detail_{assess['id']}"] = details
                                except Exception as e:
                                    st.error(f"Error: {e}")
                        
                        # Show details if loaded
                        if st.session_state.get(f"my_assess_detail_{assess['id']}"):
                            detail = st.session_state[f"my_assess_detail_{assess['id']}"]
                            st.markdown("---")
                            
                            # Step Details with full info
                            st.markdown("**📋 Step Details:**")
                            for step in detail.get('step_details', []):
                                step_passed = step.get('status') == 'PASSED'
                                step_icon = "✅" if step_passed else "❌"
                                with st.expander(f"{step_icon} {step.get('step_id')}: {step.get('description', 'N/A')[:40]}...", expanded=not step_passed):
                                    st.markdown(f"**Description:** {step.get('description', 'N/A')}")
                                    st.markdown(f"**Target Object:** `{step.get('target_object', 'N/A')}`")
                                    st.markdown(f"**Status:** {'✅ PASSED' if step_passed else '❌ FAILED'}")
                                    st.markdown(f"**Duration:** {step.get('duration', 0):.1f}s")
                                    if step.get('timestamp'):
                                        st.caption(f"Timestamp: {step.get('timestamp')}")
                            
                            # Show AI Feedback if available
                            if detail.get('feedback'):
                                st.markdown("---")
                                st.markdown("**🤖 AI Feedback:**")
                                try:
                                    feedback_data = json.loads(detail['feedback']) if isinstance(detail['feedback'], str) else detail['feedback']
                                    
                                    # Rating only (using original score)
                                    rating = feedback_data.get('overall_rating', 'N/A')
                                    rating_emoji = {
                                        'excellent': '🌟',
                                        'good': '👍',
                                        'average': '📊',
                                        'needs_improvement': '📈'
                                    }.get(rating, '📋')
                                    st.markdown(f"**Rating:** {rating_emoji} {rating.replace('_', ' ').title()}")
                                    
                                    if feedback_data.get('summary'):
                                        st.info(f"📝 {feedback_data['summary']}")
                                    
                                    if feedback_data.get('strengths'):
                                        st.markdown("**✅ Strengths:**")
                                        for s in feedback_data['strengths']:
                                            st.markdown(f"- {s}")
                                    
                                    if feedback_data.get('areas_for_improvement'):
                                        st.markdown("**📈 Areas for Improvement:**")
                                        for a in feedback_data['areas_for_improvement']:
                                            st.markdown(f"- {a}")
                                    
                                    # Learning Resources (Based on Failed Steps)
                                    if feedback_data.get('learning_resources'):
                                        st.markdown("**📚 Learning Resources (What to Study):**")
                                        for resource in feedback_data['learning_resources']:
                                            with st.expander(f"📖 {resource.get('step_name', resource.get('step_id', 'Unknown'))}", expanded=True):
                                                st.markdown(f"**What to Learn:** {resource.get('what_to_learn', 'N/A')}")
                                                st.markdown(f"**How to Practice:** {resource.get('how_to_practice', 'N/A')}")
                                    
                                    if feedback_data.get('specific_recommendations'):
                                        st.markdown("**💡 Recommendations:**")
                                        for rec in feedback_data['specific_recommendations']:
                                            priority = rec.get('priority', '')
                                            p_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(priority, '⚪')
                                            st.markdown(f"{p_icon} **{rec.get('area', '')}:** {rec.get('recommendation', '')}")
                                    
                                    if feedback_data.get('next_steps'):
                                        st.markdown("**🎯 Next Steps:**")
                                        for ns in feedback_data['next_steps']:
                                            st.markdown(f"- {ns}")
                                    
                                    if feedback_data.get('encouragement'):
                                        st.success(f"💪 {feedback_data['encouragement']}")
                                    
                                    # Statistics (collapsible)
                                    if feedback_data.get('statistics'):
                                        with st.expander("📊 Detailed Statistics", expanded=False):
                                            st.json(feedback_data['statistics'])
                                except Exception as e:
                                    st.caption(f"Feedback parsing error: {e}")
                            
                            # Show report_data if has failed_steps
                            if detail.get('report_data'):
                                report = detail['report_data']
                                if report.get('failed_steps'):
                                    st.markdown("---")
                                    st.markdown("**❌ Failed Steps:**")
                                    for failed in report['failed_steps']:
                                        with st.expander(f"🚫 {failed.get('step_id', 'Unknown')} - {failed.get('error_type', 'ERROR')}", expanded=True):
                                            st.error(f"**Error:** {failed.get('error_details', 'No details')}")
                                            st.markdown(f"**Expected:** {failed.get('description', 'N/A')}")
                                            if failed.get('wrong_step_id'):
                                                st.warning(f"**Wrong step touched:** {failed.get('wrong_step_description', failed.get('wrong_step_id'))}")
                                            if failed.get('instructions'):
                                                st.markdown("**Instructions:**")
                                                for instr in failed.get('instructions', []):
                                                    st.markdown(f"- {instr}")
                                
                                # Show error logs if any
                                if report.get('error_logs'):
                                    with st.expander("⚠️ Error Logs", expanded=False):
                                        for err in report['error_logs']:
                                            st.warning(f"**{err.get('error_type', 'ERROR')}** at {err.get('timestamp', 'N/A')}: {err.get('details', 'No details')}")
            else:
                st.caption("Click 'Load History' to view past assessments")

    # Testing mode - Admin only (Video Assessment)
    if st.session_state.mode == "Testing" and st.session_state.user_role == "admin":
        st.markdown("### 🎥 Video-Based SOP Assessment")
        st.caption("Upload a video of your work procedure for AI-powered assessment")

        # Check if SOP module is available
        if not SOP_AVAILABLE:
            st.error("⚠️ SOP Assessment module is not installed. Required packages: ultralytics, mediapipe, opencv-python")
            st.code("pip install ultralytics mediapipe opencv-python lapx", language="bash")
        else:
            # SOP Rules Section
            col_rules, col_video = st.columns([1, 2])
            
            with col_rules:
                st.markdown("#### 📋 SOP Rules")
                
                # Load default rules or upload custom
                rules_source = st.radio(
                    "Select SOP rules:",
                    ["Default Rules", "Upload Custom Rules"],
                    key="sop_rules_source",
                    horizontal=True
                )
                
                if rules_source == "Default Rules":
                    default_rules_path = Path("data/sop_rulesv3.json")
                    if default_rules_path.exists():
                        try:
                            with open(default_rules_path, 'r', encoding='utf-8') as f:
                                st.session_state.sop_rules = json.load(f)
                            st.success(f"✅ Loaded: {st.session_state.sop_rules.get('process_name', 'Unknown')}")
                        except Exception as e:
                            st.error(f"Failed to load rules: {e}")
                    else:
                        st.warning("Default rules file not found at data/sop_rules.json")
                else:
                    uploaded_rules = st.file_uploader(
                        "Upload SOP rules JSON",
                        type=["json"],
                        key="sop_rules_uploader"
                    )
                    if uploaded_rules:
                        try:
                            st.session_state.sop_rules = json.loads(uploaded_rules.read().decode('utf-8'))
                            st.success(f"✅ Loaded: {st.session_state.sop_rules.get('process_name', 'Unknown')}")
                        except Exception as e:
                            st.error(f"Invalid JSON: {e}")
                
                # Display loaded rules
                if st.session_state.sop_rules:
                    rules = st.session_state.sop_rules
                    st.markdown(f"**Process:** {rules.get('process_name', 'Unknown')}")
                    st.markdown(f"**Steps:** {len(rules.get('steps', []))}")
                    
                    with st.expander("📜 View Steps", expanded=False):
                        for i, step in enumerate(rules.get('steps', [])):
                            st.markdown(f"**{i+1}. {step.get('step_id', f'Step {i+1}')}**")
                            st.caption(f"  {step.get('description', 'No description')}")
                            st.caption(f"  Target: `{step.get('target_object', 'N/A')}`")
                
                # Activity Log Section (updates during assessment)
                st.markdown("---")
                st.markdown("#### 📊 Activity Log")
                activity_log_placeholder = st.empty()
            
            with col_video:
                st.markdown("#### 🎬 Video Upload & Assessment")
                
                uploaded_assessment_video = st.file_uploader(
                    "Upload assessment video",
                    type=["mp4", "mov", "avi", "mkv"],
                    key="assessment_video_uploader"
                )
                
                if uploaded_assessment_video:
                    video_size_mb = uploaded_assessment_video.size / (1024*1024)
                    st.info(f"🎬 **{uploaded_assessment_video.name}** ({video_size_mb:.1f} MB)")
                    
                    # Model selection
                    model_col1, model_col2 = st.columns(2)
                    with model_col1:
                        yolo_model = st.selectbox(
                            "YOLO Model",
                            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "custom"],
                            help="Select detection model (n=nano, s=small, m=medium)"
                        )
                    with model_col2:
                        confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)
                    
                    # Custom model upload
                    custom_model_path = None
                    if yolo_model == "custom":
                        uploaded_weights = st.file_uploader(
                            "Upload custom YOLO weights (.pt)",
                            type=["pt"],
                            key="custom_weights_uploader"
                        )
                        if uploaded_weights:
                            # Save weights to temp file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                                tmp.write(uploaded_weights.getbuffer())
                                custom_model_path = tmp.name
                            st.success(f"✅ Loaded: {uploaded_weights.name}")
                        else:
                            st.warning("⚠️ Please upload custom weights file")
                    
                    # Assessment controls
                    if st.session_state.sop_rules:
                        can_start = yolo_model != "custom" or custom_model_path is not None
                        if not st.session_state.assessment_running:
                            if st.button("🎯 Start Assessment", type="primary", use_container_width=True, disabled=not can_start):
                                st.session_state.assessment_running = True
                                st.session_state.assessment_result = None
                                st.session_state.custom_model_path = custom_model_path
                                st.session_state.selected_yolo_model = yolo_model
                                st.session_state.assessment_video_name = uploaded_assessment_video.name
                                
                                # Save video to temp file
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                                    tmp.write(uploaded_assessment_video.getbuffer())
                                    st.session_state.assessment_video_path = tmp.name
                                
                                add_event("sop", f"Started SOP assessment: {uploaded_assessment_video.name}")
                                st.rerun()
                        else:
                            if st.button("⏹️ Stop Assessment", type="secondary", use_container_width=True):
                                st.session_state.assessment_running = False
                                st.rerun()
                    else:
                        st.warning("⚠️ Please load SOP rules first")
                
                # Run assessment if started
                if st.session_state.assessment_running and st.session_state.assessment_video_path:
                    st.markdown("---")
                    st.markdown("#### 🔄 Assessment in Progress...")
                    
                    video_frame = st.empty()
                    
                    try:
                        # Initialize engines - use custom model if uploaded
                        model_to_use = st.session_state.get('selected_yolo_model', 'yolov8n.pt')
                        if st.session_state.get('custom_model_path'):
                            model_to_use = st.session_state.custom_model_path
                            add_event("sop", f"Using custom model: {model_to_use}")
                        
                        vision_engine = VisionEngine(model_path=model_to_use)
                        sop_engine = SOPEngine(rules_data=st.session_state.sop_rules)
                        interaction_engine = InteractionEngine()
                        
                        # Process video using generator for smooth processing
                        import cv2
                        frame_count = 0
                        last_ui_update = 0
                        activity_logs = []  # Store activity log entries
                        
                        # Get current target object for filtering
                        current_target_label = sop_engine.steps[sop_engine.current_step_index]['target_object']
                        
                        for frame, detected_objects, detected_hands in vision_engine.process_stream(st.session_state.assessment_video_path):
                            if not st.session_state.assessment_running:
                                break
                                
                            frame_count += 1
                            
                            # Update target label when step changes
                            if not sop_engine.is_completed:
                                current_target_label = sop_engine.steps[sop_engine.current_step_index]['target_object']
                            
                            # Filter objects to only relevant ones for current step
                            relevant_objects = [obj for obj in detected_objects if obj['label'] == current_target_label]
                            
                            # Check interactions between hands and relevant objects
                            interactions = interaction_engine.process(relevant_objects, detected_hands)
                            
                            # Update SOP state
                            sop_status = sop_engine.update(interactions)
                            
                            # Check for EXIT_APP signal (fatal error timeout)
                            if sop_status.get('signal') == "EXIT_APP":
                                add_event("sop", f"Assessment TERMINATED due to fatal error: {sop_status.get('error_type', 'UNKNOWN')}")
                                break
                            
                            # Update UI every 5 frames for better responsiveness
                            if frame_count - last_ui_update >= 5:
                                last_ui_update = frame_count
                                
                                # Update activity log on the left column
                                with activity_log_placeholder.container():
                                    # Check for SHOW_FATAL_ERROR signal
                                    if sop_status.get('signal') == "SHOW_FATAL_ERROR":
                                        st.error("⚠️ **CẢNH BÁO SAI QUY TRÌNH!**")
                                        st.markdown(f"**Lỗi:** {sop_status.get('error_details', 'Unknown error')}")
                                        st.markdown(f"**Hệ thống sẽ thoát sau:** {sop_status.get('countdown', 0):.1f}s")
                                        st.progress(sop_status.get('timer_ratio', 0))
                                        
                                        # Show instructions
                                        instructions = sop_status.get('instructions', [])
                                        if instructions:
                                            st.markdown("---")
                                            st.markdown("**📋 Hướng dẫn thao tác đúng:**")
                                            for inst in instructions:
                                                st.info(inst)
                                    else:
                                        # Normal display
                                        # Overall stages progress bar (always visible)
                                        step_idx = sop_status['step_index']
                                        total_steps = sop_status['total_steps']
                                        overall_progress = step_idx / total_steps if total_steps > 0 else 0
                                        st.markdown("**📊 Overall Progress:**")
                                        st.progress(overall_progress)
                                        st.caption(f"Completed: {step_idx}/{total_steps} steps")
                                        
                                        st.markdown("---")
                                        
                                        # Current step info
                                        st.markdown(f"**Step {step_idx+1}/{total_steps}**")
                                        st.markdown(f"📌 **Current:** {sop_status.get('description', 'N/A')}")
                                        st.markdown(f"🎯 **Target:** `{current_target_label}`")
                                        st.markdown(f"📍 **State:** {sop_status.get('state', 'WAITING')}")
                                        st.markdown(f"📝 **Status:** {sop_status.get('status', 'Unknown')}")
                                        
                                        # Show warning if wrong step detected
                                        if sop_status.get('wrong_step_warning'):
                                            st.warning("⚠️ Đang chạm sai đối tượng!")
                                        
                                        # Step timer progress bar (shows detection/working/cooldown progress)
                                        timer_ratio = sop_status.get('timer_ratio', 0)
                                        st.markdown("**⏱️ Step Progress:**")
                                        st.progress(timer_ratio if timer_ratio > 0 else 0.0)
                                        
                                        # Detection info
                                        st.caption(f"🖐️ Hands: {len(detected_hands)} | 📦 Objects: {len(detected_objects)} | 🤝 Interactions: {len(interactions)}")
                                        
                                        # Debug: Show detected object labels
                                        if detected_objects:
                                            obj_labels = [f"{o['label']}" for o in detected_objects]
                                            st.caption(f"📦 Detected: {', '.join(obj_labels)}")
                                        else:
                                            st.caption("📦 No objects detected")
                                        st.caption(f"Frame: {frame_count}")
                                        
                                        # Completed steps
                                        if sop_engine.report_logs:
                                            st.markdown("---")
                                            st.markdown("**✅ Completed:**")
                                            for log_entry in sop_engine.report_logs:
                                                st.success(f"{log_entry['step_id']}: {log_entry.get('duration', 0):.1f}s")
                                
                                # Draw frame with detections (clean video without overlay text)
                                display_frame = frame.copy()
                                
                                # Draw pinch points for hands (red dot - interaction point)
                                for hand in detected_hands:
                                    # Draw pinch point (between thumb and index)
                                    thumb_tip = hand.get('thumb_tip')
                                    index_tip = hand.get('index_tip')
                                    if thumb_tip and index_tip:
                                        cx = (thumb_tip[0] + index_tip[0]) // 2
                                        cy = (thumb_tip[1] + index_tip[1]) // 2
                                        cv2.circle(display_frame, (cx, cy), 8, (0, 0, 255), -1)  # Red pinch point
                                    
                                    # Draw hand bbox
                                    if 'bbox' in hand:
                                        x1, y1, x2, y2 = hand['bbox']
                                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Draw object detections
                                for obj in detected_objects:
                                    if 'bbox' in obj:
                                        x1, y1, x2, y2 = obj['bbox']
                                        # Highlight if this is the target object
                                        is_target = obj['label'] == current_target_label
                                        is_active = any(i['item_label'] == obj['label'] for i in interactions)
                                        color = (0, 255, 255) if is_target else (255, 165, 0)  # Yellow for target, orange for others
                                        thickness = 4 if is_active else 2
                                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                                        cv2.putText(display_frame, obj['label'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                
                                video_frame.image(
                                    cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                                    caption=f"Frame {frame_count}",
                                    use_container_width=True
                                )
                            
                            # Check if completed
                            if sop_status.get('is_completed', False):
                                break
                        
                        # Get final result
                        final_report = sop_engine.get_report()
                        
                        # Generate AI feedback immediately after assessment
                        add_event("sop", "Generating AI feedback...")
                        try:
                            ai_feedback = run_async(generate_ai_feedback(
                                report_data=final_report,
                                sop_rules=st.session_state.sop_rules
                            ))
                            # Include AI feedback in the result
                            final_report['ai_feedback'] = ai_feedback
                            st.session_state.ai_feedback = ai_feedback
                            add_event("sop", "AI feedback generated successfully")
                        except Exception as fb_err:
                            add_event("sop", f"AI feedback generation failed: {fb_err}")
                            final_report['ai_feedback'] = None
                        
                        st.session_state.assessment_result = final_report
                        st.session_state.assessment_running = False
                        add_event("sop", f"Assessment completed: {'PASSED' if sop_engine.is_completed else 'INCOMPLETE'}")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Assessment error: {str(e)}")
                        st.session_state.assessment_running = False
                
                # Display results
                if st.session_state.assessment_result:
                    st.markdown("---")
                    st.markdown("#### 📊 Assessment Results")
                    
                    result = st.session_state.assessment_result
                    
                    # Summary metrics with visual styling
                    score = (result['completed_steps'] / result['total_steps'] * 100) if result['total_steps'] > 0 else 0
                    is_passed = result.get('is_passed', False)
                    
                    # Result banner
                    if is_passed:
                        st.success("🎉 **ASSESSMENT PASSED!**")
                    else:
                        st.error("❌ **ASSESSMENT INCOMPLETE**")
                    
                    # Metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Score", f"{score:.0f}%")
                    with col2:
                        st.metric("Steps", f"{result['completed_steps']}/{result['total_steps']}")
                    with col3:
                        st.metric("Duration", f"{result.get('total_duration', 0):.1f}s")
                    with col4:
                        st.metric("Status", "PASSED" if is_passed else "FAILED")
                    
                    # Step details in a nice table format
                    if result.get('step_details'):
                        st.markdown("##### 📋 Step-by-Step Results")
                        for idx, step in enumerate(result['step_details']):
                            step_passed = step.get('status') == 'PASSED'
                            icon = "✅" if step_passed else "❌"
                            duration = step.get('duration', 0)
                            
                            with st.expander(f"{icon} Step {idx+1}: {step.get('step_id', 'N/A')} - {step.get('description', 'No description')[:50]}...", expanded=not step_passed):
                                col_left, col_right = st.columns([3, 1])
                                with col_left:
                                    st.markdown(f"**Step ID:** `{step.get('step_id', 'N/A')}`")
                                    st.markdown(f"**Description:** {step.get('description', 'No description')}")
                                    st.markdown(f"**Target Object:** `{step.get('target_object', 'N/A')}`")
                                    st.markdown(f"**Status:** {'✅ PASSED' if step_passed else '❌ FAILED'}")
                                    if step.get('timestamp'):
                                        st.markdown(f"**Timestamp:** {step.get('timestamp')}")
                                with col_right:
                                    st.metric("Duration", f"{duration:.1f}s")
                                
                                # Show instructions if available (from SOP rules)
                                if st.session_state.get('sop_rules'):
                                    sop_steps = st.session_state.sop_rules.get('steps', [])
                                    for sop_step in sop_steps:
                                        if sop_step.get('step_id') == step.get('step_id'):
                                            if sop_step.get('instructions'):
                                                st.markdown("**📝 Instructions:**")
                                                for instr in sop_step.get('instructions', []):
                                                    st.markdown(f"  - {instr}")
                                            if sop_step.get('common_errors'):
                                                st.markdown("**⚠️ Common Errors to Avoid:**")
                                                for err in sop_step.get('common_errors', []):
                                                    st.caption(f"  ⚠️ {err}")
                                            break
                    
                    # Display Failed Steps if any
                    if result.get('failed_steps'):
                        st.markdown("##### ❌ Failed Steps Details")
                        for failed in result['failed_steps']:
                            with st.expander(f"🚫 {failed.get('step_id', 'Unknown')} - {failed.get('error_type', 'ERROR')}", expanded=True):
                                st.error(f"**Error:** {failed.get('error_details', 'No details')}")
                                st.markdown(f"**Expected Step:** {failed.get('description', 'N/A')}")
                                st.markdown(f"**Target Object:** `{failed.get('target_object', 'N/A')}`")
                                st.markdown(f"**Step Index:** {failed.get('step_index', 'N/A')}")
                                if failed.get('timestamp'):
                                    st.markdown(f"**Timestamp:** {failed.get('timestamp')}")
                                if failed.get('wrong_step_id'):
                                    st.warning(f"**Wrong step touched:** {failed.get('wrong_step_description', failed.get('wrong_step_id'))}")
                                if failed.get('instructions'):
                                    st.markdown("**📝 Correct Instructions:**")
                                    for instr in failed.get('instructions', []):
                                        st.markdown(f"  - {instr}")
                                if failed.get('common_errors'):
                                    st.markdown("**⚠️ Common Errors:**")
                                    for err in failed.get('common_errors', []):
                                        st.caption(f"  ⚠️ {err}")
                    
                    # Display AI Feedback in details section (generated immediately after assessment)
                    feedback = result.get('ai_feedback') or st.session_state.get('ai_feedback')
                    if feedback:
                        st.markdown("##### 🤖 AI-Generated Feedback")
                        
                        # Rating only (removed AI score - using original assessment score)
                        rating = feedback.get('overall_rating', 'N/A')
                        rating_emoji = {
                            'excellent': '🌟',
                            'good': '👍',
                            'average': '📊',
                            'needs_improvement': '📈'
                        }.get(rating, '📋')
                        st.metric("Overall Rating", f"{rating_emoji} {rating.replace('_', ' ').title()}")
                        
                        # Summary
                        if feedback.get('summary'):
                            st.info(f"📝 **Summary:** {feedback['summary']}")
                        
                        # Strengths
                        if feedback.get('strengths'):
                            st.markdown("**✅ Strengths:**")
                            for strength in feedback['strengths']:
                                st.markdown(f"- {strength}")
                        
                        # Areas for Improvement
                        if feedback.get('areas_for_improvement'):
                            st.markdown("**📈 Areas for Improvement:**")
                            for area in feedback['areas_for_improvement']:
                                st.markdown(f"- {area}")
                        
                        # Learning Resources (NEW - based on failed steps)
                        if feedback.get('learning_resources'):
                            st.markdown("**📚 Learning Resources (Based on Failed Steps):**")
                            for resource in feedback['learning_resources']:
                                with st.expander(f"📖 {resource.get('step_name', resource.get('step_id', 'Unknown Step'))}", expanded=True):
                                    st.markdown(f"**What to Learn:** {resource.get('what_to_learn', 'N/A')}")
                                    st.markdown(f"**How to Practice:** {resource.get('how_to_practice', 'N/A')}")
                        
                        # Specific Recommendations
                        if feedback.get('specific_recommendations'):
                            st.markdown("**💡 Recommendations:**")
                            for rec in feedback['specific_recommendations']:
                                priority_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(rec.get('priority', ''), '⚪')
                                st.markdown(f"{priority_color} **{rec.get('area', 'General')}:** {rec.get('recommendation', '')}")
                        
                        # Next Steps
                        if feedback.get('next_steps'):
                            st.markdown("**🎯 Next Steps:**")
                            for step in feedback['next_steps']:
                                st.markdown(f"- {step}")
                        
                        # Encouragement
                        if feedback.get('encouragement'):
                            st.success(f"💪 {feedback['encouragement']}")
                        
                        # Statistics (collapsible)
                        if feedback.get('statistics'):
                            with st.expander("📊 Detailed Statistics", expanded=False):
                                stats = feedback['statistics']
                                st.json(stats)
                    else:
                        # Show option to generate feedback if not available
                        st.markdown("##### 🤖 AI Feedback")
                        st.warning("AI feedback was not generated for this assessment.")
                        if st.button("🔄 Generate AI Feedback Now", key="generate_feedback_btn"):
                            try:
                                with st.spinner("Generating AI feedback..."):
                                    ai_feedback = run_async(generate_ai_feedback(
                                        report_data=result,
                                        sop_rules=st.session_state.get('sop_rules')
                                    ))
                                    result['ai_feedback'] = ai_feedback
                                    st.session_state.assessment_result = result
                                    st.session_state.ai_feedback = ai_feedback
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Failed to generate feedback: {e}")
                    
                    # Action buttons
                    st.markdown("##### 💾 Save Assessment")
                    
                    # Admin can assign assessment to a specific user
                    st.markdown("**Assign to User:**")
                    st.caption("Select which user this assessment belongs to (for demo/testing purposes)")
                    
                    # Load users for selection if not already loaded
                    if 'assessment_users_list' not in st.session_state:
                        try:
                            users_list = run_async(get_all_users())
                            st.session_state.assessment_users_list = users_list
                        except:
                            st.session_state.assessment_users_list = []
                    
                    # Create user options
                    user_options = {}
                    for u in st.session_state.get('assessment_users_list', []):
                        if u.role.value != 'admin':  # Only non-admin users
                            user_options[f"{u.full_name} ({u.employee_id})"] = u.employee_id
                    
                    # Add current admin as option too
                    user_options[f"Self ({st.session_state.employee_id})"] = st.session_state.employee_id
                    
                    selected_user_label = st.selectbox(
                        "Assign to:",
                        options=list(user_options.keys()),
                        index=len(user_options) - 1,  # Default to self (last option)
                        key="assessment_assign_user"
                    )
                    selected_employee_id = user_options.get(selected_user_label, st.session_state.employee_id)
                    
                    col_save_db, col_download, col_reset = st.columns(3)
                    
                    with col_save_db:
                        if st.button("💾 Save to Database", use_container_width=True, type="primary", key="save_sop_db"):
                            try:
                                with st.spinner("Saving assessment to database..."):
                                    save_result = run_async(save_assessment_result(
                                        employee_id=selected_employee_id,  # Use selected user
                                        report_data=result,
                                        sop_rules=st.session_state.sop_rules,
                                        video_filename=st.session_state.get('assessment_video_name', 'video.mp4')
                                    ))
                                st.success(f"✅ Saved for **{selected_user_label}**! Session: {save_result.get('session_code', 'N/A')}")
                                add_event("sop", f"Assessment saved to DB for {selected_employee_id}: {save_result.get('session_code')}")
                            except Exception as e:
                                st.error(f"Failed to save: {e}")
                                add_event("sop", f"DB save failed: {e}")
                    
                    with col_download:
                        report_json = json.dumps(result, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="📥 Download JSON",
                            data=report_json,
                            file_name=f"report_{result['session_id']}.json",
                            mime="application/json",
                            key="download_sop_report",
                            use_container_width=True
                        )
                    
                    with col_reset:
                        if st.button("🔄 New Assessment", use_container_width=True, key="new_assessment_btn"):
                            st.session_state.assessment_result = None
                            st.session_state.assessment_video_path = None
                            st.session_state.ai_feedback = None
                            st.rerun()
                
                # Assessment History Section
                st.markdown("---")
                st.markdown("#### 📜 Assessment History")
                
                if st.button("🔄 Load History", key="load_assessment_history"):
                    try:
                        history = run_async(get_user_assessment_history(
                            st.session_state.employee_id, 
                            limit=10
                        ))
                        st.session_state.assessment_history = history
                    except Exception as e:
                        st.error(f"Failed to load history: {e}")
                
                if st.session_state.get('assessment_history'):
                    for assess in st.session_state.assessment_history:
                        status_icon = "✅" if assess['status'] == 'PASSED' else "❌"
                        score_pct = assess.get('score', 0)
                        
                        with st.expander(
                            f"{status_icon} {assess.get('process_name', 'Assessment')} - {score_pct:.0f}% | {assess.get('created_at', 'N/A')[:10] if assess.get('created_at') else 'N/A'}",
                            expanded=False
                        ):
                            col_info, col_action = st.columns([3, 1])
                            with col_info:
                                st.markdown(f"**Session:** `{assess.get('session_code', 'N/A')}`")
                                st.markdown(f"**Steps:** {assess.get('completed_steps', 0)}/{assess.get('total_steps', 0)}")
                                st.markdown(f"**Duration:** {assess.get('total_duration', 0):.1f}s")
                                st.markdown(f"**Video:** {assess.get('video_filename', 'N/A')}")
                            
                            with col_action:
                                if st.button("👁️ View Details", key=f"view_assess_{assess['id']}", use_container_width=True):
                                    try:
                                        details = run_async(get_assessment_details(assess['id']))
                                        st.session_state[f"assess_detail_{assess['id']}"] = details
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                            
                            # Show details if loaded
                            if st.session_state.get(f"assess_detail_{assess['id']}"):
                                detail = st.session_state[f"assess_detail_{assess['id']}"]
                                st.markdown("---")
                                
                                # Step Details with full info
                                st.markdown("**📋 Step Details:**")
                                for step in detail.get('step_details', []):
                                    step_passed = step.get('status') == 'PASSED'
                                    step_icon = "✅" if step_passed else "❌"
                                    with st.expander(f"{step_icon} {step.get('step_id')}: {step.get('description', 'N/A')[:40]}...", expanded=not step_passed):
                                        st.markdown(f"**Description:** {step.get('description', 'N/A')}")
                                        st.markdown(f"**Target Object:** `{step.get('target_object', 'N/A')}`")
                                        st.markdown(f"**Status:** {'✅ PASSED' if step_passed else '❌ FAILED'}")
                                        st.markdown(f"**Duration:** {step.get('duration', 0):.1f}s")
                                        if step.get('timestamp'):
                                            st.caption(f"Timestamp: {step.get('timestamp')}")
                                
                                # Show AI Feedback if available
                                if detail.get('feedback'):
                                    st.markdown("---")
                                    st.markdown("**🤖 AI Feedback:**")
                                    try:
                                        feedback_data = json.loads(detail['feedback']) if isinstance(detail['feedback'], str) else detail['feedback']
                                        
                                        # Rating only (removed AI score)
                                        rating = feedback_data.get('overall_rating', 'N/A')
                                        st.markdown(f"**Rating:** {rating.replace('_', ' ').title()}")
                                        
                                        if feedback_data.get('summary'):
                                            st.info(f"📝 {feedback_data['summary']}")
                                        
                                        if feedback_data.get('strengths'):
                                            st.markdown("**✅ Strengths:**")
                                            for s in feedback_data['strengths']:
                                                st.markdown(f"- {s}")
                                        
                                        if feedback_data.get('areas_for_improvement'):
                                            st.markdown("**📈 Areas for Improvement:**")
                                            for a in feedback_data['areas_for_improvement']:
                                                st.markdown(f"- {a}")
                                        
                                        # Learning Resources (Based on Failed Steps)
                                        if feedback_data.get('learning_resources'):
                                            st.markdown("**📚 Learning Resources (What to Study):**")
                                            for resource in feedback_data['learning_resources']:
                                                with st.expander(f"📖 {resource.get('step_name', resource.get('step_id', 'Unknown'))}", expanded=True):
                                                    st.markdown(f"**What to Learn:** {resource.get('what_to_learn', 'N/A')}")
                                                    st.markdown(f"**How to Practice:** {resource.get('how_to_practice', 'N/A')}")
                                        
                                        if feedback_data.get('specific_recommendations'):
                                            st.markdown("**💡 Recommendations:**")
                                            for rec in feedback_data['specific_recommendations']:
                                                priority = rec.get('priority', '')
                                                p_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(priority, '⚪')
                                                st.markdown(f"{p_icon} **{rec.get('area', '')}:** {rec.get('recommendation', '')}")
                                        
                                        if feedback_data.get('encouragement'):
                                            st.success(f"💪 {feedback_data['encouragement']}")
                                    except Exception as e:
                                        st.caption(f"Feedback parsing error: {e}")
                                
                                # Show report_data if has failed_steps
                                if detail.get('report_data'):
                                    report = detail['report_data']
                                    if report.get('failed_steps'):
                                        st.markdown("---")
                                        st.markdown("**❌ Failed Steps:**")
                                        for failed in report['failed_steps']:
                                            with st.expander(f"🚫 {failed.get('step_id', 'Unknown')} - {failed.get('error_type', 'ERROR')}", expanded=True):
                                                st.error(f"**Error:** {failed.get('error_details', 'No details')}")
                                                st.markdown(f"**Expected:** {failed.get('description', 'N/A')}")
                                                if failed.get('wrong_step_id'):
                                                    st.warning(f"**Wrong step:** {failed.get('wrong_step_description', failed.get('wrong_step_id'))}")
                                                if failed.get('instructions'):
                                                    st.markdown("**Instructions:**")
                                                    for instr in failed.get('instructions', []):
                                                        st.markdown(f"- {instr}")
                else:
                    st.caption("Click 'Load History' to view past assessments")

    # Admin Panel Mode (only accessible to admins)
    if st.session_state.mode == "Admin" and st.session_state.user_role == "admin":
        st.markdown("### ⚙️ Admin Panel")
        admin_tabs = st.tabs(["👥 User Management", "📋 SOP Configuration", "📊 Assessment Results"])
        
        with admin_tabs[0]:
            st.markdown("#### 👥 User Management")
            st.caption("Manage users, view progress, and assessment results")
            
            # Action buttons row
            col_refresh, col_add = st.columns([1, 1])
            with col_refresh:
                if st.button("🔄 Refresh Users", key="refresh_users", use_container_width=True):
                    try:
                        users = run_async(get_all_users())
                        st.session_state["admin_users"] = users
                        add_event("admin", f"Loaded {len(users)} users")
                    except Exception as e:
                        st.error(f"Error loading users: {e}")
            
            with col_add:
                if st.button("➕ Add New User", key="show_add_user_form", use_container_width=True):
                    st.session_state["show_add_user_form"] = not st.session_state.get("show_add_user_form", False)
            
            # Add User Form
            if st.session_state.get("show_add_user_form"):
                st.markdown("---")
                st.markdown("##### ➕ Add New User")
                with st.form("add_user_form"):
                    new_emp_id = st.text_input("Employee ID*", placeholder="e.g., EMP001")
                    new_full_name = st.text_input("Full Name*", placeholder="e.g., Nguyen Van A")
                    new_role = st.selectbox("Role", ["user", "admin"])
                    new_password = st.text_input("Password (required for admin)", type="password", placeholder="Min 6 characters")
                    
                    col_submit, col_cancel = st.columns(2)
                    with col_submit:
                        submitted = st.form_submit_button("✅ Create User", use_container_width=True, type="primary")
                    with col_cancel:
                        cancelled = st.form_submit_button("❌ Cancel", use_container_width=True)
                    
                    if submitted:
                        if not new_emp_id or not new_full_name:
                            st.error("Employee ID and Full Name are required")
                        elif new_role == "admin" and (not new_password or len(new_password) < 6):
                            st.error("Admin users require a password (min 6 characters)")
                        else:
                            try:
                                result = run_async(create_user(
                                    employee_id=new_emp_id,
                                    full_name=new_full_name,
                                    role=new_role,
                                    password=new_password if new_role == "admin" else None
                                ))
                                if result["success"]:
                                    st.success(result["message"])
                                    st.session_state["show_add_user_form"] = False
                                    st.session_state["admin_users"] = None  # Force refresh
                                    add_event("admin", f"Created user: {new_full_name} ({new_emp_id})")
                                    st.rerun()
                                else:
                                    st.error(result["message"])
                            except Exception as e:
                                st.error(f"Error creating user: {e}")
                    
                    if cancelled:
                        st.session_state["show_add_user_form"] = False
                        st.rerun()
            
            st.markdown("---")
            
            # Display users
            if st.session_state.get("admin_users"):
                users = st.session_state["admin_users"]
                st.markdown(f"**Total Users: {len(users)}**")
                
                for user in users:
                    current_role = user.role.value if user.role else "user"
                    role_icon = "🔐" if current_role == "admin" else "👤"
                    
                    with st.expander(f"{role_icon} {user.full_name} ({user.employee_id})", expanded=False):
                        # User Info Tabs
                        user_tabs = st.tabs(["📋 Info & Actions", "📚 Learning Progress", "🎥 Assessment History"])
                        
                        with user_tabs[0]:
                            col_info, col_actions = st.columns([2, 1])
                            with col_info:
                                st.markdown(f"**Name:** {user.full_name}")
                                st.markdown(f"**Employee ID:** {user.employee_id}")
                                st.markdown(f"**User ID:** `{user.id}`")
                                st.markdown(f"**Current Role:** {role_icon} {current_role.title()}")
                                created_str = user.created_at.strftime('%Y-%m-%d %H:%M') if hasattr(user, 'created_at') and user.created_at else 'N/A'
                                st.caption(f"Created: {created_str}")
                            
                            with col_actions:
                                # Role change
                                new_role = st.selectbox(
                                    "Change Role",
                                    ["user", "admin"],
                                    index=0 if current_role == "user" else 1,
                                    key=f"role_{user.id}"
                                )
                                if st.button("💾 Save Role", key=f"save_role_{user.id}", use_container_width=True):
                                    try:
                                        run_async(update_user_role(str(user.id), new_role))
                                        add_event("admin", f"Updated {user.full_name} role to {new_role}")
                                        st.success(f"Role updated to {new_role}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                                
                                # Set password for admins
                                if new_role == "admin" or current_role == "admin":
                                    st.markdown("---")
                                    new_password = st.text_input(
                                        "Set Admin Password",
                                        type="password",
                                        key=f"password_{user.id}"
                                    )
                                    if st.button("🔑 Set Password", key=f"set_pwd_{user.id}", use_container_width=True):
                                        if new_password and len(new_password) >= 6:
                                            try:
                                                run_async(set_admin_password(str(user.id), new_password))
                                                add_event("admin", f"Password set for {user.full_name}")
                                                st.success("Password updated!")
                                            except Exception as e:
                                                st.error(f"Error: {e}")
                                        else:
                                            st.warning("Password must be at least 6 characters")
                                
                                # Delete User (with confirmation)
                                st.markdown("---")
                                st.markdown("**⚠️ Danger Zone**")
                                delete_confirm = st.checkbox(f"Confirm delete", key=f"confirm_delete_{user.id}")
                                if st.button("🗑️ Delete User", key=f"delete_user_{user.id}", use_container_width=True, type="secondary", disabled=not delete_confirm):
                                    if user.employee_id == st.session_state.employee_id:
                                        st.error("Cannot delete yourself!")
                                    else:
                                        try:
                                            result = run_async(delete_user(str(user.id)))
                                            if result["success"]:
                                                st.success(result["message"])
                                                st.session_state["admin_users"] = None
                                                add_event("admin", f"Deleted user: {user.full_name}")
                                                st.rerun()
                                            else:
                                                st.error(result["message"])
                                        except Exception as e:
                                            st.error(f"Error: {e}")
                        
                        with user_tabs[1]:
                            st.markdown("##### 📚 Learning Progress")
                            if st.button("📊 Load Progress", key=f"load_progress_{user.id}", use_container_width=True):
                                try:
                                    progress = run_async(get_user_learning_progress_admin(user.employee_id))
                                    st.session_state[f"user_progress_{user.id}"] = progress
                                except Exception as e:
                                    st.error(f"Error loading progress: {e}")
                            
                            if st.session_state.get(f"user_progress_{user.id}"):
                                progress = st.session_state[f"user_progress_{user.id}"]
                                
                                if "error" in progress:
                                    st.error(progress["error"])
                                else:
                                    # Summary Statistics
                                    st.markdown("---")
                                    col_theory, col_practical = st.columns(2)
                                    
                                    with col_theory:
                                        st.markdown("**📖 Theory (Quiz) Sessions**")
                                        theory = progress.get("theory", {})
                                        st.metric("Total Sessions", theory.get("total_sessions", 0))
                                        
                                        pass_col, fail_col = st.columns(2)
                                        with pass_col:
                                            st.metric("✅ Passed", theory.get("passed", 0))
                                        with fail_col:
                                            st.metric("❌ Failed", theory.get("failed", 0))
                                        
                                        st.metric("📊 Avg Score", f"{theory.get('average_score', 0):.1f}")
                                        
                                        # Recent theory sessions
                                        if theory.get("sessions"):
                                            with st.expander("📋 Recent Quiz Sessions", expanded=False):
                                                for sess in theory["sessions"]:
                                                    status_icon = "✅" if sess["status"] == "PASSED" else "❌"
                                                    st.markdown(f"{status_icon} **Score: {sess['score']}** | {sess.get('created_at', 'N/A')}")
                                                    if sess.get("details") and sess["details"].get("question"):
                                                        wrong_count = len(sess["details"]["question"])
                                                        if wrong_count > 0:
                                                            st.caption(f"  ❌ {wrong_count} wrong answers")
                                    
                                    with col_practical:
                                        st.markdown("**🎥 Practical (SOP) Sessions**")
                                        practical = progress.get("practical", {})
                                        st.metric("Total Sessions", practical.get("total_sessions", 0))
                                        
                                        pass_col2, fail_col2 = st.columns(2)
                                        with pass_col2:
                                            st.metric("✅ Passed", practical.get("passed", 0))
                                        with fail_col2:
                                            st.metric("❌ Failed", practical.get("failed", 0))
                                        
                                        st.metric("📊 Avg Score", f"{practical.get('average_score', 0):.1f}%")
                                        
                                        # Recent practical sessions
                                        if practical.get("sessions"):
                                            with st.expander("📋 Recent SOP Sessions", expanded=False):
                                                for sess in practical["sessions"]:
                                                    status_icon = "✅" if sess["status"] == "PASSED" else "❌"
                                                    st.markdown(f"{status_icon} **{sess.get('process_name', 'Assessment')}** - {sess.get('score', 0):.0f}%")
                                                    st.caption(f"  Steps: {sess.get('completed_steps', 0)}/{sess.get('total_steps', 0)} | Duration: {sess.get('total_duration', 0):.1f}s | {sess.get('created_at', 'N/A')}")
                                    
                                    # Document Progress
                                    docs_progress = progress.get("documents_progress", [])
                                    if docs_progress:
                                        st.markdown("---")
                                        st.markdown("**📄 Document Learning Progress**")
                                        for doc in docs_progress:
                                            with st.expander(f"Document ID: {doc.get('document_id', 'N/A')}", expanded=False):
                                                progress_pct = doc.get("overall_progress", 0)
                                                st.progress(progress_pct / 100 if progress_pct > 0 else 0)
                                                st.markdown(f"**Overall Progress:** {progress_pct:.1f}%")
                                                st.markdown(f"**Chunks Studied:** {doc.get('chunks_studied', 0)} / {doc.get('total_chunks', 0)}")
                                                st.markdown(f"**Chunks Quizzed:** {doc.get('chunks_quizzed', 0)}")
                                                st.markdown(f"**Flashcards Reviewed:** {doc.get('chunks_flashcarded', 0)}")
                                                st.markdown(f"**Chunks Mastered:** {doc.get('chunks_mastered', 0)}")
                                                st.caption(f"Last activity: {doc.get('last_activity', 'N/A')}")
                        
                        with user_tabs[2]:
                            st.markdown("##### 🎥 Assessment History")
                            if st.button("🔄 Load Assessments", key=f"load_assess_{user.id}", use_container_width=True):
                                try:
                                    history = run_async(get_user_assessment_history(user.employee_id, limit=20))
                                    st.session_state[f"user_assessments_{user.id}"] = history
                                except Exception as e:
                                    st.error(f"Error: {e}")
                            
                            if st.session_state.get(f"user_assessments_{user.id}"):
                                assessments = st.session_state[f"user_assessments_{user.id}"]
                                
                                if not assessments:
                                    st.info("No assessments found for this user")
                                else:
                                    st.markdown(f"**Total: {len(assessments)} assessments**")
                                    
                                    for assess in assessments:
                                        status_icon = "✅" if assess['status'] == 'PASSED' else "❌"
                                        score_pct = assess.get('score', 0)
                                        
                                        with st.expander(
                                            f"{status_icon} {assess.get('process_name', 'Assessment')} - {score_pct:.0f}% | {assess.get('created_at', 'N/A')[:10] if assess.get('created_at') else 'N/A'}",
                                            expanded=False
                                        ):
                                            st.markdown(f"**Session:** `{assess.get('session_code', 'N/A')}`")
                                            st.markdown(f"**Steps:** {assess.get('completed_steps', 0)}/{assess.get('total_steps', 0)}")
                                            st.markdown(f"**Duration:** {assess.get('total_duration', 0):.1f}s")
                                            st.markdown(f"**Video:** {assess.get('video_filename', 'N/A')}")
                                            
                                            # View Details Button
                                            if st.button("👁️ View Full Details", key=f"view_user_assess_{user.id}_{assess['id']}", use_container_width=True):
                                                try:
                                                    details = run_async(get_assessment_details(assess['id']))
                                                    st.session_state[f"user_assess_detail_{user.id}_{assess['id']}"] = details
                                                except Exception as e:
                                                    st.error(f"Error: {e}")
                                            
                                            # Show full details if loaded
                                            if st.session_state.get(f"user_assess_detail_{user.id}_{assess['id']}"):
                                                detail = st.session_state[f"user_assess_detail_{user.id}_{assess['id']}"]
                                                st.markdown("---")
                                                
                                                # Step Details
                                                st.markdown("**📋 Step Details:**")
                                                for step in detail.get('step_details', []):
                                                    step_passed = step.get('status') == 'PASSED'
                                                    step_icon = "✅" if step_passed else "❌"
                                                    with st.expander(f"{step_icon} {step.get('step_id')}: {step.get('description', 'N/A')[:40]}...", expanded=not step_passed):
                                                        st.markdown(f"**Description:** {step.get('description', 'N/A')}")
                                                        st.markdown(f"**Target Object:** `{step.get('target_object', 'N/A')}`")
                                                        st.markdown(f"**Status:** {'✅ PASSED' if step_passed else '❌ FAILED'}")
                                                        st.markdown(f"**Duration:** {step.get('duration', 0):.1f}s")
                                                        if step.get('timestamp'):
                                                            st.caption(f"Timestamp: {step.get('timestamp')}")
                                                
                                                # AI Feedback
                                                if detail.get('feedback'):
                                                    st.markdown("---")
                                                    st.markdown("**🤖 AI Feedback:**")
                                                    try:
                                                        feedback_data = json.loads(detail['feedback']) if isinstance(detail['feedback'], str) else detail['feedback']
                                                        
                                                        rating = feedback_data.get('overall_rating', 'N/A')
                                                        rating_emoji = {'excellent': '🌟', 'good': '👍', 'average': '📊', 'needs_improvement': '📈'}.get(rating, '📋')
                                                        st.markdown(f"**Rating:** {rating_emoji} {rating.replace('_', ' ').title()}")
                                                        
                                                        if feedback_data.get('summary'):
                                                            st.info(f"📝 {feedback_data['summary']}")
                                                        
                                                        if feedback_data.get('strengths'):
                                                            st.markdown("**✅ Strengths:**")
                                                            for s in feedback_data['strengths']:
                                                                st.markdown(f"- {s}")
                                                        
                                                        if feedback_data.get('areas_for_improvement'):
                                                            st.markdown("**📈 Areas for Improvement:**")
                                                            for a in feedback_data['areas_for_improvement']:
                                                                st.markdown(f"- {a}")
                                                        
                                                        if feedback_data.get('learning_resources'):
                                                            st.markdown("**📚 Learning Resources:**")
                                                            for resource in feedback_data['learning_resources']:
                                                                with st.expander(f"📖 {resource.get('step_name', resource.get('step_id', 'Unknown'))}", expanded=True):
                                                                    st.markdown(f"**What to Learn:** {resource.get('what_to_learn', 'N/A')}")
                                                                    st.markdown(f"**How to Practice:** {resource.get('how_to_practice', 'N/A')}")
                                                        
                                                        if feedback_data.get('specific_recommendations'):
                                                            st.markdown("**💡 Recommendations:**")
                                                            for rec in feedback_data['specific_recommendations']:
                                                                priority = rec.get('priority', '')
                                                                p_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(priority, '⚪')
                                                                st.markdown(f"{p_icon} **{rec.get('area', '')}:** {rec.get('recommendation', '')}")
                                                        
                                                        if feedback_data.get('next_steps'):
                                                            st.markdown("**🎯 Next Steps:**")
                                                            for ns in feedback_data['next_steps']:
                                                                st.markdown(f"- {ns}")
                                                        
                                                        if feedback_data.get('encouragement'):
                                                            st.success(f"💪 {feedback_data['encouragement']}")
                                                        
                                                        if feedback_data.get('statistics'):
                                                            with st.expander("📊 Statistics", expanded=False):
                                                                st.json(feedback_data['statistics'])
                                                    except Exception as e:
                                                        st.caption(f"Feedback parsing error: {e}")
                                                
                                                # Failed Steps
                                                if detail.get('report_data'):
                                                    report = detail['report_data']
                                                    if report.get('failed_steps'):
                                                        st.markdown("---")
                                                        st.markdown("**❌ Failed Steps:**")
                                                        for failed in report['failed_steps']:
                                                            with st.expander(f"🚫 {failed.get('step_id', 'Unknown')} - {failed.get('error_type', 'ERROR')}", expanded=True):
                                                                st.error(f"**Error:** {failed.get('error_details', 'No details')}")
                                                                st.markdown(f"**Expected:** {failed.get('description', 'N/A')}")
                                                                if failed.get('wrong_step_id'):
                                                                    st.warning(f"**Wrong step:** {failed.get('wrong_step_description', failed.get('wrong_step_id'))}")
                                                                if failed.get('instructions'):
                                                                    st.markdown("**Instructions:**")
                                                                    for instr in failed.get('instructions', []):
                                                                        st.markdown(f"- {instr}")
                                                    
                                                    if report.get('error_logs'):
                                                        with st.expander("⚠️ Error Logs", expanded=False):
                                                            for err in report['error_logs']:
                                                                st.warning(f"**{err.get('error_type', 'ERROR')}** at {err.get('timestamp', 'N/A')}: {err.get('details', '')}")
            else:
                st.info("Click 'Refresh Users' to load user list")
        
        with admin_tabs[1]:
            st.markdown("#### 📋 SOP Configuration")
            st.caption("Configure SOP rules for video assessment")
            
            # Load current SOP rules
            sop_rules_path = Path(__file__).parent / "data" / "sop_rulesv3.json"
            
            col_load, col_save = st.columns([1, 1])
            with col_load:
                if st.button("📥 Load Current Rules", key="load_sop_rules"):
                    try:
                        with open(sop_rules_path, 'r', encoding='utf-8') as f:
                            st.session_state["admin_sop_rules"] = json.load(f)
                        add_event("admin", "Loaded SOP rules")
                        st.success("Rules loaded!")
                    except Exception as e:
                        st.error(f"Error loading rules: {e}")
            
            if st.session_state.get("admin_sop_rules"):
                rules = st.session_state["admin_sop_rules"]
                
                # Process name
                st.text_input(
                    "Process Name",
                    value=rules.get('process_name', ''),
                    key="edit_process_name"
                )
                
                # Max idle time
                st.number_input(
                    "Max Step Idle Time (seconds)",
                    value=rules.get('max_step_idle_time', 30),
                    min_value=10,
                    max_value=120,
                    key="edit_max_idle_time"
                )
                
                st.markdown("#### Steps Configuration")
                
                for i, step in enumerate(rules.get('steps', [])):
                    with st.expander(f"Step {step.get('step_id', i+1)}: {step.get('description', 'N/A')}", expanded=False):
                        st.text_input(
                            "Description",
                            value=step.get('description', ''),
                            key=f"step_desc_{i}"
                        )
                        st.text_input(
                            "Target Object",
                            value=step.get('target_object', ''),
                            key=f"step_target_{i}"
                        )
                        st.selectbox(
                            "Required Action",
                            ["HOLDING", "TOUCHING", "RELEASING"],
                            index=0,
                            key=f"step_action_{i}"
                        )
                        
                        st.text_area(
                            "Instructions (one per line)",
                            value="\n".join(step.get('instructions', [])),
                            key=f"step_instr_{i}",
                            height=150
                        )
                        
                        st.text_area(
                            "Common Errors (one per line)",
                            value="\n".join(step.get('common_errors', [])),
                            key=f"step_errors_{i}",
                            height=100
                        )
                
                # Save button
                with col_save:
                    if st.button("💾 Save Rules", key="save_sop_rules"):
                        try:
                            # Build updated rules
                            updated_rules = {
                                "process_name": st.session_state.get("edit_process_name", rules.get('process_name')),
                                "max_step_idle_time": st.session_state.get("edit_max_idle_time", 30),
                                "steps": []
                            }
                            
                            for i, step in enumerate(rules.get('steps', [])):
                                updated_step = {
                                    "step_id": step.get('step_id', i+1),
                                    "description": st.session_state.get(f"step_desc_{i}", step.get('description', '')),
                                    "required_action": st.session_state.get(f"step_action_{i}", "HOLDING"),
                                    "target_object": st.session_state.get(f"step_target_{i}", step.get('target_object', '')),
                                    "instructions": [l.strip() for l in st.session_state.get(f"step_instr_{i}", '').split('\n') if l.strip()],
                                    "common_errors": [l.strip() for l in st.session_state.get(f"step_errors_{i}", '').split('\n') if l.strip()]
                                }
                                updated_rules["steps"].append(updated_step)
                            
                            # Save to file
                            with open(sop_rules_path, 'w', encoding='utf-8') as f:
                                json.dump(updated_rules, f, indent=4, ensure_ascii=False)
                            
                            st.session_state["admin_sop_rules"] = updated_rules
                            add_event("admin", "Saved SOP rules")
                            st.success("✅ Rules saved successfully!")
                        except Exception as e:
                            st.error(f"Error saving rules: {e}")
                
                # JSON Preview
                with st.expander("📄 JSON Preview", expanded=False):
                    st.json(st.session_state.get("admin_sop_rules", {}))
            else:
                st.info("Click 'Load Current Rules' to edit SOP configuration")
        
        with admin_tabs[2]:
            st.markdown("#### 📊 All Assessment Results")
            st.caption("View and manage assessment results for all users")
            
            col_load, col_clear = st.columns([2, 1])
            
            with col_load:
                if st.button("🔄 Load All Results", key="load_all_results"):
                    try:
                        if SOP_AVAILABLE:
                            all_results = run_async(get_all_assessments(limit=100))
                            st.session_state["admin_all_results"] = all_results
                            add_event("admin", f"Loaded {len(all_results)} assessment results")
                        else:
                            st.error("SOP module not available")
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            with col_clear:
                st.markdown("")  # Spacer
            
            # Delete options
            st.markdown("---")
            st.markdown("##### 🗑️ Clear Assessment Data")
            
            clear_col1, clear_col2 = st.columns(2)
            
            with clear_col1:
                # Clear for specific user
                users_list = st.session_state.get('assessment_users_list', [])
                if not users_list:
                    try:
                        users_list = run_async(get_all_users())
                        st.session_state['assessment_users_list'] = users_list
                    except:
                        users_list = []
                
                user_options = {"Select a user...": None}
                for u in users_list:
                    user_options[f"{u.full_name} ({u.employee_id})"] = u.employee_id
                
                selected_user_to_clear = st.selectbox(
                    "Clear data for user:",
                    options=list(user_options.keys()),
                    key="clear_user_select"
                )
                
                if st.button("🗑️ Clear User Data", key="clear_user_data", type="secondary"):
                    employee_id_to_clear = user_options.get(selected_user_to_clear)
                    if employee_id_to_clear:
                        try:
                            result = run_async(delete_user_assessments(employee_id_to_clear))
                            if result['success']:
                                st.success(result['message'])
                                st.session_state["admin_all_results"] = None  # Refresh
                                add_event("admin", f"Cleared assessments for {employee_id_to_clear}")
                            else:
                                st.warning(result['message'])
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.warning("Please select a user first")
            
            with clear_col2:
                st.markdown("**⚠️ Danger Zone**")
                st.caption("This will permanently delete ALL assessment data!")
                
                confirm_delete_all = st.checkbox("I understand this cannot be undone", key="confirm_delete_all")
                
                if st.button("🗑️ Clear ALL Data", key="clear_all_data", type="primary", disabled=not confirm_delete_all):
                    try:
                        result = run_async(delete_all_assessments())
                        if result['success']:
                            st.success(result['message'])
                            st.session_state["admin_all_results"] = None
                            add_event("admin", "Cleared ALL assessment data")
                        else:
                            st.warning(result['message'])
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            st.markdown("---")
            
            if st.session_state.get("admin_all_results"):
                results = st.session_state["admin_all_results"]
                st.markdown(f"**Total Assessments: {len(results)}**")
                
                # Group by user
                user_results = {}
                for r in results:
                    user_key = f"{r.get('user_name', 'Unknown')} ({r.get('employee_id', 'N/A')})"
                    if user_key not in user_results:
                        user_results[user_key] = []
                    user_results[user_key].append(r)
                
                for user_key, assessments in user_results.items():
                    with st.expander(f"👤 {user_key} - {len(assessments)} assessments", expanded=False):
                        for assess in assessments:
                            result_icon = "✅" if assess.get('is_passed') else "❌"
                            st.markdown(f"{result_icon} **Session:** `{assess.get('session_code', 'N/A')}`")
                            col1, col2, col3 = st.columns(3)
                            col1.caption(f"Steps: {assess.get('completed_steps', 0)}/{assess.get('total_steps', 0)}")
                            col2.caption(f"Duration: {assess.get('total_duration', 0):.1f}s")
                            col3.caption(f"Date: {assess.get('created_at', 'N/A')}")
                            
                            # Show feedback if available
                            if assess.get('feedback'):
                                st.caption("🤖 AI Feedback available")
                            st.markdown("---")
            else:
                st.info("Click 'Load All Results' to view assessment data")

    # Activity log at the bottom
    st.markdown("---")
    render_events()


def main() -> None:
    init_state()
    layout()


if __name__ == "__main__":
    main()
