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
        
        # Create new user
        user = User(employee_id=employee_id, full_name=full_name)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


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
        col_nav1, col_nav2 = st.columns(2)
        if col_nav1.button("📚 Learn", use_container_width=True):
            st.session_state.mode = "Practice"
        if col_nav2.button("📝 Test", use_container_width=True):
            st.session_state.mode = "Testing"
        
        st.markdown("---")
        
        # Profile section
        st.markdown("### 👤 Profile")
        if st.session_state.user_name:
            st.write(f"**{st.session_state.user_name}**")
            st.caption(f"ID: {st.session_state.employee_id}")
        else:
            st.caption("Not signed in")
        
        if st.button("🔄 Switch Account", use_container_width=True):
            st.session_state.user_name = ""
            st.session_state.employee_id = ""
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
            
            with st.form("profile_form"):
                name = st.text_input("Full Name", st.session_state.user_name, placeholder="Enter your name")
                emp_id = st.text_input("Employee ID", st.session_state.employee_id, placeholder="Enter your employee ID")
                saved = st.form_submit_button("🚀 Sign In", use_container_width=True)
            
            if saved and name.strip() and emp_id.strip():
                st.session_state.user_name = name.strip()
                st.session_state.employee_id = emp_id.strip()
                # Save user to database
                try:
                    user = run_async(get_or_create_user(emp_id.strip(), name.strip()))
                    st.session_state.user_db_id = str(user.id)
                    add_event("info", f"User '{name.strip()}' signed in.")
                except Exception as e:
                    add_event("info", f"Profile saved (DB error: {e}).")
                st.rerun()
            elif saved:
                st.warning("Please fill in both Name and Employee ID.")
        st.stop()

    # Main content area after login
    if st.session_state.mode == "Practice":
        practice_tabs = st.tabs(["💬 Chat", "🃏 Flashcards", "📤 Upload", "⚙️ Manage"])
        
        with practice_tabs[0]:
            st.markdown("### RAG-Powered Chat")
            st.caption("Ask questions about uploaded documents (uses AI to search and answer)")
            
            # Document selection for filtering
            col_filter, col_refresh = st.columns([4, 1])
            with col_refresh:
                if st.button("🔄 Refresh", key="refresh_docs_chat"):
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
                        "Filter by documents (optional):",
                        options=list(doc_options.keys()),
                        default=[],
                        key="doc_filter_chat",
                        placeholder="All documents"
                    )
                    st.session_state.selected_doc_ids = [doc_options[d] for d in selected_docs] if selected_docs else []
                else:
                    st.info("Click 'Refresh' to load available documents.")
            
            with st.form("chat_form", clear_on_submit=True):
                user_msg = st.text_input("Ask a question", placeholder="e.g., What are the safety procedures?")
                chat_submit = st.form_submit_button("Send", use_container_width=True)
            
            if chat_submit and user_msg.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_msg.strip()})
                st.session_state.chat_pending_query = user_msg.strip()
                st.rerun()
            
            # Handle streaming response
            if st.session_state.get('chat_pending_query'):
                query = st.session_state.chat_pending_query
                st.session_state.chat_pending_query = None
                
                doc_ids = st.session_state.selected_doc_ids if st.session_state.selected_doc_ids else None
                user_id = st.session_state.user_db_id
                chat_hist = st.session_state.chat_history[:-1] if st.session_state.chat_history else None  # Exclude current query
                
                st.markdown("---")
                st.markdown("**Response:**")
                
                # Create placeholder for streaming response
                response_placeholder = st.empty()
                full_response = ""
                citations = {}
                
                try:
                    # Stream the response
                    async def stream_response():
                        nonlocal full_response, citations
                        session_factory = get_async_session()
                        async with session_factory() as session:
                            chat_service = ChatService(session)
                            chunk_ids_for_tracking = []
                            
                            async for item in chat_service.chat_stream(
                                query, 
                                document_ids=doc_ids,
                                chat_history=chat_hist
                            ):
                                if item["type"] == "text":
                                    full_response += item["content"]
                                    yield item["content"]
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
                    
                    # Use write_stream for streaming output
                    import asyncio
                    import time
                    
                    def run_stream():
                        nonlocal full_response, citations
                        async def inner():
                            nonlocal full_response, citations
                            session_factory = get_async_session()
                            async with session_factory() as session:
                                chat_service = ChatService(session)
                                chunk_ids_for_tracking = []
                                
                                async for item in chat_service.chat_stream(
                                    query, 
                                    document_ids=doc_ids,
                                    chat_history=chat_hist
                                ):
                                    if item["type"] == "text":
                                        # Break down large chunks into smaller pieces for smoother streaming
                                        text = item["content"]
                                        # Stream word by word for smoother effect
                                        words = text.split(' ')
                                        for i, word in enumerate(words):
                                            if i > 0:
                                                full_response += ' '
                                            full_response += word
                                            response_placeholder.markdown(full_response + "▌")
                                            time.sleep(0.02)  # Small delay for smooth effect
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
                        
                        return asyncio.run(inner())
                    
                    run_stream()
                    response_placeholder.markdown(full_response)
                    
                    # Update session state
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                    st.session_state.chat_reply = full_response
                    st.session_state.chat_citations = citations
                    add_event("chat", "RAG Chatbot responded (streaming).")
                    
                    # Display citations after streaming completes
                    if citations:
                        with st.expander("📚 Sources Referenced", expanded=False):
                            # Group citations by document
                            docs_dict = {}
                            doc_info = {}
                            for idx, cite in citations.items():
                                doc_id = cite.get('document_id')
                                page = cite.get('page_number', 0)
                                doc_title = cite.get('document_title', f'Document {doc_id}')
                                file_type = cite.get('file_type', 'PDF')
                                content_preview = cite.get('content_preview', '')
                                
                                doc_info[doc_id] = {'title': doc_title, 'file_type': file_type}
                                
                                if doc_id not in docs_dict:
                                    docs_dict[doc_id] = []
                                docs_dict[doc_id].append((page, idx, content_preview))
                            
                            st.markdown("Click a reference to view the source:")
                            
                            for doc_id, refs in docs_dict.items():
                                info = doc_info.get(doc_id, {})
                                doc_title = info.get('title', f'Document {doc_id}')
                                file_type = info.get('file_type', 'PDF')
                                is_video = file_type == 'VIDEO'
                                
                                icon = "🎬" if is_video else "📑"
                                st.markdown(f"**{icon} {doc_title}**")
                                
                                if is_video:
                                    # For videos, show transcript segments directly
                                    for page, source_idx, content_preview in sorted(refs):
                                        mins = page // 60
                                        secs = page % 60
                                        time_label = f"[{mins}:{secs:02d}]"
                                        with st.expander(f"⏱️ {time_label} (Nguồn {source_idx})", expanded=False):
                                            st.markdown(content_preview)
                                else:
                                    # For PDFs, show clickable page buttons
                                    # Group by page
                                    page_refs = {}
                                    for page, source_idx, _ in refs:
                                        if page not in page_refs:
                                            page_refs[page] = []
                                        page_refs[page].append(source_idx)
                                    
                                    cols = st.columns(min(len(page_refs), 6))
                                    for col_idx, (page, source_nums) in enumerate(sorted(page_refs.items())):
                                        with cols[col_idx % len(cols)]:
                                            sources_str = ", ".join(str(s) for s in source_nums)
                                            if st.button(f"Trang {page}", key=f"stream_cite_{doc_id}_{page}", use_container_width=True, help=f"Nguồn: {sources_str}"):
                                                try:
                                                    doc_content = run_async(get_document_content_for_viewer(doc_id))
                                                    st.session_state.chat_doc_viewer = {
                                                        'content': doc_content,
                                                        'page': page
                                                    }
                                                    st.rerun()
                                                except Exception as e:
                                                    st.error(f"Error loading document: {e}")
                            
                            st.caption(f"_Tham khảo {len(citations)} nguồn_")
                    
                except Exception as exc:
                    st.error(f"RAG Chat error: {exc}")
                    add_event("chat", f"Error: {exc}")
            
            # Chat display area (for previous responses when not streaming)
            elif st.session_state.chat_reply:
                st.markdown("---")
                st.markdown("**Latest Response:**")
                st.markdown(st.session_state.chat_reply)
                
                # Display citations as clickable page references
                citations = st.session_state.get('chat_citations', {})
                if citations:
                    with st.expander("📚 Sources Referenced", expanded=False):
                        # Group citations by document
                        docs_dict = {}
                        doc_info = {}
                        for idx, cite in citations.items():
                            doc_id = cite.get('document_id')
                            page = cite.get('page_number', 0)
                            doc_title = cite.get('document_title', f'Document {doc_id}')
                            file_type = cite.get('file_type', 'PDF')
                            content_preview = cite.get('content_preview', '')
                            
                            doc_info[doc_id] = {'title': doc_title, 'file_type': file_type}
                            
                            if doc_id not in docs_dict:
                                docs_dict[doc_id] = []
                            docs_dict[doc_id].append((page, idx, content_preview))
                        
                        st.markdown("Click a reference to view the source:")
                        
                        for doc_id, refs in docs_dict.items():
                            info = doc_info.get(doc_id, {})
                            doc_title = info.get('title', f'Document {doc_id}')
                            file_type = info.get('file_type', 'PDF')
                            is_video = file_type == 'VIDEO'
                            
                            icon = "🎬" if is_video else "📑"
                            st.markdown(f"**{icon} {doc_title}**")
                            
                            if is_video:
                                # For videos, show transcript segments directly with timestamps
                                for page, source_idx, content_preview in sorted(refs):
                                    mins = page // 60
                                    secs = page % 60
                                    time_label = f"[{mins}:{secs:02d}]"
                                    with st.expander(f"⏱️ {time_label} (Nguồn {source_idx})", expanded=False):
                                        st.markdown(content_preview)
                            else:
                                # For PDFs, show clickable page buttons
                                page_refs = {}
                                for page, source_idx, _ in refs:
                                    if page not in page_refs:
                                        page_refs[page] = []
                                    page_refs[page].append(source_idx)
                                
                                cols = st.columns(min(len(page_refs), 6))
                                for col_idx, (page, source_nums) in enumerate(sorted(page_refs.items())):
                                    with cols[col_idx % len(cols)]:
                                        sources_str = ", ".join(str(s) for s in source_nums)
                                        if st.button(f"Trang {page}", key=f"cite_page_{doc_id}_{page}", use_container_width=True, help=f"Nguồn: {sources_str}"):
                                            try:
                                                doc_content = run_async(get_document_content_for_viewer(doc_id))
                                                st.session_state.chat_doc_viewer = {
                                                    'content': doc_content,
                                                    'page': page
                                                }
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Error loading document: {e}")
                        
                        st.caption(f"_Tham khảo {len(citations)} nguồn_")
                
                # Show document viewer if a citation was clicked
                if st.session_state.get('chat_doc_viewer'):
                    viewer_data = st.session_state.chat_doc_viewer
                    doc = viewer_data.get('content', {})
                    target_page = viewer_data.get('page', 1)
                    
                    if doc and not doc.get('error'):
                        st.markdown("---")
                        col_title, col_close = st.columns([4, 1])
                        with col_title:
                            st.markdown(f"#### 📄 {doc.get('title', 'Document')} - Page {target_page}")
                        with col_close:
                            if st.button("✖️ Close", key="close_chat_doc_viewer"):
                                st.session_state.chat_doc_viewer = None
                                st.rerun()
                        
                        file_path = doc.get('file_path')
                        if file_path and os.path.exists(file_path):
                            try:
                                import base64
                                with open(file_path, "rb") as f:
                                    pdf_data = f.read()
                                b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
                                
                                # PDF viewer with page parameter
                                pdf_display = f'''
                                <iframe 
                                    src="data:application/pdf;base64,{b64_pdf}#page={target_page}" 
                                    width="100%" 
                                    height="500" 
                                    type="application/pdf"
                                    style="border: 1px solid #444; border-radius: 8px;">
                                </iframe>
                                '''
                                st.markdown(pdf_display, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error displaying PDF: {e}")
                        else:
                            # Fallback to text view
                            chunks = doc.get('chunks', [])
                            page_chunks = [c for c in chunks if c.get('page_number') == target_page]
                            if page_chunks:
                                text_container = st.container(height=400)
                                with text_container:
                                    for chunk in page_chunks:
                                        st.markdown(chunk.get('content', ''))
                            else:
                                st.warning(f"No content found for page {target_page}")
            
            # Recent conversation
            if len(st.session_state.chat_history) > 1:
                with st.expander("💬 Recent Conversation", expanded=False):
                    for msg in st.session_state.chat_history[-6:]:
                        if msg["role"] == "system":
                            continue
                        if msg["role"] == "user":
                            st.markdown(f"**You:** {msg['content']}")
                        else:
                            st.markdown(f"**Assistant:** {msg['content']}")

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
        
        with practice_tabs[2]:
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
        
        with practice_tabs[3]:
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
                            if is_video and doc.get('duration_seconds'):
                                st.write(f"**Duration:** {doc['duration_seconds'] // 60}:{doc['duration_seconds'] % 60:02d}")
                            else:
                                st.write(f"**Pages/Segments:** {doc['num_pages']}")
                            st.write(f"**Flashcards:** {'Generated' if doc.get('flashcards_generated') else 'Not generated'}")
                            st.write(f"**Quiz Sets:** {'Generated' if doc.get('quizzes_generated') else 'Not generated'}")
                        
                        with col_actions:
                            # View document button
                            if st.button("👁️ View", key=f"view_doc_{doc['id']}", use_container_width=True):
                                st.session_state[f"viewing_doc_{doc['id']}"] = not st.session_state.get(f"viewing_doc_{doc['id']}", False)
                                if st.session_state[f"viewing_doc_{doc['id']}"]:
                                    # Load document content
                                    try:
                                        doc_content = run_async(get_document_content_for_viewer(doc['id']))
                                        st.session_state[f"doc_content_{doc['id']}"] = doc_content
                                    except Exception as e:
                                        st.error(f"Error loading: {e}")
                                st.rerun()
                            
                            # Download button
                            if doc['file_path'] and os.path.exists(doc['file_path']):
                                try:
                                    mime_type = "video/mp4" if is_video else "application/pdf"
                                    file_ext = os.path.splitext(doc['file_path'])[1] or ('.mp4' if is_video else '.pdf')
                                    with open(doc['file_path'], "rb") as file:
                                        st.download_button(
                                            label="📥 Download",
                                            data=file.read(),
                                            file_name=f"{doc['title']}{file_ext}",
                                            mime=mime_type,
                                            key=f"download_{doc['id']}"
                                        )
                                except Exception as e:
                                    st.caption(f"File unavailable")
                        
                        # Document viewer (inline)
                        if st.session_state.get(f"viewing_doc_{doc['id']}", False):
                            doc_content = st.session_state.get(f"doc_content_{doc['id']}")
                            if doc_content and not doc_content.get('error'):
                                st.markdown("---")
                                file_path = doc_content.get('file_path')
                                
                                if is_video:
                                    # Video transcript viewer
                                    st.markdown("**🎬 Video Transcript**")
                                    chunks = doc_content.get('chunks', [])
                                    
                                    transcript_container = st.container(height=400)
                                    with transcript_container:
                                        for chunk in chunks:
                                            page_num = chunk.get('page_number', 0)
                                            content = chunk.get('content', '')
                                            mins = page_num // 60
                                            secs = page_num % 60
                                            with st.expander(f"[{mins}:{secs:02d}]"):
                                                st.markdown(content)
                                
                                elif file_path and os.path.exists(file_path):
                                    # PDF viewer
                                    st.markdown("**📄 PDF Document**")
                                    view_tab = st.radio("View mode", ["PDF View", "Text View"], horizontal=True, key=f"view_tab_{doc['id']}", label_visibility="collapsed")
                                    
                                    if view_tab == "PDF View":
                                        try:
                                            import base64
                                            with open(file_path, "rb") as f:
                                                pdf_data = f.read()
                                            b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
                                            pdf_display = f'''
                                            <iframe 
                                                src="data:application/pdf;base64,{b64_pdf}" 
                                                width="100%" 
                                                height="500" 
                                                type="application/pdf"
                                                style="border: 1px solid #444; border-radius: 8px;">
                                            </iframe>
                                            '''
                                            st.markdown(pdf_display, unsafe_allow_html=True)
                                        except Exception as e:
                                            st.error(f"Could not display PDF: {e}")
                                    else:
                                        # Text view
                                        chunks = doc_content.get('chunks', [])
                                        pages = {}
                                        for chunk in chunks:
                                            page_num = chunk.get('page_number', 1)
                                            if page_num not in pages:
                                                pages[page_num] = []
                                            pages[page_num].append(chunk.get('content', ''))
                                        
                                        page_nums = sorted(pages.keys())
                                        selected_page = st.selectbox("Page:", page_nums, key=f"page_sel_{doc['id']}")
                                        
                                        page_content = "\n\n".join(pages.get(selected_page, []))
                                        text_container = st.container(height=400)
                                        with text_container:
                                            st.markdown(page_content)
                                else:
                                    # File not found - show extracted text
                                    st.warning("Original file not found. Showing extracted text.")
                                    chunks = doc_content.get('chunks', [])
                                    if chunks:
                                        pages = {}
                                        for chunk in chunks:
                                            page_num = chunk.get('page_number', 1)
                                            if page_num not in pages:
                                                pages[page_num] = []
                                            pages[page_num].append(chunk.get('content', ''))
                                        
                                        page_nums = sorted(pages.keys())
                                        selected_page = st.selectbox("Page:", page_nums, key=f"fallback_page_{doc['id']}")
                                        page_content = "\n\n".join(pages.get(selected_page, []))
                                        
                                        text_container = st.container(height=400)
                                        with text_container:
                                            st.markdown(page_content)
                                    else:
                                        st.error("No content available.")
                            else:
                                st.error("Could not load document content.")
                        
                        st.divider()
                        st.markdown("**⚙️ Management Actions**")
                        
                        col_del_fc, col_del_quiz, col_del_doc = st.columns(3)
                        
                        with col_del_fc:
                            # Delete flashcards
                            if doc.get('flashcards_generated'):
                                if st.button("🗑️ Delete Flashcards", key=f"del_fc_{doc['id']}", use_container_width=True):
                                    st.session_state[f"confirm_del_fc_{doc['id']}"] = True
                                
                                if st.session_state.get(f"confirm_del_fc_{doc['id']}", False):
                                    st.warning("⚠️ Delete all flashcards?")
                                    col_yes, col_no = st.columns(2)
                                    with col_yes:
                                        if st.button("✅ Yes", key=f"confirm_del_fc_yes_{doc['id']}"):
                                            try:
                                                result = run_async(delete_flashcards_for_document(doc['id']))
                                                if result['success']:
                                                    st.success(f"Deleted {result['deleted_count']} flashcards")
                                                    st.session_state[f"confirm_del_fc_{doc['id']}"] = False
                                                    # Refresh docs
                                                    st.session_state.available_documents = run_async(get_all_documents())
                                                    st.rerun()
                                            except Exception as e:
                                                st.error(f"Error: {e}")
                                    with col_no:
                                        if st.button("❌ No", key=f"confirm_del_fc_no_{doc['id']}"):
                                            st.session_state[f"confirm_del_fc_{doc['id']}"] = False
                                            st.rerun()
                            else:
                                st.caption("No flashcards")
                        
                        with col_del_quiz:
                            # Delete quiz sets
                            if doc.get('quizzes_generated'):
                                if st.button("🗑️ Delete Quiz Sets", key=f"del_quiz_{doc['id']}", use_container_width=True):
                                    st.session_state[f"confirm_del_quiz_{doc['id']}"] = True
                                
                                if st.session_state.get(f"confirm_del_quiz_{doc['id']}", False):
                                    st.warning("⚠️ Delete all quiz sets?")
                                    col_yes, col_no = st.columns(2)
                                    with col_yes:
                                        if st.button("✅ Yes", key=f"confirm_del_quiz_yes_{doc['id']}"):
                                            try:
                                                result = run_async(delete_quiz_sets_for_document(doc['id']))
                                                if result['success']:
                                                    st.success(f"Deleted {result['deleted_count']} quiz sets")
                                                    st.session_state[f"confirm_del_quiz_{doc['id']}"] = False
                                                    # Refresh docs
                                                    st.session_state.available_documents = run_async(get_all_documents())
                                                    st.rerun()
                                            except Exception as e:
                                                st.error(f"Error: {e}")
                                    with col_no:
                                        if st.button("❌ No", key=f"confirm_del_quiz_no_{doc['id']}"):
                                            st.session_state[f"confirm_del_quiz_{doc['id']}"] = False
                                            st.rerun()
                            else:
                                st.caption("No quiz sets")
                        
                        with col_del_doc:
                            # Delete entire document
                            if st.button("🗑️ Delete Document", key=f"del_doc_{doc['id']}", type="primary", use_container_width=True):
                                st.session_state[f"confirm_del_doc_{doc['id']}"] = True
                            
                            if st.session_state.get(f"confirm_del_doc_{doc['id']}", False):
                                st.error("⚠️ DELETE DOCUMENT? This will remove ALL related data (flashcards, quiz sets, progress)!")
                                col_yes, col_no = st.columns(2)
                                with col_yes:
                                    if st.button("✅ DELETE", key=f"confirm_del_doc_yes_{doc['id']}"):
                                        try:
                                            result = run_async(delete_document(doc['id']))
                                            if result['success']:
                                                add_event("info", f"Deleted document: {result['title']}")
                                                st.success(f"✅ {result['message']}")
                                                st.session_state[f"confirm_del_doc_{doc['id']}"] = False
                                                # Refresh docs
                                                st.session_state.available_documents = run_async(get_all_documents())
                                                st.rerun()
                                            else:
                                                st.error(f"Error: {result.get('error', 'Unknown error')}")
                                        except Exception as e:
                                            st.error(f"Error: {e}")
                                with col_no:
                                    if st.button("❌ Cancel", key=f"confirm_del_doc_no_{doc['id']}"):
                                        st.session_state[f"confirm_del_doc_{doc['id']}"] = False
                                        st.rerun()
            else:
                st.info("No documents yet. Upload documents in the Upload tab.")

    if st.session_state.mode == "Testing":
        test_tabs = st.tabs(["📝 Quiz Sets", "🎥 Video Assessment"])
        
        # Pre-generated Quiz Tab
        with test_tabs[0]:
            st.markdown("### 📝 Quiz Assessment")
            st.caption("Take pre-generated quiz sets to test your knowledge. Pass all sets to complete the document.")
            
            # Document selection for quiz
            col_quiz_doc, col_quiz_refresh = st.columns([4, 1])
            with col_quiz_refresh:
                if st.button("🔄 Refresh", key="refresh_docs_quiz"):
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
                        key="doc_select_quiz"
                    )
                else:
                    st.info("Click 'Refresh' to load documents.")
                    selected_doc_quiz = None
            
            # Initialize quiz_sets_detail outside the conditional block
            quiz_sets_detail = []
            
            # Load and display quiz sets
            if st.session_state.available_documents and selected_doc_quiz and selected_doc_quiz != "-- Select --":
                doc_id = doc_options_quiz[selected_doc_quiz]
                
                col_gen_quiz, col_load_quiz = st.columns(2)
                with col_gen_quiz:
                    if st.button("✨ Generate", key="gen_quiz", use_container_width=True, help="Generate new quiz sets for this document"):
                        with st.spinner("Generating quiz sets (this may take a minute)..."):
                            try:
                                result = run_async(generate_content_for_document(doc_id))
                                fc_count = result.get('flashcards_count', 0)
                                quiz_count = result.get('quiz_sets_count', 0)
                                add_event("quiz", f"Generated {fc_count} flashcards & {quiz_count} quiz sets")
                                st.success(f"✅ Generated {quiz_count} quiz sets!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Generation error: {e}")
                
                with col_load_quiz:
                    if st.button("📚 Load", key="load_quiz", type="primary", use_container_width=True):
                        with st.spinner("Loading quiz sets..."):
                            try:
                                quiz_sets = run_async(get_document_quiz_sets(doc_id))
                                if quiz_sets:
                                    st.session_state.quiz_sets = quiz_sets
                                    st.session_state.selected_quiz_set = None
                                    st.session_state.quiz_answers = {}
                                    st.session_state.quiz_result = None
                                    st.session_state.current_quiz_doc_id = doc_id
                                    add_event("quiz", f"Loaded {len(quiz_sets)} quiz sets.")
                                    st.rerun()
                                else:
                                    st.warning("No quiz sets yet. Click 'Generate' first.")
                            except Exception as e:
                                st.error(f"Error loading quiz sets: {e}")
                
                # Show quiz progress for this document
                if st.session_state.employee_id:
                    try:
                        progress = run_async(get_document_progress_new(st.session_state.employee_id, doc_id))
                        quiz_prog = progress.get('quiz_progress', {})
                        passed = quiz_prog.get('passed_sets', 0)
                        total_sets = quiz_prog.get('total_sets', 0)
                        percentage = quiz_prog.get('progress_percentage', 0)
                        quiz_sets_detail = quiz_prog.get('sets_detail', [])
                        
                        st.markdown("#### 📊 Quiz Progress")
                        col_p1, col_p2 = st.columns([3, 1])
                        with col_p1:
                            st.progress(percentage / 100 if total_sets > 0 else 0)
                        with col_p2:
                            st.caption(f"{passed}/{total_sets} sets passed")
                    except:
                        pass
            
            st.markdown("---")
            
            # Build a lookup for passed quiz sets
            passed_quiz_set_ids = set()
            for detail in quiz_sets_detail:
                if detail.get('passed'):
                    passed_quiz_set_ids.add(detail.get('set_id'))
            
            # Display quiz set selection or active quiz
            if st.session_state.quiz_sets and not st.session_state.selected_quiz_set:
                st.markdown("#### Select a Quiz Set:")
                for qs in st.session_state.quiz_sets:
                    is_passed = str(qs['id']) in passed_quiz_set_ids
                    col_set, col_btn = st.columns([3, 1])
                    with col_set:
                        num_q = len(qs.get('questions', []))
                        if is_passed:
                            # Display in green for passed sets
                            st.markdown(f"✅ **:green[Set {qs['set_number']}:** {qs.get('title', 'Quiz')} ({num_q} questions)]")
                        else:
                            st.markdown(f"**Set {qs['set_number']}:** {qs.get('title', 'Quiz')} ({num_q} questions)")
                    with col_btn:
                        btn_label = f"{'✅ Passed' if is_passed else 'Start'} Set {qs['set_number']}"
                        btn_type = "secondary" if is_passed else "primary"
                        if st.button(btn_label, key=f"start_set_{qs['id']}", use_container_width=True, type=btn_type):
                            st.session_state.selected_quiz_set = qs
                            st.session_state.quiz_answers = {}
                            st.session_state.quiz_result = None
                            st.rerun()
            
            # Active quiz
            elif st.session_state.selected_quiz_set:
                qs = st.session_state.selected_quiz_set
                questions = qs.get('questions', [])
                
                st.markdown(f"### Set {qs['set_number']}: {qs.get('title', 'Quiz')}")
                
                # Back button
                if st.button("⬅️ Back to Set Selection"):
                    st.session_state.selected_quiz_set = None
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_result = None
                    st.rerun()
                
                st.markdown("---")
                
                if not st.session_state.quiz_result:
                    possible_choices = ['A', 'B', 'C', 'D']
                    
                    for i, q in enumerate(questions):
                        # Question display - use native Streamlit components for theme compatibility
                        st.markdown(f"**Question {i + 1}:** {q.get('question', 'N/A')}")
                        
                        options = q.get("options", [])
                        
                        # Display options
                        selected = st.radio(
                            f"Select answer for Q{i+1}:",
                            options=possible_choices[:len(options)],
                            format_func=lambda x, opts=options, choices=possible_choices: f"{x}. {opts[choices.index(x)]}" if choices.index(x) < len(opts) else x,
                            key=f"quiz_q_{i}",
                            index=None,
                            horizontal=True,
                            label_visibility="collapsed"
                        )
                        if selected:
                            st.session_state.quiz_answers[q['id']] = selected
                        
                        st.markdown("---")  # Separator between questions
                    
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
                                user_ans = st.session_state.quiz_answers.get(q['id'])
                                if user_ans:
                                    user_idx = possible_choices.index(user_ans)
                                    options = q.get('options', [])
                                    if user_idx < len(options):
                                        user_text = options[user_idx]
                                        if user_text == q.get('answer'):
                                            score += 1
                            
                            # Submit attempt
                            try:
                                result = run_async(submit_quiz_attempt(
                                    st.session_state.employee_id,
                                    qs['id'],
                                    st.session_state.quiz_answers,
                                    score,
                                    total
                                ))
                                st.session_state.quiz_result = {
                                    "score": score,
                                    "total": total,
                                    "status": result.get('status', 'FAILED'),
                                    "questions": questions
                                }
                                add_event("quiz", f"Quiz Set {qs['set_number']}: {score}/{total} - {result.get('status', 'FAILED')}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error submitting quiz: {e}")
                
                else:
                    # Show results
                    result = st.session_state.quiz_result
                    if result["status"] == "PASSED":
                        st.balloons()
                        st.success(f"🎉 **PASSED!** {result['score']}/{result['total']} correct (60% needed)")
                    else:
                        st.error(f"❌ **FAILED:** {result['score']}/{result['total']} correct (60% needed)")
                    
                    # Show review
                    st.markdown("#### Review Answers:")
                    possible_choices = ['A', 'B', 'C', 'D']
                    for i, q in enumerate(result.get('questions', [])):
                        user_ans = st.session_state.quiz_answers.get(q['id'])
                        user_idx = possible_choices.index(user_ans) if user_ans else -1
                        options = q.get('options', [])
                        user_text = options[user_idx] if 0 <= user_idx < len(options) else "N/A"
                        correct = q.get('answer', '')
                        is_correct = user_text == correct
                        
                        icon = "✅" if is_correct else "❌"
                        with st.expander(f"{icon} Q{i+1}: {q.get('question', '')[:50]}..."):
                            st.write(f"**Your answer:** {user_text}")
                            st.write(f"**Correct answer:** {correct}")
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
        
        # Video Assessment Tab - Stage B SOP Video Analysis
        with test_tabs[1]:
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
                            model_to_use = yolo_model
                            if yolo_model == "custom" and st.session_state.get('custom_model_path'):
                                model_to_use = st.session_state.custom_model_path
                            
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
                                
                                # Update UI less frequently (every 15 frames) for performance
                                if frame_count - last_ui_update >= 15:
                                    last_ui_update = frame_count
                                    
                                    # Update activity log on the left column
                                    with activity_log_placeholder.container():
                                        # Current status
                                        st.markdown(f"**Frame:** {frame_count}")
                                        st.markdown(f"**Step {sop_status['step_index']+1}/{sop_status['total_steps']}**")
                                        st.markdown(f"📌 **Current:** {sop_status.get('description', 'N/A')}")
                                        st.markdown(f"🎯 **Target:** `{current_target_label}`")
                                        st.markdown(f"📍 **State:** {sop_status.get('state', 'WAITING')}")
                                        st.markdown(f"📝 **Status:** {sop_status.get('status', 'Unknown')}")
                                        
                                        # Progress bar
                                        if sop_status.get('timer_ratio', 0) > 0:
                                            st.progress(sop_status['timer_ratio'])
                                        
                                        # Detection info
                                        st.caption(f"🖐️ Hands: {len(detected_hands)} | 📦 Objects: {len(detected_objects)} | 🤝 Interactions: {len(interactions)}")
                                        
                                        # Completed steps
                                        if sop_engine.report_logs:
                                            st.markdown("---")
                                            st.markdown("**✅ Completed:**")
                                            for log in sop_engine.report_logs:
                                                st.success(f"{log['step_id']}: {log.get('duration', 0):.1f}s")
                                    
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
                            st.session_state.assessment_result = sop_engine.get_report()
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
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            score = (result['completed_steps'] / result['total_steps'] * 100) if result['total_steps'] > 0 else 0
                            st.metric("Score", f"{score:.0f}%")
                        with col2:
                            st.metric("Steps Completed", f"{result['completed_steps']}/{result['total_steps']}")
                        with col3:
                            status_emoji = "✅" if result['is_passed'] else "❌"
                            st.metric("Status", f"{status_emoji} {'PASSED' if result['is_passed'] else 'FAILED'}")
                        
                        # Duration
                        if result.get('total_duration'):
                            st.info(f"⏱️ Total Time: {result['total_duration']:.1f} seconds")
                        
                        # Step details
                        if result.get('step_details'):
                            with st.expander("📋 Step Details", expanded=True):
                                for step in result['step_details']:
                                    col_step, col_dur, col_status = st.columns([3, 1, 1])
                                    with col_step:
                                        st.markdown(f"**{step['step_id']}**: {step.get('description', 'N/A')}")
                                    with col_dur:
                                        st.caption(f"{step.get('duration', 0):.1f}s")
                                    with col_status:
                                        st.markdown("✅" if step['status'] == 'PASSED' else "❌")
                        
                        # Save results
                        col_save, col_reset = st.columns(2)
                        with col_save:
                            if st.button("💾 Save Report", use_container_width=True, key="save_sop_report"):
                                try:
                                    report_path = Path("uploads/reports")
                                    report_path.mkdir(parents=True, exist_ok=True)
                                    filename = f"report_{result['session_id']}.json"
                                    filepath = report_path / filename
                                    with open(filepath, 'w', encoding='utf-8') as f:
                                        json.dump(result, f, indent=2, ensure_ascii=False)
                                    st.success(f"✅ Report saved: {filename}")
                                    add_event("sop", f"Report saved: {filename}")
                                    
                                    # Also offer download
                                    report_json = json.dumps(result, indent=2, ensure_ascii=False)
                                    st.download_button(
                                        label="📥 Download Report",
                                        data=report_json,
                                        file_name=filename,
                                        mime="application/json",
                                        key="download_sop_report"
                                    )
                                except Exception as e:
                                    st.error(f"Failed to save: {e}")
                                    add_event("sop", f"Save failed: {e}")
                        
                        with col_reset:
                            if st.button("🔄 New Assessment", use_container_width=True):
                                st.session_state.assessment_result = None
                                st.session_state.assessment_video_path = None
                                st.rerun()

    # Activity log at the bottom
    st.markdown("---")
    render_events()


def main() -> None:
    init_state()
    layout()


if __name__ == "__main__":
    main()
