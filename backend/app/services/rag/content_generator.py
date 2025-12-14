"""
Content Generation Service.
Generates comprehensive flashcard and quiz sets for documents.
These are pre-generated once and stored for consistent progress tracking.
"""
import json
import logging as log
import google.generativeai as genai
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.models.document import Document
from app.models.chunk import DocumentChunk
from app.models.learning_content import (
    Flashcard, QuizSet, QuizQuestion,
    UserFlashcardProgress, UserQuizAttempt
)
from app.core.config import settings

genai.configure(api_key=settings.GOOGLE_API_KEY)

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class ContentGenerationService:
    """
    Service for generating and managing pre-generated learning content.
    Generates flashcards and quiz sets that cover the entire document.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.7,
        )
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash', 
            generation_config=self.generation_config
        )
    
    async def get_document_chunks(self, document_id: int) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        result = await self.db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.page_number)
        )
        return result.scalars().all()
    
    async def generate_all_flashcards(self, document_id: int) -> List[Flashcard]:
        """
        Generate a comprehensive set of flashcards covering the entire document.
        Each chunk should have at least one flashcard.
        """
        chunks = await self.get_document_chunks(document_id)
        if not chunks:
            log.warning(f"No chunks found for document {document_id}")
            return []
        
        log.info(f"Generating flashcards for document {document_id} with {len(chunks)} chunks")
        
        # Process in batches to avoid token limits
        batch_size = 10
        all_flashcards = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_flashcards = await self._generate_flashcards_batch(batch, document_id, len(all_flashcards))
            all_flashcards.extend(batch_flashcards)
        
        # Save all flashcards to database
        self.db.add_all(all_flashcards)
        
        # Update document status
        doc = await self.db.get(Document, document_id)
        if doc:
            doc.flashcards_generated = True
        
        await self.db.commit()
        log.info(f"Generated and saved {len(all_flashcards)} flashcards for document {document_id}")
        
        return all_flashcards
    
    async def _generate_flashcards_batch(
        self, 
        chunks: List[DocumentChunk], 
        document_id: int,
        start_index: int
    ) -> List[Flashcard]:
        """Generate high-quality flashcards for a batch of chunks - Vietnamese."""
        context_parts = []
        chunk_id_map = {}  # Map chunk index to chunk ID
        
        for idx, chunk in enumerate(chunks):
            context_parts.append(f"[Phần {idx + 1}] {chunk.content}")
            chunk_id_map[idx + 1] = chunk.id
        
        context_text = "\n\n".join(context_parts)
        # Generate fewer but more meaningful cards - 1 quality card per chunk instead of 3
        num_cards = max(len(chunks), 5)  # 1 card per chunk, minimum 5
        
        prompt = f"""Bạn là chuyên gia đào tạo kỹ thuật cao cấp. Nhiệm vụ của bạn là tạo {num_cards} thẻ học (flash cards) CHẤT LƯỢNG CAO bằng Tiếng Việt.

VĂN BẢN NGUỒN:
---
{context_text}
---

NGUYÊN TẮC TẠO THẺ HỌC:
1. CHỈ TẠO THẺ CHO NỘI DUNG QUAN TRỌNG - không tạo thẻ cho thông tin tầm thường hoặc chi tiết không cần thiết.
2. Tập trung vào:
   - Các quy trình kỹ thuật quan trọng và các bước thực hiện
   - Các khái niệm cốt lõi mà nhân viên PHẢI nhớ
   - Thông số kỹ thuật, tiêu chuẩn an toàn quan trọng
   - Các lưu ý đặc biệt và cảnh báo
3. KHÔNG tạo thẻ cho:
   - Thông tin giới thiệu chung chung
   - Nội dung lặp lại hoặc trùng lặp ý nghĩa
   - Chi tiết phụ không ảnh hưởng đến công việc

YÊU CẦU NỘI DUNG THẺ:
1. MẶT TRƯỚC (front): Câu hỏi rõ ràng, cụ thể, tập trung vào kiến thức thực hành
2. MẶT SAU (back): Đáp án ĐẦY ĐỦ và CHI TIẾT (3-5 câu), bao gồm:
   - Giải thích trực tiếp cho câu hỏi
   - Ngữ cảnh hoặc lý do tại sao điều này quan trọng
   - Các bước cụ thể nếu là quy trình
   - Ví dụ thực tế nếu cần
3. Đánh số phần (section) tương ứng với nguồn thông tin.

CẤU TRÚC JSON:
{{
    "flashcards": [
        {{
            "section": 1,
            "front": "Câu hỏi cụ thể về kiến thức quan trọng",
            "back": "Đáp án chi tiết, đầy đủ với giải thích rõ ràng. Bao gồm ngữ cảnh và lý do quan trọng. Có thể bao gồm các bước cụ thể hoặc ví dụ minh họa."
        }}
    ]
}}

Tạo ĐÚNG {num_cards} thẻ học chất lượng cao, tập trung vào kiến thức thực sự có giá trị cho nhân viên.
"""
        
        try:
            response = self.model.generate_content(prompt)
            data = json.loads(response.text)
            flashcards_data = data.get("flashcards", [])
            
            flashcards = []
            for idx, fc in enumerate(flashcards_data):
                section = fc.get("section", 1)
                chunk_id = chunk_id_map.get(section, chunks[0].id if chunks else None)
                
                flashcard = Flashcard(
                    document_id=document_id,
                    front=fc.get("front", ""),
                    back=fc.get("back", ""),
                    chunk_ids=[chunk_id] if chunk_id else [],
                    order_index=start_index + idx
                )
                flashcards.append(flashcard)
            
            return flashcards
        except Exception as e:
            log.error(f"Error generating flashcards batch: {e}")
            return []
    
    async def generate_quiz_sets(self, document_id: int, num_sets: int = 3) -> List[QuizSet]:
        """
        Generate 3-5 quiz sets that together cover the entire document.
        Each set focuses on different sections/chunks.
        """
        chunks = await self.get_document_chunks(document_id)
        if not chunks:
            log.warning(f"No chunks found for document {document_id}")
            return []
        
        log.info(f"Generating {num_sets} quiz sets for document {document_id} with {len(chunks)} chunks")
        
        # Divide chunks into sets
        chunks_per_set = max(1, len(chunks) // num_sets)
        quiz_sets = []
        
        for set_num in range(1, num_sets + 1):
            start_idx = (set_num - 1) * chunks_per_set
            end_idx = start_idx + chunks_per_set if set_num < num_sets else len(chunks)
            set_chunks = chunks[start_idx:end_idx]
            
            if not set_chunks:
                continue
            
            quiz_set = await self._generate_quiz_set(
                document_id=document_id,
                set_number=set_num,
                chunks=set_chunks
            )
            if quiz_set:
                quiz_sets.append(quiz_set)
        
        # Update document status
        doc = await self.db.get(Document, document_id)
        if doc:
            doc.quizzes_generated = True
        
        await self.db.commit()
        log.info(f"Generated and saved {len(quiz_sets)} quiz sets for document {document_id}")
        
        return quiz_sets
    
    async def _generate_quiz_set(
        self, 
        document_id: int, 
        set_number: int,
        chunks: List[DocumentChunk],
        num_questions: int = 15
    ) -> Optional[QuizSet]:
        """Generate a single quiz set from given chunks - Vietnamese with 15-20 questions."""
        context_parts = []
        chunk_ids = []
        
        for chunk in chunks:
            context_parts.append(f"[Trang {chunk.page_number}] {chunk.content}")
            chunk_ids.append(chunk.id)
        
        context_text = "\n\n".join(context_parts)
        
        # Vietnamese quiz prompt based on original utils.py
        # Options should NOT include A/B/C/D prefix - they are added during display
        json_structure = """
{
    "title": "Tiêu đề ngắn gọn cho bộ câu hỏi",
    "questions": [
        {
            "question": "Nội dung câu hỏi",
            "options": ["Đáp án 1", "Đáp án 2", "Đáp án 3", "Đáp án 4"],
            "correct_index": 0,
            "explanation": "Giải thích ngắn gọn"
        }
    ]
}
"""
        
        prompt = f"""Bạn là chuyên gia đào tạo kỹ thuật. Nhiệm vụ của bạn là tạo {num_questions} câu hỏi trắc nghiệm (Quiz) bằng Tiếng Việt dựa trên văn bản sau.

VĂN BẢN NGUỒN:
---
{context_text}
---

YÊU CẦU:
1. Tạo đúng {num_questions} câu hỏi kiểm tra sự hiểu biết về các khái niệm và quy trình quan trọng.
2. Mỗi câu hỏi có 4 đáp án lựa chọn.
3. KHÔNG thêm tiền tố A, B, C, D vào các đáp án - chỉ viết nội dung đáp án.
4. Dùng "correct_index" để chỉ vị trí đáp án đúng (0, 1, 2, hoặc 3).
5. Bao gồm giải thích cho đáp án đúng.
6. Trả về kết quả CHỈ LÀ JSON theo cấu trúc mẫu dưới đây, không thêm bất kỳ lời dẫn nào khác:

CẤU TRÚC JSON:
{json_structure}

Tạo đúng {num_questions} câu hỏi toàn diện kiểm tra kiến thức quan trọng.
"""
        
        try:
            response = self.model.generate_content(prompt)
            data = json.loads(response.text)
            
            # Create quiz set with Vietnamese title
            quiz_set = QuizSet(
                document_id=document_id,
                set_number=set_number,
                title=data.get("title", f"Bộ câu hỏi {set_number}"),
                chunk_ids=chunk_ids
            )
            self.db.add(quiz_set)
            await self.db.flush()  # Get the quiz_set.id
            
            # Create questions
            questions_data = data.get("questions", [])
            for idx, q in enumerate(questions_data):
                options = q.get("options", [])
                
                # Handle correct answer - support both old format (answer text) and new format (correct_index)
                correct_answer = ""
                if "correct_index" in q:
                    # New format: use index to get the option text
                    correct_idx = q.get("correct_index", 0)
                    if 0 <= correct_idx < len(options):
                        correct_answer = options[correct_idx]
                elif "answer" in q:
                    # Old format: direct answer text
                    correct_answer = q.get("answer", "")
                
                # Clean options - remove any existing A/B/C/D prefixes from LLM
                cleaned_options = []
                for opt in options:
                    # Strip prefixes like "A. ", "B. ", "A) ", "B) ", etc.
                    clean_opt = opt.strip()
                    if len(clean_opt) >= 2 and clean_opt[0] in 'ABCDabcd' and clean_opt[1] in '.):':
                        clean_opt = clean_opt[2:].strip()
                    cleaned_options.append(clean_opt)
                
                # Also clean the correct answer
                clean_correct = correct_answer.strip()
                if len(clean_correct) >= 2 and clean_correct[0] in 'ABCDabcd' and clean_correct[1] in '.):':
                    clean_correct = clean_correct[2:].strip()
                
                question = QuizQuestion(
                    quiz_set_id=quiz_set.id,
                    question=q.get("question", ""),
                    options=cleaned_options,
                    correct_answer=clean_correct,
                    explanation=q.get("explanation", ""),
                    chunk_ids=chunk_ids,
                    order_index=idx
                )
                self.db.add(question)
            
            return quiz_set
        except Exception as e:
            log.error(f"Error generating quiz set {set_number}: {e}")
            return None
    
    async def get_document_flashcards(self, document_id: int) -> List[Flashcard]:
        """Get all flashcards for a document."""
        result = await self.db.execute(
            select(Flashcard)
            .where(Flashcard.document_id == document_id)
            .order_by(Flashcard.order_index)
        )
        return result.scalars().all()
    
    async def get_document_quiz_sets(self, document_id: int) -> List[Dict[str, Any]]:
        """Get all quiz sets with questions for a document."""
        result = await self.db.execute(
            select(QuizSet)
            .where(QuizSet.document_id == document_id)
            .order_by(QuizSet.set_number)
        )
        quiz_sets = result.scalars().all()
        
        sets_data = []
        for qs in quiz_sets:
            # Get questions for this set
            questions_result = await self.db.execute(
                select(QuizQuestion)
                .where(QuizQuestion.quiz_set_id == qs.id)
                .order_by(QuizQuestion.order_index)
            )
            questions = questions_result.scalars().all()
            
            sets_data.append({
                "id": str(qs.id),
                "set_number": qs.set_number,
                "title": qs.title,
                "questions": [
                    {
                        "id": str(q.id),
                        "question": q.question,
                        "options": q.options,
                        "answer": q.correct_answer,
                        "explanation": q.explanation
                    }
                    for q in questions
                ]
            })
        
        return sets_data


class ProgressTrackingService:
    """
    Service for tracking user progress on pre-generated content.
    Progress is calculated based on completed flashcards and passed quizzes.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def mark_flashcard_completed(
        self, 
        user_id: UUID, 
        flashcard_id: UUID,
        ease_factor: int = 2
    ) -> UserFlashcardProgress:
        """Mark a flashcard as completed/known by user."""
        # Check if progress record exists
        result = await self.db.execute(
            select(UserFlashcardProgress)
            .where(
                UserFlashcardProgress.user_id == user_id,
                UserFlashcardProgress.flashcard_id == flashcard_id
            )
        )
        progress = result.scalar_one_or_none()
        
        if progress:
            progress.is_completed = True
            progress.review_count += 1
            progress.ease_factor = ease_factor
        else:
            progress = UserFlashcardProgress(
                user_id=user_id,
                flashcard_id=flashcard_id,
                is_completed=True,
                review_count=1,
                ease_factor=ease_factor
            )
            self.db.add(progress)
        
        await self.db.commit()
        return progress
    
    async def mark_flashcard_reviewed(
        self, 
        user_id: UUID, 
        flashcard_id: UUID
    ) -> UserFlashcardProgress:
        """Mark a flashcard as reviewed (but not necessarily mastered)."""
        result = await self.db.execute(
            select(UserFlashcardProgress)
            .where(
                UserFlashcardProgress.user_id == user_id,
                UserFlashcardProgress.flashcard_id == flashcard_id
            )
        )
        progress = result.scalar_one_or_none()
        
        if progress:
            progress.review_count += 1
        else:
            progress = UserFlashcardProgress(
                user_id=user_id,
                flashcard_id=flashcard_id,
                is_completed=False,
                review_count=1
            )
            self.db.add(progress)
        
        await self.db.commit()
        return progress
    
    async def record_quiz_attempt(
        self, 
        user_id: UUID, 
        quiz_set_id: UUID,
        answers: Dict[str, str],
        score: int,
        total: int
    ) -> UserQuizAttempt:
        """Record a quiz attempt."""
        is_passed = (score / total) >= 0.6 if total > 0 else False
        
        attempt = UserQuizAttempt(
            user_id=user_id,
            quiz_set_id=quiz_set_id,
            score=score,
            total_questions=total,
            is_passed=is_passed,
            answers=answers
        )
        self.db.add(attempt)
        await self.db.commit()
        return attempt
    
    async def get_flashcard_progress(
        self, 
        user_id: UUID, 
        document_id: int
    ) -> Dict[str, Any]:
        """
        Get flashcard progress for a user on a document.
        Returns actual completion percentage based on marked flashcards.
        """
        # Get total flashcards for document
        total_result = await self.db.execute(
            select(func.count(Flashcard.id))
            .where(Flashcard.document_id == document_id)
        )
        total_flashcards = total_result.scalar() or 0
        
        if total_flashcards == 0:
            return {
                "total": 0,
                "completed": 0,
                "reviewed": 0,
                "progress_percentage": 0.0
            }
        
        # Get completed flashcards for this user
        completed_result = await self.db.execute(
            select(func.count(UserFlashcardProgress.id))
            .join(Flashcard, UserFlashcardProgress.flashcard_id == Flashcard.id)
            .where(
                UserFlashcardProgress.user_id == user_id,
                Flashcard.document_id == document_id,
                UserFlashcardProgress.is_completed == True
            )
        )
        completed = completed_result.scalar() or 0
        
        # Get reviewed (but not completed) flashcards
        reviewed_result = await self.db.execute(
            select(func.count(UserFlashcardProgress.id))
            .join(Flashcard, UserFlashcardProgress.flashcard_id == Flashcard.id)
            .where(
                UserFlashcardProgress.user_id == user_id,
                Flashcard.document_id == document_id
            )
        )
        reviewed = reviewed_result.scalar() or 0
        
        return {
            "total": total_flashcards,
            "completed": completed,
            "reviewed": reviewed,
            "progress_percentage": (completed / total_flashcards) * 100 if total_flashcards > 0 else 0
        }
    
    async def get_quiz_progress(
        self, 
        user_id: UUID, 
        document_id: int
    ) -> Dict[str, Any]:
        """
        Get quiz progress for a user on a document.
        Returns percentage of quiz sets passed.
        """
        # Get total quiz sets for document
        total_result = await self.db.execute(
            select(func.count(QuizSet.id))
            .where(QuizSet.document_id == document_id)
        )
        total_sets = total_result.scalar() or 0
        
        if total_sets == 0:
            return {
                "total_sets": 0,
                "passed_sets": 0,
                "attempted_sets": 0,
                "progress_percentage": 0.0,
                "sets_detail": []
            }
        
        # Get quiz sets for this document
        sets_result = await self.db.execute(
            select(QuizSet)
            .where(QuizSet.document_id == document_id)
            .order_by(QuizSet.set_number)
        )
        quiz_sets = sets_result.scalars().all()
        
        passed_sets = 0
        attempted_sets = 0
        sets_detail = []
        
        for qs in quiz_sets:
            # Get best attempt for this quiz set
            attempt_result = await self.db.execute(
                select(UserQuizAttempt)
                .where(
                    UserQuizAttempt.user_id == user_id,
                    UserQuizAttempt.quiz_set_id == qs.id
                )
                .order_by(UserQuizAttempt.score.desc())
                .limit(1)
            )
            best_attempt = attempt_result.scalar_one_or_none()
            
            set_info = {
                "set_id": str(qs.id),
                "set_number": qs.set_number,
                "title": qs.title,
                "attempted": best_attempt is not None,
                "passed": best_attempt.is_passed if best_attempt else False,
                "best_score": best_attempt.score if best_attempt else None,
                "total_questions": best_attempt.total_questions if best_attempt else None
            }
            sets_detail.append(set_info)
            
            if best_attempt:
                attempted_sets += 1
                if best_attempt.is_passed:
                    passed_sets += 1
        
        return {
            "total_sets": total_sets,
            "passed_sets": passed_sets,
            "attempted_sets": attempted_sets,
            "progress_percentage": (passed_sets / total_sets) * 100 if total_sets > 0 else 0,
            "sets_detail": sets_detail
        }
    
    async def get_overall_document_progress(
        self, 
        user_id: UUID, 
        document_id: int
    ) -> Dict[str, Any]:
        """
        Get combined progress for a document.
        Overall progress = (flashcard_progress + quiz_progress) / 2
        """
        flashcard_progress = await self.get_flashcard_progress(user_id, document_id)
        quiz_progress = await self.get_quiz_progress(user_id, document_id)
        
        # Overall is average of both metrics
        overall = (
            flashcard_progress["progress_percentage"] + 
            quiz_progress["progress_percentage"]
        ) / 2
        
        return {
            "document_id": document_id,
            "overall_progress": overall,
            "flashcard_progress": flashcard_progress,
            "quiz_progress": quiz_progress
        }
