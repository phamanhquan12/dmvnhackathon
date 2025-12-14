# app/services/sop/feedback_service.py
"""
AI-powered Feedback Service for SOP Assessment
Generates contextual feedback using RAG for errors during assessment
"""
import json
import logging as log
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.services.rag.vectorizer import Vectorizer
from app.models.chunk import DocumentChunk
from app.models.document import Document

# Configure GenAI
genai.configure(api_key=settings.GOOGLE_API_KEY)

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class FeedbackService:
    """
    Service for generating AI-powered feedback for SOP assessment errors.
    Uses RAG to find relevant documentation and generates contextual advice.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.vectorizer = Vectorizer()
        self.feedback_config = genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.3,  # Lower temp for more accurate feedback
        )
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash', 
            generation_config=self.feedback_config
        )
    
    async def generate_error_feedback(
        self,
        error_type: str,
        error_details: str,
        step_info: Dict[str, Any],
        wrong_step_info: Optional[Dict[str, Any]] = None,
        sop_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate AI-powered feedback for an error during assessment.
        
        Args:
            error_type: Type of error (WRONG_ORDER, TIMEOUT, MISSED_STEP)
            error_details: Human-readable error description
            step_info: Information about the expected step
            wrong_step_info: Information about the wrong step touched (if applicable)
            sop_rules: Full SOP rules for context
            
        Returns:
            Dictionary with feedback, suggestions, and document references
        """
        # Build context from step info
        step_instructions = step_info.get('instructions', [])
        step_errors = step_info.get('common_errors', [])
        
        # Search for relevant documents
        search_query = self._build_search_query(error_type, step_info, wrong_step_info)
        relevant_docs = await self._search_relevant_docs(search_query)
        
        # Generate AI feedback
        feedback = await self._generate_feedback(
            error_type=error_type,
            error_details=error_details,
            step_info=step_info,
            wrong_step_info=wrong_step_info,
            step_instructions=step_instructions,
            step_errors=step_errors,
            relevant_docs=relevant_docs
        )
        
        return feedback
    
    def _build_search_query(
        self,
        error_type: str,
        step_info: Dict[str, Any],
        wrong_step_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build search query for finding relevant documents."""
        step_desc = step_info.get('description', '')
        
        if error_type == "WRONG_ORDER":
            wrong_desc = wrong_step_info.get('description', '') if wrong_step_info else ''
            return f"quy trình lắp ráp thứ tự đúng {step_desc} {wrong_desc} lỗi sai bước"
        elif error_type == "TIMEOUT":
            return f"hướng dẫn thực hiện {step_desc} cách làm nhanh hiệu quả"
        else:
            return f"quy trình lắp ráp {step_desc} hướng dẫn chi tiết"
    
    async def _search_relevant_docs(
        self, 
        query: str, 
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Search for relevant document chunks."""
        try:
            # Generate embedding for query
            query_embedding = await self.vectorizer.get_embedding(query)
            
            # Search for similar chunks
            result = await self.db.execute(
                select(DocumentChunk, Document)
                .join(Document, DocumentChunk.document_id == Document.id)
                .order_by(
                    DocumentChunk.embedding.cosine_distance(query_embedding)
                )
                .limit(top_k)
            )
            
            chunks = result.fetchall()
            
            docs = []
            for chunk, document in chunks:
                docs.append({
                    "content": chunk.content,
                    "document_name": document.file_name,
                    "document_id": str(document.id),
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index
                })
            
            return docs
            
        except Exception as e:
            log.error(f"Error searching documents: {e}")
            return []
    
    async def _generate_feedback(
        self,
        error_type: str,
        error_details: str,
        step_info: Dict[str, Any],
        wrong_step_info: Optional[Dict[str, Any]],
        step_instructions: List[str],
        step_errors: List[str],
        relevant_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate AI feedback using Gemini."""
        
        # Build document context
        doc_context = ""
        if relevant_docs:
            doc_parts = []
            for doc in relevant_docs:
                doc_parts.append(f"- {doc['content'][:500]}... (Tài liệu: {doc['document_name']}, Trang: {doc.get('page_number', 'N/A')})")
            doc_context = "\n".join(doc_parts)
        
        # Build step context
        instructions_text = "\n".join(f"- {instr}" for instr in step_instructions) if step_instructions else "Không có hướng dẫn chi tiết"
        errors_text = "\n".join(f"- {err}" for err in step_errors) if step_errors else "Không có thông tin lỗi phổ biến"
        
        wrong_step_context = ""
        if wrong_step_info:
            wrong_step_context = f"""
Bước sai đã thực hiện:
- ID: {wrong_step_info.get('step_id', 'N/A')}
- Mô tả: {wrong_step_info.get('description', 'N/A')}
"""
        
        prompt = f"""Bạn là chuyên gia đào tạo quy trình lắp ráp công nghiệp. Hãy phân tích lỗi sau và đưa ra phản hồi hữu ích cho người học.

LOẠI LỖI: {error_type}
CHI TIẾT LỖI: {error_details}

BÍC CẦN THỰC HIỆN:
- ID: {step_info.get('step_id', 'N/A')}
- Mô tả: {step_info.get('description', 'N/A')}
- Đối tượng: {step_info.get('target_object', 'N/A')}

HƯỚNG DẪN THỰC HIỆN BƯỚC NÀY:
{instructions_text}

LỖI PHỔ BIẾN CỦA BƯỚC NÀY:
{errors_text}
{wrong_step_context}

TÀI LIỆU LIÊN QUAN:
{doc_context if doc_context else "Không tìm thấy tài liệu liên quan"}

Hãy tạo phản hồi JSON với cấu trúc sau:
{{
    "feedback_message": "Thông điệp phản hồi chính cho người học (ngắn gọn, rõ ràng, khuyến khích)",
    "error_explanation": "Giải thích tại sao đây là lỗi và hậu quả có thể xảy ra",
    "correction_steps": ["Bước 1 để sửa lỗi", "Bước 2...", "..."],
    "tips": ["Mẹo 1 để tránh lỗi tương tự", "Mẹo 2...", "..."],
    "document_references": [
        {{
            "document_name": "Tên tài liệu",
            "page_number": "Số trang hoặc null",
            "description": "Mô tả ngắn về nội dung liên quan"
        }}
    ],
    "severity": "low/medium/high",
    "encouragement": "Lời động viên cho người học"
}}

Lưu ý:
- Sử dụng tiếng Việt tự nhiên, thân thiện
- Tập trung vào việc giúp người học cải thiện
- Đưa ra lời khuyên cụ thể và thực tế
"""
        
        try:
            response = await self.model.generate_content_async(prompt)
            feedback_data = json.loads(response.text)
            
            # Add source document info if available
            if relevant_docs:
                feedback_data["source_documents"] = [
                    {
                        "document_id": doc["document_id"],
                        "document_name": doc["document_name"],
                        "page_number": doc.get("page_number")
                    }
                    for doc in relevant_docs
                ]
            
            return feedback_data
            
        except Exception as e:
            log.error(f"Error generating feedback: {e}")
            # Return basic fallback feedback
            return self._generate_fallback_feedback(
                error_type, error_details, step_info, step_instructions, step_errors
            )
    
    def _generate_fallback_feedback(
        self,
        error_type: str,
        error_details: str,
        step_info: Dict[str, Any],
        step_instructions: List[str],
        step_errors: List[str]
    ) -> Dict[str, Any]:
        """Generate fallback feedback when AI generation fails."""
        
        if error_type == "WRONG_ORDER":
            feedback_message = f"Bạn đã thực hiện sai thứ tự. Hãy thực hiện bước: {step_info.get('description', '')}"
            severity = "high"
        elif error_type == "TIMEOUT":
            feedback_message = f"Bạn đã mất quá nhiều thời gian. Hãy tiếp tục với bước: {step_info.get('description', '')}"
            severity = "medium"
        else:
            feedback_message = f"Đã xảy ra lỗi. Hãy kiểm tra lại bước: {step_info.get('description', '')}"
            severity = "low"
        
        return {
            "feedback_message": feedback_message,
            "error_explanation": error_details,
            "correction_steps": step_instructions[:3] if step_instructions else ["Xem lại hướng dẫn và thử lại"],
            "tips": step_errors[:2] if step_errors else ["Chú ý thực hiện đúng thứ tự"],
            "document_references": [],
            "severity": severity,
            "encouragement": "Đừng lo lắng! Hãy bình tĩnh và thử lại. Bạn sẽ làm được!"
        }
    
    async def generate_completion_summary(
        self,
        report: Dict[str, Any],
        sop_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a summary and recommendations after assessment completion.
        
        Args:
            report: The full assessment report from SOPEngine
            sop_rules: The SOP rules used for assessment
            
        Returns:
            Dictionary with summary, performance analysis, and recommendations
        """
        step_details = report.get('step_details', [])
        error_logs = report.get('error_logs', [])
        failed_steps = report.get('failed_steps', [])  # Get failed steps with descriptions
        total_duration = report.get('total_duration', 0)
        total_steps = report.get('total_steps', 0)
        completed_steps = report.get('completed_steps', 0)
        is_passed = report.get('is_passed', False)
        
        # Calculate original score (completed/total * 100)
        original_score = (completed_steps / total_steps * 100) if total_steps > 0 else 0
        
        # Calculate statistics
        step_times = [step['duration'] for step in step_details]
        avg_time = sum(step_times) / len(step_times) if step_times else 0
        slowest_step = max(step_details, key=lambda x: x['duration']) if step_details else None
        
        # Build failed steps info for AI to recommend learning
        failed_steps_info = ""
        if failed_steps:
            failed_steps_info = "\nCÁC BƯỚC THẤT BẠI (CẦN HỌC LẠI):\n"
            for fs in failed_steps:
                failed_steps_info += f"""
- Bước {fs.get('step_id', 'N/A')}: {fs.get('description', 'N/A')}
  + Lỗi: {fs.get('error_type', 'UNKNOWN')} - {fs.get('error_details', '')}
  + Đối tượng mục tiêu: {fs.get('target_object', 'N/A')}
  + Hướng dẫn đúng: {', '.join(fs.get('instructions', ['Không có']))}
"""
                if fs.get('wrong_step_id'):
                    failed_steps_info += f"  + Bước sai đã chạm: {fs.get('wrong_step_description', fs.get('wrong_step_id'))}\n"
        
        prompt = f"""Bạn là chuyên gia đánh giá đào tạo SOP (Standard Operating Procedure). 
Hãy phân tích kết quả bài kiểm tra và đưa ra nhận xét chi tiết, đặc biệt là HƯỚNG DẪN HỌC LẠI cho các bước thất bại.

KẾT QUẢ: {'ĐẠT' if is_passed else 'KHÔNG ĐẠT'}
ĐIỂM SỐ: {original_score:.0f}% ({completed_steps}/{total_steps} bước)
TỔNG THỜI GIAN: {total_duration:.1f}s
SỐ LỖI: {len(error_logs)}

CHI TIẾT CÁC BƯỚC ĐÃ HOÀN THÀNH:
{json.dumps(step_details, ensure_ascii=False, indent=2)}
{failed_steps_info}
CÁC LỖI ĐÃ GHI NHẬN:
{json.dumps(error_logs, ensure_ascii=False, indent=2) if error_logs else "Không có lỗi"}

Hãy tạo phản hồi JSON với cấu trúc sau:
{{
    "overall_rating": "excellent/good/average/needs_improvement",
    "summary": "Tóm tắt ngắn gọn về kết quả, nêu rõ các bước đã làm tốt và bước nào cần cải thiện",
    "strengths": ["Điểm mạnh cụ thể 1", "Điểm mạnh cụ thể 2"],
    "areas_for_improvement": ["Bước/kỹ năng cần cải thiện 1 (dựa trên failed_steps)", "..."],
    "specific_recommendations": [
        {{
            "area": "Tên bước/kỹ năng cần học lại (VD: Bước 2 - Lắp hai tay)",
            "recommendation": "Hướng dẫn chi tiết cách học lại bước này, các tài liệu tham khảo",
            "priority": "high/medium/low"
        }}
    ],
    "learning_resources": [
        {{
            "step_id": "ID bước cần học",
            "step_name": "Tên bước",
            "what_to_learn": "Nội dung cần học",
            "how_to_practice": "Cách luyện tập"
        }}
    ],
    "next_steps": ["Hành động cụ thể tiếp theo 1", "Hành động cụ thể 2"],
    "encouragement": "Lời động viên phù hợp với kết quả"
}}

QUAN TRỌNG:
- Với mỗi bước thất bại, hãy đưa ra hướng dẫn học lại cụ thể trong specific_recommendations
- Trong learning_resources, liệt kê chi tiết các bước cần học lại
- Đánh giá overall_rating dựa trên điểm số: excellent (>=90%), good (>=70%), average (>=50%), needs_improvement (<50%)
- Tập trung vào việc giúp người học biết CẦN HỌC GÌ và LÀM THẾ NÀO
"""
        
        try:
            response = await self.model.generate_content_async(prompt)
            summary_data = json.loads(response.text)
            
            # Add raw statistics (use original score, not AI score)
            summary_data["statistics"] = {
                "total_duration": total_duration,
                "average_step_time": round(avg_time, 2),
                "steps_completed": completed_steps,
                "total_steps": total_steps,
                "error_count": len(error_logs),
                "failed_steps_count": len(failed_steps),
                "slowest_step": slowest_step['step_id'] if slowest_step else None,
                "slowest_step_time": slowest_step['duration'] if slowest_step else None
            }
            
            # Override AI score with original score
            summary_data["score"] = round(original_score)
            
            return summary_data
            
        except Exception as e:
            log.error(f"Error generating completion summary: {e}")
            # Return basic fallback summary
            return {
                "overall_rating": "average" if result == "PASSED" else "needs_improvement",
                "score": 70 if result == "PASSED" else 40,
                "summary": f"Hoàn thành bài đánh giá trong {total_duration}s với {len(error_logs)} lỗi.",
                "strengths": ["Đã hoàn thành bài đánh giá"] if result == "PASSED" else [],
                "areas_for_improvement": [f"Giảm số lỗi ({len(error_logs)} lỗi)"] if error_logs else [],
                "specific_recommendations": [],
                "next_steps": ["Xem lại các bước gặp lỗi", "Luyện tập thêm"],
                "encouragement": "Tiếp tục cố gắng! Mỗi lần luyện tập là một cơ hội để tiến bộ.",
                "statistics": {
                    "total_duration": total_duration,
                    "average_step_time": round(avg_time, 2),
                    "steps_completed": len(step_details),
                    "total_steps": report.get('total_steps', 0),
                    "error_count": len(error_logs)
                }
            }
