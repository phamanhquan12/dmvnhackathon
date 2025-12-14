# Example Usage (Async)
import json
import logging as log
import google.generativeai as genai
from app.core.database import async_session
from app.services.rag.utils import create_quiz_prompt, create_flash_cards_prompt
from typing import List, Optional, Tuple, Dict, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.document import Document, FileTypeEnum
from app.models.chunk import DocumentChunk
from app.models.session import TheorySession
from app.services.rag.vectorizer import Vectorizer
from app.core.config import settings

# Configure GenAI
genai.configure(api_key=settings.GOOGLE_API_KEY)

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ChatService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.vectorizer = Vectorizer()
        # Using gemini-2.5-flash as main model
        self.chat_config = genai.GenerationConfig(
            response_mime_type="text/plain",
            temperature=0.2,  # Lower temp for more accurate answers
        )
        self.query_transform_config = genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.3,
        )
        self.quiz_config = genai.GenerationConfig(
            response_mime_type= "application/json",
            temperature=0.8,
        )
        self.flash_cards_config = genai.GenerationConfig(
            response_mime_type= "application/json",
            temperature=0.8,
        )
        self.model = genai.GenerativeModel('gemini-2.5-flash', generation_config=self.chat_config)
        self.query_model = genai.GenerativeModel('gemini-2.5-flash', generation_config=self.query_transform_config)
        self.quiz_model = genai.GenerativeModel('gemini-2.5-flash', generation_config=self.quiz_config)
        self.flash_cards_model = genai.GenerativeModel('gemini-2.5-flash', generation_config=self.flash_cards_config)
    
    async def transform_query(
        self, 
        query: str, 
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> List[str]:
        """
        Transform user query into multiple search queries for better retrieval.
        Uses chat history context if available.
        """
        history_context = ""
        last_topic = ""
        if chat_history and len(chat_history) > 0:
            # Use last 3 exchanges for context
            recent_history = chat_history[-6:]  # 3 exchanges = 6 messages
            history_parts = []
            for msg in recent_history:
                role = "User" if msg.get('role') == 'user' else "Assistant"
                content = msg.get('content', '')[:500]  # More context
                history_parts.append(f"{role}: {content}")
                if msg.get('role') == 'user':
                    last_topic = msg.get('content', '')[:100]
            history_context = f"\n\nLịch sử hội thoại gần đây:\n" + "\n".join(history_parts)
        
        prompt = f"""Bạn là chuyên gia tìm kiếm thông tin. Nhiệm vụ của bạn là phân tích câu hỏi của người dùng và tạo các truy vấn tìm kiếm tối ưu.
{history_context}

Câu hỏi hiện tại: "{query}"

Yêu cầu:
1. Phân tích ý định và ngữ cảnh của câu hỏi
2. **QUAN TRỌNG**: Nếu câu hỏi ngắn gọn hoặc thiếu ngữ cảnh (như "bước 1", "nó là gì", "tiếp theo"):
   - PHẢI kết hợp với lịch sử hội thoại để hiểu ngữ cảnh đầy đủ
   - Mở rộng truy vấn để bao gồm chủ đề đang thảo luận
   - Ví dụ: nếu đang nói về "lắp ráp robot" và user hỏi "bước 1 là gì" → tìm kiếm "bước 1 lắp ráp robot quy trình"
3. Tạo 3-5 truy vấn tìm kiếm khác nhau:
   - Truy vấn đầy đủ ngữ cảnh (kết hợp câu hỏi + chủ đề từ lịch sử)
   - Truy vấn mở rộng (thêm từ khóa liên quan)
   - Truy vấn cụ thể (nếu có thể xác định khái niệm cụ thể)
   - Nếu câu hỏi về quy trình/các bước, tạo truy vấn cho từng bước
4. **TUYỆT ĐỐI KHÔNG** tạo truy vấn quá ngắn như chỉ "bước 1" - phải có ngữ cảnh

Trả về JSON:
{{
    "queries": ["truy_van_1", "truy_van_2", "truy_van_3", "truy_van_4", "truy_van_5"],
    "intent": "mô_tả_ngắn_ý_định",
    "expanded_query": "câu_hỏi_đầy_đủ_với_ngữ_cảnh"
}}
"""
        
        try:
            response = self.query_model.generate_content(prompt)
            data = json.loads(response.text)
            queries = data.get("queries", [query])
            expanded = data.get("expanded_query", query)
            
            # Always include the expanded query if available
            if expanded and expanded not in queries:
                queries.insert(0, expanded)
            
            log.info(f"Transformed query '{query}' into: {queries}")
            return queries if queries else [query]
        except Exception as e:
            log.warning(f"Query transformation failed: {e}, using original query")
            # Fallback: if there's history context, combine query with last topic
            if last_topic and len(query) < 30:
                return [f"{query} {last_topic}", query]
            return [query]
    
    async def get_document_chunk_count(self, document_ids: List[int]) -> int:
        """Get total chunk count for selected documents."""
        try:
            result = await self.db.execute(
                select(DocumentChunk.id).where(DocumentChunk.document_id.in_(document_ids))
            )
            return len(result.scalars().all())
        except:
            return 0
        
    async def retrieve_context(
        self, 
        query: str, 
        limit: int = 5, 
        document_ids: Optional[List[int]] = None
    ) -> Tuple[List[DocumentChunk], List[str]]:
        """
        Embeds the query and retrieves the most relevant document chunks from the database.
        Optionally filters by document IDs.
        Returns both chunks and their IDs for progress tracking.
        """
        query_embedding = self.vectorizer.embed_query(query)[0]
        if not query_embedding:
            log.warning("Failed to generate embedding for query.")
            return [], []

        try:
            # Search using cosine distance (pgvector)
            if document_ids:
                # Filter by specific document IDs
                stmt = select(DocumentChunk).where(
                    DocumentChunk.document_id.in_(document_ids)
                ).order_by(
                    DocumentChunk.embedding.cosine_distance(query_embedding)
                ).limit(limit)
            else:
                # Search all documents
                stmt = select(DocumentChunk).order_by(
                    DocumentChunk.embedding.cosine_distance(query_embedding)
                ).limit(limit)

            result = await self.db.execute(stmt)
            chunks = result.scalars().all()
            chunk_ids = [c.id for c in chunks]
            log.info(f"Retrieved {len(chunks)} chunks for query: {query} (filtered by doc_ids: {document_ids})")
            return chunks, chunk_ids
        except Exception as e:
            log.error(f"Error retrieving context: {e}")
            return [], []
    
    async def retrieve_all_chunks(
        self,
        document_ids: List[int],
        limit: int = 30
    ) -> Tuple[List[DocumentChunk], List[str]]:
        """
        Retrieve all chunks from specific documents, sorted by page number.
        Used when we want comprehensive coverage of a small document.
        """
        try:
            stmt = select(DocumentChunk).where(
                DocumentChunk.document_id.in_(document_ids)
            ).order_by(
                DocumentChunk.document_id,
                DocumentChunk.page_number
            ).limit(limit)
            
            result = await self.db.execute(stmt)
            chunks = result.scalars().all()
            chunk_ids = [str(c.id) for c in chunks]
            log.info(f"Retrieved all {len(chunks)} chunks from documents: {document_ids}")
            return chunks, chunk_ids
        except Exception as e:
            log.error(f"Error retrieving all chunks: {e}")
            return [], []
    
    async def retrieve_with_transformation(
        self,
        query: str,
        limit: int = 8,
        document_ids: Optional[List[int]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[List[DocumentChunk], List[str]]:
        """
        Enhanced retrieval with query transformation.
        Uses multiple queries and deduplicates results.
        """
        # Transform the query
        queries = await self.transform_query(query, chat_history)
        
        all_chunks = []
        all_chunk_ids = set()
        seen_content = set()
        
        # Get more chunks per query when we have multiple queries
        chunks_per_query = max(limit // len(queries) + 3, 6)
        
        for q in queries:
            chunks, chunk_ids = await self.retrieve_context(q, limit=chunks_per_query, document_ids=document_ids)
            for chunk, cid in zip(chunks, chunk_ids):
                # Deduplicate by chunk id and similar content
                content_key = chunk.content[:100]  # Use first 100 chars as key
                if cid not in all_chunk_ids and content_key not in seen_content:
                    all_chunks.append(chunk)
                    all_chunk_ids.add(cid)
                    seen_content.add(content_key)
        
        # Sort chunks by page number within each document for sequential reading
        all_chunks.sort(key=lambda c: (c.document_id, c.page_number))
        
        # Limit total results
        all_chunks = all_chunks[:limit]
        
        return all_chunks, [str(c.id) for c in all_chunks]
    
    async def format_context_with_citations(
        self, 
        chunks: List[DocumentChunk]
    ) -> Tuple[str, Dict[int, Dict[str, Any]]]:
        """
        Format context with citation markers and build citation metadata.
        Returns formatted context and citation lookup dictionary.
        """
        context_parts = []
        citations = {}
        
        # Fetch document titles and file types for all unique document IDs
        doc_ids = list(set(chunk.document_id for chunk in chunks))
        doc_info = {}
        try:
            result = await self.db.execute(
                select(Document.id, Document.title, Document.file_type).where(Document.id.in_(doc_ids))
            )
            for doc_id, title, file_type in result.all():
                doc_info[doc_id] = {"title": title, "file_type": file_type}
        except Exception as e:
            log.warning(f"Could not fetch document info: {e}")
        
        for idx, chunk in enumerate(chunks, 1):
            info = doc_info.get(chunk.document_id, {"title": f"Document {chunk.document_id}", "file_type": "PDF"})
            doc_title = info["title"]
            file_type = info["file_type"]
            
            # Build citation metadata
            citations[idx] = {
                "document_id": chunk.document_id,
                "document_title": doc_title,
                "file_type": file_type,
                "page_number": chunk.page_number,
                "chunk_id": str(chunk.id),
                "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            }
            
            # Format context with citation marker
            if file_type == "VIDEO":
                # For videos, page_number represents timestamp in seconds
                mins = chunk.page_number // 60
                secs = chunk.page_number % 60
                context_parts.append(f"[Nguồn {idx} - {doc_title} - [{mins}:{secs:02d}]]\n{chunk.content}")
            else:
                context_parts.append(f"[Nguồn {idx} - {doc_title} - Trang {chunk.page_number}]\n{chunk.content}")
        
        return "\n\n---\n\n".join(context_parts), citations

    async def get_all_documents(self) -> List[Document]:
        """
        Retrieves all documents from the database for UI selection.
        """
        try:
            result = await self.db.execute(select(Document))
            documents = result.scalars().all()
            log.info(f"Retrieved {len(documents)} documents from database.")
            return documents
        except Exception as e:
            log.error(f"Error retrieving documents: {e}")
            return []

    async def chat(
        self, 
        query: str, 
        document_ids: Optional[List[int]] = None,
        user_id: Optional[UUID] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, List[str], Dict[int, Dict[str, Any]]]:
        """
        Enhanced RAG flow with query transformation, chat history, and citations.
        Returns response, chunk IDs, and citation metadata.
        """
        # Check if we should retrieve all chunks (for small documents)
        use_all_chunks = False
        if document_ids:
            total_chunks = await self.get_document_chunk_count(document_ids)
            # If document(s) have 30 or fewer chunks, retrieve all for comprehensive coverage
            if total_chunks <= 30:
                use_all_chunks = True
                log.info(f"Small document detected ({total_chunks} chunks), retrieving all chunks")
        
        # 1. Retrieve chunks
        if use_all_chunks and document_ids:
            chunks, chunk_ids = await self.retrieve_all_chunks(document_ids, limit=30)
        else:
            chunks, chunk_ids = await self.retrieve_with_transformation(
                query, 
                limit=20,  # Get more context for comprehensive answers about procedures
                document_ids=document_ids,
                chat_history=chat_history
            )
        
        if not chunks:
            return "Tôi không tìm thấy thông tin liên quan trong tài liệu để trả lời câu hỏi của bạn.", [], {}

        # 2. Format context with citations
        context_text, citations = await self.format_context_with_citations(chunks)
        
        # 3. Build chat history context
        history_context = ""
        if chat_history and len(chat_history) > 0:
            recent_history = chat_history[-6:]  # Last 3 exchanges
            history_parts = []
            for msg in recent_history:
                role = "Người dùng" if msg.get('role') == 'user' else "Trợ lý"
                content = msg.get('content', '')[:300]  # Limit length
                history_parts.append(f"{role}: {content}")
            history_context = "\n\nLịch sử hội thoại gần đây:\n" + "\n".join(history_parts) + "\n"
        
        # 4. Construct enhanced prompt
        prompt = f"""Bạn là trợ lý kỹ thuật chuyên nghiệp của DENSO. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên tài liệu được cung cấp.
{history_context}
NGUYÊN TẮC TRẢ LỜI:
1. Trả lời CHI TIẾT và ĐẦY ĐỦ - cung cấp thông tin toàn diện, không trả lời quá ngắn gọn
2. Sử dụng cấu trúc rõ ràng với bullet points hoặc đánh số khi liệt kê
3. LUÔN TRÍCH DẪN NGUỒN bằng cách thêm [Nguồn X] sau thông tin lấy từ nguồn đó
4. Nếu có nhiều nguồn nói về cùng một điều, tổng hợp và trích dẫn tất cả các nguồn liên quan
5. Nếu thông tin không có trong tài liệu, hãy nói rõ điều đó
6. Trả lời bằng Tiếng Việt
7. **QUY TRÌNH/CÁC BƯỚC**: Khi câu hỏi về quy trình hoặc các bước thực hiện:
   - Tìm và liệt kê TẤT CẢ các bước từ Bước 1 đến bước cuối cùng
   - Kiểm tra kỹ tài liệu để không bỏ sót bước nào
   - Trình bày theo thứ tự: Bước 1, Bước 2, Bước 3, Bước 4...
   - Nếu thiếu thông tin về bước nào, ghi rõ "Bước X: Không tìm thấy thông tin chi tiết trong tài liệu"
8. Với các thông số kỹ thuật, cung cấp giá trị cụ thể nếu có

TÀI LIỆU THAM KHẢO:
---
{context_text}
---

Câu hỏi: {query}

Trả lời chi tiết và trích dẫn nguồn:"""

        # 5. Generate Response
        try:
            log.info("Sending enhanced prompt to LLM...")
            response = self.model.generate_content(prompt)
            return response.text, chunk_ids, citations
        except Exception as e:
            log.error(f"Error generating response from LLM: {e}")
            return "Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu của bạn.", chunk_ids, citations
    
    async def chat_stream(
        self, 
        query: str, 
        document_ids: Optional[List[int]] = None,
        user_id: Optional[UUID] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ):
        """
        Streaming version of chat - yields response chunks as they're generated.
        Returns a generator for streaming and also yields citations at the end.
        """
        # Check if we should retrieve all chunks (for small documents)
        use_all_chunks = False
        if document_ids:
            total_chunks = await self.get_document_chunk_count(document_ids)
            if total_chunks <= 30:
                use_all_chunks = True
                log.info(f"Small document detected ({total_chunks} chunks), retrieving all chunks")
        
        # 1. Retrieve chunks
        if use_all_chunks and document_ids:
            chunks, chunk_ids = await self.retrieve_all_chunks(document_ids, limit=30)
        else:
            chunks, chunk_ids = await self.retrieve_with_transformation(
                query, 
                limit=20,
                document_ids=document_ids,
                chat_history=chat_history
            )
        
        if not chunks:
            yield {"type": "text", "content": "Tôi không tìm thấy thông tin liên quan trong tài liệu để trả lời câu hỏi của bạn."}
            yield {"type": "citations", "data": {}, "chunk_ids": []}
            return

        # 2. Format context with citations
        context_text, citations = await self.format_context_with_citations(chunks)
        
        # 3. Build chat history context
        history_context = ""
        if chat_history and len(chat_history) > 0:
            recent_history = chat_history[-6:]
            history_parts = []
            for msg in recent_history:
                role = "Người dùng" if msg.get('role') == 'user' else "Trợ lý"
                content = msg.get('content', '')[:300]
                history_parts.append(f"{role}: {content}")
            history_context = "\n\nLịch sử hội thoại gần đây:\n" + "\n".join(history_parts) + "\n"
        
        # 4. Construct enhanced prompt
        prompt = f"""Bạn là trợ lý kỹ thuật chuyên nghiệp của DENSO. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên tài liệu được cung cấp.
{history_context}
NGUYÊN TẮC TRẢ LỜI:
1. Trả lời CHI TIẾT và ĐẦY ĐỦ - cung cấp thông tin toàn diện, không trả lời quá ngắn gọn
2. Sử dụng cấu trúc rõ ràng với bullet points hoặc đánh số khi liệt kê
3. LUÔN TRÍCH DẪN NGUỒN bằng cách thêm [Nguồn X] sau thông tin lấy từ nguồn đó
4. Nếu có nhiều nguồn nói về cùng một điều, tổng hợp và trích dẫn tất cả các nguồn liên quan
5. Nếu thông tin không có trong tài liệu, hãy nói rõ điều đó
6. Trả lời bằng Tiếng Việt
7. **QUY TRÌNH/CÁC BƯỚC**: Khi câu hỏi về quy trình hoặc các bước thực hiện:
   - Tìm và liệt kê TẤT CẢ các bước từ Bước 1 đến bước cuối cùng
   - Kiểm tra kỹ tài liệu để không bỏ sót bước nào
   - Trình bày theo thứ tự: Bước 1, Bước 2, Bước 3, Bước 4...
   - Nếu thiếu thông tin về bước nào, ghi rõ "Bước X: Không tìm thấy thông tin chi tiết trong tài liệu"
8. Với các thông số kỹ thuật, cung cấp giá trị cụ thể nếu có

TÀI LIỆU THAM KHẢO:
---
{context_text}
---

Câu hỏi: {query}

Trả lời chi tiết và trích dẫn nguồn:"""

        # 5. Generate Response with streaming
        try:
            log.info("Sending enhanced prompt to LLM with streaming...")
            response = self.model.generate_content(prompt, stream=True)
            
            for chunk in response:
                if chunk.text:
                    yield {"type": "text", "content": chunk.text}
            
            # Yield citations at the end
            yield {"type": "citations", "data": citations, "chunk_ids": chunk_ids}
            
        except Exception as e:
            log.error(f"Error generating streaming response from LLM: {e}")
            yield {"type": "text", "content": "Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu của bạn."}
            yield {"type": "citations", "data": {}, "chunk_ids": []}
    
    # Legacy method for backward compatibility
    async def chat_simple(
        self, 
        query: str, 
        document_ids: Optional[List[int]] = None,
        user_id: Optional[UUID] = None
    ) -> Tuple[str, List[str]]:
        """Simple chat without history - for backward compatibility."""
        response, chunk_ids, _ = await self.chat(query, document_ids, user_id, None)
        return response, chunk_ids
        
    async def quiz(self, document_ids: Optional[List[int]] = None) -> Tuple[List[dict], List[str]]:
        """
        Generates quiz questions based on document chunks.
        Optionally filters by document IDs.
        Returns quiz data and chunk IDs for progress tracking.
        """
        if document_ids:
            all_chunks = await self.db.execute(
                select(DocumentChunk).where(DocumentChunk.document_id.in_(document_ids))
            )
        else:
            all_chunks = await self.db.execute(select(DocumentChunk))
        
        chunks = all_chunks.scalars().all()
        chunk_ids = [c.id for c in chunks]
        log.info(f"Total chunks available for quiz generation: {len(chunks)} (filtered by doc_ids: {document_ids})")
        context_parts = []
        for c in chunks:
            context_parts.append(f"[Page {c.page_number}] {c.content}")
        
        context_text = "\n\n".join(context_parts)
        
        prompt = create_quiz_prompt(context_text)
        try:
            log.info("Sending quiz prompt to LLM...")
            # Use sync version to avoid event loop conflicts with Streamlit
            response = self.quiz_model.generate_content(prompt)
            log.info("Quiz generated successfully.")
            return json.loads(response.text), chunk_ids
        except Exception as e:
            log.error(f"Error generating quiz from LLM: {e}")
            return [], chunk_ids
    async def grading(self, quiz_response : List[dict], employee_id: str):
        """
        grade based on user answers, and save necessary data into theory session
        """
        possible_choices = ['A', 'B', 'C', 'D']
        score = 0
        wrong_questions_answers = {'question': [], 'user_answer': [], 'correct_answer': [], 'explanation': []}
        for i, q in enumerate(quiz_response):
            question = q.get("question")
            options = q.get("options", [])
            log.info(f"Question {i+1}: {question}")
            for option in options:
                log.info(option)
            user_answer = input(f"Your answer for question {i+1} (A/B/C/D): ")
            if user_answer not in possible_choices:
                log.warning(f"Invalid answer '{user_answer}' for question {i+1}. Skipping grading for this question.")
                continue
            correct_answer = q.get("answer")    
            explaination = q.get("explanation", "")
            user_answer_index = possible_choices.index(user_answer)
            user_answer_text = options[user_answer_index]
            if user_answer_text == correct_answer:
                log.info(f"Question {i+1}: Correct")
                score += 1
            else:
                wrong_questions_answers['question'].append(question)
                wrong_questions_answers['user_answer'].append(user_answer_text)
                wrong_questions_answers['correct_answer'].append(correct_answer)
                wrong_questions_answers['explanation'].append(explaination)
                log.info(f"Question {i+1}: Incorrect. Correct answer: {correct_answer}. Explanation: {explaination}")
        log.info(f"Your total score: {score} out of {len(quiz_response)}")
        status = "PASSED" if len(quiz_response)//score >= 0.6 else "FAILED"
        # Save to TheorySession
        session_obj = TheorySession(
            user_id = employee_id,
            score = score,
            status = status,
            details = wrong_questions_answers
        )
        self.db.add(session_obj)
        await self.db.commit()
        await self.db.refresh(session_obj)
        log.info(f"Saved TheorySession with ID: {session_obj.id} for user: {employee_id}")
        return {
            "score": score,
            "status": status,
            "details": wrong_questions_answers
            }
            


    async def pipeline(self) -> str:
        query = input("Enter your query (or 'quiz:' to generate quiz, 'quit' to exit): ")
        if query.startswith("quiz"):
            quiz_data, _ = await self.quiz()
            return quiz_data
        elif query.__contains__("quit"):
            return "Exiting pipeline."
        else:
            response, _ = await self.chat(query)
            return response
            
    async def flash_cards(self, id: Optional[int] = None, title: Optional[str] = None) -> Tuple[List[dict], List[str]]:
        """
        Generate flash cards based on id'd documents or title.
        Returns flashcards and chunk IDs for progress tracking.
        """
        if id:
            all_chunks = await self.db.execute(select(DocumentChunk).where(DocumentChunk.document_id == id))
        elif title:
            all_chunks = await self.db.execute(
                select(DocumentChunk).join_from(
                    DocumentChunk, 
                    Document, 
                    Document.id == DocumentChunk.document_id
                ).where(Document.title == title)
            )
        else:
            return [], []

        chunks = all_chunks.scalars().all()
        chunk_ids = [c.id for c in chunks]
        log.info(f"Total chunks available for flash card generation: {len(chunks)}")
        context_parts = []
        for c in chunks:
            context_parts.append(f"[Page {c.page_number}] {c.content}")
        
        context_text = "\n\n".join(context_parts)
        
        prompt = create_flash_cards_prompt(context_text)
        try:
            log.info("Sending flash cards prompt to LLM...")
            # Use sync version to avoid event loop conflicts with Streamlit
            response = self.flash_cards_model.generate_content(prompt)
            log.info("Flash cards generated successfully.")
            return json.loads(response.text), chunk_ids
        except Exception as e:
            log.error(f"Error generating flash cards from LLM: {e}")
            return [], chunk_ids
if __name__ == "__main__":
    import asyncio
    from app.core.database import async_session
    
        # async def test_chat():
        #     async with async_session() as session:
        #         chat_service = ChatService(session)
        #         query = "What is the document about?" # Example query
        #         print(f"Query: {query}")
        #         response = await chat_service.chat(query)
        #         print(f"Response: {response}")

    # asyncio.run(test_chat())
    async def test():
        async with async_session() as session:
            chat_service = ChatService(session)
            while True:
                response = await chat_service.pipeline()
                log.info(f"Pipeline response: {response}")
                if response == "Exiting pipeline.":
                    break
    asyncio.run(test())