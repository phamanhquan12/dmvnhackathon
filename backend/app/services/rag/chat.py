# Example Usage (Async)
import json
import logging as log
import google.generativeai as genai
from app.core.database import async_session
from app.services.rag.utils import create_quiz_prompt, create_flash_cards_prompt
from typing import List, Optional
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
        # Using gemini-1.5-flash as it is the current standard Flash model. 
        # If "2.5 Flash" becomes available, update the model name here.
        self.chat_config = genai.GenerationConfig(
            response_mime_type="text/plain",
            temperature=0.1,
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
        self.quiz_model = genai.GenerativeModel('gemini-2.5-flash', generation_config=self.quiz_config)
        self.flash_cards_model = genai.GenerativeModel('gemini-2.5-flash', generation_config=self.flash_cards_config)
        
    async def retrieve_context(self, query: str, limit: int = 5, document_ids: Optional[List[int]] = None) -> List[DocumentChunk]:
        """
        Embeds the query and retrieves the most relevant document chunks from the database.
        Optionally filters by document IDs.
        """
        query_embedding = self.vectorizer.embed_query(query)[0]
        if not query_embedding:
            log.warning("Failed to generate embedding for query.")
            return []

        try:
            # Search using cosine distance (pgvector)
            # Note: pgvector's cosine_distance operator is <=>
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
            log.info(f"Retrieved {len(chunks)} chunks for query: {query} (filtered by doc_ids: {document_ids})")
            return chunks
        except Exception as e:
            log.error(f"Error retrieving context: {e}")
            return []

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

    async def chat(self, query: str, document_ids: Optional[List[int]] = None) -> str:
        """
        Orchestrates the RAG flow: Retrieve -> Generate.
        Optionally filters retrieval by document IDs.
        """
        # 1. Retrieve relevant chunks (with optional document filtering)
        chunks = await self.retrieve_context(query, document_ids=document_ids)
        
        if not chunks:
            return "I couldn't find any relevant information in the documents to answer your question."

        # 2. Construct Prompt
        # We include page numbers for "Golden Link" potential (though just text here)
        context_parts = []
        for c in chunks:
            context_parts.append(f"[Page {c.page_number}] {c.content}")
        
        context_text = "\n\n".join(context_parts)
        
        prompt = f"""You are a helpful technical assistant. 
Use the following context to answer the user's question.
If the answer is not in the context, say you don't know.

Context:
{context_text}

User Question: {query}

Answer:"""

        # 3. Generate Response
        try:
            log.info("Sending prompt to LLM...")
            # Use sync version to avoid event loop conflicts with Streamlit
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            log.error(f"Error generating response from LLM: {e}")
            return "Sorry, I encountered an error while processing your request."
        
    async def quiz(self, document_ids: Optional[List[int]] = None) -> str:
        """
        Generates quiz questions based on document chunks.
        Optionally filters by document IDs.
        """
        if document_ids:
            all_chunks = await self.db.execute(
                select(DocumentChunk).where(DocumentChunk.document_id.in_(document_ids))
            )
        else:
            all_chunks = await self.db.execute(select(DocumentChunk))
        
        chunks = all_chunks.scalars().all()
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
            return json.loads(response.text)
        except Exception as e:
            log.error(f"Error generating quiz from LLM: {e}")
            return "Sorry, I encountered an error while generating the quiz."
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
            return await self.quiz()
        elif query.__contains__("quit"):
            return "Exiting pipeline."
        else:
            return await self.chat(query)
    async def flash_cards(self, id : Optional[int] = None, title : Optional[str] = None) -> str:
        """
        Generate flash cards based on id'd documents or title
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
            return "Please provide either document ID or title for flash card generation."

        chunks = all_chunks.scalars().all()
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
            return json.loads(response.text)
        except Exception as e:
            log.error(f"Error generating flash cards from LLM: {e}")
            return "Sorry, I encountered an error while generating the flash cards."
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