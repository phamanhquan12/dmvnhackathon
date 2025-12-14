import logging as log
import asyncio
import os
import shutil
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.document import Document, FileTypeEnum
from app.models.chunk import DocumentChunk
from app.services.rag.loaders import DocumentLoader
from app.services.rag.splitters import TextProcessor, HybridChunker
from app.services.rag.vectorizer import Vectorizer
from app.core.database import async_session

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Permanent storage directory for uploaded documents
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


class IngestionService:
    def __init__(self, db: AsyncSession, use_semantic_chunking: bool = True):
        self.db = db
        self.vectorizer = Vectorizer()
        # Use hybrid chunker for better semantic preservation
        self.chunker = HybridChunker(
            max_chunk_size=1000,
            min_chunk_size=200,
            overlap=100,
            use_semantic=use_semantic_chunking
        )
        # Keep text processor for backward compatibility
        self.processor = TextProcessor()
        self.use_semantic = use_semantic_chunking

    async def process_document(
        self, 
        file_path: str, 
        filename: str, 
        file_type: FileTypeEnum = FileTypeEnum.PDF,
        subtitle_path: Optional[str] = None
    ):
        """Process and ingest a document (PDF or Video) into the database."""
        # Clean up filename for title
        title = filename
        if file_type == FileTypeEnum.PDF:
            title = filename.replace('.pdf', '')
        elif file_type == FileTypeEnum.VIDEO:
            for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
                title = title.replace(ext, '')
        
        # Copy file to permanent storage
        permanent_path = os.path.join(UPLOAD_DIR, filename)
        try:
            shutil.copy2(file_path, permanent_path)
            log.info(f"Copied file to permanent storage: {permanent_path}")
        except Exception as e:
            log.warning(f"Could not copy to permanent storage: {e}")
            permanent_path = file_path  # Fallback to original path
        
        # Load document
        loader = DocumentLoader(file_path, file_type, subtitle_path)
        loaded_docs = loader.load()
        
        if not loaded_docs:
            log.error(f"No content loaded from {file_path}")
            return {
                "document_id": None,
                "title": title,
                "num_chunks": 0,
                "status": "FAILED",
                "error": "No content could be extracted from the file"
            }
        
        log.info(f"Loaded {len(loaded_docs)} raw document pages/segments from {file_path}")
        
        # Extract metadata
        metadata = loaded_docs[0].metadata if loaded_docs else {}
        
        # Determine num_pages/duration based on file type
        if file_type == FileTypeEnum.VIDEO:
            duration = metadata.get('total_duration', 0)
            num_pages = len(loaded_docs)  # Number of transcript chunks
            duration_seconds = int(duration) if duration else None
        else:
            num_pages = metadata.get('total_pages', len(loaded_docs))
            duration_seconds = None
        
        # Create document record with permanent path
        document_obj = Document(
            title=title,
            file_type=file_type.value,
            file_path=permanent_path,
            num_pages=num_pages,
            duration_seconds=duration_seconds
        )
        self.db.add(document_obj)
        await self.db.commit()
        await self.db.refresh(document_obj)
        log.info(f"Created Document record with ID: {document_obj.id}")

        # Process chunks
        if self.use_semantic and file_type == FileTypeEnum.PDF:
            # Use semantic chunking for PDFs
            chunk_dicts = self.chunker.chunk_documents(loaded_docs)
            log.info(f"Semantic chunking completed: {len(chunk_dicts)} chunks")
            
            # Extract content for vectorization
            chunk_contents = [c['content'] for c in chunk_dicts]
            vectors = self.vectorizer.embed_texts(chunk_contents)
            log.info(f"Vectorization completed.")
            
            db_chunks = []
            for i, chunk_dict in enumerate(chunk_dicts):
                # Build content with header context if available
                content = chunk_dict['content']
                if chunk_dict.get('header'):
                    content = f"[{chunk_dict['header']}]\n{content}"
                
                chunk_obj = DocumentChunk(
                    document_id=document_obj.id,
                    content=content,
                    page_number=chunk_dict.get('page_number', 1),
                    embedding=vectors[i]
                )
                db_chunks.append(chunk_obj)
        else:
            # Use simple splitting for videos or when semantic is disabled
            if file_type == FileTypeEnum.VIDEO:
                # Videos are already chunked by time segments
                chunks = loaded_docs
                log.info(f"Using {len(chunks)} video transcript segments")
            else:
                chunks = self.processor.split_documents(loaded_docs)
                log.info(f"Simple splitting completed.")

            # Vectorize
            chunk_texts = [c.page_content for c in chunks]
            vectors = self.vectorizer.embed_texts(chunk_texts)
            log.info(f"Vectorization completed.")
            
            db_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = chunk.metadata
                
                # For videos, include timestamp info in content
                if file_type == FileTypeEnum.VIDEO:
                    start_time = chunk_metadata.get('start_time', 0)
                    end_time = chunk_metadata.get('end_time', 0)
                    content = f"[{self._format_time(start_time)} - {self._format_time(end_time)}]\n{chunk.page_content}"
                else:
                    content = chunk.page_content
                
                chunk_obj = DocumentChunk(
                    document_id=document_obj.id,
                    content=content,
                    page_number=int(chunk_metadata.get('page_label', chunk_metadata.get('page', 1))),
                    embedding=vectors[i]
                )
                db_chunks.append(chunk_obj)
        
        self.db.add_all(db_chunks)
        await self.db.commit()
        log.info(f"Inserted {len(db_chunks)} chunks into the database for Document ID: {document_obj.id}")
        
        return {
            "document_id": document_obj.id,
            "title": document_obj.title,
            "num_chunks": len(db_chunks),
            "file_type": file_type.value,
            "chunking_method": "semantic" if (self.use_semantic and file_type == FileTypeEnum.PDF) else "simple",
            "duration_seconds": duration_seconds,
            "status": "SUCCESS"
        }
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


async def main_service():
    async with async_session() as session:
        service = IngestionService(session)
        yield service


if __name__ == "__main__":
    from app.core.database import async_session
    
    async def test_ingestion():
        file_path = 'test2.pdf'
        file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), file_path)
        if os.path.exists(file_path):
            log.info(f"File found at path: {file_path}")

        async with async_session() as session:
            service = IngestionService(session)
            try:
                results = await service.process_document(file_path, 'statement', FileTypeEnum.PDF)
                log.info(f"Ingestion results: {results}")
            except Exception as e:
                log.error(f"Error during ingestion: {e}")
    
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )
    asyncio.run(test_ingestion())