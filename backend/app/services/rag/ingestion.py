import logging as log
import asyncio
import os
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.document import Document, FileTypeEnum
from app.models.chunk import DocumentChunk
from app.services.rag.loaders import DocumentLoader
from app.services.rag.splitters import TextProcessor
from app.services.rag.vectorizer import Vectorizer
from app.core.database import async_session
log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log.info(f"")
class IngestionService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.vectorizer = Vectorizer()
        self.processor = TextProcessor()
        self.loader = DocumentLoader

    async def process_document(self, file_path: str, filename: str, file_type: FileTypeEnum = FileTypeEnum.PDF):
        filename = filename.replace('.pdf', '')
        loaded_docs = self.loader(file_path, file_type).load()
        log.info(f"Loaded {len(loaded_docs)} raw documents pages from {file_path}")
        metadata = loaded_docs[0].metadata if loaded_docs else {}
        document_obj = Document(
            title = filename,
            file_type = file_type.value,
            file_path = file_path,
            num_pages = metadata.get('total_pages', 'N/A')
        )
        self.db.add(document_obj)
        await self.db.commit()
        await self.db.refresh(document_obj)
        log.info(f"Created Document record with ID: {document_obj.id}")

        chunks = self.processor.split_documents(loaded_docs)
        log.info(f"Splitting completed.")

        vectors = self.vectorizer.embed_documents(chunks) 
        log.info(f"Vectorization completed.")
        db_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = chunk.metadata
            log.info(f"Page number : {chunk_metadata.get('page_label', 'N/A')}")
            chunk_obj = DocumentChunk(
                document_id = document_obj.id,
                content = chunk.page_content,
                page_number = int(chunk_metadata.get('page_label', -1)),
                embedding = vectors[i]
            )
            db_chunks.append(chunk_obj)
        self.db.add_all(db_chunks)
        await self.db.commit()
        log.info(f"Inserted {len(chunks)} chunks into the database for Document ID: {document_obj.id}")
        return {
            "document_id": document_obj.id,
            "title": document_obj.title,
            "num_chunks": len(chunks),
            "status" : "SUCCESS"
        }
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