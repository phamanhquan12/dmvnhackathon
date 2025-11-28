import google.generativeai as genai
import logging as log
from typing import Optional
from langchain_core.documents import Document
from app.core.config import settings
genai.configure(api_key=settings.GOOGLE_API_KEY)
log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# genai.GenerationConfig(
#     response_mime_type="text/plain",
#     max_output_tokens=1024
# )
class Vectorizer:
    def __init__(self):
        self.embeddings = genai
        self.model_name = "models/text-embedding-004"
    def embed_documents(self, texts : list[Document], title: Optional[str] = None) -> list[list[float]]:
        title = title.replace('.pdf', '') if title else "untitled densomind document"
        try:

            embedded_dict = self.embeddings.embed_content(
                model = self.model_name,
                content = [t.page_content for t in texts],
                task_type = "retrieval_document",
                title = title
            )
            log.info(f"Embedded {len(embedded_dict['embedding'])} documents with title: {title}")
            return embedded_dict['embedding']
        except Exception as e:
            log.error(f"Error embedding documents with title {title}: {e}")
            return []
    def embed_query(self, text: str) -> list[float]:
        try:
            embedded_dict = self.embeddings.embed_content(
                model = self.model_name,
                content = [text],
                task_type = "retrieval_query"
            )
            log.info(f"Embedded query: {text}")
            return embedded_dict['embedding']
        except Exception as e:
            log.error(f"Error embedding query: {text}: {e}")
            return []