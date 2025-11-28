from langchain_community.document_loaders import PyPDFLoader
from app.models.document import FileTypeEnum
from typing import List
from langchain_core.documents import Document
import logging as log
log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DocumentLoader:
    def __init__(self, file_path: str, file_type: FileTypeEnum):
        self.file_path = file_path
        self.file_type = file_type

    def load(self) -> List[Document]:
        if self.file_type == FileTypeEnum.PDF:
            try:
                loader = PyPDFLoader(self.file_path)
                log.info(f"Found path: {self.file_path}")
                return loader.load()
            except Exception as e:
                log.error(f"Error loading PDF file: {e}")
                return []
        else:
            log.error(f"Unsupported file type: {self.file_type}")
            return 