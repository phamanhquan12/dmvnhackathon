import logging as log
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
class TextProcessor:
    def __init__(self, chunk_size=750, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def split_documents(self, raw_documents : List[Document]) -> List[Document]:
        """Chunk documents into smaller pieces."""
        try:
            splitted_docs = self.splitter.split_documents(raw_documents)
            log.info(f"Split {len(raw_documents)} documents into {len(splitted_docs)} chunks.")
            return splitted_docs
        except Exception as e:
            log.error(f"Error splitting documents: {e}")
            return []