from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from loguru import logger

from src.utils.config import settings

try:
    import pymupdf4llm
    HAS_MUPDF = True
except ImportError:
    HAS_MUPDF = False


def load_documents_from_folder(folder_path: str) -> list:
    path = Path(folder_path)
    documents = []
    
    for file in path.iterdir():
        if file.suffix.lower() == '.pdf':
            logger.info(f"Загрузка PDF: {file.name}")
            if HAS_MUPDF:
                md_text = pymupdf4llm.to_markdown(str(file))
                doc = Document(page_content=md_text, metadata={"source": str(file)})
                documents.append(doc)
            else:
                loader = PyPDFLoader(str(file))
                documents.extend(loader.load())
        elif file.suffix.lower() == '.txt':
            logger.info(f"Загрузка TXT: {file.name}")
            loader = TextLoader(str(file), encoding='utf-8')
            documents.extend(loader.load())
            
    logger.success(f"Всего загружено документов: {len(documents)}")
    return documents


def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    logger.success(f"Документы разбиты на {len(chunks)} чанков.")
    return chunks


def save_to_qdrant(chunks: list):
    logger.info("Инициализация эмбеддинг-модели и подключение к Qdrant...")
    
    embeddings = OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url
    )
    
    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.qdrant_collection_name,
        url=settings.qdrant_url,
        force_recreate=True
    )
    
    logger.success(f"Данные успешно загружены в коллекцию '{settings.qdrant_collection_name}' в Qdrant!")
    return vector_store