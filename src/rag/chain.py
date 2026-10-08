from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from loguru import logger

from src.utils.config import settings


def get_rag_chain(top_k: int = 3):
    embeddings = OllamaEmbeddings(
        model=settings.ollama_embedding_model, 
        base_url=settings.ollama_base_url
    )
    
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=settings.qdrant_collection_name,
        url=settings.qdrant_url,
        prefer_grpc=False
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k * 2})

    llm = ChatOllama(
        model=settings.ollama_model, 
        base_url=settings.ollama_base_url, 
        temperature=0
    )

    template = """Ты — эксперт по анализу документов. Твоя задача — ответить на вопрос СТРОГО на основе предоставленного контекста.

    ПРАВИЛА:
    1. Используй ТОЛЬКО информацию из контекста ниже
    2. Если ответа нет в контексте, скажи: "В предоставленных документах нет ответа на этот вопрос"
    3. Не добавляй информацию из своих знаний
    4. Отвечай подробно, цитируя контекст где возможно

    КОНТЕКСТ:
    {context}

    ВОПРОС: {question}

    ОТВЕТ:"""
    prompt = PromptTemplate.from_template(template)

    return retriever, llm, prompt


def ask_question(question: str, top_k: int = 3) -> str:
    retriever, llm, prompt = get_rag_chain(top_k=top_k)
    
    logger.info(f"Ищем релевантные фрагменты для вопроса: '{question}'")
    
    docs = retriever.invoke(question)
    if not docs:
        return "Я не нашел никакой информации по вашему вопросу в базе знаний."
        
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    logger.info("Генерируем ответ с помощью Qwen...")
    formatted_prompt = prompt.format(context=context, question=question)
    
    response = llm.invoke(formatted_prompt)
    
    return response.content