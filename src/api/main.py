from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger

from src.rag.chain import ask_question

app = FastAPI(
    title="Production RAG API",
    description="API для вопросов по базе знаний (RAG)",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources_count: int


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "production-rag"}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    logger.info(f"Получен вопрос через API: {request.question}")
    
    try:
        answer = ask_question(request.question, top_k=request.top_k)
        return AnswerResponse(
            question=request.question,
            answer=answer,
            sources_count=request.top_k
        )
    except Exception as e:
        logger.error(f"Ошибка при обработке вопроса: {e}")
        raise HTTPException(status_code=500, detail=str(e))