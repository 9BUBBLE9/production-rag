import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from src.rag.chain import ask_question


def main():
    question = "Что такое RAG и из каких основных компонентов состоит эта система?"
    
    logger.info(f"Задаем вопрос: {question}")
    print("-" * 50)
    
    answer = ask_question(question)
    
    print("-" * 50)
    logger.success(f"Ответ модели:\n{answer}")


if __name__ == "__main__":
    main()