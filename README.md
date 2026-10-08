# Production RAG

Локальная RAG-система с полным циклом: ingestion, vector search, generation, API, Telegram-бот и автоматическая оценка качества через RAGAS.

Работает полностью на локальном железе - без облачных API и платных ключей.

## Стек технологий

- **LLM**: Qwen 2.5 7B (через Ollama, локально)
- **Embeddings**: nomic-embed-text (через Ollama)
- **Vector DB**: Qdrant (Docker)
- **Cache**: Redis (Docker)
- **Backend**: FastAPI + Uvicorn
- **Bot**: aiogram 3.x
- **Orchestration**: LangChain
- **Evaluation**: RAGAS
- **PDF parsing**: PyMuPDF4LLM

## Функциональность

- Парсинг PDF и TXT документов с сохранением структуры (MuPDF)
- Семантический поиск по векторной базе
- Генерация ответов с защитой от галлюцинаций
- REST API с автоматической Swagger-документацией
- Telegram-бот для интерактивного общения
- Автоматическая оценка качества через RAGAS (4 метрики)

## Быстрый старт

### Предварительные требования

- Python 3.11+
- Docker Desktop
- Ollama (https://ollama.com)

### Установка

1. Клонируйте репозиторий:

        git clone <repo-url>
        cd production-rag

2. Создайте виртуальное окружение и установите зависимости:

        python -m venv .venv
        .venv\Scripts\Activate.ps1
        pip install -r requirements.txt

3. Скачайте модели через Ollama:

        ollama pull qwen2.5:7b
        ollama pull nomic-embed-text

4. Создайте файл `.env` на основе `.env.example`:

        cp .env.example .env

   Укажите токен Telegram-бота (получить у @BotFather).

5. Запустите инфраструктуру:

        docker compose up -d

### Запуск

Ingestion pipeline:

        python scripts/run_ingestion.py

FastAPI сервер (Swagger UI на http://localhost:8000/docs):

        python scripts/run_api.py

Telegram-бот:

        python scripts/run_bot.py

Оценка качества через RAGAS:

        python src/evaluation/ragas_eval.py

## Оценка качества (RAGAS)

Проведено A/B-тестирование пайплайна на 5 вопросах по статье "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020).

## Планы по улучшению

- **Hybrid Search (BM25 + Dense)** - для повышения Context Precision на точных терминах
- **Query Translation** - автоматический перевод запроса на язык исходных документов перед поиском
- **Advanced parsing** - интеграция Marker или Unstructured.io для сложных PDF
- **Conversation history** - поддержка многошаговых диалогов в боте
- **Deploy на Hugging Face Spaces** - публичное демо через Gradio