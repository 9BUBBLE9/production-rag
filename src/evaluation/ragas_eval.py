import sys
import os
from datasets import Dataset
from loguru import logger
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama, OllamaEmbeddings

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config import settings
from src.rag.chain import get_rag_chain

TEST_DATA = [
    {
        "question": "What are the two types of RAG models described in the paper?",
        "ground_truth": "RAG-Token and RAG-Sequence models",
    },
    {
        "question": "What problem of parametric memory does RAG address?",
        "ground_truth": "Parametric memory is fixed after training and cannot be easily updated",
    },
    {
        "question": "What is the role of the retriever in RAG?",
        "ground_truth": "The retriever finds relevant documents from a corpus to condition the generator",
    },
    {
        "question": "What datasets were used to evaluate RAG in the paper?",
        "ground_truth": "Natural Questions, TriviaQA, and CuratedTREC",
    },
    {
        "question": "What is the main advantage of RAG over fine-tuning?",
        "ground_truth": "RAG allows updating knowledge without retraining the model",
    },
]

def run_evaluation():
    logger.info("Запуск оценки качества RAG через RAGAS...")
    
    retriever, llm, prompt = get_rag_chain(top_k=5)
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for item in TEST_DATA:
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        logger.info(f"Обработка вопроса: {question}")
        
        docs = retriever.invoke(question)
        context_list = [doc.page_content for doc in docs]
        
        context_text = "\n\n---\n\n".join(context_list)
        formatted_prompt = prompt.format(context=context_text, question=question)
        response = llm.invoke(formatted_prompt)
        
        questions.append(question)
        answers.append(response.content)
        contexts.append(context_list)
        ground_truths.append(ground_truth)
    
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    
    logger.info("Вычисление метрик RAGAS с использованием локальной модели...")
    
    eval_llm = LangchainLLMWrapper(ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url))
    eval_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=settings.ollama_embedding_model, base_url=settings.ollama_base_url))

    run_config = RunConfig(
        timeout=180, 
        max_workers=1,
        max_retries=2,
    )
    
    results = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=run_config,
    )
    
    logger.info("Результаты оценки:")
    print(results)
    
    return results

if __name__ == "__main__":
    run_evaluation()