import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from loguru import logger
from src.ingestion.pipeline import load_documents_from_folder, split_documents, save_to_qdrant


def main():
    logger.info("Запуск Ingestion Pipeline...")
    
    raw_data_path = os.path.join(project_root, "data", "raw")
    
    if not os.path.exists(raw_data_path) or not os.listdir(raw_data_path):
        logger.error(f"Папка {raw_data_path} пуста или не существует! Положите туда PDF или TXT файлы.")
        return

    documents = load_documents_from_folder(raw_data_path)
    if not documents:
        logger.error("Не удалось загрузить ни одного документа.")
        return

    chunks = split_documents(documents)

    save_to_qdrant(chunks)
    
    logger.info("Ingestion Pipeline завершен успешно!")


if __name__ == "__main__":
    main()