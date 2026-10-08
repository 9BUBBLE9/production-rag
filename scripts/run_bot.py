import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from loguru import logger
from src.utils.config import settings
from src.bot.telegram_bot import start_bot


def main():
    if not settings.telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN не задан в .env!")
        return
    
    logger.info("Запуск Telegram-бота...")
    start_bot(settings.telegram_bot_token)


if __name__ == "__main__":
    main()