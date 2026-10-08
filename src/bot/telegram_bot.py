import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart, Command
from loguru import logger

from src.rag.chain import ask_question


def start_bot(token: str):
    bot = Bot(token=token)
    dp = Dispatcher()
    
    @dp.message(CommandStart())
    async def cmd_start(message: Message):
        await message.answer(
            "Привет! Я — RAG-ассистент.\n\n"
            "Задай мне любой вопрос по загруженной базе знаний, "
            "и я отвечу, опираясь только на документы.\n\n"
            "Напиши /help, чтобы узнать больше."
        )
    
    @dp.message(Command("help"))
    async def cmd_help(message: Message):
        await message.answer(
            "<b>Как пользоваться:</b>\n\n"
            "Просто напиши мне вопрос текстом, например:\n"
            "• Что такое RAG?\n"
            "• Из каких компонентов состоит система?\n"
            "• Какие проблемы решает этот метод?\n\n"
            "Я найду ответ в базе знаний и пришлю тебе."
        )
    
    @dp.message(F.text)
    async def handle_question(message: Message):
        question = message.text
        logger.info(f"Получен вопрос от @{message.from_user.username}: {question}")
        
        await message.answer("Ищу ответ в базе знаний...")
        
        try:
            answer = await asyncio.to_thread(ask_question, question)
            
            if len(answer) > 4000:
                answer = answer[:4000] + "\n\n...(ответ обрезан из-за длины)"
            
            await message.answer(f"<b>Ответ:</b>\n\n{answer}", parse_mode="HTML")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке вопроса: {e}")
            await message.answer(
                "Произошла ошибка при обработке вопроса. "
                "Попробуй ещё раз позже."
            )
    
    logger.info(" Запуск Telegram-бота...")
    asyncio.run(dp.start_polling(bot))