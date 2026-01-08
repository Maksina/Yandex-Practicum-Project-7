# telegram_bot.py
import logging
from rag_engine import RAGEngine
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# === Конфигурация Telegram-бота ===
TELEGRAM_BOT_TOKEN = "TOKEN_HERE" 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("RAGTelegramBot")

class RAGTelegramBot:
    def __init__(self, token: str):
        self.engine = RAGEngine()
        self.app = ApplicationBuilder().token(token).build()

        # Регистрация обработчиков
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.handle_message))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Привет! Задай мне вопрос по вселенной 'Туман над городом'.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_message = update.message.text
        logger.info(f"Получено сообщение от {update.effective_user.id}: {user_message}")

        try:
            response = self.engine.query(user_message)
            await update.message.reply_text(response)
        except Exception as e:
            logger.error(f"Ошибка при обработке сообщения: {e}")
            await update.message.reply_text("Произошла ошибка при обработке запроса. Попробуйте позже.")

    def run_polling(self):
        logger.info("🚀 Запускаем Telegram-бота в режиме polling...")
        self.app.run_polling()

if __name__ == "__main__":
    bot = RAGTelegramBot(token=TELEGRAM_BOT_TOKEN)
    bot.run_polling()