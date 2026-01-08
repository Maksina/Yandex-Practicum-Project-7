# rag_bot_ollama.py
import logging
from rag_engine import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("RAGConsole")

if __name__ == "__main__":
    engine = RAGEngine()
    logger.info("🚀 RAG-бот (консоль) готов к работе!")

    while True:
        try:
            user_input = input("\n❓ Ваш вопрос: ").strip()
            if user_input.lower() in ("выход", "exit", "quit"):
                logger.info("Завершение работы")
                break
            if not user_input:
                continue
            response = engine.query(user_input)
            print(f"\n💬 Ответ:\n{response}\n")
        except KeyboardInterrupt:
            logger.info("Принудительное завершение")
            break
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            print(f"\n⚠️ Ошибка: {e}\n")