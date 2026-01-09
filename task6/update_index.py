# update_index.py
import os
import sys
import time
import json
import hashlib
from datetime import datetime
from pathlib import Path

# === Настройка импорта из task3 ===
TASK3_PATH = os.path.join(os.path.dirname(__file__), '..', 'task3')
sys.path.insert(0, TASK3_PATH)

from chunking import load_single_txt_file
from langchain_chroma import Chroma
from build_index import Qwen3Embeddings
import torch
import logging

# === Конфигурация ===
SOURCE_DIR = "../task2/knowledge_base"  # Папка с документами
CHROMA_PATH = "../task3/chroma_db"      # Путь к векторной базе из task3
MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
INDEX_LOG_FILE = "./logs/index_update.log"  # Лог в папке task6
INDEX_META_FILE = "./logs/index_meta.json"  # Метаданные в папке task6

device = "cuda" if torch.cuda.is_available() else "cpu"

# Настройка логирования
os.makedirs(os.path.dirname(INDEX_LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(INDEX_LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def get_file_hash(filepath: Path) -> str:
    """Вычисляет SHA256 хеш файла."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def load_index_meta() -> dict:
    """Загружает метаданные индекса из файла."""
    if os.path.exists(INDEX_META_FILE):
        with open(INDEX_META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"files": {}}

def save_index_meta(meta: dict):
    """Сохраняет метаданные индекса."""
    with open(INDEX_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def scan_for_updates(source_dir: str) -> tuple[list[Path], list[str]]:
    """Сканирует папку и возвращает список новых или изменённых файлов."""
    source_path = Path(source_dir)
    if not source_path.exists():
        logger.error(f"Директория {source_dir} не существует.")
        return [], []

    indexed_meta = load_index_meta()
    new_or_changed = []
    indexed_files = set(indexed_meta["files"].keys())

    for file_path in source_path.glob("*.txt"):
        # Используем относительный путь от source_path
        rel_path_str = str(file_path.relative_to(source_path))
        current_hash = get_file_hash(file_path)

        if rel_path_str not in indexed_files or indexed_meta["files"][rel_path_str] != current_hash:
            new_or_changed.append(file_path)
            logger.info(f"Найден новый или изменённый файл: {file_path.name}")

    return new_or_changed, []

def update_chroma_db(new_chunks):
    """Добавляет новые чанки в ChromaDB."""
    if not os.path.exists(CHROMA_PATH):
        logger.warning("ChromaDB не существует. Индекс создастся при первом запуске build_index.py.")
        return

    embeddings = Qwen3Embeddings(MODEL_NAME, device=device)
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="rag_qwen_kb"
    )

    # Добавление новых чанков
    if new_chunks:
        # === Уникализируем ID глобально ===
        for i, c in enumerate(new_chunks):
            old_id = c.metadata.get("chunk_id")
            new_id = f"kb_{int(time.time() * 1000000)}_{i}"  # Используем микросекунды + счётчик
            c.metadata["chunk_id"] = new_id
        ids = [str(c.metadata.get("chunk_id")) for c in new_chunks]

        db.add_documents(documents=new_chunks, ids=ids)
        logger.info(f"Добавлено {len(new_chunks)} новых чанков в базу.")

def main():
    start_time = time.time()
    logger.info("="*50)
    logger.info("Запуск обновления индекса из knowledge_base...")

    # 1. Найти новые/изменённые файлы
    new_files, _ = scan_for_updates(SOURCE_DIR)

    if not new_files:
        logger.info("Нет изменений в knowledge_base. Обновление пропущено.")
        return

    # 2. Загрузить и разбить каждый файл отдельно
    all_new_chunks = []
    for file in new_files:
        try:
            logger.info(f"Обработка файла: {file.name}")
            temp_chunks = load_single_txt_file(file_path=file)
            all_new_chunks.extend(temp_chunks)
        except Exception as e:
            logger.error(f"Ошибка при обработке файла {file}: {e}")

    logger.info(f"Получено {len(all_new_chunks)} новых чанков.")

    # 3. Обновить ChromaDB
    update_chroma_db(new_chunks=all_new_chunks)

    # 4. Обновить метаданные индекса
    meta = load_index_meta()
    knowledge_path = Path(SOURCE_DIR)
    for file in new_files:
        rel_path_str = str(file.relative_to(knowledge_path))
        meta["files"][rel_path_str] = get_file_hash(file)
    save_index_meta(meta)

    end_time = time.time()
    duration = round(end_time - start_time, 2)

    total_docs = 0
    if os.path.exists(CHROMA_PATH):
        embeddings = Qwen3Embeddings(MODEL_NAME, device=device)
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name="rag_qwen_kb"
        )
        total_docs = db._collection.count()

    logger.info(f"✅ Индекс обновлён за {duration} секунд.")
    logger.info(f"📊 Всего чанков в индексе: {total_docs}")
    logger.info("="*50)

if __name__ == "__main__":
    main()