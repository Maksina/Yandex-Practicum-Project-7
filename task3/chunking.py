# chunking.py
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from transformers import AutoTokenizer
import os
import re

def load_and_split_txt_files(
    knowledge_dir: str = "../task2/knowledge_base",
    chunk_size: int = 512,
    chunk_overlap: int = 50
):
    """
    Загружает .txt файлы и разбивает их на чанки с токен-точным контролем.
    Автоматически определяет тип документа: персонаж или эпизод.
    """
    knowledge_path = Path(knowledge_dir)
    if not knowledge_path.exists():
        raise FileNotFoundError(f"Директория {knowledge_dir} не найдена.")

    txt_files = list(knowledge_path.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"Не найдено .txt файлов в {knowledge_dir}")

    print(f"📂 Загружаем {len(txt_files)} .txt файлов...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-Embedding-4B",
        trust_remote_code=True
    )
    def len_fn(text: str) -> int:
        return len(tokenizer.encode(text))

    chunks = []
    chunk_id_counter = 0

    for file_path in txt_files:
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            if not docs:
                continue

            full_text = "\n\n".join([d.page_content for d in docs])
            filename = file_path.stem

            # === Определяем тип документа ===
            if filename.startswith("Глава_"):
                doc_type = "episode"
                # Приводим название к читаемому виду: "Глава_восьмая._Битва..." → "Глава восьмая. Битва..."
                title = filename.replace("_", " ").replace(".", ".", 1)  # аккуратная замена
                # Уберём лишние точки после цифр, если нужно
                title = re.sub(r'(\d+)\.\s*', r'\1. ', title)
                prefix = f"[Эпизод: {title}]"
            else:
                doc_type = "character"
                title = filename.replace("_", " ")
                prefix = f"[Персонаж: {title}]"

            # === Разбиение ===
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len_fn,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
                keep_separator=True
            )

            raw_chunks = text_splitter.split_text(full_text)

            for chunk_text in raw_chunks:
                chunk_with_context = f"{prefix}\n\n{chunk_text}"
                chunks.append(Document(
                    page_content=chunk_with_context,
                    metadata={
                        "source": str(file_path),
                        "title": title,
                        "doc_type": doc_type,
                        "chunk_id": chunk_id_counter
                    }
                ))
                chunk_id_counter += 1

        except Exception as e:
            print(f"⚠️ Ошибка при загрузке {file_path}: {e}")

    print(f"✅ Получено {len(chunks)} чанков.")
    return chunks