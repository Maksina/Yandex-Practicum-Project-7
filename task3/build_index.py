# build_index.py
from chunking import load_and_split_txt_files
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os
import time

# === Конфигурация ===
KNOWLEDGE_DIR = "../task2/knowledge_base"
CHROMA_PATH = "./chroma_db"
MODEL_NAME = "Qwen/Qwen3-Embedding-4B"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Используем устройство: {device}")


# === Кастомный эмбеддинг-класс для Qwen3-Embedding-4B ===
class Qwen3Embeddings(Embeddings):
    def __init__(self, model_name: str, device: str = "cpu"):
        print("Загружаем токенизатор и модель...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        ).eval()
        self.device = device

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        batch_size = 4
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=8192
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']

                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask

                mean_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
                all_embeddings.extend(mean_embeddings.cpu().numpy().astype("float32").tolist())
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# === Основной pipeline ===
if __name__ == "__main__":
    print("1️⃣ Загружаем и разбиваем документы...")
    chunks = load_and_split_txt_files(knowledge_dir=KNOWLEDGE_DIR)

    print("2️⃣ Инициализируем Qwen3-Embedding-4B...")
    start_load = time.time()
    embeddings = Qwen3Embeddings(MODEL_NAME, device=device)
    load_time = time.time() - start_load
    print(f"✅ Модель загружена за {load_time:.2f} секунд")

    print("3️⃣ Создаём векторную базу ChromaDB...")
    start_index = time.time()

    if os.path.exists(CHROMA_PATH):
        print(f"⚠️ Обнаружена существующая база: {CHROMA_PATH}. Перезаписываем.")
        import shutil
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="rag_qwen_kb"
    )

    index_time = time.time() - start_index
    print(f"✅ Векторный индекс сохранён в: {os.path.abspath(CHROMA_PATH)}")
    print(f"📊 Чанков в индексе: {len(chunks)}")
    print(f"⏱️ Время индексации: {index_time:.2f} секунд")
