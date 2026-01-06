# test_query.py
from langchain_chroma import Chroma
from build_index import Qwen3Embeddings  # импортируем ваш кастомный эмбеддинг-класс
import torch

# Конфигурация
CHROMA_PATH = "./chroma_db"
MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Загружаем ту же модель эмбеддингов
embeddings = Qwen3Embeddings(MODEL_NAME, device=device)

# Загружаем ChromaDB
db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
    collection_name="rag_qwen_kb"
)

# Примеры запросов
queries = [
    "Чью роль исполняет Милли Олегби Браун?",
    "Как называется сериал?",
    "Кто бывшая девушка Романа Дуброва?"
]

for query in queries:
    print(f"\n🔍 Запрос: {query}")
    results = db.similarity_search_with_score(query, k=3)  # топ-3 результата

    for i, (doc, score) in enumerate(results):
        print(f"\n  → Результат {i+1} (score: {score:.4f})")
        print(f"    Источник: {doc.metadata.get('source', 'N/A')}")
        print(f"    Текст: {doc.page_content[:300]}...")