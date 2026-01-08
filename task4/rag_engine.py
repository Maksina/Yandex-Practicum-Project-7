# rag_engine.py
import logging
import requests
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch

# === Конфигурация ===
CHROMA_PATH = "../task3/chroma_db"
OLLAMA_MODEL_NAME = "qwen3:4b-instruct"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
COLLECTION_NAME = "rag_qwen_kb"
TOP_K = 20
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.8
LOG_CHUNKS = False

logger = logging.getLogger("RAGEngine")

class Qwen3Embeddings(Embeddings):
    def __init__(self, model_name: str, device: str = "cpu"):
        print("Загружаем токенизатор и модель для эмбеддингов...")
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

class RAGEngine:
    def __init__(self):
        self.embedding_function = Qwen3Embeddings(model_name=EMBEDDING_MODEL_NAME, device="cuda")
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embedding_function,
            collection_name=COLLECTION_NAME
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": TOP_K})

        self.system_prompt = self._load_system_prompt()
        self.few_shot_examples = self._load_few_shot_examples()

    def _load_system_prompt(self) -> str:
        prompts_dir = Path("prompts")
        return (prompts_dir / "system_prompt.txt").read_text(encoding="utf-8").strip()

    def _load_few_shot_examples(self) -> str:
        prompts_dir = Path("prompts")
        return (prompts_dir / "few_shot_examples.txt").read_text(encoding="utf-8").strip()

    def query(self, user_query: str) -> str:
        logger.info(f"Получен запрос: {user_query[:60]}{'...' if len(user_query) > 60 else ''}")

        docs = self.retriever.invoke(user_query)
        if LOG_CHUNKS:
            for idx, doc in enumerate(docs):
                logger.info(f"Chunk {idx + 1} (source: {doc.metadata.get('source')}):\n{doc.page_content[:500]}...\n")

        context = "\n\n".join([doc.page_content.strip() for doc in docs])
        logger.info(f"Общий размер контекста: {len(context)} символов.")

        user_message = f"Context:\n{context}\n\nQuestion: {user_query}\n\n{self.few_shot_examples}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        response = self._call_ollama(messages)
        return response.strip()

    def _call_ollama(self, messages: list[dict]) -> str:
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": OLLAMA_MODEL_NAME,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "stop": []
            }
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            # print("DEBUG: Raw Ollama response:", result)
            return result["message"]["content"]
        else:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")