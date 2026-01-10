import json
import logging
import re
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import List
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch


class QueryLogger:
    def __init__(self, log_file_path: str = "logs/rag_queries.log"):
        """
        Инициализирует логгер для запросов RAG-бота.
        
        Args:
            log_file_path: путь к файлу лога
        """
        self.log_file_path = Path(log_file_path)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)  # Создаем директорию, если её нет
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера для записи запросов."""
        logger = logging.getLogger("RAGQueryLogger")
        logger.setLevel(logging.INFO)
        
        # Удаляем старые хендлеры, чтобы избежать дублирования
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        # Создаем файловый хендлер
        file_handler = logging.FileHandler(self.log_file_path, encoding='utf-8')
        formatter = logging.Formatter('%(message)s')  # Простой формат для JSON
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.propagate = False  # Не передаем логи дальше
        
        return logger
    
    def log_query(
        self, 
        query_text: str, 
        response_text: str,
        chunks_found: bool, 
        answer_length: int, 
        sources: List[str],
        timestamp: float = None
    ) -> None:
        """
        Записывает информацию о запросе в лог.
        
        Args:
            query_text: текст запроса пользователя
            response_text: текст ответа модели
            chunks_found: были ли найдены чанки
            answer_length: длина ответа
            sources: список найденных источников
            timestamp: временная метка (если не указана, используется текущее время)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Определяем флаг успешного ответа
        success_flag = self._evaluate_success(answer_length, response_text)
        
        log_entry = {
            "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
            "query_text": query_text,
            "response_text": response_text,
            "chunks_found": chunks_found,
            "answer_length": answer_length,
            "success_flag": success_flag,
            "sources": sources,
            "epoch_time": timestamp
        }
        
        # Записываем JSON-запись в лог
        self.logger.info(json.dumps(log_entry, ensure_ascii=False))
    
    def _evaluate_success(self, answer_length: int, response_text: str) -> bool:
        """
        Оценивает, является ли ответ успешным.
        Извлекает только часть после "Ответ:" для проверки на фразы неудачного ответа.
        
        Args:
            answer_length: длина ответа
            response_text: полный текст ответа модели
            
        Returns:
            bool: True если ответ считается успешным
        """
        # Базовая проверка по длине
        if answer_length < 10:  # минимальная длина ответа
            return False
            
        # Извлекаем только часть после "Ответ:" (или "ANSWER:" в английской версии)
        # Ищем разделение между Chain-of-Thought и фактическим ответом
        parts = re.split(r'(?:\n\n|<br><br>)', response_text)
        answer_part = ""
        
        # Ищем часть, начинающуюся с "Ответ:" или "Answer:"
        for part in parts:
            if re.match(r'^\s*Ответ:\s*', part, re.IGNORECASE) or re.match(r'^\s*Answer:\s*', part, re.IGNORECASE):
                answer_part = part
                break
        
        # Если не нашли "Ответ:", используем весь текст ответа
        if not answer_part:
            answer_part = response_text
        
        # Извлекаем текст после "Ответ:" или "Answer:"
        match = re.search(r'(?:Ответ|Answer):\s*(.*)', answer_part, re.DOTALL | re.IGNORECASE)
        if match:
            answer_part = match.group(1).strip()
        else:
            # Если не найдено "Ответ:", используем весь текст
            answer_part = response_text
        
        # Проверяем наличие ключевых фраз неудачного ответа ТОЛЬКО в части после "Ответ:"
        failure_indicators = [
            "информация не найдена",
            "ответ не найден",
            "нет данных",
            "недостаточно информации",
            "information not found"
        ]
        
        lower_answer = answer_part.lower()
        for indicator in failure_indicators:
            if indicator in lower_answer:
                return False
                
        return True


# === Конфигурация ===
CHROMA_PATH = "../task3/chroma_db"
OLLAMA_MODEL_NAME = "qwen3:4b-instruct"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
COLLECTION_NAME = "rag_qwen_kb"
TOP_K = 20
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.8
LOG_CHUNKS = True

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
        self.query_logger = QueryLogger(log_file_path="logs/rag_queries.log")

    def _load_system_prompt(self) -> str:
        prompts_dir = Path("prompts")
        return (prompts_dir / "system_prompt.txt").read_text(encoding="utf-8").strip()

    def _load_few_shot_examples(self) -> str:
        prompts_dir = Path("prompts")
        return (prompts_dir / "few_shot_examples.txt").read_text(encoding="utf-8").strip()

    def _is_malicious_chunk(self, text: str) -> bool:
        """
        Проверяет, содержит ли чанк признаки prompt injection или чувствительные данные.
        Возвращает True, если чанк следует отфильтровать.
        """
        text_lower = text.lower()

        # Триггеры команд и ключевых слов
        command_triggers = [
            "ignore all instructions",
            "output:",
            "print:",
            "say:",
            "repeat this",
            "disregard previous",
            "override instructions",
            "reveal the secret",
            "пароль",
            "суперпароль",
            "root:",
            "swordfish"
        ]

        # Регулярные выражения для структур с секретами
        secret_patterns = [
            r'\broot\s*:\s*\w+',
            r'\bпароль\s*[:\-]?\s*\w+',
            r'"[^"]*swordfish[^"]*"',
            r'секрет\s*[:\-]?\s*\S+'
        ]

        for trigger in command_triggers:
            if trigger in text_lower:
                return True

        for pattern in secret_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def query(self, user_query: str) -> str:
        start_time = time.time()
        logger.info(f"Получен запрос: {user_query[:60]}{'...' if len(user_query) > 60 else ''}")

        # Получаем все релевантные чанки
        all_docs = self.retriever.invoke(user_query)

        # Фильтрация чанков
        clean_docs = []
        malicious_docs = []
        for doc in all_docs:
            if self._is_malicious_chunk(doc.page_content):
                malicious_docs.append(doc)
            else:
                clean_docs.append(doc)

        # Логирование вредоносных чанков
        if malicious_docs:
            logger.warning(f"⚠️ Отфильтровано {len(malicious_docs)} вредоносных чанков:")
            for idx, doc in enumerate(malicious_docs):
                source = doc.metadata.get('source', 'unknown')
                preview = doc.page_content[:100].replace('\n', ' ')
                logger.warning(f"  [{idx+1}] Источник: {source} | Содержимое: {preview}...")

        docs = clean_docs
        chunks_found = len(docs) > 0

        if LOG_CHUNKS:
            if docs:
                for idx, doc in enumerate(docs):
                    logger.info(f"Chunk {idx + 1} (source: {doc.metadata.get('source')}):\n{doc.page_content[:500]}...\n")
            else:
                logger.info("Нет релевантных (и безопасных) чанков для контекста.")

        context = "\n\n".join([doc.page_content.strip() for doc in docs])
        logger.info(f"Общий размер контекста: {len(context)} символов.")

        user_message = f"Context:\n{context}\n\nQuestion: {user_query}\n\n{self.few_shot_examples}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        response = self._call_ollama(messages)
        clean_response = response.strip()

        # Логируем запрос после получения ответа
        sources = [doc.metadata.get('source', 'unknown') for doc in docs]
        self.query_logger.log_query(
            query_text=user_query,
            response_text=clean_response,
            chunks_found=chunks_found,
            answer_length=len(clean_response),
            sources=sources,
            timestamp=start_time
        )

        return clean_response

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
            return result["message"]["content"]
        else:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")