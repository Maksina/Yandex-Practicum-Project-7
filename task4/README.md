# Реализация RAG-бота с техниками промптинга

Технологический стек:
- ЯП: Python 3
- LLM: qwen3:4b-instruct
- Эмбеддинги: Qwen3-Embedding-4B  
- Векторная база: ChromaDB  

## Промпты

1. [system_prompt.txt](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task4/prompts/system_prompt.txt) - Системный промпт с указание CoT.
2. [few_shot_examples.txt](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task4/prompts/few_shot_examples.txt) - Few-shot

## Реализация

1. [rag_engine.py](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task4/rag_engine.py) - Ядро сервиса, в котором происходит поиск в векторной базе и обращение к LLM.
2. [rag_bot_ollama.py](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task4/rag_bot_ollama.py) - Консольный интерфейс, который использует rag_engine.py
3. [telegram_bot.py](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task4/telegram_bot.py) - Телеграм бот, который использует rag_engine.py. Необходимо вставить свой токен в скрипт.

## Скриншоты

1. [Успешный ответ](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task4/screenshots/%D0%A3%D1%81%D0%BF%D0%B5%D1%85.png) 
2. [Ответ, что нет информации](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task4/screenshots/%D0%9D%D0%B5%D1%82%20%D0%B8%D0%BD%D1%84%D0%BE%D1%80%D0%BC%D0%B0%D1%86%D0%B8%D0%B8.png)

## Примечание

Без ollama, с их алгоритмом использования CUDA-ядер и квантирования, время ответа LLM занимало ~5 минут, а с ollama < 5 секунд.