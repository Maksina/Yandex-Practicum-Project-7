# Создание векторного индекса базы знаний

## Информация

**Модель эмбеддингов**
- Название: Qwen/Qwen3-Embedding-4B
- Репозиторий: https://huggingface.co/Qwen/Qwen3-Embedding-4B
- Нормализация: включена

**База знаний**
- Директория: /knowledge_base/
- Форматы: .txt

**Статистика индекса**
- Количество чанков: 434
- Время индексации: 35 секунд (на NVIDIA RTX 5070)

## Запуск

1. [check_gpu.py](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task3/check_gpu.py) - Скрипт для проверки возможности использования CUDA-ядер
2. [chunking.py](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task3/chunking.py) - Скрипт-функция для разделения текстовых файлов на чанки
3. [test_chunking.py](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task3/test_chunking.py) - Скрипт проверки работоспособности chunking.py
4. [build_index.py](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task3/build_index.py) - Скрипт для формирования индекса
5. [test_query.py](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task3/test_query.py) - Скрипт для проверки векторного поиска
6. [chroma_db](https://github.com/Maksina/Yandex-Practicum-Project-7/tree/RAG/task3/chroma_db) - Итоговая векторная база