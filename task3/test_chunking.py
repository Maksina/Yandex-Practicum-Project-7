# test_chunking.py
from chunking import load_and_split_txt_files

chunks = load_and_split_txt_files()

# Покажем первый чанк
print("Пример чанка:")
print(f"Источник: {chunks[0].metadata['source']}")
print(f"Chunk ID: {chunks[0].metadata['chunk_id']}")
print(f"Текст:\n{chunks[0].page_content[:300]}...\n")