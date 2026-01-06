import os
import json
from pathlib import Path

def load_replacements(mapping_path: str):
    """Загружает список замен из JSON-файла и сортирует по длине ключа (по убыванию)."""
    with open(mapping_path, 'r', encoding='utf-8') as f:
        replacements = json.load(f)
    
    # Убедимся, что это список словарей с ключами "original" и "new"
    if not all('original' in r and 'new' in r for r in replacements):
        raise ValueError("Каждый элемент в JSON должен содержать 'original' и 'new'")
    
    # Сортируем по длине original (в обратном порядке), чтобы сначала заменять более длинные фразы
    replacements.sort(key=lambda x: len(x['original']), reverse=True)
    return replacements

def replace_in_text(text: str, replacements: list) -> str:
    """Выполняет последовательную замену всех терминов в тексте."""
    for item in replacements:
        text = text.replace(item['original'], item['new'])
    return text

def main():
    # Пути
    mapping_file = Path('../terms_map.json')
    input_dir = Path('../original_txt')
    output_dir = Path('../knowledge_base')

    # Проверка существования входной директории
    if not input_dir.exists():
        raise FileNotFoundError(f"Директория {input_dir} не найдена.")
    
    # Создание выходной директории, если её нет
    output_dir.mkdir(parents=True, exist_ok=True)

    # Загрузка маппинга
    print("Загрузка маппинга замен...")
    replacements = load_replacements(mapping_file)
    print(f"Загружено {len(replacements)} правил замены.")

    # Обработка всех .txt файлов
    txt_files = list(input_dir.glob('*.txt'))
    if not txt_files:
        print(f"В папке {input_dir} не найдено .txt файлов.")
        return

    for txt_file in txt_files:
        print(f"Обработка: {txt_file.name}")
        try:
            # Чтение файла
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Замена
            updated_content = replace_in_text(content, replacements)

            # Запись в выходную папку
            output_file = output_dir / txt_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)

        except Exception as e:
            print(f"Ошибка при обработке {txt_file.name}: {e}")

    print("✅ Обработка завершена. Файлы сохранены в ../knowledge_base")

if __name__ == "__main__":
    main()