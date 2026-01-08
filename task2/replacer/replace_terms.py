import os
import json
from pathlib import Path

def load_replacements(mapping_path: str):
    """Загружает список замен из JSON-файла и сортирует по длине ключа (по убыванию)."""
    with open(mapping_path, 'r', encoding='utf-8') as f:
        replacements = json.load(f)
    
    if not all('original' in r and 'new' in r for r in replacements):
        raise ValueError("Каждый элемент в JSON должен содержать 'original' и 'new'")
    
    replacements.sort(key=lambda x: len(x['original']), reverse=True)
    return replacements

def replace_in_text(text: str, replacements: list) -> str:
    """Выполняет последовательную замену всех терминов в тексте."""
    for item in replacements:
        text = text.replace(item['original'], item['new'])
    return text

def replace_in_filename(filename: str, replacements: list) -> str:
    """Заменяет термины в имени файла (без расширения)."""
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    new_stem = replace_in_text(stem, replacements)
    return new_stem + suffix

def main():
    mapping_file = Path('../terms_map.json')
    input_dir = Path('../original_txt')
    output_dir = Path('../knowledge_base')

    if not input_dir.exists():
        raise FileNotFoundError(f"Директория {input_dir} не найдена.")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Загрузка маппинга замен...")
    replacements = load_replacements(mapping_file)
    print(f"Загружено {len(replacements)} правил замены.")

    txt_files = list(input_dir.glob('*.txt'))
    if not txt_files:
        print(f"В папке {input_dir} не найдено .txt файлов.")
        return

    for txt_file in txt_files:
        try:
            # Чтение содержимого
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Замена в содержимом
            updated_content = replace_in_text(content, replacements)

            # Замена в имени файла
            new_filename = replace_in_filename(txt_file.name, replacements)
            output_file = output_dir / new_filename

            # Предупреждение при конфликте имён (опционально можно обработать иначе)
            if output_file.exists():
                print(f"⚠️  Внимание: файл с именем {new_filename} уже существует. Перезапись.")

            # Запись обновлённого файла с новым именем
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)

            print(f"✅ Обработан: {txt_file.name} → {new_filename}")

        except Exception as e:
            print(f"❌ Ошибка при обработке {txt_file.name}: {e}")

    print("✅ Обработка завершена. Файлы сохранены в ../knowledge_base")

if __name__ == "__main__":
    main()