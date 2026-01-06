#!/usr/bin/env python3
import os
import sys
import re
import hashlib
from pathlib import Path
from bs4 import BeautifulSoup

# --- Настройки ---
HTML_FOLDER = Path("../HTML")          # Папка с исходными HTML
KB_DIR = Path("../original_txt")       # Папка для сохранения текстов
KB_DIR.mkdir(exist_ok=True)

# --- Вспомогательные функции ---
def sanitize_for_filename(name: str) -> str:
    name = re.sub(r'[^\w\-\.\(\)]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_') or "untitled"

def get_filename_from_title(html: str, source_path: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        title_text = title_tag.string.strip()
        base_name = title_text.split('|', 1)[0].rstrip() if '|' in title_text else title_text
        base_name = base_name.replace(' ', '_')
        safe_name = sanitize_for_filename(base_name)
        if safe_name:
            return f"{safe_name}.txt"
    # Резерв: хеш от имени файла
    return f"{hashlib.md5(source_path.encode()).hexdigest()}.txt"

def extract_text_from_content_block(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Ищем строго: <div id="content" class="page-content">
    content_element = soup.find("div", id="content", class_="page-content")
    if not content_element:
        raise ValueError("Блок <div id='content' class='page-content'> не найден")

    # Удаляем "Содержание"
    for toc in content_element.select('#toc, .toc'):
        toc.decompose()

    # Удаляем технические элементы
    for tag in content_element(["script", "style", "nav", "footer", "aside", "header", "form", "button", "noscript"]):
        tag.decompose()

    # Стоп-заголовки (регистронезависимо)
    stop_headings = {"за кадром", "цитаты", "интересные факты", "сноски", "галерея"}
    stop_tag = None

    for h in content_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        headline_span = h.find("span", class_="mw-headline")
        heading_text = (headline_span.get_text(strip=True) if headline_span else h.get_text(strip=True)).lower()
        if heading_text in stop_headings:
            stop_tag = h
            break

    # Обрезаем HTML до стоп-заголовка
    if stop_tag:
        content_html = str(content_element)
        stop_tag_html = str(stop_tag)
        pos = content_html.find(stop_tag_html)
        if pos != -1:
            clean_html = content_html[:pos]
            content_for_text = BeautifulSoup(clean_html, "html.parser")
        else:
            content_for_text = content_element
    else:
        content_for_text = content_element

    # Извлечение текста с сохранением абзацев
    block_tags = {'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote', 'pre', 'section', 'article', 'ul', 'ol', 'table'}
    for tag in content_for_text.find_all(block_tags):
        marker = soup.new_string("\x00PARA\x00")
        tag.insert_before(marker)

    text = content_for_text.get_text(separator=" ", strip=True)
    text = text.replace("\x00PARA\x00", "\n\n")
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

# --- Основная логика ---
def process_html_file(file_path: Path):
    try:
        html = file_path.read_text(encoding="utf-8")
        filename = get_filename_from_title(html, str(file_path))
        output_path = KB_DIR / filename

        text = extract_text_from_content_block(html)
        if not text:
            print(f"⚠️  Пустой текст: {file_path.name}")
            return

        output_path.write_text(text, encoding="utf-8")
        print(f"✅ {file_path.name} → {output_path.name}")
    except Exception as e:
        print(f"❌ Ошибка при обработке {file_path}: {e}")

def main():
    html_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HTML_FOLDER
    if not html_folder.exists():
        print(f"Ошибка: папка не найдена — {html_folder}", file=sys.stderr)
        sys.exit(1)

    html_files = list(html_folder.rglob("*.html")) + list(html_folder.rglob("*.htm"))
    if not html_files:
        print(f"В папке {html_folder} не найдено HTML-файлов", file=sys.stderr)
        sys.exit(1)

    print(f"Найдено {len(html_files)} HTML-файлов. Начинаю обработку...\n")
    for file_path in sorted(html_files):
        process_html_file(file_path)

    print(f"\n✅ Готово! Тексты сохранены в: {KB_DIR}")

if __name__ == "__main__":
    main()