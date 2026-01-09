# Автоматическое ежедневное обновление базы знаний

## Что было сделано

1. Реализован [update_index.py](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task6/update_index.py) - скрипт обновления индекса
2. Обновлен [chunking.py](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task3/chunking.py) - добавлена функция *load_single_txt_file*, которая загружает только один файл.
3. Обновлен [build_index.py](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task3/build_index.py) - добавлена функция записи index_meta.json
4. Реализован [setup_task.ps1](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task6/setup_task.ps1) - скрипт-планировщик ежедневного выполнения update_index.py на windows
5. [Архитектурная диаграмма](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task6/schemas) 
6. [Новый файл для тестирования](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task6/newfiles) - для проверки обновления необходимо переместить файлы в "../task2/knowledge_base" и запустить update_index.py.

## Обновление индекса

- Обновление индекса происходит с помощью [update_index.py](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task6/update_index.py).  
- Скрипт проверяет изменения/добавление файлов в "../task2/knowledge_base", если файлы добавились или удалились, то запускается процесс получения чанков и обновления БД.  
- Проверка файлов исползует хэш, который записывается в [index_meta.json](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task6/logs/index_meta.json)  
- Логи записываются в [index_update.log](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task6/logs/index_update.log)  

## Ежедневное обновление базы знаний

Для запуска необходимо настроить [setup_task.ps1](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task6/setup_task.ps1): указать путь до директории, время запуска.

Запуск производится в PowerShell(от им.адм.) командой:
```console
.\setup_task.ps1 
```

Удалить запланирование задание можно командой:
```console
Unregister-ScheduledTask -TaskName "RAG Daily Index Update" -Confirm:$false
```

## Архитектурная диаграмма

![Image Диаграмма](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task6/schemas/Диаграмма.png)