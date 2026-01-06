# База знаний

За основу взят сериал "Очень странные дела" [fandom.com](https://strangerthings.fandom.com/ru/wiki/%D0%9E%D1%87%D0%B5%D0%BD%D1%8C_%D0%A1%D1%82%D1%80%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5_%D0%94%D0%B5%D0%BB%D0%B0_%D0%92%D0%B8%D0%BA%D0%B8)

В качестве базы используем описание серий и главных персонажей.

## Подготовка данных

fandom.com защищен капчей, поэтому классический парсинг не сработал.
HTML-страницы были собраны и сохранены вручную.

[Каталог HTML](https://github.com/Maksina/Yandex-Practicum-Project-7/tree/RAG/task2/HTML)

## Очистка данных

Парсинг HTML страниц реализован с помощью python-скрипта, который убирает разметку и лишние блоки информации, сохраняет полученную информацию в папку original_txt.

[Parser python](https://github.com/Maksina/Yandex-Practicum-Project-7/tree/RAG/task2/parser)

## Маппинг определений

Сгенерирован маппинг с оригинальных имен и названий на вымышленные.

[terms_map.json](https://github.com/Maksina/Yandex-Practicum-Project-7/blob/RAG/task2/terms_map.json)

## Замена по маппингу

Разработан Python-скрипт, который заменяет все имена и названия в файлах папки original_txt, используя маппинг в terms_map.json, и сохранет в knoweledge_base.

[Replacer python](https://github.com/Maksina/Yandex-Practicum-Project-7/tree/RAG/task2/replacer)


## Финальная база знаний

[knoweledge_base](https://github.com/Maksina/Yandex-Practicum-Project-7/tree/RAG/task2/knowledge_base)
