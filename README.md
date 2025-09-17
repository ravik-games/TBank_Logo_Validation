# Валидатор для проекта по детекции логотипа Т-Банка

**Дополнение к проекту [ravik-games/TBank_Logo_Detection](https://github.com/ravik-games/TBank-Logo-Detection)**

Датасет и все модели можно скачать по ссылке:
https://disk.yandex.ru/d/Nbin45aEzEmGSg

## Установка и запуск

Для работы требуется PyYAML и Ultralytics, датасет и модель. Веса модели (YOLOv11n) находятся в папке `models`.

**Установка:**
- `pip install -r requirements.txt`

**Быстрый запуск:**
- `python validate_yolo.py --model path/to/model.pt --data path/to/dataset`

Результаты будут сохранены в папку `output`
