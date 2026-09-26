# Kinopoisk Tone

Классификатор тональности отзывов на фильмы (Bad / Neutral / Good) на основе
[RuModernBERT-base](https://huggingface.co/deepvk/RuModernBERT-base), дообученной на отзывах с Кинопоиска.
Веб-интерфейс на Gradio и HTTP API на FastAPI.

## Результаты
<img width="2993" height="1808" alt="image" src="https://github.com/user-attachments/assets/d32ca51d-9d39-424d-82e9-e75685855ebf" />


              precision    recall  f1-score   support

         Bad      0.822     0.796     0.809       500
     Neutral      0.594     0.588     0.591       500
        Good      0.758     0.790     0.774       500

    accuracy                          0.725      1500
   macro avg      0.725     0.725     0.725      1500
weighted avg      0.725     0.725     0.725      1500

## Структура

- `trans.ipynb`: обучение (Google Colab, GPU)
- `load.py`: сервис (модель, интерфейс, API)
- `Dockerfile`, `docker-compose.yml`: запуск в контейнере
- `requirements.txt`, `run.sh`: запуск без Docker

Датасеты и веса модели в репозиторий не входят.

## Запуск

1. Клонировать репозиторий и скачать модель (`[<ссылка>](https://huggingface.co/deepvk/RuModernBERT-base))`) в `models/ru-modernbert-sentiment/`.

2. Запустить:

```bash
docker compose up --build
```

3. Открыть http://127.0.0.1:8000

## API

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Отличный фильм, рекомендую"}'
```
```json
{"label": "Good", "probs": {"Bad": 0.02, "Neutral": 0.10, "Good": 0.88}}
```

Метки: `0 = Bad`, `1 = Neutral`, `2 = Good`. Путь к модели задаётся переменной `MODEL_PATH`.

## Обучение

Данные: 10 500 / 1 500 / 1 500 отзывов (train / validation / test).
Параметры: `max_length=1024`, 3 эпохи, batch 8, lr 2e-5, лучший чекпоинт по F1.

## Ограничения

Модель обучена на длинных отзывах (в среднем около 400 токенов). На коротких фразах
чаще выдаёт `Neutral`. К третьей эпохе заметно переобучение. Наиболее сложный класс: Neutral (F1 0.59).

