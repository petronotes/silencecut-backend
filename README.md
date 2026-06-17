# SilenceCut — Backend

FastAPI + FFmpeg. Разворачивается на Railway бесплатно.

## Деплой на Railway

1. Зарегистрируйся на [railway.app](https://railway.app)
2. **New Project → Deploy from GitHub repo**
3. Settings → **Root Directory: `backend`**
4. Railway автоматически найдёт nixpacks.toml и установит Python 3.11 + FFmpeg
5. После деплоя скопируй URL: `https://xxx.up.railway.app`
6. Вставь URL в поле «Сервер» в HTML-файле фронтенда

## Локальный запуск

```bash
# Установи FFmpeg: https://ffmpeg.org/download.html
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# Фронтенд → поле Сервер: http://localhost:8000
```

## Эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Проверка FFmpeg |
| POST | `/process` | Загрузить видео + параметры |
| GET | `/status/{id}` | Прогресс задачи |
| GET | `/download/{id}` | Скачать MP4 |
