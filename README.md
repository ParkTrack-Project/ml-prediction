# parktrack-ml

ML-микросервис для прогнозирования загруженности парковок. Работает как автономный процесс: сам обучается, сам генерирует прогнозы и публикует их в ParkTrack API.

Всё взаимодействие — только через ParkTrack REST API. Прямого доступа к БД нет.

---

## Как это работает

Сервис запускается и делает три вещи по расписанию:

- **каждые 30 минут** — генерирует прогнозы на ближайшие 24 часа для всех активных зон и постит их в `POST /forecasts/new`
- **каждые 5 минут** — подтягивает актуальную погоду
- **ежедневно в 02:00 UTC** — переобучает модель на свежих данных

При первом запуске сервис сам обучает модель, если артефакта нет.

---

## Структура проекта

```
parktrack_ml/
├── api_client.py   — HTTP-клиент к ParkTrack API (Bearer auth)
├── config.py       — все настройки и константы
├── data_loader.py  — загрузка данных через API
├── features.py     — построение признаков
├── model.py        — LightGBM-обёртка + LR-fallback
├── train.py        — скрипт обучения
├── predict.py      — логика предсказания
├── forecaster.py   — генерация и публикация прогнозов
├── weather.py      — работа с погодными данными
├── interfaces.py   — публичный API пакета
└── service.py      — точка входа (планировщик)
Dockerfile.train    — образ для разового обучения
Dockerfile.predict  — образ основного сервиса
docker-compose.yml
.env.example
requirements.txt
```

---

## Быстрый старт

### 1. Настроить окружение

```bash
cp .env.example .env
# Заполнить API_URL и API_TOKEN
```

### 2. Локально (без Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Разовое обучение
python -m parktrack_ml.train

# Запуск сервиса (обучение + прогнозы по расписанию)
python -m parktrack_ml.service
```

### 3. Docker Compose

```bash
# Разовое обучение
docker compose --profile train up --build

# Основной сервис (работает постоянно)
docker compose --profile predict up -d --build
```

### 4. Проверить что сервис работает

```bash
# Логи
docker logs <container_name> -f

# Разовый тест предсказания прямо в контейнере
docker exec <container_name> python -c "
from parktrack_ml.interfaces import predict
from datetime import datetime, timezone
r = predict(zone_id=3, predicted_for=datetime.now(timezone.utc))
print(r)
"
```

---

## Переменные окружения

| Переменная        | Обязательна | По умолчанию | Описание |
|-------------------|-------------|--------------|----------|
| `API_URL`         | да          | —            | Базовый URL ParkTrack API |
| `API_TOKEN`       | да          | —            | Bearer-токен |
| `MODEL_PATH`      | нет         | `./models`   | Путь к папке с артефактами модели |
| `TRAIN_DAYS_BACK` | нет         | `200`        | Глубина истории для обучения (дни) |

---

## Интеграция в deploy

Добавить в основной `docker-compose.yml`:

```yaml
services:
  ml:
    image: ghcr.io/parktrack-project/parktrack-ml-predict:development
    env_file: .env          # API_URL, API_TOKEN
    environment:
      MODEL_PATH: /app/models
    volumes:
      - ml_models:/app/models
    restart: unless-stopped

volumes:
  ml_models:
```

Сервис не требует открытых портов — он только ходит в ParkTrack API.

---

## Использование пакета напрямую (Python)

Если нужно вызвать предсказание из другого Python-сервиса:

```python
from parktrack_ml.interfaces import predict
from datetime import datetime, timezone

result = predict(
    zone_id=3,
    predicted_for=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
)

print(result.occupancy_class)         # "Low" | "Medium" | "High"
print(result.predicted_occupied)      # 4  (машин из capacity)
print(result.capacity)                # 10
print(result.confidence)              # 0.89
print(result.probability_free_space)  # 0.95
print(result.prob_low)                # 0.67
print(result.prob_medium)             # 0.31
print(result.prob_high)               # 0.02
```

---

## Модель

- **Алгоритм**: LightGBM (multiclass)
- **Классы**: Low (< 33%), Medium (33–67%), High (> 67%)
- **Признаков**: 28 — история загруженности (lag/MA), время (с циклическим кодированием), погода, зона, праздники
- **Данных для обучения**: ~18 000 часовых наблюдений (150 дней по 15 зонам)
- **Точность на валидации**: 90.9% (temporal split 80/20)
- **Артефакты**: `models/model.lgb`, `models/zone_meta.json`
