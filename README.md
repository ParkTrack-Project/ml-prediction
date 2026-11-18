# parktrack-ml

ML-микросервис для прогнозирования загруженности парковок. Работает как автономный процесс: сам обучается, сам генерирует прогнозы и публикует их в ParkTrack API.

Всё взаимодействие — только через ParkTrack REST API. Прямого доступа к БД нет.

---

## Как это работает

Сервис запускается и делает три вещи по расписанию:

- **каждые 30 минут** — генерирует прогнозы на ближайшие 48 часов для всех активных зон и постит их в `POST /forecasts/new`
- **ежечасно на :05 UTC** — подтягивает актуальную погоду
- **ежедневно в 02:00 UTC** — переобучает модель на свежих данных

При первом запуске сервис сам обучает модель, если артефакта нет.

---

## Структура проекта

```
parktrack_ml/
├── api_client.py   — HTTP-клиент к ParkTrack API (Bearer auth)
├── config.py       — все настройки и константы
├── data_loader.py  — загрузка данных через API
├── features_h.py   — признаки относительно фактического момента наблюдения
├── model_h.py      — модели горизонтов и атомарные артефакты
├── backtest.py     — последовательная проверка на отложенных периодах
├── features.py     — признаки старой модели v3
├── model.py        — старая модель v3
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
| `FORECAST_HOURS`  | нет         | `48`         | Дальность публикуемых прогнозов |
| `LGBM_THREADS`    | нет         | `4`          | Потоки обучения LightGBM |

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
from datetime import datetime, timedelta, timezone

result = predict(
    zone_id=3,
    predicted_for=datetime.now(timezone.utc) + timedelta(hours=1),
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

## Модель (v4.1)

На горизонтах 1, 3, 6, 12, 24, 48 и 72 часа обучаются три LightGBM: классификатор загрузки, регрессор доли занятых мест и бинарный классификатор приближения к полной загрузке. Всего 21 бустер. Выбор модели зависит от фактического возраста последнего завершённого часа; свежие лаги берутся относительно этого наблюдения.

30 признаков включают историю, тренды, покрытие лагов, категорию зоны и профили. Временные индексы хранятся в UTC, календарь считается по `Europe/Moscow`. Пропуски истории заполняются только вперёд максимум на два часа; целевые значения не заполняются. Профили используют последние 45 дней, отдельный профиль часа - 14 дней. Обучающие профили используют только прошлое каждого временного блока.

Некорректные счётчики отбрасываются. Пустая зона с `confidence=0` остаётся валидным наблюдением. Часовое среднее не взвешивается по confidence; вес часа при обучении равен `clip(sqrt(n_obs), 1, 8)`. Недоступная, пустая или слишком старая история блокирует публикацию зоны.

Последние 14 дней используются для выбора числа деревьев и температуры классификатора. Сохраняются именно выбранные модели, без последующего рефита. Это метрики подбора; независимая оценка выполняется в `backtest.py`. Фильтры зон, вместимости и профили фиксируются до validation. Между периодами исключаются прогнозы с origin из предыдущего периода.

`occupancy_class` определяется из округлённого `predicted_occupied / capacity`. `prob_low/medium/high` остаются оценками отдельного классификатора; их наиболее вероятный класс может отличаться. `confidence` - вероятность возвращённого класса. `probability_free_space` пока оценивает событие по **средней часовой загрузке**, не вероятность свободного места в конкретную минуту; бинарная голова отдельно не калибровалась.

Прогнозы уникальных часов вычисляются пакетами по горизонту. Получасовые точки интерполируются между соседними часами. Каждый запуск публикует новую версию с единым `generated_at`; HTTP 409 означает повтор той же тройки `(zone_id, generated_at, predicted_for)` и учитывается отдельно.

`models/model_v4.json` атомарно переключает полный комплект в `models/runs/<id>/`. Внутри находятся `manifest.json` и 21 файл `v4_h*_{cls,reg,free}.lgb` со всеми деревьями и весами листьев; SHA256 проверяется при загрузке. Старый комплект остаётся доступен при неудачном сохранении или reload. При переходе с v3/v4 необходимо переобучение.

```bash
python -m parktrack_ml.backtest --hourly-csv data.csv.gz --split 2026-07-01 2026-08-15 --output predictions.csv.gz
python -m parktrack_ml.train --hourly-csv data.csv.gz
python -m unittest discover -s tests -q
```

Бэктест показывает качество всего потока и общих строк с baseline, покрытие данных, MAE, классификацию и калибровку. Преимущество перед baseline проверяется по каждому горизонту и периоду; оно не гарантируется настройками модели. Разброс предсказаний не заменяет оценку ошибки.

Полная схема и ограничения: [PIPELINE.md](PIPELINE.md). Старые `evaluate.py` и `visualize.py` работают только с v3 (`models/model.lgb`).
