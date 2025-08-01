# LipVibe

<img width="1280" height="853" alt="image" src="https://github.com/user-attachments/assets/1e57a543-9fb4-4428-b3ff-3c7fc52ba4c7" />

<img width="1280" height="853" alt="image" src="https://github.com/user-attachments/assets/97246fc3-f9b1-4007-9c0d-890961aad75c" />

## Описание

LipVibe - это комплексное решение для анализа цветовых характеристик губ и подбора губной помады. Проект состоит из двух основных компонентов:

- Клиентская часть (веб-интерфейс для загрузки фотографий и отображения рекомендаций)
- Серверная часть (API для анализа изображений и подбора помады)

## Требования

- Python 3.8 или выше
- Node.js 18 или выше
- npm 9 или выше
- OpenCV (для обработки изображений)

## Структура проекта

```
lipvibe/
├── client/          # Клиентская часть (Astro.js + TypeScript)
└── service/         # Серверная часть (Python)
```

## Установка и запуск

### 1. Клонирование репозитория

```bash
git clone [URL репозитория]
cd lipvibe
```

### 2. Настройка серверной части

```bash
cd service
python -m venv venv
# Для Windows:
venv\Scripts\activate
# Для Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Настройка клиентской части

```bash
cd client
npm install
```

### 4. Запуск приложения

#### Запуск сервера

```bash
cd service
# Для Windows:
venv\Scripts\activate
# Для Linux/Mac:
source venv/bin/activate

python app.py
```

Сервер будет доступен по адресу: `http://localhost:8000`

#### Запуск клиента

```bash
cd client
npm run dev
```

Клиент будет доступен по адресу: `http://localhost:4321`

## Дополнительная информация

- Подробная документация по клиентской части находится в `client/README.md`
- Подробная документация по серверной части находится в `service/README.md`

## Технологии

### Клиентская часть

- Astro.js
- TypeScript
- TailwindCSS
- React

### Серверная часть

- Python
- Flask
- OpenCV (для анализа изображений)
- SQLite (для хранения базы данных помад)

## Функциональность

- Загрузка фотографий губ
- Анализ цветовых характеристик
- Подбор подходящих оттенков помады
- Рекомендации брендов
- Сохранение истории подборов

