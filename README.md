# DeepSpeech2 PyTorch Pipeline

Репозиторий для обучения и инференса модели на основе DeepSpeech2 на датасете LibriSpeech.
Включает продвинутый CTC beam search декодер, полный скрипт инференса, визуализацию метрик и структурированные модули.

---

## Быстрое начало

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Скачивание предобученной модели

Скачайте файл `final_model.pt` по ссылке:  
**https://drive.google.com/file/d/1Sjv4p5Y4pKUC4hiYINzFM5HqI9cCd14-/view?usp=sharing**  
Положите его в папку `name_project/checkpoints/` перед инференсом.

### 3. Подготовка данных

Скачайте LibriSpeech (`train-clean-100`, `test-clean`) с [официального сайта](https://www.openslr.org/12). Папку разместите по пути, указанному в config.py (обычно `./data/LibriSpeech`). Либо он установится автоматически при запуске обучения

### 4. Обучение модели

```bash
python train.py
```
Чекпоинты сохраняются в `./checkpoints/`. Логи – в `./logs/`.

### 5. Визуализация обучения

```bash
python visualize_training.py
```
Построит графики метрик обучения (Loss, WER, CER) и сохранит их в `./checkpoints/`.

**Пример графика:**

![training_plot.jpg](training_plot.png)

---

## Инференс предобученной модели

- Для одного WAV-файла:
```bash
python inference.py --wav myaudio.wav
```
- Для случайных примеров из test-clean:
```bash
python inference.py
```
По умолчанию используется модель из `checkpoints/final_model.pt` с beam search декодированием.

---

## О метриках и обучение

- Полный лог обучения (logs) показывает:
    - **Train Loss** снижается с 2.5 до ~0.15
    - **Test Loss** — до уровня 0.35
    - **Word Error Rate (WER):** от 99% (на старте) до лучших 21.55% (на тесте)
    - **Character Error Rate (CER):** от 49% до лучших 6.93%
- Модель уверенно учится, не переобучается, показывает обобщение на тестовом датасете.

---

## Структура репозитория

```
├── train.py                 # Обучение
├── inference.py             # Инференс (WAV/тестовый)
├── visualize_training.py    # Визуализация метрик
├── logging_utils.py         # сбор логов
│
├── config.py
├── requirements.txt
│
├── data/
│   ├── dataset.py
│   └── text_transform.py
├── model/
│   ├── blocks.py
│   ├── model.py
│   └── decoder.py         # BeamCTCDecoder, GreedyDecoder
├── trainer/
│   └── trainer.py
```

---


## Автор

Гареев Равиль
Год: 2025