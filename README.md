# Классификация неискренних вопросов Quora

Проект по определению токсичных/неискренних вопросов на платформе Quora. Сравниваются три подхода: TF-IDF + Logistic Regression, TF-IDF + LightGBM и BERT-base. Лучший результат показал BERT с ROC-AUC 0.965.

**Ссылка на проект:** [Google Colab](https://colab.research.google.com/drive/1-2yNFb1Pq50PcPFw9VKDEIyUAwWIGxYq#scrollTo=_LBNzAVT8RXH)
**Ссылка на датасет:** [Google Colab](https://www.kaggle.com/datasets/ojasgolatkar/quora-insincere-questions-classification/data)

---

## Этапы работы

**Технологии:** Python, PyTorch, Transformers (Hugging Face), Scikit-learn, LightGBM, BERT-base, Pandas, Matplotlib

### 1. **Загрузка и анализ данных**
   - Датасет: 1.3M тренировочных и 375k тестовых вопросов
   - Сильный дисбаланс классов: искренних вопросов значительно больше
   - Для ускорения экспериментов использована стратифицированная выборка 100k примеров

### 2. **TF-IDF + классические модели**
   - Подбор гиперпараметров TF-IDF: GridSearch по max_features, ngram_range, max_df, min_df
   - Сравнение двух алгоритмов: Logistic Regression и LightGBM
   - Лучшая модель: Logistic Regression (C=1, solver='liblinear')

### 3. **BERT**
   - Модель: `bert-base-uncased` с dropout 0.2
   - Аугментация текста: случайное удаление или перестановка слов (30% вероятности)
   - Fine-tuning: 3 эпохи, lr=2e-5, batch_size=32

---

## Результаты

| Модель | ROC-AUC (val) | Лучшие параметры |
|--------|---------------|------------------|
| TF-IDF + Logistic Regression | 0.940 | C=1, solver='liblinear' |
| TF-IDF + LightGBM | 0.913 | lr=0.1, max_depth=10, n_estimators=200 |
| **BERT-base (fine-tuned)** | **0.965** | 3 эпохи, lr=2e-5 |

---

## Процесс обучения

### Logistic Regression
- CV AUC: 0.934, Val AUC: 0.940
- Стабильное обучение, нет переобучения
- Лучшие параметры: C=1, solver='liblinear'

### LightGBM
- CV AUC: 0.909, Val AUC: 0.913
- Уступает LR на разреженных TF-IDF признаках
- Лучшие параметры: learning_rate=0.1, max_depth=10, n_estimators=200, num_leaves=31

### BERT
- Начало: train loss=0.417, val AUC=0.958
- Эпоха 2: train loss=0.308, val AUC=0.964
- Эпоха 3: train loss=0.268, val AUC=0.965

---

## Анализ важности слов (Logistic Regression)

| Топ-10 слов: неискренность | Коэффициент | Топ-10 слов: искренность | Коэффициент |
|----------------------------|-------------|---------------------------|-------------|
| trump | 8.743 | best | -3.384 |
| liberals | 8.410 | affect | -2.851 |
| indians | 8.351 | tips | -2.645 |
| muslims | 8.344 | difference | -2.618 |
| women | 8.067 | study | -2.504 |
| americans | 7.751 | engineering | -2.420 |
| democrats | 7.486 | company | -2.375 |
| jews | 6.349 | characteristics | -2.355 |
| girls | 6.262 | books | -2.245 |
| castrated | 6.200 | invest | -2.186 |

---

## Аугментация текста для BERT

```python
def augment_text(text):
    # 30% вероятности применить аугментацию
    # Удаление случайного слова или перестановка двух слов
    # Только для текстов длиннее 5 слов
```

**Зачем:** Увеличивает разнообразие обучающих примеров, снижает переобучение

---

## Что я узнала

1. **Logistic Regression отлично работает с TF-IDF:** на разреженных матрицах LR показывает результат 0.940, обходя LightGBM

2. **BERT значительно сильнее:** даже на маленькой выборке (20k примеров) BERT достигает 0.965, что на 2.5% лучше LR

3. **Аугментация полезна для BERT:** случайные удаления/перестановки слов помогают модели лучше обобщать

4. **LightGBM не подходит для TF-IDF:** деревья хуже работают с разреженными матрицами, чем линейные модели

---

## Ключевые выводы

- **Для быстрого прототипа:** Logistic Regression + TF-IDF (ROC-AUC 0.940)
- **Для максимального качества:** BERT-base (ROC-AUC 0.965)
