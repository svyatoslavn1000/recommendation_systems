# Двухуровневая рекомендательная система на `python`

Рекомендательная система использует own recommender для формирования первоначальных рекомендаций и catboost (YetiRankPairwise) для ранжирования.

Все константы введены в файле `./settings.yaml`

Демонстрация работы приведена в файле `./main.ipynb`

Исходный код находится в директории `./src`

Ниже приведен базовый пример использования системы.



* [Импорт](#first-bullet)
* [Префильтрация](#second-bullet)
* [Модель 1 уровня: рекомендательная](#third-bullet)
* [Модель 2 уровня: ранжирующая](#forth-bullet)
* [Оценка результатов](#fith-bullet)
* [Сохранение результатов](#sixth-bullet)

---

***

***

# Блок импорта <a class="anchor" id="first-bullet"></a>

* импорт библиотек
* загрузка констант из файла `settings.yaml`

```python
from src.utils import *
globals().update(load_settings(True))
```

***

# Блок предоброботки данных <a class="anchor" id="second-bullet"></a>

* загрузка датасетов
* первичная трансформация датасетов
* добавление новых фич
* конвертация текстовых фичей в цифровые для модели второго уровня
* разделение и подготовка датасетов для формирования рекомендаций и последующего ранжирования

```python
data = Dataset()
data.data_prefilter()
data.data_split()
```

***

# Блок формирования рекомендаций <a class="anchor" id="third-bullet"></a>

* train_test_split
* получение рекомендаций по собственным покупкам (с добавлением популярных товаров)

```python
recommender = MainRecommender(data.data_train_lvl_1)
recommender.preprocessing(data, t='train')
recommender.preprocessing(data, t='test')
```

***

# Блок ранжирования полученных рекомендаций <a class="anchor" id="forth-bullet"></a>

* форматируем датасеты под формат (class Pool) CatBoost
* обучаем CatBoostRanker с функцией потерь YetiRankPairwise
* получаем предсказанные вероятности для ранжирования
* ранжируем рекомендации от модели первого уровня

```python
recommender.ranker_fit()
```

***

# Блок оценки полученных результатов <a class="anchor" id="fith-bullet"></a>

```python
recommender.evaluate_2models()
```

***

# Блок сохранения результата <a class="anchor" id="sixth-bullet"></a>

* повторяем все вышеописанные действия для "боевой" базы
* сохраняем результат в .xml

```python
data.data_test_split()
recommender = MainRecommender(data.data_train_lvl_1_real)
recommender.preprocessing(data, t='train', training=False)
recommender.preprocessing(data, t='test', training=False)
recommender.ranker_fit(training=False)
recommender.evaluate_2models(training=False)
```

***
