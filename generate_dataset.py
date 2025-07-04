import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Установка seed для воспроизводимости
np.random.seed(42)

# Количество записей
n_samples = 1000

# Генерация данных
data = {
    'total_square': np.random.normal(75, 25, n_samples).round(1),  # общая площадь
    'living_square': np.random.normal(50, 20, n_samples).round(1),  # жилая площадь
    'kitchen_square': np.random.normal(12, 4, n_samples).round(1),  # площадь кухни
    'rooms': np.random.randint(1, 6, n_samples),  # количество комнат
    'floor': np.random.randint(1, 25, n_samples),  # этаж
    'floors_total': np.random.randint(5, 30, n_samples),  # всего этажей
    'distance_to_metro': np.random.normal(1000, 500, n_samples).round(0),  # расстояние до метро
    'distance_to_center': np.random.normal(5000, 2000, n_samples).round(0),  # расстояние до центра
    'year_built': np.random.randint(1960, 2024, n_samples),  # год постройки
    'price': None  # цена будет рассчитана позже
}

# Создание DataFrame
df = pd.DataFrame(data)

# Расчет базовой цены с учетом всех факторов
base_price = 100000  # базовая цена за квадратный метр

# Влияние площади
df['price'] = df['total_square'] * base_price

# Влияние количества комнат
df['price'] *= (1 + df['rooms'] * 0.1)

# Влияние этажа (средние этажи ценятся выше)
df['price'] *= (1 + np.sin((df['floor'] / df['floors_total']) * np.pi) * 0.2)

# Влияние расстояния до метро (ближе - дороже)
df['price'] *= (1 + np.exp(-df['distance_to_metro'] / 2000))

# Влияние расстояния до центра
df['price'] *= (1 + np.exp(-df['distance_to_center'] / 5000))

# Влияние года постройки (новые дома дороже)
df['price'] *= (1 + (df['year_built'] - 1960) / 1000)

# Добавление случайного шума
df['price'] *= np.random.normal(1, 0.1, n_samples)

# Округление цены
df['price'] = df['price'].round(0)

# Сохранение в CSV
df.to_csv('real_estate_dataset.csv', index=False)

print("Датасет успешно создан и сохранен в 'real_estate_dataset.csv'")
print("\nПример данных:")
print(df.head())
print("\nСтатистика:")
print(df.describe()) 