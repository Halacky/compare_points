import numpy as np
from scipy.spatial.distance import cdist

def find_nearest_points(points1, points2, metric='euclidean'):
    """
    Находит ближайшие точки во втором облаке для каждой точки в первом облаке.
    
    :param points1: Первый набор точек.
    :param points2: Второй набор точек.
    :param metric: Метрика расстояния ('euclidean', 'cityblock' и т.д.).
    :return: Список пар (точка из первого облака, ближайшая точка из второго облака).
    """
    distances = cdist(points1, points2, metric=metric)
    nearest_indices = np.argmin(distances, axis=1)
    nearest_points = points2[nearest_indices]
    pairs = list(zip(points1, nearest_points))
    return pairs

# Генерация облаков точек
points1 = np.random.rand(3, 3)  # Первое облако из 3 точек
points2 = np.random.rand(15, 3)  # Второе облако из 15 точек

# Поиск ближайших точек
pairs = find_nearest_points(points1, points2, metric='euclidean')

# Вывод пар сопоставленных точек
for point1, point2 in pairs:
    print(f"({point1}, {point2})")


import matplotlib.pyplot as plt

def plot_points(points, title=None):
    """Функция для визуализации облака точек."""
    plt.scatter(points[:, 0], points[:, 1], s=50)  # Исправлено здесь
    plt.title(title)
    plt.show()

# Генерация облаков точек
points1 = np.random.rand(3, 3)  # Первое облако из 3 точек
points2 = np.random.rand(15, 3)  # Второе облако из 15 точек

# Поиск ближайших точек
pairs = find_nearest_points(points1, points2, metric='euclidean')

# Визуализация каждого облака отдельно
plot_points(points1, "Облако точек 1")
plot_points(points2, "Облако точек 2")

# Визуализация сопоставленных точек
plt.figure(figsize=(8, 6))
plt.scatter(points1[:, 0], points1[:, 1], color='blue', label='Облако 1')  # Исправлено здесь
plt.scatter(points2[:, 0], points2[:, 1], color='red', label='Облако 2')  # Исправлено здесь
for point1, point2 in pairs:
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], color='green')  # Исправлено здесь
plt.legend()
plt.title('Сопоставленные точки')
plt.show()
