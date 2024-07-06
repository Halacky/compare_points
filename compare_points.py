import numpy as np
from scipy.spatial.distance import cdist

def find_nearest_points(points1, points2, metric='euclidean'):
    """
    Находит ближайшие точки во втором облаке для каждой точки в первом облаке,
    гарантируя, что каждая точка во втором облаке может быть ближайшей только для одной точки из первого облака,
    и учитывая условие, что ближайшей может считаться только та точка, минимальное расстояние до которой меньше 1.
    
    :param points1: Первый набор точек.
    :param points2: Второй набор точек.
    :param metric: Метрика расстояния ('euclidean', 'cityblock' и т.д.).
    :return: Список пар (точка из первого облака, ближайшая точка из второго облака), удовлетворяющих условиям.
    """
    # Инициализация списка пар
    valid_pairs = []
    
    # Повторяем процесс для каждой точки в первом облаке
    for point1 in points1:
        # Вычисляем расстояния от текущей точки первого облака до всех точек второго облака
        distances = cdist(np.array([point1]), points2, metric=metric)
        
        # Находим индекс ближайшей точки во втором облаке
        nearest_index = np.argmin(distances)
        
        # Проверяем, удовлетворяет ли минимальное расстояние условию (меньше 1)
        min_distance = distances[0][nearest_index]
        if min_distance < 1:
            # Добавляем пару (точка из первого облака, ближайшая точка из второго облака) в список
            valid_pairs.append((point1, points2[nearest_index]))
            
            # Удаляем найденную ближайшую точку из второго облака, чтобы она не могла быть выбрана снова
            points2 = np.delete(points2, nearest_index, axis=0)
    
    return valid_pairs

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
