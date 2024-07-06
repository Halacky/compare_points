import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

# Пример данных: координаты точек в облаках A и B
cloud_A = np.array([(197, 162), (317, 155), (360, 416)])
cloud_B = np.array([(155, 193), (221, 195)])

distances = distance.cdist(cloud_A, cloud_B, 'euclidean')

# Решение задачи о назначениях (Assignment Problem) с использованием метода Венгерского (Hungarian algorithm)
row_indices, col_indices = linear_sum_assignment(distances)

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(cloud_A[:, 0], cloud_A[:, 1], c='blue', label='Облако A')
plt.scatter(cloud_B[:, 0], cloud_B[:, 1], c='red', label='Облако B')

# Соединение соответствующих точек
for row, col in zip(row_indices, col_indices):
    plt.plot([cloud_A[row, 0], cloud_B[col, 0]], [cloud_A[row, 1], cloud_B[col, 1]], 'k--')

plt.gca().invert_yaxis()
# Настройка графика
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Сопоставление точек из облака A и облака B')
plt.legend()
plt.grid(True)
plt.show()
