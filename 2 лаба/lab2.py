import numpy as np
import matplotlib.pyplot as plt
import timeit
from sklearn.cluster import KMeans
import random

# Фиксируем генераторы случайных чисел для воспроизводимости
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Генерация данных: 50 городов с координатами от 0 до 100
n_cities = 50
points = np.random.rand(n_cities, 2) * 100
points_list = [(x, y) for x, y in points]

# Функция высчитывает евклидово расстояние
def euclidean_distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

# 1. Собственный алгоритм
def my_k_means(points, K, init_centers=None, max_iter=100):
    n = len(points)
    if K >= n:
        return list(range(n)), points, 0.0

    # Если начальные центры не заданы, выбираем случайно (с фикс. seed)
    if init_centers is None:
        indices = random.sample(range(n), K)
        centers = [points[i] for i in indices]
    else:
        centers = init_centers.copy()

    for _ in range(max_iter):
        labels = []
        for p in points:
            best_dist = float('inf')
            best_idx = 0
            for idx, c in enumerate(centers):
                dist = euclidean_distance(p, c)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            labels.append(best_idx)

        new_centers = []
        for k in range(K):
            cluster_points = [points[i] for i in range(n) if labels[i] == k]
            if cluster_points:
                avg_x = sum(p[0] for p in cluster_points) / len(cluster_points)
                avg_y = sum(p[1] for p in cluster_points) / len(cluster_points)
                new_centers.append((avg_x, avg_y))
            else:
                new_centers.append(centers[k])

        converged = True
        for i in range(K):
            if new_centers[i] != centers[i]:
                converged = False
                break
        centers = new_centers
        if converged:
            break

    inertia = 0.0
    for i, p in enumerate(points):
        c = centers[labels[i]]
        inertia += euclidean_distance(p, c) ** 2
    return labels, centers, inertia

# 2. Ручная реализация K‑means
def kmeans_manual(points, K, init_centers=None, max_iter=100):
    points = np.array(points)
    n = len(points)

    if init_centers is None:
        # fallback – случайный выбор
        indices = np.random.choice(n, K, replace=False)
        centers = points[indices].copy()
    else:
        centers = np.array(init_centers)

    for _ in range(max_iter):
        # Вычисление расстояний от каждой точки до всех центров
        diff = points[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        labels = np.argmin(dists, axis=1)

        new_centers = np.array([points[labels == k].mean(axis=0) for k in range(K)])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    inertia = np.sum((points - centers[labels]) ** 2)
    return labels.tolist(), centers.tolist(), inertia


# Ручной ввод K
try:
    K = int(input("Введите выбранное количество кластеров K (по графику локтя): "))
except:
    K = 4
    print("Используется K=4 по умолчанию")

# Генерация начальных центры для всех алгоритмов
indices = random.sample(range(n_cities), K)
initial_centers = [points_list[i] for i in indices]
print(f"Начальные центры (индексы {indices}):")
for i, c in enumerate(initial_centers):
    print(f"  Центр {i}: ({c[0]:.2f}, {c[1]:.2f})")

# Запуск трёх алгоритмов с начальными центрами
print(f"\nКластеризация с K={K}")

# Собственный алгоритм
start = timeit.default_timer()
labels_custom, centers_custom, inertia_custom = my_k_means(points_list, K, init_centers=initial_centers)
time_custom = timeit.default_timer() - start

# Ручной K‑means
start = timeit.default_timer()
labels_manual, centers_manual, inertia_manual = kmeans_manual(points_list, K, init_centers=initial_centers)
time_manual = timeit.default_timer() - start

# Sklearn K‑means (передаём начальные центры)
start = timeit.default_timer()
km = KMeans(n_clusters=K, init=np.array(initial_centers), n_init=1, random_state=SEED)
labels_sklearn = km.fit_predict(points_list)
inertia_sklearn = km.inertia_
time_sklearn = timeit.default_timer() - start

# Вывод результатов сравнения
print("\nРезультаты:")
print(f"{'Алгоритм':<20} {'Время (с)':<12} {'Оценка качества кластера':<12}")
print("-" * 44)
print(f"{'Собственный':<20} {time_custom:<12.6f} {inertia_custom:<12.2f}")
print(f"{'K-means (ручной)':<20} {time_manual:<12.6f} {inertia_manual:<12.2f}")
print(f"{'K-means (sklearn)':<20} {time_sklearn:<12.6f} {inertia_sklearn:<12.2f}")

# Визуализация графиков
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.scatter([p[0] for p in points_list], [p[1] for p in points_list], c=labels_custom, cmap='viridis')
plt.scatter([c[0] for c in centers_custom], [c[1] for c in centers_custom], marker='x', c='red', s=200)
plt.title("Собственный алгоритм")

plt.subplot(1, 3, 2)
plt.scatter([p[0] for p in points_list], [p[1] for p in points_list], c=labels_manual, cmap='viridis')
plt.scatter([c[0] for c in centers_manual], [c[1] for c in centers_manual], marker='x', c='red', s=200)
plt.title("Ручной K-means")

plt.subplot(1, 3, 3)
plt.scatter([p[0] for p in points_list], [p[1] for p in points_list], c=labels_sklearn, cmap='viridis')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker='x', c='red', s=200)
plt.title("Sklearn K-means")

plt.tight_layout()
plt.show()