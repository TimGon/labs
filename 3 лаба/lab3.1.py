import time
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Функция загрузки данных из файла
def load_dataset(filename):
    """
    Загружает данные из текстового файла, где каждая строка содержит:
    продукт сладость хруст класс
    (поля разделены пробелами или табуляцией)
    Возвращает список кортежей (продукт, сладость, хруст, класс)
    """
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # пропускаем пустые строки
            parts = line.split()
            if len(parts) < 4:
                continue  # недостаточно полей – пропускаем
            product = parts[0]
            try:
                sweet = int(parts[1])
                crunch = int(parts[2])
            except ValueError:
                continue  # если не числа – пропускаем
            cls = parts[3]
            data.append((product, sweet, crunch, cls))
    return data

# 2. Подготовка данных для k-NN (признаки + метки)
def prepare_data(data_list, exclude_class=None):
    """
    Преобразует список (продукт, сладость, хруст, класс) в формат
    [((сладость, хруст), класс)].
    Если указан exclude_class, записи с этим классом исключаются.
    """
    features_labels = []
    for _, sweet, crunch, cls in data_list:
        if exclude_class and cls == exclude_class:
            continue
        features_labels.append(((sweet, crunch), cls))
    return features_labels

# 3. Евклидово расстояние между двумя точками
def euclidean_distance(p1, p2):
    """p1 и p2 – кортежи (сладость, хруст)"""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

# 4. Ручная реализация k-NN (без sklearn)
def predict_knn(train_data, test_point, k):
    """
    train_data: список [((сладость, хруст), класс)]
    test_point: (сладость, хруст)
    k: количество соседей
    Возвращает предсказанный класс
    """
    # Список (расстояние, класс) для всех точек обучения
    distances = []
    for features, label in train_data:
        dist = euclidean_distance(features, test_point)
        distances.append((dist, label))
    # Сортируем по расстоянию
    distances.sort(key=lambda x: x[0])
    # Берём k ближайших
    k_nearest = [label for _, label in distances[:k]]
    # Голосование: выбираем самый частый класс
    vote = Counter(k_nearest).most_common(1)[0][0]
    return vote

# 5. Тестовые примеры для двух экспериментов
def get_test_samples(version='old'):
    """
    Возвращает список ((сладость, хруст), ожидаемый_класс)
    version='old' – только Фрукт, Овощ, Протеин
    version='new' – добавляются примеры Злаков
    """
    if version == 'old':
        return [
            ((3, 4), "Овощ"),
            ((2, 6), "Овощ"),
            ((7, 2), "Фрукт"),
            ((1, 3), "Протеин"),
            ((5, 5), "Фрукт"),
        ]
    else:
        return [
            ((3, 4), "Овощ"),
            ((2, 6), "Овощ"),
            ((7, 2), "Фрукт"),
            ((1, 3), "Протеин"),
            ((4, 8), "Злаки"),
            ((3, 9), "Злаки"),
        ]

# 6. Визуализация данных
def visualize(train_data, title):
    """
    train_data: список [((сладость, хруст), класс)]
    title: заголовок графика
    """
    colors = {'Фрукт': 'red', 'Овощ': 'green', 'Протеин': 'blue', 'Злаки': 'orange'}
    plt.figure(figsize=(8, 6))

    # Обучающие точки (каждая точка – свой цвет по классу)
    for (sweet, crunch), label in train_data:
        plt.scatter(sweet, crunch, c=colors[label], label=label, alpha=1, s=150, linewidth=1.5)

    # Убираем дублирование в легенде (одна запись на класс)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Название классов")
    plt.xlabel("Сладость")
    plt.ylabel("Хруст")
    plt.title(title)
    plt.grid(True)
    plt.show()

# 7. Проведение эксперимента для заданного набора данных и k
def run_experiment(train_data_list, test_samples, experiment_name, k):
    """
    train_data_list: список (продукт, сладость, хруст, класс) для обучения
    test_samples: список ((сладость, хруст), истинный_класс)
    experiment_name: строка для вывода
    k: число соседей (введённое пользователем)
    """
    print(f"\n========== {experiment_name} (k={k}) ==========")

    # Преобразуем обучающие данные в удобный формат
    train_features_labels = prepare_data(train_data_list)
    test_points = [p for p, _ in test_samples]
    true_labels = [l for _, l in test_samples]

    # --- Ручной k-NN ---
    start = time.time()
    manual_preds = [predict_knn(train_features_labels, p, k) for p in test_points]
    manual_time = time.time() - start

    # --- sklearn k-NN ---
    X_train = [feat for feat, _ in train_features_labels]
    y_train = [label for _, label in train_features_labels]
    X_test = test_points

    start = time.time()
    knn_sk = KNeighborsClassifier(n_neighbors=k)
    knn_sk.fit(X_train, y_train)
    sk_time = time.time() - start

    # Вычисляем точность
    manual_acc = sum(1 for i, p in enumerate(manual_preds) if p == true_labels[i]) / len(true_labels)
    sk_acc = knn_sk.score(X_test, true_labels)

    print(f"Ручной k-NN   : точность = {manual_acc:.2f}, время = {manual_time:.5f} сек")
    print(f"sklearn k-NN  : точность = {sk_acc:.2f}, время = {sk_time:.5f} сек")

    # Визуализируем (на основе ручных предсказаний)
    visualize(train_features_labels, f"{experiment_name}")
    return manual_acc, sk_acc, manual_time, sk_time

# Шаг 1: загружаем исходный датасет из файла
input_file = "food_data.txt"
original_data = ''
try:
    original_data = load_dataset(input_file)
    print(f"Загружено {len(original_data)} записей из '{input_file}'.")
except FileNotFoundError:
    print(f"Ошибка: файл '{input_file}' не найден.")
    print("Убедитесь, что файл существует и имеет формат:")
    print("продукт,сладость,хруст,класс")

# Шаг 2: пользователь вводит значение k
try:
    k = int(input("Введите значение k (количество соседей): "))
    if k <= 0:
        raise ValueError
except ValueError:
    print("Ошибка: k должно быть положительным целым числом.")

# Шаг 3: эксперимент 1 – без класса Злаки
old_train_data = [row for row in original_data if row[3] != "Злаки"]  # исходные без злаков
test_old = get_test_samples('old')
run_experiment(old_train_data, test_old, "Классификация продуктов", k)

# Шаг 4: эксперимент 2 – с классом Злаки
extended_data = [row for row in original_data]
test_new = get_test_samples('new')
run_experiment(extended_data, test_new, "Классификация продуктов", k)
