import time
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, make_scorer
import pandas as pd

# 1. Датасет и использование их
df = pd.DataFrame(
    {
        "Продукт": ["Яблоко", "Салат","Бекон","Банан","Орехи","Рыба","Сыр","Виноград","Морковь","Апельсин"],
        "Сладость": [7, 2, 1, 9, 1, 1, 1, 8, 2, 6],
        "Хруст": [7, 5, 2, 1, 5, 1, 1, 1, 8, 1],
        "Класс": ["Фрукт", "Овощ", "Протеин","Фрукт","Протеин", "Протеин", "Протеин", "Фрукт", "Овощ", "Фрукт"]
    }
)

# 2. Подготовка данных для k-NN (признаки + метки)
def prepare_data(data_list, exclude_class=None):
    """
    Преобразует список (продукт, сладость, хруст, класс) в формат
    [((сладость, хруст), класс)].
    Если указан exclude_class, записи с этим классом исключаются.
    """
    features_labels = []
    for _, row in data_list.iterrows():
        cls = row["Класс"]
        if exclude_class and cls == exclude_class:
            continue
        features_labels.append(((row["Сладость"], row["Хруст"]), cls))
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
# 7. Кросс-валидация
def cross_validate_sklearn(df, k_values, n_folds=5):
    X = df[["Сладость", "Хруст"]].values
    y = df["Класс"].values
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {}

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        # cross_validate возвращает словарь с ключами 'test_score', 'fit_time', 'score_time'
        cv_results = cross_validate(knn, X, y, cv=kf, scoring=make_scorer(accuracy_score),
                                    return_train_score=False)
        test_scores = cv_results['test_score']
        results[k] = {'mean': np.mean(test_scores), 'std': np.std(test_scores)}
    return results
# 8. Проведение эксперимента для заданного набора данных и k
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

    start = time.time()
    knn_sk = KNeighborsClassifier(n_neighbors=k)
    knn_sk.fit(X_train, y_train)
    sk_preds = knn_sk.predict(test_points)
    sk_time = time.time() - start

    # Вычисляем точность
    manual_acc = sum(1 for i, p in enumerate(manual_preds) if p == true_labels[i]) / len(true_labels)
    sk_acc = accuracy_score(true_labels, sk_preds)

    print(f"Ручной k-NN   : точность = {manual_acc:.2f}, время = {manual_time:.5f} сек")
    print(f"sklearn k-NN  : точность = {sk_acc:.2f}, время = {sk_time:.5f} сек")

    # Визуализируем (на основе ручных предсказаний)
    visualize(train_features_labels, f"{experiment_name}")
    return manual_acc, sk_acc, manual_time, sk_time

# Шаг 1: пользователь вводит значение k
k=0
try:
    k = int(input("Введите значение k (количество соседей): "))
    if k <= 0:
        raise ValueError
except ValueError:
    print("Ошибка: k должно быть положительным целым числом.")

# Шаг 2: Проводим Кросс-валидацию данных
# Формируем список k для проверки (вокруг введённого значения)
neighborhood = 2
k_min = max(1, k - neighborhood)
k_max = k + neighborhood
k_values = list(range(k_min, k_max + 1))

X = df[["Сладость", "Хруст"]].values
y = df["Класс"].values

# Создаём 5-кратный KFold с перемешиванием
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n--- Кросс-валидация (5-fold) для k = {k_values} ---")
for test_k in k_values:
    knn = KNeighborsClassifier(n_neighbors=test_k)
    scores = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
    print(f"k={test_k}: средняя точность = {scores.mean():.3f} (+/- {scores.std():.3f})")

# Шаг 3: эксперимент 1 – без класса Злаки
df_no_grains = df[df["Класс"] != "Злаки"].copy()
test_old = get_test_samples('old')
run_experiment(df_no_grains, test_old, "Классификация продуктов", k)

# Шаг 4: эксперимент 2 – с классом Злаки
# Добавляем продукты класса Злаки (так как в исходном датасете их нет)
extra_grains = pd.DataFrame([
    ["Мюсли", 4, 9, "Злаки"],
    ["Овсянка", 3, 8, "Злаки"],
    ["Гранола", 5, 9, "Злаки"]
], columns=["Продукт", "Сладость", "Хруст", "Класс"])
df_extended = pd.concat([df, extra_grains], ignore_index=True)

test_new = get_test_samples('new')
run_experiment(df_extended, test_new, "Классификация продуктов", k)
