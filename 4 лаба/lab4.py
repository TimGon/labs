import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, silhouette_samples)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. Загрузка данных
train_df = pd.read_csv('disease_train.csv')
test_df = pd.read_csv('disease_public_test.csv')
true_labels_df = pd.read_csv('disease_sample_submission.csv')  # содержит Y для теста

# Целевая переменная и признаки
X = train_df.drop('Y', axis=1)
y = train_df['Y']
X_test = test_df  # без Y
y_test_true = true_labels_df['Y']   # истинные метки тестовой выборки

print("Размер обучающей выборки:", X.shape)
print("Размер тестовой выборки:", X_test.shape)

# 2. Предварительный анализ (пропуски, типы)
print("\nИнформация о данных:")
print(train_df.info())
print("\nПропуски в обучающем наборе:\n", train_df.isnull().sum())

# 3. Обучение на сырых данных (без предобработки)
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

svc_raw = LinearSVC(class_weight='balanced', random_state=42, max_iter=5000)
svc_raw.fit(X_train_raw, y_train_raw)
y_pred_raw = svc_raw.predict(X_val_raw)

print("\n=== Результаты на сырых данных (валидация) ===")
print(f"Accuracy: {accuracy_score(y_val_raw, y_pred_raw):.4f}")
print(f"Precision: {precision_score(y_val_raw, y_pred_raw):.4f}")
print(f"Recall: {recall_score(y_val_raw, y_pred_raw):.4f}")
print(f"F1-score: {f1_score(y_val_raw, y_pred_raw):.4f}")
print("Confusion matrix:")
print(confusion_matrix(y_val_raw, y_pred_raw))

# 4. Предобработка (масштабирование)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 5. Обучение на масштабированных данных (простое разбиение)
X_train_sc, X_val_sc, y_train_sc, y_val_sc = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

svc_sc = LinearSVC(class_weight='balanced', random_state=42, max_iter=5000)
svc_sc.fit(X_train_sc, y_train_sc)
y_pred_sc = svc_sc.predict(X_val_sc)

print("\n=== Результаты на масштабированных данных (валидация) ===")
print(f"Accuracy: {accuracy_score(y_val_sc, y_pred_sc):.4f}")
print(f"Precision: {precision_score(y_val_sc, y_pred_sc):.4f}")
print(f"Recall: {recall_score(y_val_sc, y_pred_sc):.4f}")
print(f"F1-score: {f1_score(y_val_sc, y_pred_sc):.4f}")
print("Confusion matrix:")
print(confusion_matrix(y_val_sc, y_pred_sc))

# 6. Кросс-валидация (5‑кратная) на масштабированных данных
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(svc_sc, X_scaled, y, cv=cv, scoring='accuracy')
print(f"\nКросс-валидация (5-кратная) для масштабированных данных:")
print(f"Средняя accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 7. Оценка на тестовом наборе (сравнение с истинными метками)
svc_final = LinearSVC(class_weight='balanced', random_state=42, max_iter=5000)
svc_final.fit(X_scaled, y)
y_pred_test = svc_final.predict(X_test_scaled)

#Вывод метрик тестовой выборки
print("\n=== Оценка на тестовой выборке (масштабированные данные) ===")
print(f"Accuracy: {accuracy_score(y_test_true, y_pred_test):.4f}")
print(f"Precision: {precision_score(y_test_true, y_pred_test):.4f}")
print(f"Recall: {recall_score(y_test_true, y_pred_test):.4f}")
print(f"F1-score: {f1_score(y_test_true, y_pred_test):.4f}")
print("Confusion matrix:")
print(confusion_matrix(y_test_true, y_pred_test))

# 8. ГРАФИЧЕСКИЙ АНАЛИЗ
# 8.1 Матрица ошибок на тестовых данных
cm = confusion_matrix(y_test_true, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Здоров (0)', 'Болен (1)'])
disp.plot(cmap='Blues')
plt.title('Матрица ошибок (LinearSVC на тестовых данных)')
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# 8.2 Кластеризация для n_clusters=2
kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(X_scaled)
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)

y_lower = 10
for i in range(2):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster
    y_lower = y_upper + 10

# 8.3 Визуализация кластеров в 2D (PCA) и сравнение с истинными метками
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sc1 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6, edgecolors='k')
plt.title('Кластеры (KMeans)')
plt.colorbar(sc1, label='Кластер')

plt.subplot(1, 2, 2)
sc2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='bwr', alpha=0.6, edgecolors='k')
plt.title('Истинные метки (0=здоров, 1=болен)')
plt.colorbar(sc2, label='Метка')
plt.tight_layout()
plt.savefig('clustering_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 8.4 Кривая обучения (learning curve)
train_sizes, train_scores, test_scores = learning_curve(
    svc_final, X_scaled, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Тренировочная точность')
plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Валидационная точность (CV)')
plt.title('Кривая обучения LinearSVC')
plt.xlabel('Размер обучающей выборки')
plt.ylabel('Точность (accuracy)')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('learning_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# 9. Метрики кластеризации (обучение уже выполнено в п. 8.4-8.5, просто выводим)
print("\n=== Кластеризация KMeans (n_clusters=2) ===")
print(f"Silhouette Score: {silhouette_avg:.4f}")
ari = adjusted_rand_score(y, cluster_labels)
nmi = normalized_mutual_info_score(y, cluster_labels)
print(f"Adjusted Rand Index: {ari:.4f}")
print(f"Normalized Mutual Information: {nmi:.4f}")

# 10. Сводная таблица результатов
results = {
    'Модель/Данные': ['Raw (no scaling)', 'Scaled (train/val split)', 'Scaled (5-fold CV)', 'Scaled (test set)', 'KMeans clustering'],
    'Accuracy': [accuracy_score(y_val_raw, y_pred_raw),
                 accuracy_score(y_val_sc, y_pred_sc),
                 cv_scores.mean(),
                 accuracy_score(y_test_true, y_pred_test),
                 'N/A'],
    'Precision': [precision_score(y_val_raw, y_pred_raw),
                  precision_score(y_val_sc, y_pred_sc),
                  'N/A',
                  precision_score(y_test_true, y_pred_test),
                  'N/A'],
    'Recall': [recall_score(y_val_raw, y_pred_raw),
               recall_score(y_val_sc, y_pred_sc),
               'N/A',
               recall_score(y_test_true, y_pred_test),
               'N/A'],
    'F1': [f1_score(y_val_raw, y_pred_raw),
           f1_score(y_val_sc, y_pred_sc),
           'N/A',
           f1_score(y_test_true, y_pred_test),
           'N/A'],
    'Silhouette': ['N/A', 'N/A', 'N/A', 'N/A', silhouette_avg]
}

results_df = pd.DataFrame(results)
print("\n=== Сводная таблица результатов ===")
print(results_df.to_string(index=False))
results_df.to_csv('classification_results.csv', index=False)