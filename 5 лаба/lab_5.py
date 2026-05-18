import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.cross_decomposition import CCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. ЗАГРУЗКА И ПЕРВИЧНАЯ ОБРАБОТКА ДАННЫХ (НЕДВИЖИМОСТЬ)

df = pd.read_csv('ml_moscow_flats (1).csv')
print("Исходные признаки:", df.columns.tolist())

# Выбор признаков в соответствии с заданием
# wallsMaterial, floorNumber, floorsTotal, totalArea, kitchenArea, latitude, longitude
feature_cols = ['wallsMaterial', 'floorNumber', 'floorsTotal', 'totalArea', 'kitchenArea', 'latitude', 'longitude']
target_col = 'price'

# Проверка наличия всех столбцов
missing_cols = [col for col in feature_cols if col not in df.columns]
if missing_cols:
    raise KeyError(f"В датасете отсутствуют столбцы: {missing_cols}")

X_raw = df[feature_cols].copy()
y_raw = df[target_col].copy()

# Обработка пропусков
num_cols = X_raw.select_dtypes(include=[np.number]).columns
cat_cols = ['wallsMaterial']  # категориальный признак

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Раздельная обработка: числовые - медиана, категориальные - мода
X_raw[num_cols] = num_imputer.fit_transform(X_raw[num_cols])
X_raw[cat_cols] = cat_imputer.fit_transform(X_raw[cat_cols])

# 2. ПРЕДОБРАЗОВАНИЕ ПРИЗНАКОВ (OneHotEncoder + StandardScaler)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ])

X_preprocessed = preprocessor.fit_transform(X_raw)
feature_names = (list(num_cols) +
                 [f'wallsMaterial_{cat}' for cat in preprocessor.named_transformers_['cat'].categories_[0][1:]])

print(f"Число признаков после обработки: {X_preprocessed.shape[1]}")

# 3. ОБУЧЕНИЕ НА СЫРЫХ ДАННЫХ

# Для сырых данных не масштабируем числовые, но категориальный кодируем
X_raw_encoded = pd.get_dummies(X_raw, columns=['wallsMaterial'], drop_first=True)
X_raw_encoded = X_raw_encoded.values.astype(np.float64)
y = y_raw.values.reshape(-1, 1)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw_encoded, y, test_size=0.2, random_state=42)

cca_raw = CCA(n_components=1)
cca_raw.fit(X_train_raw, y_train)
y_pred_train_raw = cca_raw.predict(X_train_raw)
y_pred_test_raw = cca_raw.predict(X_test_raw)

def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

mse_tr, mae_tr, r2_tr = evaluate(y_train, y_pred_train_raw)
mse_te, mae_te, r2_te = evaluate(y_test, y_pred_test_raw)

print("\n=== СЫРЫЕ ДАННЫЕ (OneHot, без масштабирования) ===")
print(f"Train: MSE={mse_tr:.2f}, MAE={mae_tr:.2f}, R2={r2_tr:.4f}")
print(f"Test:  MSE={mse_te:.2f}, MAE={mae_te:.2f}, R2={r2_te:.4f}")

# 4. ОБУЧЕНИЕ НА ОЧИЩЕННЫХ ДАННЫХ (масштабирование + OneHot)
X_train_pre, X_test_pre, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

cca_pre = CCA(n_components=1)
cca_pre.fit(X_train_pre, y_train)
y_pred_train_pre = cca_pre.predict(X_train_pre)
y_pred_test_pre = cca_pre.predict(X_test_pre)

mse_tr_pre, mae_tr_pre, r2_tr_pre = evaluate(y_train, y_pred_train_pre)
mse_te_pre, mae_te_pre, r2_te_pre = evaluate(y_test, y_pred_test_pre)

print("\n=== ОЧИЩЕННЫЕ ДАННЫЕ (OneHot + StandardScaler) ===")
print(f"Train: MSE={mse_tr_pre:.2f}, MAE={mae_tr_pre:.2f}, R2={r2_tr_pre:.4f}")
print(f"Test:  MSE={mse_te_pre:.2f}, MAE={mae_te_pre:.2f}, R2={r2_te_pre:.4f}")

# 5. СРАВНЕНИЕ РАЗНЫХ МЕТОДОВ РАЗБИЕНИЯ

methods = {
    'random_80_20': train_test_split(X_preprocessed, y, test_size=0.2, random_state=42),
    'random_70_30': train_test_split(X_preprocessed, y, test_size=0.3, random_state=42),
}

# Стратифицированное разбиение по квантилям цены (чтобы сохранить распределение)
y_flat = y.ravel()
quantiles = pd.qcut(y_flat, q=5, labels=False, duplicates='drop')
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in strat_split.split(X_preprocessed, quantiles):
    X_tr_strat, X_te_strat = X_preprocessed[train_idx], X_preprocessed[test_idx]
    y_tr_strat, y_te_strat = y[train_idx], y[test_idx]
methods['stratified_80_20'] = (X_tr_strat, X_te_strat, y_tr_strat, y_te_strat)

results = {}
for name, (X_tr, X_te, y_tr, y_te) in methods.items():
    cca_temp = CCA(n_components=1)
    cca_temp.fit(X_tr, y_tr)
    y_pred = cca_temp.predict(X_te)
    mse, mae, r2 = evaluate(y_te, y_pred)
    results[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}

print("\n=== СРАВНЕНИЕ МЕТОДОВ РАЗБИЕНИЯ ===")
results_df = pd.DataFrame(results).T
print(results_df.round(4))

# 6. КРОСС-ВАЛИДАЦИЯ (5-fold)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mse = []
cv_mae = []
cv_r2 = []

for train_idx, val_idx in kfold.split(X_preprocessed):
    X_tr, X_val = X_preprocessed[train_idx], X_preprocessed[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    cca_cv = CCA(n_components=1)
    cca_cv.fit(X_tr, y_tr)
    y_pred_val = cca_cv.predict(X_val)
    cv_mse.append(mean_squared_error(y_val, y_pred_val))
    cv_mae.append(mean_absolute_error(y_val, y_pred_val))
    cv_r2.append(r2_score(y_val, y_pred_val))

print("\n=== 5-FOLD КРОСС-ВАЛИДАЦИЯ ===")
print(f"MSE: {np.mean(cv_mse):.2f} ± {np.std(cv_mse):.2f}")
print(f"MAE: {np.mean(cv_mae):.2f} ± {np.std(cv_mae):.2f}")
print(f"R2:  {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")

# 7. ГРАФИК РЕАЛЬНЫХ VS ПРЕДСКАЗАННЫХ (на очищенных данных)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test_pre, alpha=0.5, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Реальная цена (руб)')
plt.ylabel('Предсказанная цена (руб)')
plt.title('CCA: реальные vs предсказанные цены (очищенные данные)')
plt.tight_layout()
plt.show()