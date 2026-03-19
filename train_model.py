import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from category_encoders import TargetEncoder
import warnings
warnings.filterwarnings('ignore')

print("📂 Завантажуємо дані...")
df = pd.read_csv('train_ML_READY.csv')

X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(df)) 

print("\n🧠 ПОЧИНАЄМО НАВЧАННЯ (Ансамбль XGBoost + CatBoost)...")
fold = 1
for train_index, val_index in skf.split(X, y):
    print(f"   ⏳ Сесія {fold}/5: Навчання двох моделей...")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # --- Розумне кодування (Target Encoding) ---
    cat_cols = ['gender', 'reg_country', 'traffic_type']
    te = TargetEncoder(cols=cat_cols)
    X_train = te.fit_transform(X_train, y_train)
    X_val = te.transform(X_val)
    
    # --- Детектив 1: XGBoost ---
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=10,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict_proba(X_val)[:, 1]
    
    # --- Детектив 2: LightGBM (від Microsoft) ---
    lgb_model = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=10, # Та сама параноя для балансу
        random_state=42,
        n_jobs=-1,
        verbose=-1           # Вимикаємо системний спам у терміналі
    )
    lgb_model.fit(X_train, y_train)
    
    # Отримуємо думку другого детектива
    lgb_preds = lgb_model.predict_proba(X_val)[:, 1]
    
    # --- КОНСИЛІУМ: Усереднюємо думки XGBoost та LightGBM ---
    oof_predictions[val_index] = (xgb_preds + lgb_preds) / 2.0
    
    fold += 1

print("\n🎯 Всі 5 сесій завершено. Шукаємо ідеальний спільний поріг...")
best_f1 = 0
best_thresh = 0

for thresh in np.arange(0.1, 0.9, 0.05):
    preds = (oof_predictions >= thresh).astype(int)
    current_f1 = f1_score(y, preds)
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_thresh = thresh

print("\n" + "="*50)
print(f"🔥 ЧЕСНИЙ F1-Score (Ансамбль): {best_f1:.4f} (при порозі {best_thresh:.2f})")
print("="*50 + "\n")

final_preds = (oof_predictions >= best_thresh).astype(int)
print("Детальний звіт по всіх екзаменаційних сесіях:")
print(classification_report(y, final_preds))