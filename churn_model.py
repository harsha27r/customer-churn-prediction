import duckdb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 1. Load data ──────────────────────────────────────────────────────────────
con = duckdb.connect('C:/Users/harsh/dbt-ecommerce-pipeline/ecommerce_pipeline/dev.duckdb')

df = con.execute("""
    SELECT customer_unique_id, total_orders, lifetime_value,
           avg_order_value, total_items_purchased,
           first_order_date, last_order_date
    FROM mart_customer_orders
    WHERE first_order_date IS NOT NULL
""").fetchdf()

print(f"Loaded {len(df)} customers")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
df['days_since_last_order'] = (
    pd.Timestamp('2018-09-01') - pd.to_datetime(df['last_order_date'])
).dt.days

df['customer_lifespan_days'] = (
    pd.to_datetime(df['last_order_date']) - pd.to_datetime(df['first_order_date'])
).dt.days

df['churned'] = ((df['total_orders'] == 1) & (df['days_since_last_order'] > 180)).astype(int)

print(f"Churn rate: {df['churned'].mean():.1%}")

# ── 3. Train/Test Split ───────────────────────────────────────────────────────
features = ['total_orders', 'lifetime_value', 'avg_order_value',
            'total_items_purchased', 'days_since_last_order', 'customer_lifespan_days']

X = df[features].fillna(0)
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ── 4. Train Model ────────────────────────────────────────────────────────────
model = XGBClassifier(n_estimators=100, max_depth=4,
                      learning_rate=0.1, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n── Model Performance ──")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# ── 6. Feature Importance Plot ────────────────────────────────────────────────
importance = pd.Series(model.feature_importances_, index=features).sort_values()

plt.figure(figsize=(8, 5))
importance.plot(kind='barh', color='steelblue')
plt.title('Feature Importance — Customer Churn Model')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('C:/Users/harsh/customer-churn-prediction/feature_importance.png', dpi=150)
plt.close()
print("Saved: feature_importance.png")
print("\nDone.")