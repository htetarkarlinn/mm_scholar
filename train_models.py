import os
import sqlite3
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "mm_scholar.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CAT_FEATURES = ["country_of_study", "level", "field_of_study", "funding_type"]
NUM_FEATURES = ["min_gpa", "min_ielts"]
FEATURES     = CAT_FEATURES + NUM_FEATURES
TARGET       = "scholarship_name"

# load from database
conn = sqlite3.connect(DB_PATH)
df   = pd.read_sql("SELECT * FROM scholarships", conn)
conn.close()
print(f"Loaded {len(df)} rows")

# drop singleton classes before encoding so encoders only know trainable classes
counts = df[TARGET].map(df[TARGET].value_counts())
df     = df[counts >= 2].reset_index(drop=True)
print(f"After dropping singletons: {len(df)} rows, {df[TARGET].nunique()} classes")

# encode categorical features and target
encoders = {}
for col in CAT_FEATURES + [TARGET]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# numeric features: fill any NaN with 0 (no requirement)
for col in NUM_FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# split features and target
X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

# scale all features to 0-1 range
scaler  = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# GridSearchCV — k-NN
print("\nk-NN (GridSearchCV 5-fold)")
knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid={"n_neighbors": [3, 5, 7, 9, 11, 15], "metric": ["euclidean", "manhattan"]},
    cv=5, scoring="accuracy", n_jobs=-1
)
knn_grid.fit(X_train, y_train)
knn_model = knn_grid.best_estimator_
knn_acc   = accuracy_score(y_test, knn_model.predict(X_test))
print(f"  Best: {knn_grid.best_params_}  →  {knn_acc*100:.2f}%")

# GridSearchCV — Decision Tree
print("\nDecision Tree (GridSearchCV 5-fold)")
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={"max_depth": [3, 5, 7, 9, 11, 15], "min_samples_split": [2, 5, 10]},
    cv=5, scoring="accuracy", n_jobs=-1
)
dt_grid.fit(X_train, y_train)
dt_model = dt_grid.best_estimator_
dt_acc   = accuracy_score(y_test, dt_model.predict(X_test))
print(f"  Best: {dt_grid.best_params_}  →  {dt_acc*100:.2f}%")

# GridSearchCV — Random Forest
print("\nRandom Forest (GridSearchCV 5-fold)")
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
    cv=5, scoring="accuracy", n_jobs=-1
)
rf_grid.fit(X_train, y_train)
rf_model = rf_grid.best_estimator_
rf_acc   = accuracy_score(y_test, rf_model.predict(X_test))
print(f"  Best: {rf_grid.best_params_}  →  {rf_acc*100:.2f}%")

# save everything
joblib.dump(knn_model, os.path.join(MODELS_DIR, "knn_model.pkl"))
joblib.dump(dt_model,  os.path.join(MODELS_DIR, "dt_model.pkl"))
joblib.dump(rf_model,  os.path.join(MODELS_DIR, "rf_model.pkl"))
joblib.dump(encoders,  os.path.join(MODELS_DIR, "encoders.pkl"))
joblib.dump(scaler,    os.path.join(MODELS_DIR, "scaler.pkl"))

# summary
print("\n=== Summary ===")
print(f"{'Model':<20} {'Best Params':<40} {'Accuracy':>10}")
print("-" * 72)
print(f"{'k-NN':<20} {str(knn_grid.best_params_):<40} {knn_acc*100:>9.2f}%")
print(f"{'Decision Tree':<20} {str(dt_grid.best_params_):<40} {dt_acc*100:>9.2f}%")
print(f"{'Random Forest':<20} {str(rf_grid.best_params_):<40} {rf_acc*100:>9.2f}%")
print("-" * 72)

scores  = {"k-NN": knn_acc, "Decision Tree": dt_acc, "Random Forest": rf_acc}
objects = {"k-NN": knn_model, "Decision Tree": dt_model, "Random Forest": rf_model}
best    = max(scores, key=scores.get)
print(f"Best model: {best}  ({scores[best]*100:.2f}%)")

joblib.dump(objects[best], os.path.join(MODELS_DIR, "best_model.pkl"))

import json
with open(os.path.join(MODELS_DIR, "best_model_info.json"), "w") as f:
    json.dump({"name": best, "accuracy": round(scores[best] * 100, 2)}, f, indent=2)

print(f"Saved best_model.pkl  ({best})")
print("\nSaved to models/")
