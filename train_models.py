import os
import sqlite3
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "mm_scholar.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = ["country_of_study", "level", "field_of_study", "funding_type"]
TARGET   = "scholarship_name"

# load from database
conn = sqlite3.connect(DB_PATH)
df   = pd.read_sql("SELECT * FROM scholarships", conn)
conn.close()
print(f"Loaded {len(df)} rows")

# encode categorical features and target
encoders = {}
for col in FEATURES + [TARGET]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# drop classes with only 1 sample — stratified split requires ≥2 per class
counts = df[TARGET].map(df[TARGET].value_counts())
df = df[counts >= 2].reset_index(drop=True)
print(f"After dropping singletons: {len(df)} rows, {df[TARGET].nunique()} classes")

# split features and target
X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

# scale features to 0-1 range
scaler  = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# train k-NN — test k = 3, 5, 7, 9
print("\nk-NN")
knn_results = {}
for k in [3, 5, 7, 9]:
    m   = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    knn_results[k] = (m, acc)
    print(f"  k={k}  →  {acc*100:.2f}%")

best_k        = max(knn_results, key=lambda k: knn_results[k][1])
knn_model, knn_acc = knn_results[best_k]
print(f"  Best k={best_k}  ({knn_acc*100:.2f}%)")

# train Decision Tree — test depth = 3, 5, 7
print("\nDecision Tree")
dt_results = {}
for depth in [3, 5, 7, 9, 11]:
    m   = DecisionTreeClassifier(max_depth=depth, random_state=42)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    dt_results[depth] = (m, acc)
    print(f"  depth={depth}  →  {acc*100:.2f}%")

best_depth       = max(dt_results, key=lambda d: dt_results[d][1])
dt_model, dt_acc = dt_results[best_depth]
print(f"  Best depth={best_depth}  ({dt_acc*100:.2f}%)")

# train Random Forest — test 50, 100 trees
print("\nRandom Forest")
rf_results = {}
for n in [50, 100]:
    m   = RandomForestClassifier(n_estimators=n, random_state=42)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    rf_results[n] = (m, acc)
    print(f"  n_estimators={n}  →  {acc*100:.2f}%")

best_n           = max(rf_results, key=lambda n: rf_results[n][1])
rf_model, rf_acc = rf_results[best_n]
print(f"  Best n={best_n}  ({rf_acc*100:.2f}%)")

# save everything
joblib.dump(knn_model, os.path.join(MODELS_DIR, "knn_model.pkl"))
joblib.dump(dt_model,  os.path.join(MODELS_DIR, "dt_model.pkl"))
joblib.dump(rf_model,  os.path.join(MODELS_DIR, "rf_model.pkl"))
joblib.dump(encoders,  os.path.join(MODELS_DIR, "encoders.pkl"))
joblib.dump(scaler,    os.path.join(MODELS_DIR, "scaler.pkl"))

# summary
print("\n=== Summary ===")
print(f"{'Model':<20} {'Params':<18} {'Accuracy':>10}")
print("-" * 50)
print(f"{'k-NN':<20} {'k='+str(best_k):<18} {knn_acc*100:>9.2f}%")
print(f"{'Decision Tree':<20} {'depth='+str(best_depth):<18} {dt_acc*100:>9.2f}%")
print(f"{'Random Forest':<20} {'n='+str(best_n):<18} {rf_acc*100:>9.2f}%")
print("-" * 50)

scores = {"k-NN": knn_acc, "Decision Tree": dt_acc, "Random Forest": rf_acc}
best   = max(scores, key=scores.get)
print(f"Best model: {best}  ({scores[best]*100:.2f}%)")
print("\nSaved to models/")