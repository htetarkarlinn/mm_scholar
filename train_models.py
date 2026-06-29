import os
import json
import sqlite3
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "mm_scholar.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CAT_FEATURES = ["country_of_study", "level", "field_of_study", "funding_type"]
NUM_FEATURES = ["min_gpa", "min_ielts"]
FEATURES     = CAT_FEATURES + NUM_FEATURES
TARGET       = "scholarship_name"

conn = sqlite3.connect(DB_PATH)
df   = pd.read_sql("SELECT * FROM scholarships", conn)
conn.close()
print(f"Loaded {len(df)} rows")

counts = df[TARGET].map(df[TARGET].value_counts())
df     = df[counts >= 2].reset_index(drop=True)
print(f"After dropping singletons: {len(df)} rows, {df[TARGET].nunique()} classes")

for col in NUM_FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Shared target + CAT encoders (used by all models) 
encoders = {}
for col in CAT_FEATURES + [TARGET]:
    le = LabelEncoder()
    le.fit(df[col])
    encoders[col] = le

y = encoders[TARGET].transform(df[TARGET])

# Label-encode CAT features for tree baselines
df_le = df.copy()
for col in CAT_FEATURES:
    df_le[col] = encoders[col].transform(df_le[col])

X     = df_le[FEATURES].values.astype(float)   # label-encoded, for DT/RF/GB
X_knn = df[FEATURES]                            # raw strings + floats, for k-NN OHE pipeline

# Single deterministic split — same row indices used for both k-NN and tree models
idx = np.arange(len(df))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

X_train,     X_test     = X[train_idx],              X[test_idx]
X_knn_train, X_knn_test = X_knn.iloc[train_idx],     X_knn.iloc[test_idx]
y_train,     y_test     = y[train_idx],               y[test_idx]
print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

#  FIX 1+2: k-NN — OHE for categoricals, MinMaxScaler for numerics,
#    weights='distance' so predict_proba is continuous (not k-vote fractions) 
knn_pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ("num", MinMaxScaler(), NUM_FEATURES),
])
knn_pipe = Pipeline([
    ("pre", knn_pre),
    ("clf", KNeighborsClassifier(weights="distance")),
])

print("\nk-NN (GridSearchCV 5-fold)")
knn_grid = GridSearchCV(
    knn_pipe,
    param_grid={
        "clf__n_neighbors": [3, 5, 7, 9, 11, 15],
        "clf__metric":      ["euclidean", "manhattan"],
    },
    cv=5, scoring="accuracy", n_jobs=-1,
)
knn_grid.fit(X_knn_train, y_train)
knn_model = knn_grid.best_estimator_
knn_acc   = accuracy_score(y_test, knn_model.predict(X_knn_test))
print(f"  Best: {knn_grid.best_params_}  →  {knn_acc*100:.2f}%")

# FIX 3: Tree baselines — scaler inside pipeline so GridSearchCV folds
# cannot leak a pre-fitted scaler into the test folds 

print("\nDecision Tree (GridSearchCV 5-fold)")
dt_grid = GridSearchCV(
    Pipeline([("scaler", MinMaxScaler()), ("clf", DecisionTreeClassifier(random_state=42))]),
    param_grid={"clf__max_depth": [3, 5, 7, 9, 11, 15], "clf__min_samples_split": [2, 5, 10]},
    cv=5, scoring="accuracy", n_jobs=-1,
)
dt_grid.fit(X_train, y_train)
dt_model = dt_grid.best_estimator_
dt_acc   = accuracy_score(y_test, dt_model.predict(X_test))
print(f"  Best: {dt_grid.best_params_}  →  {dt_acc*100:.2f}%")

print("\nRandom Forest (GridSearchCV 5-fold)")
rf_grid = GridSearchCV(
    Pipeline([("scaler", MinMaxScaler()), ("clf", RandomForestClassifier(random_state=42))]),
    param_grid={"clf__n_estimators": [50, 100, 200], "clf__max_depth": [None, 10, 20]},
    cv=5, scoring="accuracy", n_jobs=-1,
)
rf_grid.fit(X_train, y_train)
rf_model = rf_grid.best_estimator_
rf_acc   = accuracy_score(y_test, rf_model.predict(X_test))
print(f"  Best: {rf_grid.best_params_}  →  {rf_acc*100:.2f}%")

print("\nGradient Boosting (GridSearchCV 5-fold)")
gb_grid = GridSearchCV(
    Pipeline([("scaler", MinMaxScaler()), ("clf", GradientBoostingClassifier(random_state=42))]),
    param_grid={
        "clf__n_estimators":  [50, 100],
        "clf__max_depth":     [3, 5],
        "clf__learning_rate": [0.1, 0.2],
    },
    cv=5, scoring="accuracy", n_jobs=-1,
)
gb_grid.fit(X_train, y_train)
gb_model = gb_grid.best_estimator_
gb_acc   = accuracy_score(y_test, gb_model.predict(X_test))
print(f"  Best: {gb_grid.best_params_}  →  {gb_acc*100:.2f}%")

# save models + encoders (no standalone scaler.pkl — every model carries its own)
joblib.dump(knn_model, os.path.join(MODELS_DIR, "knn_model.pkl"))
joblib.dump(dt_model,  os.path.join(MODELS_DIR, "dt_model.pkl"))
joblib.dump(rf_model,  os.path.join(MODELS_DIR, "rf_model.pkl"))
joblib.dump(gb_model,  os.path.join(MODELS_DIR, "gb_model.pkl"))
joblib.dump(encoders,  os.path.join(MODELS_DIR, "encoders.pkl"))

print("\n=== Summary ===")
print(f"{'Model':<20} {'Best Params':<40} {'Accuracy':>10}")
print("-" * 72)
print(f"{'k-NN':<20} {str(knn_grid.best_params_):<40} {knn_acc*100:>9.2f}%")
print(f"{'Decision Tree':<20} {str(dt_grid.best_params_):<40} {dt_acc*100:>9.2f}%")
print(f"{'Random Forest':<20} {str(rf_grid.best_params_):<40} {rf_acc*100:>9.2f}%")
print(f"{'Gradient Boosting':<20} {str(gb_grid.best_params_):<40} {gb_acc*100:>9.2f}%")
print("-" * 72)
print("k-NN is the production recommendation model (similarity-based matching).")
print("DT, RF, GB are academic comparison baselines.")

scores  = {"Decision Tree": dt_acc, "Random Forest": rf_acc, "Gradient Boosting": gb_acc}
best    = max(scores, key=scores.get)
print(f"Best classifier baseline: {best}  ({scores[best]*100:.2f}%)")

with open(os.path.join(MODELS_DIR, "best_model_info.json"), "w") as f:
    json.dump({"name": best, "accuracy": round(scores[best] * 100, 2)}, f, indent=2)

print("\nSaved to models/")
