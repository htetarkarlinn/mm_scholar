import os
import json
import sqlite3
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay, top_k_accuracy_score)
from sklearn.tree import export_text, plot_tree

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "mm_scholar.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CAT_FEATURES = ["country_of_study", "level", "field_of_study", "funding_type"]
NUM_FEATURES = ["min_gpa", "min_ielts"]
FEATURES     = CAT_FEATURES + NUM_FEATURES
TARGET       = "scholarship_name"

BLUE  = "#1F4E79"
BLUE2 = "#2E75B6"
GREEN = "#1D9E75"
RED   = "#E24B4A"

# load data
conn = sqlite3.connect(DB_PATH)
df   = pd.read_sql("SELECT * FROM scholarships", conn)
conn.close()
print(f"Loaded {len(df)} rows")

# load models and encoders
knn      = joblib.load(os.path.join(MODELS_DIR, "knn_model.pkl"))
dt       = joblib.load(os.path.join(MODELS_DIR, "dt_model.pkl"))
rf       = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
gb       = joblib.load(os.path.join(MODELS_DIR, "gb_model.pkl"))
encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

# drop singleton classes (same filter applied during training)
counts = df[TARGET].map(df[TARGET].value_counts())
df = df[counts >= 2].reset_index(drop=True)

# encode categorical features using saved encoders
for col in CAT_FEATURES + [TARGET]:
    df[col] = encoders[col].transform(df[col])

# numeric features: fill NaN with 0
for col in NUM_FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)


# --- 1. Accuracy comparison bar chart ---
models      = {"k-NN": knn, "Decision Tree": dt, "Random Forest": rf, "Gradient Boosting": gb}
accuracies  = {name: accuracy_score(y_test, m.predict(X_test_s))
               for name, m in models.items()}

fig, ax = plt.subplots(figsize=(8, 5))
colors  = [GREEN if v == max(accuracies.values()) else BLUE2
           for v in accuracies.values()]
bars = ax.bar(accuracies.keys(), [v * 100 for v in accuracies.values()],
              color=colors, width=0.5)
ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
for bar, val in zip(bars, accuracies.values()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val*100:.2f}%", ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: accuracy_comparison.png")


# --- 2. Full metrics table ---
print("\n=== Model Evaluation Metrics ===")
print(f"{'Model':<20} {'Top-1 Acc':>10} {'Top-3 Acc':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 72)
eval_metrics = {}
for name, m in models.items():
    y_pred  = m.predict(X_test_s)
    y_proba = m.predict_proba(X_test_s)
    acc     = accuracy_score(y_test, y_pred)
    top3    = top_k_accuracy_score(y_test, y_proba, k=3, labels=m.classes_)
    prec    = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec     = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1      = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    eval_metrics[name] = {"accuracy": acc, "top3_accuracy": top3,
                          "precision": prec, "recall": rec, "f1": f1}
    print(f"{name:<20} {acc*100:>9.2f}% {top3*100:>9.2f}% {prec*100:>9.2f}% {rec*100:>9.2f}% {f1*100:>9.2f}%")
print("-" * 72)


# --- 3. Cross-validation scores ---
# Pipeline wraps MinMaxScaler + model so the scaler is re-fitted on each
# fold's training split only — no leakage from test folds into the scaler.
print("\n=== 5-Fold Cross Validation ===")
for name, m in models.items():
    pipe   = Pipeline([("scaler", MinMaxScaler()), ("clf", clone(m))])
    scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    eval_metrics[name]["cv_mean"] = scores.mean()
    eval_metrics[name]["cv_std"]  = scores.std()
    print(f"{name:<20} mean={scores.mean()*100:.2f}%  std={scores.std()*100:.2f}%")

params_map = {
    "k-NN":               f"k={knn.n_neighbors}",
    "Decision Tree":      f"depth={dt.max_depth}",
    "Random Forest":      f"n={rf.n_estimators}",
    "Gradient Boosting":  f"n={gb.n_estimators}, lr={gb.learning_rate}",
}
models_list = [
    {
        "model":         name,
        "params":        params_map[name],
        "accuracy":      round(v["accuracy"]      * 100, 2),
        "top3_accuracy": round(v["top3_accuracy"] * 100, 2),
        "precision":     round(v["precision"]     * 100, 2),
        "recall":        round(v["recall"]        * 100, 2),
        "f1":            round(v["f1"]            * 100, 2),
        "cv_mean":       round(v["cv_mean"]       * 100, 2),
        "cv_std":        round(v["cv_std"]        * 100, 2),
    }
    for name, v in eval_metrics.items()
]
with open(os.path.join(MODELS_DIR, "best_model_info.json")) as _f:
    _best = json.load(_f)
with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
    json.dump({
        "knn_role":        "production_ranker",
        "best_model_name": _best["name"],
        "best_model_acc":  _best["accuracy"],
        "models":          models_list,
    }, f, indent=2)
print("Saved: models/metrics.json")


# --- 4. Confusion matrix for best classifier baseline ---
_baseline_models = {"Decision Tree": dt, "Random Forest": rf, "Gradient Boosting": gb}
_best_baseline   = max(_baseline_models, key=lambda n: eval_metrics[n]["accuracy"])
y_pred_best = _baseline_models[_best_baseline].predict(X_test_s)
cm        = confusion_matrix(y_test, y_pred_best)
fig, ax   = plt.subplots(figsize=(14, 12))
disp      = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=90)
ax.set_title(f"Confusion Matrix — {_best_baseline} (Best Classifier Baseline)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_dt.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: confusion_matrix_dt.png")


# --- 5. Decision Tree visualization (top 3 levels) ---
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(dt, max_depth=3,
          feature_names=FEATURES,
          filled=True, rounded=True,
          fontsize=9, ax=ax)
ax.set_title("Decision Tree Structure — Comparison Baseline (top 3 levels)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "decision_tree_plot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: decision_tree_plot.png")


# --- 6. Feature importance from Random Forest ---
importances = rf.feature_importances_
fig, ax     = plt.subplots(figsize=(8, 4))
colors      = [GREEN if v == max(importances) else BLUE2 for v in importances]
ax.barh(FEATURES, importances, color=colors)
ax.set_title("Feature Importance — Random Forest", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance score")
for i, v in enumerate(importances):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: feature_importance.png")


# --- 7. Robustness check ---
# Pipeline keeps scaler inside the loop so each seed gets its own scaler
# fitted on that seed's training split only — consistent with CV treatment.
print("\n=== Robustness Check (different random seeds) ===")
for name, m in models.items():
    accs = []
    for seed in [0, 7, 21, 42, 99]:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        pipe = Pipeline([("scaler", MinMaxScaler()), ("clf", clone(m))])
        pipe.fit(Xtr, ytr)
        accs.append(accuracy_score(yte, pipe.predict(Xte)))
    print(f"{name:<20} min={min(accs)*100:.2f}%  max={max(accs)*100:.2f}%  mean={sum(accs)/len(accs)*100:.2f}%")

print("\nEvaluation complete. Charts saved to static/eda/")