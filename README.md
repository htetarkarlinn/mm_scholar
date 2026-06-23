# MM Scholar

A free, AI-assisted scholarship recommendation web application for Myanmar students. Students enter their study goals and academic profile, then receive eligible scholarships ranked by k-NN similarity matching and explained on demand by Google Gemini.

**Live:** [mm-scholar.onrender.com](https://mm-scholar.onrender.com)

---

## What it does

* Recommends scholarships from a curated dataset of 87 verified scholarships across 30 countries
* Filters scholarships using hard eligibility constraints such as study country, level, funding type, GPA, and IELTS score
* Uses a four-level SQL fallback chain to reduce empty results while preserving GPA and IELTS eligibility thresholds
* Ranks eligible scholarships using a distance-weighted k-Nearest Neighbours (k-NN) model
* Shows a match percentage for each recommended scholarship based on k-NN class probability
* Generates personalised AI explanations on demand using Google Gemini 2.5 Flash
* Collects student feedback with star ratings and optional comments
* Stores feedback in SQLite during local development and PostgreSQL in production on Render

---

## Tech stack

| Layer                        | Technology                                      |
| ---------------------------- | ----------------------------------------------- |
| Web framework                | Flask 3.0                                       |
| Templates                    | Jinja2                                          |
| Frontend                     | Bootstrap 5.3 + custom CSS                      |
| Production ranker            | k-NN, scikit-learn                              |
| Comparison baselines         | Decision Tree, Random Forest, Gradient Boosting |
| AI explanations              | Google Gemini 2.5 Flash                         |
| Local database               | SQLite                                          |
| Production feedback database | PostgreSQL via psycopg2                         |
| Deployment                   | Gunicorn on Render                              |

---

## Prerequisites

* Python 3.10+
* A Google Gemini API key from [Google AI Studio](https://aistudio.google.com)
* Git

---

## Local setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/htetarkarlinn/mm_scholar.git
cd mm_scholar
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
ADMIN_PASSWORD=your_admin_password_here
```

For local development, `DATABASE_URL` is optional. If it is not set, the application uses SQLite.

### 4. Create the SQLite database

This converts `data/scholarships_dataset.csv` into the local SQLite database:

```bash
python convert_to_sqlite.py
```

### 5. Train the models

This trains k-NN, Decision Tree, Random Forest, and Gradient Boosting models and saves the model files to `models/`:

```bash
python train_models.py
```

### 6. Evaluate the models

This generates `models/metrics.json`, which is displayed on the `/compare` route:

```bash
python evaluate_models.py
```

### 7. Run the app

```bash
flask --app app run --port 5001
```

Open http://localhost:5001

---

## Deployment on Render

### 1. Push to GitHub

```bash
git add .
git commit -m "Update project"
git push
```

### 2. Create a PostgreSQL database on Render

Render Dashboard → **New** → **PostgreSQL**

Recommended settings:

| Field  | Value           |
| ------ | --------------- |
| Name   | `mm-scholar-db` |
| Region | Singapore       |
| Plan   | Free            |

After creation, copy the **Internal Database URL**.

### 3. Create a Web Service on Render

Render Dashboard → **New** → **Web Service** → connect the GitHub repository.

Use these settings:

| Field         | Value                             |
| ------------- | --------------------------------- |
| Runtime       | Python 3                          |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn app:app`                |
| Plan          | Free                              |

Add these environment variables:

| Key              | Value                                        |
| ---------------- | -------------------------------------------- |
| `DATABASE_URL`   | Internal Database URL from Render PostgreSQL |
| `GEMINI_API_KEY` | Your Google Gemini API key                   |
| `ADMIN_PASSWORD` | Password for the `/admin` route              |

The application detects `DATABASE_URL` in production and uses PostgreSQL for feedback storage. If `DATABASE_URL` is not present, it uses SQLite locally.

> **Free tier note:** Render's free web service may sleep after inactivity. The first request after sleep can take around 30 seconds to wake up.

---

## Project structure

```text
mm_scholar/
├── app.py                       # Flask routes and request handling
├── config.py                    # Environment configuration and logging setup
├── convert_to_sqlite.py         # CSV to SQLite ingestion pipeline
├── train_models.py              # Trains k-NN and baseline models
├── evaluate_models.py           # Generates metrics.json and evaluation results
├── eda.py                       # Exploratory data analysis charts
├── requirements.txt
├── Procfile                     # Gunicorn entry point for Render
├── data/
│   └── scholarships_dataset.csv # Source dataset
├── models/
│   ├── knn_model.pkl            # Production k-NN ranker
│   ├── dt_model.pkl             # Decision Tree baseline
│   ├── rf_model.pkl             # Random Forest baseline
│   ├── gb_model.pkl             # Gradient Boosting baseline
│   ├── encoders.pkl             # Target label encoder / model metadata
│   ├── best_model_info.json     # Best baseline model information
│   └── metrics.json             # Evaluation results displayed by /compare
├── repositories/
│   ├── scholarship_repo.py      # Scholarship catalogue queries
│   └── feedback_repo.py         # SQLite/PostgreSQL feedback adapter
├── services/
│   ├── recommendation_service.py # Validation, SQL fallback, k-NN ranking
│   └── explanation_service.py    # Gemini explanation generation and cache
├── static/
│   ├── style.css
│   ├── hero_bg.jpg
│   ├── favicon.png
│   └── eda/
└── templates/
    ├── index.html
    ├── results.html
    ├── about.html
    ├── compare.html
    ├── browse.html
    ├── feedback_results.html
    ├── thank_you.html
    ├── admin.html
    ├── admin_edit_feedback.html
    ├── 400.html
    ├── 404.html
    └── 500.html
```

---

## ML pipeline

### Features

| Feature            | Type        | Processing    |
| ------------------ | ----------- | ------------- |
| `country_of_study` | Categorical | OneHotEncoder |
| `level`            | Categorical | OneHotEncoder |
| `field_of_study`   | Categorical | OneHotEncoder |
| `funding_type`     | Categorical | OneHotEncoder |
| `min_gpa`          | Numeric     | MinMaxScaler  |
| `min_ielts`        | Numeric     | MinMaxScaler  |

The target label is `scholarship_name`, encoded with `LabelEncoder`.

### Recommendation logic

1. **Input validation**
   The student profile is validated server-side before processing. GPA must be within 0.0–4.0 and IELTS must be within 0.0–9.0.

2. **SQL eligibility filtering**
   The system applies a four-level fallback chain:

   * Level 1: exact match on all provided filters
   * Level 2: relax funding type
   * Level 3: match country while retaining GPA and IELTS thresholds
   * Level 4: fully-funded scholarships at the student's study level while retaining GPA and IELTS thresholds

3. **k-NN ranking**
   The student's profile is passed to the trained k-NN pipeline. Candidate scholarships are scored using `predict_proba()` and ranked by class probability.

4. **Match percentage**
   The displayed match percentage is calculated as:

   ```text
   scholarship_class_probability × 100
   ```

5. **AI explanation**
   Explanations are generated separately on demand. When a student clicks “Why this matches me?”, the system checks the in-memory cache first. On a cache miss, it calls Google Gemini and stores the generated explanation.

---

## Evaluation results

| Model             | Role              | Top-1 Acc. | Top-3 Acc. | CV Mean |
| ----------------- | ----------------- | ---------: | ---------: | ------: |
| k-NN              | Production ranker |     54.35% |     74.64% |  68.11% |
| Decision Tree     | Baseline          |     81.16% |     90.58% |  84.72% |
| Random Forest     | Baseline          |     79.71% |     98.55% |  82.97% |
| Gradient Boosting | Baseline          |     80.43% |     97.83% |  83.56% |

Top-3 accuracy is the primary metric because MM Scholar returns up to three recommendations per query.

Although the tree-based baselines achieve higher classification accuracy, k-NN is used as the production ranker because its distance-weighted probability provides a more interpretable similarity-based match score for students.

---

## Main routes

| Route               | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `/`                 | Homepage and recommendation form                             |
| `/recommend`        | Processes student profile and returns ranked recommendations |
| `/explain`          | Generates AI explanation for a selected scholarship          |
| `/scholarships`     | Browse full scholarship catalogue                            |
| `/compare`          | View model comparison metrics                                |
| `/feedback-results` | View public feedback summary                                 |
| `/admin`            | Admin dashboard protected by HTTP Basic Authentication       |
| `/health`           | Health check endpoint for deployment monitoring              |

---

## Developer

**Htet Arkar Linn**
Hanoi University of Science and Technology (HUST), Vietnam
Bachelor of Engineering in Information Technology
EU Mobility Programme for Myanmar (EMPM), Class of 2026

[GitHub](https://github.com/htetarkarlinn/mm_scholar) · [LinkedIn](https://www.linkedin.com/in/htet-arkar-linn/)
