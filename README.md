# VITyarthi-AIML-Project
# 🎓 Student Performance Predictor

A machine learning project that predicts student exam scores based on study habits, attendance, sleep, and other lifestyle factors.

Built as a **BYOP (Bring Your Own Project)** submission for the course *Fundamentals of AI and ML*.

---

## 📌 Problem Statement

Student academic performance is influenced by many factors beyond raw intelligence — study hours, attendance, sleep, parental education, and access to resources all play a role. This project builds a predictive model to estimate a student's exam score from these features, helping educators identify at-risk students early.

---

## 📁 Project Structure

```
student-performance-predictor/
│
├── student_performance_predictor.py   # Main ML pipeline
├── student_performance.csv            # Dataset (100 student records)
├── eda_plots.png                      # Generated EDA visualizations
├── model_comparison.png               # Model benchmarking chart
├── actual_vs_predicted.png            # Best model prediction plot
├── feature_importance.png             # Feature importance chart
└── README.md                          # This file
```

---

## 📊 Dataset

The dataset (`student_performance.csv`) contains **100 student records** with the following features:

| Column | Description |
|---|---|
| `student_id` | Unique student identifier |
| `gender` | Male / Female |
| `age` | Age of the student (18–22) |
| `study_hours_per_day` | Average daily study hours |
| `attendance_percentage` | Class attendance (%) |
| `sleep_hours` | Average sleep per night |
| `previous_score` | Score in previous exam |
| `extracurricular` | Participates in activities (Yes/No) |
| `internet_access` | Has internet access (Yes/No) |
| `parent_education_level` | High School / Graduate / Postgraduate |
| `exam_score` | **Target variable** – Final exam score |

---

## 🧠 ML Concepts Used

- Exploratory Data Analysis (EDA)
- Label Encoding & Feature Scaling
- Train/Test Split
- Regression Models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor ⭐ (best performer)
  - Gradient Boosting Regressor
- Model Evaluation: MAE, RMSE, R² Score
- Feature Importance Analysis

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/student-performance-predictor.git
cd student-performance-predictor
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

> Python 3.8+ recommended.

### 3. Run the Project

```bash
python student_performance_predictor.py
```

---

## 📈 Sample Output

```
======================================================
       STUDENT PERFORMANCE PREDICTOR
======================================================

📂 Dataset Shape : 100 rows × 11 columns

  🤖 Linear Regression     → R²: 0.9823
  🤖 Decision Tree         → R²: 0.9741
  🤖 Random Forest         → R²: 0.9901  ← Best
  🤖 Gradient Boosting     → R²: 0.9887

🏆 Best Model: Random Forest  (R² = 0.9901)

  🎓 Predicted Exam Score: 81.4 / 100
```

Four plots are auto-saved:
- `eda_plots.png` — distributions, scatter plots, heatmap
- `model_comparison.png` — R², MAE, RMSE side-by-side
- `actual_vs_predicted.png` — best model accuracy
- `feature_importance.png` — top contributing features

---

## 🔍 Key Findings

- **Study hours** and **previous score** are the strongest predictors of exam performance
- **Attendance percentage** is highly correlated with final score
- Students with postgraduate parents score consistently higher on average
- Random Forest outperforms other models with R² > 0.99 on this dataset

---

## 🚀 How to Predict for a New Student

Edit the `predict_new_student()` function in the script:

```python
new_student = pd.DataFrame([{
    "gender":                 1,    # 1=Male, 0=Female
    "age":                    19,
    "study_hours_per_day":    6,
    "attendance_percentage":  88,
    "sleep_hours":            7,
    "previous_score":         75,
    "extracurricular":        1,    # 1=Yes, 0=No
    "internet_access":        1,
    "parent_education_level": 1,
}])
```

---

## 🛠 Technologies Used

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Pandas | Data loading & manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Visualization |
| Scikit-learn | ML models & evaluation |

---

## 👤 Author

**[Your Name]**
VIT — Fundamentals of AI and ML (BYOP Submission)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
