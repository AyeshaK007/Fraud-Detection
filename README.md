# Fraud Detection Using Machine Learning 📊

An end-to-end machine-learning project for detecting potentially fraudulent financial transactions in a highly imbalanced dataset.

## 🔍 Project overview

The project covers the full workflow from data preparation and exploratory analysis to model training and an interactive prediction interface.

### What I worked on

- Explored a dataset containing 6.3M+ transaction records
- Investigated skewed transaction distributions and feature relationships
- Built preprocessing with a Scikit-learn `ColumnTransformer`
- Encoded categorical features and standardized numerical features
- Addressed severe class imbalance with `class_weight="balanced"`
- Trained a Logistic Regression model
- Evaluated model performance
- Built a Streamlit interface for interactive fraud predictions

## 🛠️ Tech stack

- **Python**
- **Pandas / NumPy**
- **Matplotlib / Seaborn**
- **Scikit-learn**
- **Streamlit**
- **Joblib**
- Jupyter / VS Code

## 📈 Key result

The current project reports approximately **94% overall accuracy** while explicitly accounting for the fact that fraud represents only about **0.13%** of the dataset.

> Accuracy alone is not enough for an imbalanced fraud-detection problem. The imbalance-handling strategy is an important part of this project.

## 📸 Recommended portfolio screenshots

Add these to `docs/screenshots/`:

1. Dataset / EDA visualization
2. Feature or correlation analysis
3. Model evaluation
4. Streamlit prediction interface
5. Example prediction result

## 🚀 Run locally

```bash
git clone https://github.com/AyeshaK007/Fraud-Detection.git
cd Fraud-Detection
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
streamlit run "fraud detection.py"
```

The original dataset is excluded from version control via `.gitignore`.

## 📌 Project status

Portfolio data-science project. Future improvements could include comparing multiple models, reporting precision/recall and PR-AUC, tuning thresholds, and adding stronger validation for the minority class.

## 👩🏻‍💻 Author

**AyeshaK007** — BSCS student building toward data science while developing strong frontend and software-engineering skills.
