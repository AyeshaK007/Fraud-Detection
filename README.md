#  Fraud Detection Using Machine Learning

A comprehensive end-to-end data science project focused on predicting fraudulent financial transactions with high accuracy. This project includes data cleaning, exploratory data analysis (EDA), handling massive class imbalance, pipeline design, and deploying a live Streamlit web application.

##  Key Features & Capabilities
* **Massive Dataset Handling:** Analyzed a synthetic financial transaction dataset consisting of over 6.3 million rows of transaction logs.
* **Exploratory Data Analysis (EDA):** Leveraged log transformations to handle highly skewed transaction distributions and built a detailed correlation matrix to uncover multi-collinearity.
* **Advanced Pipeline Optimization:** Constructed an automated Scikit-Learn `ColumnTransformer` pipeline that standardizes numerical features while applying one-hot encoding to transaction categories.
* **Interactive Web Dashboard:** Built and deployed a live Streamlit web application where a user can input transaction variables and instantly receive a fraud assessment from the underlying model.

##  Tech Stack & Libraries
* **Language & IDE:** Python (Visual Studio Code / Jupyter Notebooks)
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Logistic Regression, Pipelines, Train-Test Split)
* **Deployment:** Streamlit, Joblib

##  Core Performance Metrics
* **Overall Accuracy:** ~94% 
* **Class Imbalance Strategy:** Handled the extreme class imbalance (where fraud accounts for only ~0.13% of total data) directly at the algorithm level by adjusting the model's `class_weight` parameter to `'balanced'`.

##  How to Run the App
1. Clone the repository: `git clone https://github.com/AyeshaK007/Fraud-Detection.git`
2. Ensure you have the required packages installed (`pandas`, `scikit-learn`, `streamlit`, `joblib`).
3. Launch the web interface using:
   ```bash
   streamlit run fraud detection.py
