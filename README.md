# 🧠 Diabetes Progression Prediction Using Machine Learning

This project applies multiple machine learning models to predict diabetes progression based on a real-world dataset. It includes preprocessing, feature selection, model evaluation, hyperparameter tuning, and result visualization to identify the best regression model.

📎 **Notebook Link**: [Google Colab - CODTECH-TASK2](https://colab.research.google.com/drive/1uLTaRcTLYYdm47XLmq5-f-mw_e1pdS0s)

---

## 📊 Dataset

* **Source**: `load_diabetes()` from `sklearn.datasets`
* **Samples**: 442
* **Features**: 10 numerical features related to patient data
* **Target**: Disease progression one year after baseline

---

## 🧰 Tools & Libraries Used

* **NumPy, Pandas** – Data handling and transformation
* **Matplotlib, Seaborn** – Data visualization
* **Scikit-learn** – Modeling, evaluation, and hyperparameter tuning
* **Joblib** – Model saving
* **Google Colab** – Cloud-based notebook execution

---

## 📈 Workflow Summary

### 1. **Data Exploration**

* Overview of features and target variable
* Check for missing values
* Visualize distribution of the target

### 2. **Preprocessing**

* Standardization using `StandardScaler`
* Train-test split (80/20)
* Feature selection using `SelectKBest`

### 3. **Model Training**

* Trained and evaluated:

  * **Linear Regression**
  * **Ridge Regression**
  * **Random Forest Regressor**
* Metrics: **MSE** and **R² Score**

### 4. **Hyperparameter Tuning**

* Performed `GridSearchCV` for Random Forest
* Tuned hyperparameters: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
* Evaluated best model on test set

### 5. **Model Comparison & Visualization**

* Compared models based on MSE and R²
* Plotted:

  * Feature importance
  * Actual vs Predicted values
  * Cross-validation performance

### 6. **Model Saving**

* Saved the best performing model (`Random Forest`) using `joblib`

---

## 🏆 Results

| Model                    | Test MSE   | R² Score    |
| ------------------------ | ---------- | ----------- |
| Linear Regression        | \~xxxx     | \~xx.xx     |
| Ridge Regression         | \~xxxx     | \~xx.xx     |
| **Random Forest (Best)** | **\~xxxx** | **\~xx.xx** |

* Top 5 features selected for best performance
* Final model saved as: `best_diabetes_model.pkl`

---

## ✅ How to Run

### Requirements

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

### Run Notebook

1. Clone or open the Colab notebook
2. Execute all cells sequentially
3. The model will be trained, evaluated, and saved locally

---

## 📃 License
This project is open for educational and personal use. Please credit if reused.

---

## 🙋‍♂️ Author
TANMAY GUHA

Email:- tanmayguha15@gmail.com

---
