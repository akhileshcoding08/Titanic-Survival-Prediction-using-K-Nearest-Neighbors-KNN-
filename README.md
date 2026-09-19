# 🚢 Titanic Survival Prediction using K-Nearest Neighbors (KNN)

A machine learning project that predicts passenger survival on the Titanic using the **K-Nearest Neighbors (KNN)** classification algorithm. The project covers data cleaning verification, exploratory data analysis (EDA), feature scaling, model training, and evaluation.

---

## 📌 Project Overview

This project uses a pre-cleaned and one-hot encoded version of the classic Titanic dataset to build a classification model that predicts whether a passenger survived (`1`) or did not survive (`0`). The workflow is implemented end-to-end in a Jupyter Notebook, from data inspection to model evaluation using the F1 score.

---

## 📂 Repository Structure

```
titanic-knn-classification/
│
├── data_cleaned.csv                   # Pre-processed, one-hot encoded Titanic dataset
├── Titanic_Data_Cleaned_KNN.ipynb     # Main Jupyter Notebook (EDA + KNN model)
├── README.md                          # Project documentation
└── Titanic_KNN_Project_Documentation.docx  # Detailed Word documentation
```

---

## 📊 Dataset

The dataset (`data_cleaned.csv`) contains **891 passenger records** with **25 columns**, including the target variable and one-hot encoded categorical features:

| Column | Description |
|---|---|
| `Survived` | Target variable — 0 = Did not survive, 1 = Survived |
| `Age` | Passenger age |
| `Fare` | Ticket fare |
| `Pclass_1/2/3` | One-hot encoded passenger class |
| `Sex_female` / `Sex_male` | One-hot encoded gender |
| `SibSp_0` … `SibSp_8` | One-hot encoded number of siblings/spouses aboard |
| `Parch_0` … `Parch_6` | One-hot encoded number of parents/children aboard |
| `Embarked_C` / `Embarked_Q` / `Embarked_S` | One-hot encoded port of embarkation |

> The dataset provided is already cleaned and encoded — no missing values are present.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:**
  - `pandas`, `numpy` — data handling
  - `matplotlib`, `seaborn` — data visualization
  - `scikit-learn` — preprocessing, model building & evaluation

---

## 🔍 Project Workflow

1. **Import Libraries** — Load pandas, numpy, matplotlib, and seaborn.
2. **Load Dataset** — Read `data_cleaned.csv` into a DataFrame.
3. **Data Quality Checks**
   - Check for missing values (`isna().sum()`) → none found.
   - Check for duplicate rows (`duplicated().sum()`) → **111 duplicates found**.
   - Remove duplicates using `drop_duplicates()`.
4. **Data Inspection** — Use `.info()` and `.describe()` to understand structure and statistics of the cleaned data (780 rows remain after de-duplication).
5. **Outlier Detection** — Generate boxplots for every numeric column to visually inspect outliers.
6. **Correlation Analysis** — Compute a correlation matrix and visualize it with a heatmap to understand relationships between features.
7. **Feature/Target Split** — Separate the dataset into features (`X`) and target (`y = Survived`).
8. **Feature Scaling** — Apply `MinMaxScaler` to normalize all feature values between 0 and 1.
9. **Train-Test Split** — Split data into 80% training and 20% testing sets (`random_state=56`).
10. **Model Building** — Train a `KNeighborsClassifier` with `n_neighbors=10`.
11. **Model Evaluation** — Evaluate performance using the **F1 score**.

---

## 📈 Results

| Metric | Value |
|---|---|
| Algorithm | K-Nearest Neighbors (k = 10) |
| Train/Test Split | 80% / 20% |
| **F1 Score** | **0.717** |

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed along with the following libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/titanic-knn-classification.git
   cd titanic-knn-classification
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Titanic_Data_Cleaned_KNN.ipynb
   ```

3. Run all cells sequentially to reproduce the analysis and results.

---

## 🔮 Future Improvements

- Hyperparameter tuning for `n_neighbors` using `GridSearchCV`
- Compare KNN with other classifiers (Logistic Regression, Random Forest, SVM)
- Add cross-validation for more robust performance estimates
- Handle class imbalance if present
- Deploy the model via a simple Flask/Streamlit web app

---

## 🙋 Author

Feel free to connect for feedback, suggestions, or collaboration opportunities.
