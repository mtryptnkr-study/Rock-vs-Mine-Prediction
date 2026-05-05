# Sonar Rock vs. Mine Prediction

This project uses machine learning to classify underwater objects as either **Rocks** or **Mines** based on sonar return signals. It features an optimized pipeline using **SMOTE** for data augmentation and **GridSearchCV** for hyperparameter tuning to maximize predictive performance on a small, high-dimensional dataset.

## 📌 Problem Statement
The goal is to build a model that can accurately distinguish between metal cylinders (representing naval mines) and cylindrical rocks. With only **208 instances** and **60 features**, the primary challenge is preventing overfitting while ensuring high classification accuracy.

## 📊 Dataset Overview
The dataset used is the **UCI Sonar Dataset**:
* **Total Samples:** 208
* **Features:** 60 numerical attributes (representing energy levels across different frequency bands).
* **Target Labels:** 
    * `M` -> Mine (Metal Cylinder)
    * `R` -> Rock

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** 
    * `Scikit-learn` (Machine Learning & Evaluation)
    * `Imbalanced-learn` (SMOTE for Data Augmentation)
    * `Pandas` & `NumPy` (Data Manipulation)
    * `Matplotlib/Seaborn` (Visualization)

## 🚀 Workflow & Optimization
To improve upon baseline results, the project followed an iterative optimization process:

1. **Baseline Modeling:** Establishing initial scores using Logistic Regression and Random Forest.
2. **Data Augmentation (SMOTE):** Since the dataset is small, I used **Synthetic Minority Over-sampling Technique (SMOTE)** to generate synthetic instances in the training set, helping the model learn more robust decision boundaries.
3. **Pipeline Construction:** Integrated an `imblearn` Pipeline to wrap the SMOTE and Classifier steps together. This ensures that augmentation happens **only during training folds**, preventing data leakage.
4. **Hyperparameter Tuning:** Conducted an exhaustive search via **GridSearchCV** and **RandomizedSearchCV** to find the optimal `max_depth`, `n_estimators`, and `min_samples_split` for the Random Forest model.

## 📈 Performance Comparison


| Model Approach | Accuracy |
| :--- | :--- |
| Logistic Regression | 82.35% |
| Random Forest | 85.71% |
| Random Forest + SMOTE | 85.71% |
| **Hypertuned RF (GridSearchCV + Pipeline)** | **86.00%** |

## ⚙️ How to Use
1. **Clone the repository:**
   ```bash
   git clone https://github.com
   ```
2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn
   ```
3. **Inference:**
   Use the `predict_sonar(input_data)` function provided in the script to pass a list of 60 frequency readings and get an instant classification result.

## 📝 Conclusion
By combining **SMOTE** to handle the low instance count and **GridSearchCV** to fine-tune the model parameters, the final pipeline achieved a more stable and accurate result than the baseline models. This project demonstrates how to handle small-scale datasets using modern synthetic data techniques and rigorous cross-validation.
