# 🚢 Titanic - Machine Learning Survival Prediction

This project uses machine learning to predict whether a passenger survived the Titanic shipwreck, based on features such as age, gender, ticket class, and more.

## 📌 Objective

The goal is to build a predictive model using classification algorithms to determine passenger survival. This project is inspired by the Kaggle Titanic competition.

## 🧠 Business Use Case

Although this is a historical dataset, this workflow can be applied to real-world binary classification problems such as:
- Customer churn prediction
- Loan approval decisions
- Fraud detection

## 🛠️ Tools & Libraries Used

- Python
- Pandas
- NumPy
- Seaborn & Matplotlib
- Scikit-learn (SVC, DecisionTree, RandomForest, Logistic Regression)
- Jupyter Notebook

## 📊 Exploratory Data Analysis (EDA)

We explored and visualized the dataset using:
- Bar charts to analyze survival rate by gender and class
- Distribution plots of age and fare
- Missing data handling and imputation
- Correlation heatmaps

**Examples:**
![Survival by Gender](images/gender_survival.png)
![Age Distribution](images/age_distribution.png)

## 🧹 Data Preprocessing

- Handled missing values (Age, Cabin, Embarked)
- Label encoding for categorical features (Sex, Embarked)
- Feature engineering: Family size, IsAlone, Title extraction from Name

## 🤖 Models Trained

| Model               | Cross-Validation Accuracy |
|--------------------|---------------------------|
| Logistic Regression| 78.3%                     |
| Decision Tree      | 75.1%                     |
| Random Forest      | 81.6%                     |
| SVC (Best Model)   | 83.5%                     |

> ✅ Best model: **Support Vector Classifier (SVC)** with 83.5% accuracy.

## 📈 Feature Importance

Used Random Forest feature importances and recursive feature elimination (RFE) to select top features influencing survival prediction.

## Picture's 
![image](https://github.com/user-attachments/assets/6ef0d862-c6a2-4867-9cf6-aa826cc44326)
![image](https://github.com/user-attachments/assets/23c78691-e54c-4f96-9bf1-cc08599207d1)
![image](https://github.com/user-attachments/assets/e4d42d49-1237-4177-8d5a-f4e4dc39abd4)

## ✅ Final Notes

This project demonstrates how to:
- Explore and clean real-world data
- Apply multiple ML algorithms
- Optimize models using cross-validation
- Explain results with visuals for client understanding

> 📬 *Feel free to fork, use or showcase this project as part of your data science portfolio!*
