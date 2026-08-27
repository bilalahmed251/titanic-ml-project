# Advanced Titanic Survival ML Pipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Initializing Advanced Titanic Pipeline...")
df = pd.DataFrame({
    'Pclass': np.random.choice([1, 2, 3], 891),
    'Sex': np.random.choice([0, 1], 891),
    'Age': np.random.uniform(1, 80, 891),
    'Fare': np.random.uniform(7, 500, 891),
    'SibSp': np.random.choice([0,1,2,3], 891),
    'Parch': np.random.choice([0,1,2], 891),
    'Survived': np.random.choice([0, 1], 891)
})

# Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Gradient Boosting Classifier...")
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbc.fit(X_train, y_train)

print("Cross-Validation Score:", np.mean(cross_val_score(gbc, X_train, y_train, cv=5)))

print("Test Evaluation...")
preds = gbc.predict(X_test)
print(classification_report(y_test, preds))

joblib.dump(gbc, 'titanic_gbc_model.pkl')
print("Advanced Titanic ML Pipeline Complete!")
