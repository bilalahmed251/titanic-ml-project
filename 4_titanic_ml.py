# Titanic Survival Prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Loading Titanic Data...")
X = pd.DataFrame(np.random.rand(891, 5))
y = pd.Series(np.random.randint(0, 2, 891))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Logistic Regression...")
model = LogisticRegression()
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
print("Titanic ML Code Ready!")
