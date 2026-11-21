# Titanic Survival Prediction Project

# ------------------------------
# 1. Import libraries
# ------------------------------



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# ------------------------------
# 2. Load dataset
# ------------------------------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic = pd.read_csv(url)

print(titanic.head())
print(titanic.info())
print(titanic.isnull().sum())

# ------------------------------
# 3. Select useful features
# ------------------------------
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

X = titanic[features]
y = titanic[target]

# ------------------------------
# 4. Preprocessing pipeline
# ------------------------------

numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
categorical_features = ['Sex', 'Embarked']

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ------------------------------
# 5. Split dataset
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ------------------------------
# 6. Build models
# ------------------------------

# Logistic Regression
log_reg = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression())
])

# Decision Tree
dt = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", DecisionTreeClassifier(max_depth=5))
])

# Random Forest (Best)
rf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])

# ------------------------------
# 7. Train models
# ------------------------------
log_reg.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# ------------------------------
# 8. Evaluate results
# ------------------------------
models = [("Logistic Regression", log_reg),
          ("Decision Tree", dt),
          ("Random Forest", rf)]

for name, model in models:
    predictions = model.predict(X_test)
    print("\n📌 Model:", name)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))
