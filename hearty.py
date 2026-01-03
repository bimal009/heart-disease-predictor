

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('heart.csv')

df.head()

df.info()

sns.histplot(df['Age'],kde=True)

sns.histplot(df['RestingBP'],kde=True)

sns.histplot(df['RestingECG'],kde=True)



df['RestingBP'] = df['RestingBP'].replace(0, df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()).round(2)
df['Cholesterol'] = df['Cholesterol'].replace(0, df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()).round(2)

sns.histplot(df['Cholesterol'],kde=True)



sns.histplot(df['RestingBP'],kde=True)

sns.countplot(x=df['Sex'],hue=df['HeartDisease'])

sns.countplot(x=df['ST_Slope'],hue=df['HeartDisease'])

sns.countplot(x=df['ChestPainType'],hue=df['HeartDisease'])

sns.countplot(x=df['RestingECG'],hue=df['HeartDisease'])

sns.countplot(x=df['ExerciseAngina'],hue=df['HeartDisease'])

df_encoded=pd.get_dummies(df,drop_first=True)

df_encoded=df_encoded.astype(int)



from sklearn.preprocessing import StandardScaler
num_cols=['Age','Cholesterol','RestingBP','MaxHR','Oldpeak']
scaler=StandardScaler()
df_encoded[num_cols]=scaler.fit_transform(df_encoded[num_cols])

df_encoded.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM (RBF Kernel)": SVC(probability=True)
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'F1 Score': round(f1, 4)
    })

results

import joblib

joblib.dump(models['KNN'], 'heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
# Save a plain list of column names (call tolist(), not the method)
joblib.dump(X.columns.tolist(), 'columns.pkl')

