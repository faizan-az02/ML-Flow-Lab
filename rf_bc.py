import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Breast_Cancer_Experiments")

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():

    n_estimators = 200

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="weighted")
    recall = recall_score(y_test, preds, average="weighted")

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", n_estimators)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, "model")

print("Random Forest run completed")