import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlflow-demo")

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():

    max_iter = 200

    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="weighted")
    recall = recall_score(y_test, preds, average="weighted")

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", max_iter)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    mlflow.sklearn.log_model(model, "model")

print("Logistic Regression run completed")