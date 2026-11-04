from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Подключаем скрипт к запущенному серверу MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris_classification_experiment")

# Загружаем датасет Iris, сохранённый в CSV
data = pd.read_csv("data/dataset.csv")
X = data.drop(columns=["target"])
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Эти параметры будем менять для трёх запусков
n_estimators = 150
max_depth = 15

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Сохраняем матрицу ошибок
    Path("plots").mkdir(exist_ok=True)
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png")
    plt.close()

    # Сохраняем важность признаков
    importance = pd.Series(
        model.feature_importances_,
        index=X.columns,
    ).sort_values()
    importance.plot(kind="barh", title="Feature importance")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png")
    plt.close()

    mlflow.log_artifact("plots/confusion_matrix.png", artifact_path="plots")
    mlflow.log_artifact("plots/feature_importance.png", artifact_path="plots")
    mlflow.sklearn.log_model(model, artifact_path="random_forest_model")

    print(f"Параметры: n_estimators={n_estimators}, max_depth={max_depth}")
    print(f"Accuracy: {accuracy:.3f}; F1: {f1:.3f}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")