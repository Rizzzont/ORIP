import mlflow
import mlflow.sklearn
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model_uri = "models:/iris_production_model@staging"
print(f"Загрузка модели: {model_uri}")

model = mlflow.sklearn.load_model(model_uri)

initial_type = [("float_input", FloatTensorType([None, 4]))]
onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    target_opset=15,
)

output_path = "iris_model.onnx"
with open(output_path, "wb") as file:
    file.write(onnx_model.SerializeToString())

print(f"Экспорт завершён: {output_path}")