import numpy as np
import onnxruntime as ort


def main():
    print("Запуск модели через ONNX Runtime...")

    session = ort.InferenceSession(
        "iris_model.onnx",
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    print(f"Вход модели: {input_name}")

    # Четыре признака Iris для двух объектов
    sample = np.array(
        [
            [5.1, 3.5, 1.4, 0.2],
            [6.7, 3.0, 5.2, 2.3],
        ],
        dtype=np.float32,
    )

    outputs = session.run(None, {input_name: sample})

    print("Предсказанные классы:", outputs[0])
    if len(outputs) > 1:
        print("Вероятности классов:", outputs[1])


if __name__ == "__main__":
    main()