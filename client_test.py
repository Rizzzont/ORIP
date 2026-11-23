import time

import requests


API_URL = "http://127.0.0.1:8000/v1/predict"

payload = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


def send_request():
    start_time = time.time()
    response = requests.post(API_URL, json=payload, timeout=10)
    latency = (time.time() - start_time) * 1000

    if response.status_code == 200:
        data = response.json()
        print(
            f"[УСПЕХ] Статус: {response.status_code} | "
            f"Задержка: {latency:.2f} мс"
        )
        print(
            f"Предсказание: класс {data['predicted_class']} | "
            f"Статус модели: {data['model_status']}\n"
        )
    else:
        print(f"[ОШИБКА] Статус: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    print("Старт автоматического тестирования API Iris\n")

    for i in range(5):
        print(f"Отправка запроса №{i + 1}...")
        send_request()
        time.sleep(0.5)