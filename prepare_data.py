from pathlib import Path

from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
dataset = iris.frame.rename(columns={"target": "target"})

Path("data").mkdir(exist_ok=True)
dataset.to_csv("data/dataset.csv", index=False)

print(f"Датасет сохранён: {dataset.shape[0]} строк, "
      f"{dataset.shape[1]} столбцов")
print(dataset.head())