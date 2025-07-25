# 🚀 Flowserve

**Flowserve** is a modular pipeline framework for building, training, and serving machine learning models in Python — with zero boilerplate and full flexibility.

From loading data to launching a REST API, Flowserve lets you define the full journey in one intuitive pipeline.

---

## 🧠 What It Does

Flowserve makes it easy to:

- 🔄 Chain together modular pipeline steps (Load → Adapt → Model → Serve)
- 📊 Automatically infer the task type (classification or regression)
- 🧠 Auto-select or manually pass ML models (e.g., sklearn, XGBoost)
- 🪄 Serve your model as a REST API (coming soon)
- 📈 Track experiments with Weights & Biases (optional)

---

## 🔧 Example Usage

```python
from flowserve.steps.load import Load
from flowserve.steps.model import Model
from flowserve.pipeline import Pipeline

pipeline = Pipeline([
    Load(resource="https://raw.githubusercontent.com/openai/sample-datasets/main/customers.csv"),
    Model(target="purchased", model_type="classification", infer_model=True)
])

pipeline.execute()
````

---

## 📦 Components

### ✅ `Load`

Load data from:

* Local CSV or Excel files
* Remote URLs (direct file links)
* Cloud databases (PostgreSQL, MySQL, etc. — support coming soon)

### ✅ `Model`

* Infers whether it's a classification or regression task
* Supports automatic or user-supplied models
* Logs performance metrics (accuracy or RMSE)
* Optional integration with `wandb` for experiment tracking

### 🔜 `Serve` *(Coming Soon)*

* Expose trained model as a REST API with FastAPI
* Serve predictions via `/predict` endpoint

---

## 🛠 Installation

```bash
pip install -e .
```

### Required Packages

* `polars`
* `scikit-learn`
* `requests`
* `joblib`
* *(Optional)* `wandb`

---

## 🧪 Sample Dataset

You can test Flowserve using this sample CSV:

📥 [`customers.csv`](https://raw.githubusercontent.com/openai/sample-datasets/main/customers.csv)

```csv
age,income,has_credit_card,purchased
25,45000,yes,no
32,54000,no,yes
45,72000,yes,yes
...
```

---

## 🗺️ Roadmap

* [x] Basic pipeline execution
* [x] Local + URL data loading
* [x] Model training & evaluation
* [ ] S3/Cloud storage loading
* [ ] Model serialization
* [ ] RESTful model serving
* [ ] CLI interface (`flowserve run pipeline.py`)
* [ ] DAG-based dependency support

---

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute new steps, integrations, or performance improvements, please open a pull request or create an issue.

---

## 📝 License

MIT License © 2025 \[Your Name]
