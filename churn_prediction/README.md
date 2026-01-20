# Churn Prediction - Simple ML Pipeline

Minimal, production-ready churn prediction pipeline with 5 modules.

## Structure

```
churn_prediction/
├── data.py          # Data loading
├── models.py        # Preprocessing + ML models
├── train.py         # Training script
├── api.py           # FastAPI backend
├── app.py           # Streamlit frontend
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost fastapi uvicorn streamlit requests pydantic matplotlib seaborn pyyaml
```

### 2. Train Models

```bash
python churn_prediction/train.py
```

Models saved to `churn_prediction/saved/`

### 3. Start API

```bash
uvicorn churn_prediction.api:app --reload
```

API: `http://localhost:8000` | Docs: `http://localhost:8000/docs`

### 4. Launch UI

```bash
streamlit run churn_prediction/app.py
```

UI: `http://localhost:8501`

## Usage

### API Example

```python
import requests

response = requests.post(
    "http://localhost:8000/predict?model=xgboost",
    json={
        "acquisition_channel": "phone",
        "fiber_or_adsl": "fiber",
        "has_retention": False,
        "offer": "#11:Freebox Revolution with TV 3999eur",
        "sub_offer": "11.4:Freebox Revolution with TV 3999eur",
        "recruit_year_month": "2016-01",
        "total_bill": 1500.0,
        "cancel_year_month": None,
        "duration_month": None
    }
)
print(response.json())
```

## Models

- **Baseline**: Logistic Regression (fast, interpretable)
- **XGBoost**: Gradient Boosting (higher accuracy)

Both handle class imbalance automatically.
