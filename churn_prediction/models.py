"""Preprocessing and ML models."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class Preprocessor:
    """Simple preprocessor for churn data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_cols = None
        self.cat_cols = None
        self.num_cols = None
    
    def _create_features(self, df):
        """Create engineered features."""
        df = df.copy()
        df['cancel_year_month'] = df['cancel_year_month'].replace('N/A', np.nan)
        df['duration_month'] = pd.to_numeric(df['duration_month'].replace('N/A', np.nan), errors='coerce')
        
        if 'churned' not in df.columns:
            df['churned'] = df['cancel_year_month'].notna().astype(int)
        
        df['recruit_date'] = pd.to_datetime(df['recruit_year_month'], format='%Y-%m')
        df['cancel_date'] = pd.to_datetime(df['cancel_year_month'], format='%Y-%m', errors='coerce')
        ref_date = pd.to_datetime('2022-01-13')
        
        df['tenure_months'] = df.apply(
            lambda row: (row['cancel_date'] - row['recruit_date']).days / 30.44 
            if pd.notna(row.get('cancel_date')) 
            else (ref_date - row['recruit_date']).days / 30.44, axis=1
        )
        
        mask = df['churned'] == 1
        df.loc[mask, 'tenure_months'] = df.loc[mask, 'duration_month'].fillna(df.loc[mask, 'tenure_months'])
        
        df['monthly_revenue_rate'] = (df['total_bill'] / df['tenure_months'].replace(0, np.nan)).fillna(0)
        return df
    
    def fit(self, df):
        """Fit preprocessor."""
        df = self._create_features(df)
        exclude = ['churned', 'recruit_date', 'cancel_date', 'cancel_year_month', 'recruit_year_month', 'duration_month']
        self.feature_cols = [c for c in df.columns if c not in exclude]
        self.cat_cols = df[self.feature_cols].select_dtypes(include=['object', 'bool']).columns.tolist()
        self.num_cols = df[self.feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        for col in self.cat_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.encoders[col] = le
        
        if self.num_cols:
            self.scaler.fit(df[self.num_cols].fillna(0))
        return self
    
    def transform(self, df):
        """Transform data."""
        df = self._create_features(df)
        target = df['churned'] if 'churned' in df.columns else None
        
        for col in self.cat_cols:
            if col in df.columns:
                le = self.encoders[col]
                df[col] = df[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])
        
        if self.num_cols:
            for col in self.num_cols:
                if col not in df.columns:
                    df[col] = 0
            df[self.num_cols] = self.scaler.transform(df[self.num_cols].fillna(0))
        
        return df[self.feature_cols], target


def get_baseline_model():
    """Get baseline logistic regression model."""
    return LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')


def get_xgboost_model():
    """Get XGBoost model."""
    if not HAS_XGB:
        raise ImportError("Install xgboost: pip install xgboost")
    return xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)

