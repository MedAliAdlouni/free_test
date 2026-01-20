"""Training script for churn prediction models."""

import pickle
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from churn_prediction.data import load_data
from churn_prediction.models import Preprocessor, get_baseline_model, get_xgboost_model
from sklearn.model_selection import train_test_split


def evaluate(y_true, y_pred, y_proba, name):
    """Evaluate model and print metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba[:, 1]) if y_proba is not None else None
    }
    print(f"\n{name} Results:")
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k:12s}: {v:.4f}")
    return metrics


def plot_confusion_matrix(y_true, y_pred, name, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title(f'{name} - Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main training pipeline."""
    print("Loading data...")
    df = load_data()
    
    print("Preprocessing...")
    preprocessor = Preprocessor()
    preprocessor.fit(df)
    X, y = preprocessor.transform(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train baseline
    print("\nTraining baseline model...")
    baseline = get_baseline_model()
    baseline.fit(X_train, y_train)
    y_val_pred = baseline.predict(X_val)
    y_val_proba = baseline.predict_proba(X_val)
    evaluate(y_val, y_val_pred, y_val_proba, "Baseline (Val)")
    
    # Train XGBoost
    print("\nTraining XGBoost model...")
    xgb_model = get_xgboost_model()
    # Handle class imbalance
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    if pos_count > 0:
        xgb_model.set_params(scale_pos_weight=neg_count / pos_count)
    
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_val_pred_xgb = xgb_model.predict(X_val)
    y_val_proba_xgb = xgb_model.predict_proba(X_val)
    evaluate(y_val, y_val_pred_xgb, y_val_proba_xgb, "XGBoost (Val)")
    
    # Test evaluation
    print("\n" + "="*50)
    print("Test Set Evaluation:")
    print("="*50)
    
    y_test_pred = baseline.predict(X_test)
    y_test_proba = baseline.predict_proba(X_test)
    evaluate(y_test, y_test_pred, y_test_proba, "Baseline")
    
    y_test_pred_xgb = xgb_model.predict(X_test)
    y_test_proba_xgb = xgb_model.predict_proba(X_test)
    evaluate(y_test, y_test_pred_xgb, y_test_proba_xgb, "XGBoost")
    
    # Save models
    save_dir = Path(__file__).parent / 'saved'
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / 'preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    with open(save_dir / 'baseline.pkl', 'wb') as f:
        pickle.dump(baseline, f)
    with open(save_dir / 'xgboost.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    
    print(f"\nModels saved to {save_dir}")
    
    # Save plots
    plot_dir = Path(__file__).parent / 'plots'
    plot_dir.mkdir(exist_ok=True)
    plot_confusion_matrix(y_test, y_test_pred, "Baseline", plot_dir / 'baseline_cm.png')
    plot_confusion_matrix(y_test, y_test_pred_xgb, "XGBoost", plot_dir / 'xgboost_cm.png')
    print(f"Plots saved to {plot_dir}")


if __name__ == '__main__':
    main()
