import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, auc, RocCurveDisplay)
from xgboost import plot_importance

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else None
    }
    
    # ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'AUC = {metrics["ROC-AUC"]:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
    
    return metrics

def generate_roc_comparison(model_results, X_test, y_test, output_path):
    """Generate ROC curve comparison for all models"""
    plt.figure(figsize=(10, 8))
    
    for model_name, model in model_results.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:  # Handle SVM
            decision = model.decision_function(X_test)
            y_proba = (decision - decision.min()) / (decision.max() - decision.min())
            
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2.5, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate', fontsize=12, weight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, weight='bold')
    plt.title('ROC Curve Comparison', fontsize=14, weight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_anova_feature_importance(selector, feature_names, output_path):
    """Plot ANOVA F-score feature importance"""
    plt.figure(figsize=(12, 8))

    if hasattr(selector, 'scores_'):
        scores = selector.scores_
    elif isinstance(selector, np.ndarray):
        scores = selector
    else:
        raise ValueError("Unsupported selector type")
    
    sorted_idx = scores.argsort()[::-1]
    sorted_scores = scores[sorted_idx][:20]
    sorted_features = [feature_names[i] for i in sorted_idx][:20]
    
    plt.barh(sorted_features, sorted_scores, color='#3498db', height=0.7)
    plt.xlabel('F-Score', fontsize=12, weight='bold')
    plt.ylabel('Features', fontsize=12, weight='bold')
    plt.title('Top 20 Features by ANOVA F-Score', fontsize=14, weight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_xgb_feature_importance(model, feature_names, output_path):
    """Plot XGBoost feature importance"""
    plt.figure(figsize=(12, 8))
    try:
        importance = model.feature_importances_
    except AttributeError:
        # Try to get importance from booster
        importance = model.get_booster().get_score(importance_type='weight')
        # Convert to array if needed
        if isinstance(importance, dict):
            importance = np.array([importance.get(f, 0) for f in feature_names])
    sorted_idx = importance.argsort()[::-1]
    sorted_imp = importance[sorted_idx][:15]
    sorted_features = [feature_names[i] for i in sorted_idx][:15]
    
    plt.barh(sorted_features, sorted_imp, color='#e74c3c', height=0.7)
    plt.xlabel('Importance Score', fontsize=12, weight='bold')
    plt.ylabel('Features', fontsize=12, weight='bold')
    plt.title('XGBoost Feature Importance', fontsize=14, weight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()