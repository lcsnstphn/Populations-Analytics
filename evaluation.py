import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

def evaluate_all_models(models, X_test, y_test):
    metrics_list = []
    roc_curves = {}
    conf_matrices = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        metrics_list.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'AUC': auc
        })
        
        # ROC Curve computations
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves[name] = (fpr, tpr, auc)
        
        # Confusion Matrix
        conf_matrices[name] = confusion_matrix(y_test, y_pred)
        
    return pd.DataFrame(metrics_list), roc_curves, conf_matrices

def plot_roc_curves(roc_curves):
    fig = go.Figure()
    for name, (fpr, tpr, auc) in roc_curves.items():
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC = {auc:.3f})'))
    
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='grey'), showlegend=False))
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curves',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_white'
    )
    return fig

def plot_confusion_matrix(cm, model_name):
    fig = px.imshow(
        cm, text_auto=True,
        labels=dict(x="Predicted Label", y="True Label"),
        x=['Class 0', 'Class 1'],
        y=['Class 0', 'Class 1'],
        title=f'{model_name} Confusion Matrix',
        color_continuous_scale='Blues'
    )
    return fig
