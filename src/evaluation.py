"""
Model Evaluation Utilities

This module contains functions for comprehensive model evaluation,
including metrics, visualizations, and analysis.

Extracted from notebooks/baseline_models.ipynb fit_predict_join() function
and evaluation sections.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)


class ModelEvaluator:
    """
    Comprehensive model evaluation class.

    Provides metrics, visualizations, and analysis for fraud detection models.
    """

    def __init__(self, model_name="Model"):
        """
        Initialize evaluator.

        Parameters
        ----------
        model_name : str
            Name of the model being evaluated
        """
        self.model_name = model_name
        self.metrics = {}

    def evaluate(self, y_true, y_pred, y_proba=None, print_report=True):
        """
        Compute all evaluation metrics.

        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_proba : array-like, optional
            Predicted probabilities for positive class
        print_report : bool
            Whether to print the classification report

        Returns
        -------
        dict
            Dictionary of metrics
        """
        # Basic metrics
        self.metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }

        # Probability-based metrics
        if y_proba is not None:
            self.metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            self.metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        else:
            # Use predictions as proxy
            self.metrics['pr_auc'] = average_precision_score(y_true, y_pred)

        if print_report:
            print(f"\n{'='*60}")
            print(f"{self.model_name} Evaluation Results")
            print(f"{'='*60}")
            print(f"Accuracy:  {self.metrics['accuracy']:.4f}")
            print(f"Precision: {self.metrics['precision']:.4f}")
            print(f"Recall:    {self.metrics['recall']:.4f}")
            print(f"F1 Score:  {self.metrics['f1']:.4f}")
            if y_proba is not None:
                print(f"ROC-AUC:   {self.metrics['roc_auc']:.4f}")
            print(f"PR-AUC:    {self.metrics['pr_auc']:.4f}")
            print(f"\n{classification_report(y_true, y_pred, digits=4)}")

        return self.metrics

    def plot_confusion_matrix(self, y_true, y_pred, figsize=(8, 6), save_path=None):
        """
        Plot confusion matrix.

        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'{self.model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        plt.show()

    def plot_roc_curve(self, y_true, y_proba, figsize=(8, 6), save_path=None):
        """
        Plot ROC curve.

        Parameters
        ----------
        y_true : array-like
            True labels
        y_proba : array-like
            Predicted probabilities
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)

        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.model_name} - ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")

        plt.show()

    def plot_precision_recall_curve(self, y_true, y_proba, figsize=(8, 6), save_path=None):
        """
        Plot Precision-Recall curve.

        Parameters
        ----------
        y_true : array-like
            True labels
        y_proba : array-like
            Predicted probabilities
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)

        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.axhline(y=y_true.mean(), color='red', linestyle='--',
                   label=f'Baseline (fraud rate = {y_true.mean():.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{self.model_name} - Precision-Recall Curve')
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR curve saved to {save_path}")

        plt.show()

    def plot_feature_importance(self, feature_importance_df, top_n=20,
                                figsize=(10, 8), save_path=None):
        """
        Plot feature importance.

        Parameters
        ----------
        feature_importance_df : pd.DataFrame
            DataFrame with 'feature' and 'importance' or 'coefficient' columns
        top_n : int
            Number of top features to plot
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        # Get top features
        top_features = feature_importance_df.head(top_n).copy()

        # Determine value column
        value_col = 'importance' if 'importance' in top_features.columns else 'coefficient'

        # Sort by absolute value for better visualization
        top_features['abs_value'] = top_features[value_col].abs()
        top_features = top_features.sort_values('abs_value')

        plt.figure(figsize=figsize)
        colors = ['green' if x > 0 else 'red' for x in top_features[value_col]]
        plt.barh(range(len(top_features)), top_features[value_col], color=colors, alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel(value_col.capitalize())
        plt.title(f'{self.model_name} - Top {top_n} Features')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")

        plt.show()


def compare_models(results_dict):
    """
    Compare multiple models side by side.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping model names to their metrics dictionaries

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    comparison = []

    for model_name, metrics in results_dict.items():
        row = {'Model': model_name}
        row.update(metrics)
        comparison.append(row)

    comparison_df = pd.DataFrame(comparison)

    # Reorder columns
    col_order = ['Model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    col_order = [col for col in col_order if col in comparison_df.columns]
    comparison_df = comparison_df[col_order]

    return comparison_df


def evaluate_overfitting(model, X_train, y_train, X_test, y_test):
    """
    Check for overfitting by comparing train and test performance.

    Parameters
    ----------
    model : trained model
        Model with predict and predict_proba methods
    X_train, y_train : array-like
        Training data
    X_test, y_test : array-like
        Test data

    Returns
    -------
    pd.DataFrame
        Train vs test metrics
    """
    # Train predictions
    y_train_pred = model.predict(X_train)
    train_metrics = {
        'Split': 'Train',
        'Accuracy': accuracy_score(y_train, y_train_pred),
        'Precision': precision_score(y_train, y_train_pred, zero_division=0),
        'Recall': recall_score(y_train, y_train_pred, zero_division=0),
        'F1': f1_score(y_train, y_train_pred, zero_division=0)
    }

    if hasattr(model, 'predict_proba'):
        y_train_proba = model.predict_proba(X_train)[:, 1]
        train_metrics['ROC-AUC'] = roc_auc_score(y_train, y_train_proba)

    # Test predictions
    y_test_pred = model.predict(X_test)
    test_metrics = {
        'Split': 'Test',
        'Accuracy': accuracy_score(y_test, y_test_pred),
        'Precision': precision_score(y_test, y_test_pred, zero_division=0),
        'Recall': recall_score(y_test, y_test_pred, zero_division=0),
        'F1': f1_score(y_test, y_test_pred, zero_division=0)
    }

    if hasattr(model, 'predict_proba'):
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_metrics['ROC-AUC'] = roc_auc_score(y_test, y_test_proba)

    # Create comparison dataframe
    overfitting_df = pd.DataFrame([train_metrics, test_metrics])

    # Calculate difference
    diff_row = {'Split': 'Difference (Train - Test)'}
    for col in overfitting_df.columns:
        if col != 'Split':
            diff_row[col] = train_metrics[col] - test_metrics[col]
    overfitting_df = pd.concat([overfitting_df, pd.DataFrame([diff_row])], ignore_index=True)

    return overfitting_df


if __name__ == "__main__":
    print("Model Evaluation Utilities")
    print("\nExample usage:")
    print("evaluator = ModelEvaluator(model_name='Logistic Regression')")
    print("metrics = evaluator.evaluate(y_test, y_pred, y_proba)")
    print("evaluator.plot_confusion_matrix(y_test, y_pred)")
    print("evaluator.plot_roc_curve(y_test, y_proba)")
