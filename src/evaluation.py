"""
Model Evaluation Utilities

This module contains functions for comprehensive model evaluation,
including metrics, visualizations, and analysis.

Extracted from notebooks/baseline_models.ipynb fit_predict_join() function
and evaluation sections.
"""

from src.config import TARGET_COL, TEST_DATA_PATH
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
import streamlit as st
from typing import Dict


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
        self.test = pd.read_csv(TEST_DATA_PATH)
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
    
    def evaluate_business(
            self,
            y_pred: np.ndarray,
            recovery_rate: float = 0.7,
            chargeback_rate_caught: float = 0.4,
            chargeback_rate_missed: float = 0.7,
            processing_fee: float = 10,
            chargeback_penalty: float = 50,
            manual_review_cost: float = 8,
            customer_lifetime_value: float = 100,
            abandonment_rate: float = 0.3,
            new_customer_churn_rate: float = 0.5,
            repeat_customer_churn_rate: float = 0.15,
            investigation_cost: float = 200,
            investigation_rate: float = 0.5
    ) -> Dict:
        """
        Translate model performance metrics into business value.
     
        Args:
            y_pred: Model predictions (0=legitimate, 1=fraud)
            recovery_rate: % of caught fraud that is actually prevented (default: 0.7)
            chargeback_rate_caught: % of caught fraud that would've been chargebacks (default: 0.4)
            chargeback_rate_missed: % of missed fraud that becomes chargebacks (default: 0.7)
            processing_fee: Bank fee per chargeback (default: $25)
            chargeback_penalty: Additional penalty if high chargeback rate (default: $50)
            manual_review_cost: Cost to manually review flagged transaction (default: $8)
            customer_lifetime_value: Average CLV (default: $2000)
            abandonment_rate: % customers who abandon after false decline (default: 0.3)
            new_customer_churn_rate: % new customers who never return after FP (default: 0.5)
            repeat_customer_churn_rate: % repeat customers who churn after FP (default: 0.15)
            investigation_cost: Cost to investigate fraud incident (default: $200)
            investigation_rate: % of missed fraud that gets investigated (default: 0.5)
        
        Returns:
            Dictionary containing all business metrics
        """
        test = self.test
        test["pred"] = y_pred
        avg_purchase_value = test["purchase_value"].mean()

        tp_tn = test[test[TARGET_COL] == y_pred]
        fp_fn = test[test[TARGET_COL] != y_pred]

        tp = tp_tn[tp_tn[TARGET_COL] == 1]
        tn = tp_tn[tp_tn[TARGET_COL] == 0]
        fp = fp_fn[fp_fn[TARGET_COL] == 0]
        fn = fp_fn[fp_fn[TARGET_COL] == 1]

        # TN - Legitimate Correctly Approved
        tn_count = len(tn)
        realised_revenue = tn["purchase_value"].sum()
        satisfaction_value  = tn_count * 2
        tn_efficiency_gain = tn_count * 7.50
        tn_metrics = {
            'count': int(tn_count),
            'revenue_realized': float(realised_revenue),
            'satisfaction_value': float(satisfaction_value),
            'efficiency_gain': float(tn_efficiency_gain),
            'total_benefit': float(satisfaction_value + tn_efficiency_gain)
        }

        # TP - Fraud Correctly Caught
        tp_count = len(tp)
        tp_total_value = tp["purchase_value"].sum()
        tp_average_value = tp["purchase_value"].mean() if tp_count > 0 else avg_purchase_value
        
        # 1. Revenue protected (we stopped the fraud before completion)
        tp_protected_revenue = tp_total_value * recovery_rate

        # 2. Chargeback prevention
        # Not all caught fraud would become chargebacks (some are testing, etc. )
        # Chargeback cost = Transaction value + Processing fee + Penalty (cost to cover administrative costs of the chargeback process)
        tp_avg_chargeback_cost = tp_average_value + processing_fee + chargeback_penalty
        tp_chargebacks_prevented_count = tp_count * chargeback_rate_caught
        tp_chargeback_prevention = tp_chargebacks_prevented_count * tp_avg_chargeback_cost

        # 3. Customer trust value (protecting legitimate cardholders)
        tp_customers_protected = tp_count * 0.85
        tp_trust_value = tp_customers_protected * customer_lifetime_value * 0.05
        
        tp_total_benefit = tp_protected_revenue + tp_chargeback_prevention + tp_trust_value

        tp_metrics = {
            'count': int(tp_count),
            'total_transaction_value': float(tp_total_value),
            'avg_transaction_value': float(tp_average_value),
            'revenue_protected': float(tp_protected_revenue),
            'chargebacks_prevented_count': float(tp_chargebacks_prevented_count),
            'customer_trust_value': float(tp_trust_value),
            'total_benefit': float(tp_total_benefit)
        }
        
        # FP - Legitimate Flagged as Fraud 
        fp_count = len(fp)
        fp_total_value = fp["purchase_value"].sum()
        fp_avg_value = fp["purchase_value"].mean() if fp_count > 0 else avg_purchase_value

        # 1. Manual review cost (direct operational cost)
        fp_manual_review_cost = fp_count * manual_review_cost

        # 2. Customer friction - abandoned purchases
        fp_abandoned_count = fp_count * abandonment_rate
        fp_friction_cost = fp_abandoned_count * fp_avg_value

        # 3. Customer lifetime value loss (permanent churn)
        fp_new_customers = fp["is_ip_single_device"].sum()
        fp_repeat_customers = fp_count - fp_new_customers

        fp_clv_loss = (
            fp_new_customers * new_customer_churn_rate * customer_lifetime_value +
            fp_repeat_customers * repeat_customer_churn_rate * customer_lifetime_value
        )

        # 4. Brand reputation damage
        fp_reputation_cost = fp_count * 0.08 * 100 # 8% share experience, $100 per incident
        fp_total_cost = (
            fp_manual_review_cost +
            fp_friction_cost +
            fp_clv_loss +
            fp_reputation_cost
        )

        fp_metrics = {
            'count': int(fp_count),
            'total_transaction_value': float(fp_total_value),
            'avg_transaction_value': float(fp_avg_value),
            'manual_review_cost': float(fp_manual_review_cost),
            'friction_cost': float(fp_friction_cost),
            'clv_loss': float(fp_clv_loss),
            'reputation_cost': float(fp_reputation_cost),
            'total_cost': float(fp_total_cost)
        }

        # FN - Fraud Missed
        fn_count = len(fn)
        fn_total_value = fn["purchase_value"].sum()
        fn_avg_value = fn['purchase_value'].mean() if fn_count > 0 else avg_purchase_value

        # 1. Revenue loss (merchandise/service already delivered)
        # Some fraud is recovered through insurance, law enforcement, etc.
        fn_missed_recovery_rate = 0.2  # Only 20% of missed fraud recovered
        fn_revenue_loss = fn_total_value * (1 - fn_missed_recovery_rate)

        # 2. Chargeback costs (THE BIG ONE for missed fraud)
        # Higher rate because customers dispute when they see unauthorized charges
        fn_avg_chargeback_cost = fn_avg_value + processing_fee + chargeback_penalty
        fn_chargeback_count = fn_count * chargeback_rate_missed
        fn_chargeback_cost = fn_chargeback_count * fn_avg_chargeback_cost

        # 3. Investigation costs
        fn_investigation_cost = fn_count * investigation_rate * investigation_cost

        # 4. Regulatory costs (fines for fraud patterns)
        fn_regulatory_fine_prob = 0.05 # 5% chance of fining per incident
        fn_avg_regulatory_fine = 10000
        fn_regulatory_cost = fn_count * fn_regulatory_fine_prob * fn_avg_regulatory_fine

        fn_total_cost = (
            fn_revenue_loss +
            fn_chargeback_cost +
            fn_investigation_cost +
            fn_regulatory_cost
        )

        fn_metrics = {
            'count': int(fn_count),
            'total_transaction_value': float(fn_total_value),
            'avg_transaction_value': float(fn_avg_value),
            'revenue_loss': float(fn_revenue_loss),
            'chargeback_cost': float(fn_chargeback_cost),
            'chargeback_count': float(fn_chargeback_count),
            'investigation_cost': float(fn_investigation_cost),
            'regulatory_cost': float(fn_regulatory_cost),
            'total_cost': float(fn_total_cost)
        }

        # Summary 
        total_transactions = len(test)
        fraud_rate = (test["class"] == 1).sum() / total_transactions
         # Calculate rates
        fraud_detection_rate = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        false_positive_rate = fp_count / (fp_count + tn_count) if (fp_count + tn_count) > 0 else 0
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        
        # Financial summary
        total_benefits = tp_total_benefit + tn_metrics['total_benefit']
        total_costs = fp_total_cost + fn_total_cost
        net_business_value = total_benefits - total_costs
        
        # ROI calculation (investment = operational costs)
        investment = fp_manual_review_cost
        roi_percentage = (net_business_value / investment * 100) if investment > 0 else 0
        
        summary = {
            'total_transactions': int(total_transactions),
            'fraud_rate': float(fraud_rate),
            'fraud_detection_rate': float(fraud_detection_rate),
            'false_positive_rate': float(false_positive_rate),
            'precision': float(precision),
            'total_benefits': float(total_benefits),
            'total_costs': float(total_costs),
            'net_business_value': float(net_business_value),
            'roi_percentage': float(roi_percentage)
        }

        results = {
            'true_positives': tp_metrics,
            'true_negatives': tn_metrics,
            'false_positives': fp_metrics,
            'false_negatives': fn_metrics,
            'summary': summary,
            'parameters': {
                'recovery_rate': recovery_rate,
                'chargeback_rate_caught': chargeback_rate_caught,
                'chargeback_rate_missed': chargeback_rate_missed,
                'processing_fee': processing_fee,
                'chargeback_penalty': chargeback_penalty,
                'manual_review_cost': manual_review_cost,
                'customer_lifetime_value': customer_lifetime_value
            }
        }

        return results

        
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
