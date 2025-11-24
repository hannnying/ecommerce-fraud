"""
Training Script for Fraud Detection Models

This script provides a command-line interface to train and evaluate
fraud detection models with support for custom parameters, resampling,
and feature selection.

Usage:
------
# Train best model (Logistic Regression)
python train.py

# Train specific model with resampling
python train.py --model random_forest --resampling smote

# Train with specific columns
python train.py --columns "age,purchase_value,source,browser"

# Train with custom parameters
python train.py --model logistic_regression --params '{"C": 0.01, "max_iter": 500}'

# Train and save model
python train.py --model logistic_regression --save

# Compare multiple models
python train.py --compare
"""

import argparse
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models import FraudDetectionModel
from src.preprocessing import load_and_preprocess_data
from src.evaluation import ModelEvaluator, compare_models, evaluate_overfitting
from src.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    MODELS_DIR,
    TARGET_COL,
    MODEL_CONFIGS,
    FIGURES_DIR
)


def train_single_model(
        columns=None,
        resampling_type='random_undersampling',
        model_type='logistic_regression',
        custom_params=None,
        save_model=False,
        show_plots=False):
    """
    Train a single fraud detection model.

    Args:
        columns (list): List of strings specifying columns to use for modelling.
        resampling_type (str): Type of model used for resampling: 'randon_undersamping', 'enn', 'smote', 'smoteenn'.
        model_type (str): Type of model: 'logistic_regression', 'random_forest', 'xgboost'
        custom_params (dict): Custom hyperparameters to override defaults
        save_model (bool): Whether to save the trained model
        show_plots (bool): Whether to display evaluation plots

    Returns: (model, metrics, evaluator)
    """
    print(f"\n{'='*70}")
    print(f"Training {MODEL_CONFIGS[model_type]['display_name']}")
    print(f"Description: {MODEL_CONFIGS[model_type]['description']}")
    print(f"{'='*70}\n")

    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(
        str(TRAIN_DATA_PATH),
        str(TEST_DATA_PATH),
        target_col=TARGET_COL
    )

    # Initialize model
    print(f"\nInitializing {model_type} model...")
    print(f"Resampling technique: {resampling_type}")
    if columns:
        print(f"Using selected columns: {columns}")
    if custom_params:
        print(f"Custom parameters: {custom_params}")

    model = FraudDetectionModel(
        columns=columns,
        resampling_type=resampling_type,
        model_type=model_type,
        custom_params=custom_params
    )

    # Train model
    print("Training model...")
    model.fit(X_train, y_train)
    print("Training complete!")

    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate model
    evaluator = ModelEvaluator(model_name=MODEL_CONFIGS[model_type]['display_name'])
    metrics = evaluator.evaluate(y_test, y_pred, y_proba, print_report=True)
    evaluator.evaluate_business(y_pred)

    # Check for overfitting
    print(f"\n{'='*70}")
    print("Overfitting Analysis (Train vs Test Performance)")
    print(f"{'='*70}")
    overfitting_results = evaluate_overfitting(
        model.model, X_train, y_train, X_test, y_test
    )
    print(overfitting_results.to_string(index=False))

    # Feature importance
    print(f"\n{'='*70}")
    print("Top 10 Most Important Features")
    print(f"{'='*70}")
    print(model.get_feature_importance(top_n=10).to_string(index=False))

    if show_plots:
        # Plot evaluation metrics
        print("\nGenerating evaluation plots...")

        # Confusion Matrix
        save_path = FIGURES_DIR / f"{model_type}_confusion_matrix.png"
        evaluator.plot_confusion_matrix(y_test, y_pred, save_path=str(save_path))

        # ROC Curve
        save_path = FIGURES_DIR / f"{model_type}_roc_curve.png"
        evaluator.plot_roc_curve(y_test, y_proba, save_path=str(save_path))

        # PR Curve
        save_path = FIGURES_DIR / f"{model_type}_pr_curve.png"
        evaluator.plot_precision_recall_curve(y_test, y_proba, save_path=str(save_path))

        # Feature Importance
        save_path = FIGURES_DIR / f"{model_type}_feature_importance.png"
        evaluator.plot_feature_importance(
            model.feature_importance_,
            top_n=20,
            save_path=str(save_path)
        )

    # Save model
    if save_model:
        model_path = MODELS_DIR / f"{model_type}_model.pkl"
        model.save(str(model_path))
        print(f"\nModel saved to {model_path}")

    return model, metrics, evaluator


def compare_all_models(resampling_type='random_undersampling'):
    """
    Train and compare all available models.

    Args:
        resampling_type (str): Type of resampling to use for all models

    Returns
    -------
    pd.DataFrame
        Comparison table of all models
    """
    print(f"\n{'='*70}")
    print("Training and Comparing All Models")
    print(f"Resampling technique: {resampling_type}")
    print(f"{'='*70}\n")

    results = {}

    for model_type in MODEL_CONFIGS.keys():
        try:
            print(f"\n--- Training {model_type} ---")
            _, metrics, _ = train_single_model(
                model_type=model_type,
                resampling_type=resampling_type,
                save_model=False,
                show_plots=False
            )
            results[MODEL_CONFIGS[model_type]['display_name']] = metrics
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            continue

    # Create comparison table
    print(f"\n{'='*70}")
    print("Model Comparison Results")
    print(f"{'='*70}\n")

    comparison_df = compare_models(results)
    print(comparison_df.to_string(index=False))

    # Save comparison
    comparison_path = FIGURES_DIR.parent / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison saved to {comparison_path}")

    return comparison_df


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train fraud detection models with custom parameters and resampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                                    # Train best model (Logistic Regression)
  python train.py --model random_forest              # Train Random Forest
  python train.py --resampling smote                 # Train with SMOTE resampling
  python train.py --columns "age,purchase_value"     # Train with selected columns
  python train.py --params '{"C": 0.01}'             # Train with custom parameters
  python train.py --model logistic_regression --save # Train and save model
  python train.py --compare --resampling enn         # Compare all models with ENN
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default='logistic_regression',
        choices=list(MODEL_CONFIGS.keys()),
        help='Model type to train'
    )

    parser.add_argument(
        '--resampling',
        type=str,
        default='random_undersampling',
        choices=['random_undersampling', 'enn', 'smote', 'smoteenn'],
        help='Resampling technique to handle class imbalance (default: random_undersampling)'
    )

    parser.add_argument(
        '--columns',
        type=str,
        default=None,
        help='Comma-separated list of column names to use for training (default: all columns)'
    )

    parser.add_argument(
        '--params',
        type=str,
        default=None,
        help='Custom model parameters as JSON string, e.g., \'{"C": 0.01, "max_iter": 500}\''
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help='Save trained model to disk'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='Train and compare all models'
    )

    args = parser.parse_args()

    # Create necessary directories
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    # Process columns argument
    columns = None
    if args.columns:
        columns = [col.strip() for col in args.columns.split(',')]
        print(f"Selected columns: {columns}")

    # Process custom parameters
    custom_params = None
    if args.params:
        try:
            custom_params = json.loads(args.params)
            print(f"Custom parameters: {custom_params}")
        except json.JSONDecodeError as e:
            print(f"Error parsing custom parameters: {e}")
            print("Parameters should be valid JSON, e.g., '{\"C\": 0.01, \"max_iter\": 500}'")
            sys.exit(1)

    try:
        if args.compare:
            # Compare all models
            compare_all_models(resampling_type=args.resampling)
        else:
            # Train single model
            train_single_model(
                columns=columns,
                resampling_type=args.resampling,
                model_type=args.model,
                custom_params=custom_params,
                save_model=args.save,
                show_plots=not args.no_plots
            )

        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
