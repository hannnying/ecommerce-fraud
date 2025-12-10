import argparse
import pickle
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

import pandas as pd
from src.models import FraudDetectionModel
from src.data_prep_v2 import FraudFeatureEngineer
from src.preprocessing import FraudDataPreprocessor
from src.evaluation import ModelEvaluator, compare_models, evaluate_overfitting
from src.config import (
    RAW_DATA_PATH,
    MODELS_DIR,
    TARGET_COL,
    TRAIN_DATA_PATH,
    MODEL_CONFIGS,
    FIGURES_DIR
) 
from sklearn.model_selection import train_test_split

def train_single_model(
        columns=None,
        resampling_type='random_undersampling',
        model_type='logistic_regression',
        custom_params=None,
        save_model=False
):
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

    # Load raw data
    df = pd.read_csv(RAW_DATA_PATH)

    # split into train and test and save 
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET_COL])
    train_df.to_csv("data/raw_train.csv")
    test_df.to_csv("data/raw_test.csv")

    # Feature engineering
    feature_engineer = FraudFeatureEngineer(
        ip_country_path="data/IpAddress_to_Country.csv",
        gdp_data_path="data/gdp_usd.xlsx"
    )
    train_processed = feature_engineer.fit_transform(train_df, TARGET_COL)

    with open("models/feature_engineer.pkl", 'wb') as f:
        pickle.dump(feature_engineer, f)

    # Further preprocessing of data
    preprocessor = FraudDataPreprocessor()

    X_train, y_train = preprocessor.preprocess(train_processed)
    
    X_train_further_processed = preprocessor.fit_transform(X_train)

    with open("models/preprocessor.pkl", 'wb') as f:
        pickle.dump(preprocessor, f)
    
    X_train_further_processed.to_csv(TRAIN_DATA_PATH)

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
    X_train_further_processed = pd.read_csv(TRAIN_DATA_PATH, index_col=[0])
    model.fit(X_train_further_processed, y_train)
    print("Training complete!")

    # Save model
    if save_model:
        model_path = MODELS_DIR / f"{model_type}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            print(f"\nModel saved to {model_path}")

    return model

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

        # Train single model
        train_single_model(
            columns=columns,
            resampling_type=args.resampling,
            model_type=args.model,
            custom_params=custom_params,
            save_model=args.save
        )

        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
