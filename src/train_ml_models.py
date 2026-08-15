import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import xgboost as xgb
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_features(features_path, target_threshold_percentile=90):
    """
    Load features from multi-band GeoTIFF and prepare for machine learning.

    Parameters:
    features_path (str or Path): Path to the multi-band GeoTIFF features.tif
    target_threshold_percentile (int): Percentile threshold for creating binary target (LST anomaly)

    Returns:
    tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    print("Loading features from GeoTIFF...")
    with rasterio.open(features_path) as src:
        # Read all bands
        bands = []
        feature_names = []
        for i in range(1, src.count + 1):
            band = src.read(i)
            bands.append(band)
            # Get band description if available, otherwise use index
            desc = src.descriptions[i-1] if src.descriptions[i-1] else f"Band_{i}"
            feature_names.append(desc)

        # Get metadata for masking
        meta = src.meta

    # Stack bands into a 3D array (bands, height, width)
    feature_stack = np.stack(bands, axis=0)
    print(f"Loaded {len(feature_stack)} bands with shape {feature_stack.shape}")

    # Identify LST band (should be first band based on our engineering)
    lst_band = feature_stack[0]  # Band 0: LST

    # Create binary target: extreme heat event (LST > threshold percentile)
    # Only consider valid pixels (not NaN)
    valid_mask = ~np.isnan(lst_band)
    if np.sum(valid_mask) == 0:
        raise ValueError("No valid LST pixels found")

    lst_valid = lst_band[valid_mask]
    threshold = np.percentile(lst_valid, target_threshold_percentile)
    print(f"LST threshold for {target_threshold_percentile}th percentile: {threshold:.2f}°C")

    # Create target binary array (1 = extreme heat, 0 = normal)
    target = np.full_like(lst_band, np.nan)
    target[valid_mask] = (lst_band[valid_mask] > threshold).astype(int)

    # Prepare feature matrix: flatten bands and mask invalid pixels
    # We'll use the same mask for all features (where LST is valid)
    X_list = []
    for band in feature_stack:
        X_list.append(band[valid_mask])

    X = np.column_stack(X_list)  # Shape: (n_valid_pixels, n_bands)
    y = target[valid_mask]       # Shape: (n_valid_pixels,)

    print(f"Prepared {X.shape[0]} valid samples with {X.shape[1]} features")
    print(f"Target distribution: {np.bincount(y.astype(int))} (0=no extreme heat, 1=extreme heat)")

    return X, y, feature_names

def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Train and evaluate multiple models: baseline, bagging, boosting, stacking.

    Returns:
    dict: Dictionary containing model performances and trained models
    """
    results = {}
    models = {}

    # Standardize features (important for logistic regression and SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Baseline: Logistic Regression
    print("\nTraining Baseline: Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

    results['logistic_regression'] = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba_lr),
        'f1': f1_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr),
        'recall': recall_score(y_test, y_pred_lr),
        'confusion_matrix': confusion_matrix(y_test, y_pred_lr)
    }
    models['logistic_regression'] = {'model': lr, 'scaler': scaler}
    print(f"  ROC-AUC: {results['logistic_regression']['roc_auc']:.3f}")
    print(f"  F1-Score: {results['logistic_regression']['f1']:.3f}")

    # 2. Bagging: Random Forest
    print("\nTraining Bagging: Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)  # RF doesn't require scaling
    y_pred_rf = rf.predict(X_test)
    y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

    results['random_forest'] = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba_rf),
        'f1': f1_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
        'confusion_matrix': confusion_matrix(y_test, y_pred_rf),
        'feature_importances': dict(zip(feature_names, rf.feature_importances_))
    }
    models['random_forest'] = {'model': rf}
    print(f"  ROC-AUC: {results['random_forest']['roc_auc']:.3f}")
    print(f"  F1-Score: {results['random_forest']['f1']:.3f}")

    # 3. Boosting: XGBoost
    print("\nTraining Boosting: XGBoost...")
    # Handle class imbalance if present
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

    results['xgboost'] = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba_xgb),
        'f1': f1_score(y_test, y_pred_xgb),
        'precision': precision_score(y_test, y_pred_xgb),
        'recall': recall_score(y_test, y_pred_xgb),
        'confusion_matrix': confusion_matrix(y_test, y_pred_xgb)
    }
    models['xgboost'] = {'model': xgb_model}
    print(f"  ROC-AUC: {results['xgboost']['roc_auc']:.3f}")
    print(f"  F1-Score: {results['xgboost']['f1']:.3f}")

    # 4. Stacking Ensemble
    print("\nTraining Stacking Ensemble...")
    # Define base models
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=50, random_state=42,
                                 scale_pos_weight=scale_pos_weight,
                                 use_label_encoder=False,
                                 eval_metric='logloss'))
    ]
    # Define meta-model
    meta_model = LogisticRegression(random_state=42, max_iter=1000)

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )

    stacking_clf.fit(X_train, y_train)
    y_pred_stack = stacking_clf.predict(X_test)
    y_pred_proba_stack = stacking_clf.predict_proba(X_test)[:, 1]

    results['stacking'] = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba_stack),
        'f1': f1_score(y_test, y_pred_stack),
        'precision': precision_score(y_test, y_pred_stack),
        'recall': recall_score(y_test, y_pred_stack),
        'confusion_matrix': confusion_matrix(y_test, y_pred_stack)
    }
    models['stacking'] = {'model': stacking_clf}
    print(f"  ROC-AUC: {results['stacking']['roc_auc']:.3f}")
    print(f"  F1-Score: {results['stacking']['f1']:.3f}")

    return results, models

def main():
    print("=== Urban Heat Island Machine Learning Modeling ===")

    # Paths - relative to project root (parent of src directory)
    project_root = Path(__file__).parent.parent
    features_path = project_root / "results" / "features.tif"
    if not features_path.exists():
        print("ERROR: Features file not found at", features_path)
        print("Please run the feature engineering script first.")
        sys.exit(1)

    # Load and prepare data
    try:
        X, y, feature_names = load_and_prepare_features(features_path, target_threshold_percentile=90)
    except Exception as e:
        print(f"ERROR loading features: {e}")
        sys.exit(1)

    # Split into train and test sets (using random split for simplicity - in practice use spatial blocking)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Train and evaluate models
    results, models = train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names)

    # Print summary
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'ROC-AUC':<10} {'F1-Score':<10} {'Precision':<10} {'Recall':<10}")
    print("-"*60)
    for model_name in ['logistic_regression', 'random_forest', 'xgboost', 'stacking']:
        if model_name in results:
            res = results[model_name]
            print(f"{model_name:<20} {res['roc_auc']:<10.3f} {res['f1']:<10.3f} "
                  f"{res['precision']:<10.3f} {res['recall']:<10.3f}")

    # Determine best model based on ROC-AUC
    best_model = max(results.keys(), key=lambda k: results[k]['roc_auc'])
    print("-"*60)
    print(f"Best model based on ROC-AUC: {best_model}")
    print(f"Best ROC-AUC: {results[best_model]['roc_auc']:.3f}")

    # Feature importance from Random Forest (if available)
    if 'random_forest' in results and 'feature_importances' in results['random_forest']:
        print("\nFeature Importance (Random Forest):")
        importances = results['random_forest']['feature_importances']
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_features:
            print(f"  {feat}: {imp:.3f}")

    # Save results to file
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # Save model performance summary
    with open(results_dir / "model_performance.txt", "w") as f:
        f.write("Urban Heat Island ML Model Performance\n")
        f.write("="*50 + "\n\n")
        f.write(f"Target: Extreme heat event (LST > 90th percentile)\n")
        f.write(f"Total samples: {X.shape[0]}\n")
        f.write(f"Train samples: {X_train.shape[0]}\n")
        f.write(f"Test samples: {X_test.shape[0]}\n\n")

        f.write("Model Performance:\n")
        f.write("-"*30 + "\n")
        f.write(f"{'Model':<20} {'ROC-AUC':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}\n")
        f.write("-"*30 + "\n")
        for model_name in ['logistic_regression', 'random_forest', 'xgboost', 'stacking']:
            if model_name in results:
                res = results[model_name]
                f.write(f"{model_name:<20} {res['roc_auc']:<10.3f} {res['f1']:<10.3f} "
                       f"{res['precision']:<10.3f} {res['recall']:<10.3f}\n")

        f.write("\nBest Model: " + best_model + f" (ROC-AUC: {results[best_model]['roc_auc']:.3f})\n\n")

        if 'random_forest' in results and 'feature_importances' in results['random_forest']:
            f.write("Feature Importance (Random Forest):\n")
            f.write("-"*30 + "\n")
            for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{feat}: {imp:.3f}\n")

    print(f"\nModel performance saved to: {results_dir / 'model_performance.txt'}")
    print("\n=== Machine Learning Modeling Complete ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)