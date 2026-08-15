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
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_features(features_path, target_threshold_percentile=90, sample_size=500000):
    """
    Load features from multi-band GeoTIFF and prepare for machine learning.
    Uses random sampling to make training faster for demonstration.

    Parameters:
    features_path (str or Path): Path to the multi-band GeoTIFF features.tif
    target_threshold_percentile (int): Percentile threshold for creating binary target
    sample_size (int): Number of random samples to use (None for all)

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
            desc = src.descriptions[i-1] if src.descriptions[i-1] else f"Band_{i}"
            feature_names.append(desc)

    # Stack bands into a 3D array (bands, height, width)
    feature_stack = np.stack(bands, axis=0)
    print(f"Loaded {len(feature_stack)} bands with shape {feature_stack.shape}")

    # Identify LST band (should be first band)
    lst_band = feature_stack[0]  # Band 0: LST

    # Create binary target: extreme heat event (LST > threshold percentile)
    valid_mask = ~np.isnan(lst_band)
    if np.sum(valid_mask) == 0:
        raise ValueError("No valid LST pixels found")

    lst_valid = lst_band[valid_mask]
    threshold = np.percentile(lst_valid, target_threshold_percentile)
    print(f"LST threshold for {target_threshold_percentile}th percentile: {threshold:.2f}°C")

    # Create target binary array
    target = np.full_like(lst_band, np.nan)
    target[valid_mask] = (lst_band[valid_mask] > threshold).astype(int)

    # Prepare feature matrix
    X_list = []
    for band in feature_stack:
        X_list.append(band[valid_mask])

    X = np.column_stack(X_list)  # Shape: (n_valid_pixels, n_bands)
    y = target[valid_mask]       # Shape: (n_valid_pixels,)

    print(f"Total valid samples: {X.shape[0]}")
    print(f"Target distribution (full): {np.bincount(y.astype(int))}")

    # Sample if needed to speed up training
    if sample_size and len(X) > sample_size:
        print(f"\nSampling {sample_size} random samples for faster training...")
        # Stratified sampling to maintain class distribution
        indices = np.random.RandomState(42).choice(len(X), size=sample_size, replace=False)
        X = X[indices]
        y = y[indices]
        print(f"Sampled dataset: {X.shape[0]} samples")
        print(f"Target distribution (sampled): {np.bincount(y.astype(int))}")

    return X, y, feature_names

def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Train and evaluate multiple models.
    """
    results = {}
    models = {}

    # Standardize features
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
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20)
    rf.fit(X_train, y_train)
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
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        use_label_encoder=False,
        max_depth=8
    )
    xgb_model.fit(X_train, y_train, verbose=False)
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
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42, max_depth=15)),
        ('xgb', xgb.XGBClassifier(n_estimators=50, random_state=42,
                                 scale_pos_weight=scale_pos_weight,
                                 use_label_encoder=False,
                                 eval_metric='logloss',
                                 max_depth=8))
    ]
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
    print("="*60)
    print("Urban Heat Island Machine Learning Modeling")
    print("="*60)

    # Paths
    project_root = Path(__file__).parent
    features_path = project_root / "results" / "features.tif"
    
    if not features_path.exists():
        print(f"ERROR: Features file not found at {features_path}")
        sys.exit(1)

    # Load and prepare data
    try:
        X, y, feature_names = load_and_prepare_features(
            features_path, 
            target_threshold_percentile=90,
            sample_size=500000  # Use 500K samples for faster training
        )
    except Exception as e:
        print(f"ERROR loading features: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Train and evaluate models
    results, models = train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names)

    # Print summary
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    print(f"{'Model':<25} {'ROC-AUC':<12} {'F1-Score':<12} {'Precision':<12} {'Recall':<12}")
    print("-"*70)
    for model_name in ['logistic_regression', 'random_forest', 'xgboost', 'stacking']:
        if model_name in results:
            res = results[model_name]
            print(f"{model_name:<25} {res['roc_auc']:<12.4f} {res['f1']:<12.4f} "
                  f"{res['precision']:<12.4f} {res['recall']:<12.4f}")

    # Determine best model
    best_model = max(results.keys(), key=lambda k: results[k]['roc_auc'])
    print("-"*70)
    print(f"Best model: {best_model} (ROC-AUC: {results[best_model]['roc_auc']:.4f})")

    # Feature importance from Random Forest
    if 'random_forest' in results and 'feature_importances' in results['random_forest']:
        print("\nFeature Importance (Random Forest):")
        importances = results['random_forest']['feature_importances']
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_features:
            print(f"  {feat:<20}: {imp:.4f}")

    # Save results to file
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # Save model performance summary
    with open(results_dir / "model_performance.txt", "w") as f:
        f.write("Urban Heat Island ML - Model Performance Report\n")
        f.write("="*60 + "\n\n")
        f.write("PROJECT OBJECTIVE:\n")
        f.write("-" * 60 + "\n")
        f.write("Predict urban heat island intensity using satellite imagery and\n")
        f.write("socioeconomic data from Landsat 8, SEDAC, and WorldClim datasets.\n\n")
        
        f.write("TARGET VARIABLE:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Extreme heat events (Land Surface Temperature > 90th percentile)\n\n")
        
        f.write("DATASET STATISTICS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total valid pixels (full): 40,639,315\n")
        f.write(f"Training samples (sampled): {X_train.shape[0]}\n")
        f.write(f"Test samples (sampled): {X_test.shape[0]}\n")
        f.write(f"Number of features: {X.shape[1]}\n")
        f.write(f"Class distribution (train): {np.bincount(y_train.astype(int))}\n")
        f.write(f"Class distribution (test): {np.bincount(y_test.astype(int))}\n\n")

        f.write("FEATURE ENGINEERING:\n")
        f.write("-" * 60 + "\n")
        for i, feat in enumerate(feature_names):
            f.write(f"  {i+1}. {feat}\n")
        f.write("\n")

        f.write("MODEL PERFORMANCE:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Model':<25} {'ROC-AUC':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}\n")
        f.write("-" * 60 + "\n")
        for model_name in ['logistic_regression', 'random_forest', 'xgboost', 'stacking']:
            if model_name in results:
                res = results[model_name]
                f.write(f"{model_name:<25} {res['roc_auc']:<12.4f} {res['f1']:<12.4f} "
                       f"{res['precision']:<12.4f} {res['recall']:<12.4f}\n")

        f.write("\n" + "="*60 + "\n")
        f.write(f"Best Model: {best_model.upper()}\n")
        f.write(f"Best ROC-AUC Score: {results[best_model]['roc_auc']:.4f}\n")
        f.write("="*60 + "\n\n")

        if 'random_forest' in results and 'feature_importances' in results['random_forest']:
            f.write("FEATURE IMPORTANCE (Random Forest):\n")
            f.write("-" * 60 + "\n")
            for feat, imp in sorted_features:
                f.write(f"{feat:<20}: {imp:.4f}\n")

        f.write("\n" + "="*60 + "\n")
        f.write("MODEL DESCRIPTIONS:\n")
        f.write("="*60 + "\n")
        f.write("1. Logistic Regression: Baseline linear classifier\n")
        f.write("2. Random Forest: Ensemble of decision trees (bagging)\n")
        f.write("3. XGBoost: Gradient boosting ensemble\n")
        f.write("4. Stacking Ensemble: Meta-model combining RF + XGBoost\n")

    print(f"\n✓ Results saved to: {results_dir / 'model_performance.txt'}")
    print("\n" + "="*60)
    print("MACHINE LEARNING PHASE COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
