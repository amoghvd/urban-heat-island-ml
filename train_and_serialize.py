#!/usr/bin/env python
"""
Urban Heat Island ML - Model Training & Serialization for Deployment
Trains models and saves them for production inference
"""
import rasterio
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
import xgboost as xgb
import joblib
import pickle
import sys

print("="*70)
print("URBAN HEAT ISLAND ML - MODEL TRAINING & DEPLOYMENT")
print("="*70)

project_root = Path(__file__).parent
features_path = project_root / "results" / "features.tif"
models_dir = project_root / "models"
models_dir.mkdir(exist_ok=True)

print("\n[1/6] Loading features...")
with rasterio.open(features_path) as src:
    bands = [src.read(i) for i in range(1, min(7, src.count + 1))]
    feature_names = ["LST", "NDVI", "NDBI", "Brightness", "PopDensity", "Bio1"]

feature_stack = np.stack(bands)
print(f"  Loaded {len(bands)} bands: {feature_stack.shape}")

print("\n[2/6] Preparing data...")
lst_band = feature_stack[0]
valid_mask = ~np.isnan(lst_band)
lst_valid = lst_band[valid_mask]

threshold = np.percentile(lst_valid, 90)
target = (lst_band[valid_mask] > threshold).astype(int)

X = np.column_stack([band[valid_mask] for band in feature_stack])
y = target

print(f"  Total samples: {len(X):,}")
print(f"  Class distribution: {np.bincount(y.astype(int))}")

# Use subset for faster training
sample_size = min(30000, len(X))
indices = np.random.RandomState(42).choice(len(X), size=sample_size, replace=False)
X = X[indices]
y = y[indices]

print(f"  Using {len(X):,} samples")

print("\n[3/6] Train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n[4/6] Training models...")

# Prepare scalers
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Random Forest
print("  - Random Forest...", end="", flush=True)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
rf_model.fit(X_train, y_train)
rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
print(f" AUC={rf_auc:.4f}")

# 2. Logistic Regression
print("  - Logistic Regression...", end="", flush=True)
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test_scaled)[:, 1])
print(f" AUC={lr_auc:.4f}")

# 3. XGBoost
print("  - XGBoost...", end="", flush=True)
scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
xgb_model = xgb.XGBClassifier(n_estimators=50, random_state=42, 
                              scale_pos_weight=scale_pos, max_depth=6, verbose=0)
xgb_model.fit(X_train, y_train)
xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
print(f" AUC={xgb_auc:.4f}")

# 4. Voting Ensemble
print("  - Voting Ensemble...", end="", flush=True)
voting_model = VotingClassifier(
    estimators=[('rf', rf_model), ('lr', lr_model), ('xgb', xgb_model)],
    voting='soft'
)
voting_model.fit(X_train, y_train)
voting_auc = roc_auc_score(y_test, voting_model.predict_proba(X_test)[:, 1])
print(f" AUC={voting_auc:.4f}")

print("\n[5/6] Saving models...")

# Save models with metadata
models_metadata = {
    'feature_names': feature_names,
    'threshold': threshold,
    'scaler': scaler,
    'feature_importance': dict(zip(feature_names, rf_model.feature_importances_)),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'performance': {
        'random_forest': {'auc': rf_auc},
        'logistic_regression': {'auc': lr_auc},
        'xgboost': {'auc': xgb_auc},
        'voting_ensemble': {'auc': voting_auc}
    }
}

# Save individual models
joblib.dump(rf_model, models_dir / "random_forest.pkl")
joblib.dump(lr_model, models_dir / "logistic_regression.pkl")
joblib.dump(xgb_model, models_dir / "xgboost.pkl")
joblib.dump(voting_model, models_dir / "voting_ensemble.pkl")
joblib.dump(scaler, models_dir / "scaler.pkl")

# Save metadata
with open(models_dir / "metadata.pkl", 'wb') as f:
    pickle.dump(models_metadata, f)

print(f"  ✓ Saved 5 models to {models_dir}/")
print(f"  ✓ Saved metadata and scaler")

print("\n[6/6] Model summary:")
print(f"  Best Model: Random Forest (AUC={rf_auc:.4f})")
print(f"  Ensemble Model: Voting (AUC={voting_auc:.4f})")
print(f"  Feature Importance:")
for feat, imp in sorted(models_metadata['feature_importance'].items(), 
                       key=lambda x: x[1], reverse=True)[:3]:
    print(f"    {feat}: {imp:.4f}")

print("\n" + "="*70)
print("✓ MODELS TRAINED AND SERIALIZED - READY FOR DEPLOYMENT")
print("="*70)
print(f"\nDeployment files created in: {models_dir}/")
print("\nNext: Run 'python deploy_predictions.py' to generate predictions")
