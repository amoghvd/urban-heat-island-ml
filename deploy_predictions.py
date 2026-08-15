#!/usr/bin/env python
"""
Urban Heat Island ML - Inference/Prediction Deployment Script
Loads trained models and generates predictions on full dataset
"""
import rasterio
from rasterio.transform import Affine
import numpy as np
from pathlib import Path
import joblib
import pickle
import sys

print("="*70)
print("URBAN HEAT ISLAND ML - PREDICTION & DEPLOYMENT")
print("="*70)

project_root = Path(__file__).parent
features_path = project_root / "results" / "features.tif"
models_dir = project_root / "models"
output_dir = project_root / "predictions"
output_dir.mkdir(exist_ok=True)

print("\n[1/5] Loading trained models...")
try:
    rf_model = joblib.load(models_dir / "random_forest.pkl")
    voting_model = joblib.load(models_dir / "voting_ensemble.pkl")
    scaler = joblib.load(models_dir / "scaler.pkl")
    
    with open(models_dir / "metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"  ✓ Loaded Random Forest (best model)")
    print(f"  ✓ Loaded Voting Ensemble")
    print(f"  ✓ Loaded metadata")
except FileNotFoundError:
    print("  ERROR: Models not found. Run 'python train_and_serialize.py' first.")
    sys.exit(1)

print("\n[2/5] Loading feature data...")
with rasterio.open(features_path) as src:
    # Read all bands
    bands = [src.read(i) for i in range(1, src.count + 1)]
    feature_stack = np.stack(bands)
    
    # Store metadata for output
    profile = src.profile
    transform = src.transform
    crs = src.crs
    
    print(f"  Loaded {len(bands)} feature bands")
    print(f"  Shape: {feature_stack.shape}")
    print(f"  Projection: {crs}")

print("\n[3/5] Generating predictions...")

# Prepare data
bands_shape = feature_stack.shape[1:]  # (height, width)
feature_stack_flat = feature_stack.reshape(len(bands), -1).T  # (n_pixels, n_features)

# Create output arrays
predictions_rf = np.full(feature_stack.shape[1:], np.nan)
predictions_voting = np.full(feature_stack.shape[1:], np.nan)
probabilities_rf = np.full(feature_stack.shape[1:], np.nan)
probabilities_voting = np.full(feature_stack.shape[1:], np.nan)

# Find valid pixels (LST band is first)
lst_band = feature_stack[0]
valid_mask = ~np.isnan(lst_band)

print(f"  Valid pixels: {np.sum(valid_mask):,}")

# Make predictions on valid pixels
valid_features = feature_stack_flat[valid_mask.flatten()]

print("  Running Random Forest predictions...", end="", flush=True)
predictions_rf_valid = rf_model.predict(valid_features)
probabilities_rf_valid = rf_model.predict_proba(valid_features)[:, 1]
print(" ✓")

print("  Running Voting Ensemble predictions...", end="", flush=True)
predictions_voting_valid = voting_model.predict(valid_features)
probabilities_voting_valid = voting_model.predict_proba(valid_features)[:, 1]
print(" ✓")

# Reshape back to 2D
predictions_rf[valid_mask] = predictions_rf_valid
probabilities_rf[valid_mask] = probabilities_rf_valid
predictions_voting[valid_mask] = predictions_voting_valid
probabilities_voting[valid_mask] = probabilities_voting_valid

print(f"  Predictions generated for {np.sum(valid_mask):,} pixels")

print("\n[4/5] Saving prediction maps...")

# Update profile for new outputs
output_profile = profile.copy()
output_profile.update(count=1, dtype=rasterio.float32)

# Save Random Forest predictions
with rasterio.open(
    output_dir / "predictions_rf.tif", 'w',
    **output_profile
) as dst:
    dst.write(probabilities_rf.astype(rasterio.float32), 1)
    dst.update_tags(1, description="RF Probability (Extreme Heat)")

# Save Voting Ensemble predictions
with rasterio.open(
    output_dir / "predictions_voting.tif", 'w',
    **output_profile
) as dst:
    dst.write(probabilities_voting.astype(rasterio.float32), 1)
    dst.update_tags(1, description="Voting Probability (Extreme Heat)")

# Save binary predictions (RF)
with rasterio.open(
    output_dir / "predictions_binary_rf.tif", 'w',
    **output_profile.copy().update(dtype=rasterio.uint8)
) as dst:
    dst.write(predictions_rf.astype(rasterio.uint8), 1)
    dst.update_tags(1, description="RF Binary Prediction (1=Extreme, 0=Normal)")

print(f"  ✓ predictions_rf.tif (probability map)")
print(f"  ✓ predictions_voting.tif (probability map)")
print(f"  ✓ predictions_binary_rf.tif (binary predictions)")

print("\n[5/5] Generating summary statistics...")

# Calculate statistics
extreme_heat_count_rf = np.sum(predictions_rf[valid_mask] == 1)
extreme_heat_pct_rf = 100 * extreme_heat_count_rf / np.sum(valid_mask)

extreme_heat_count_voting = np.sum(predictions_voting[valid_mask] == 1)
extreme_heat_pct_voting = 100 * extreme_heat_count_voting / np.sum(valid_mask)

# Mean confidence
mean_conf_rf = np.mean(probabilities_rf[valid_mask])
mean_conf_voting = np.mean(probabilities_voting[valid_mask])

print(f"  Random Forest Results:")
print(f"    - Extreme heat pixels: {extreme_heat_count_rf:,} ({extreme_heat_pct_rf:.2f}%)")
print(f"    - Mean confidence: {mean_conf_rf:.4f}")

print(f"  Voting Ensemble Results:")
print(f"    - Extreme heat pixels: {extreme_heat_count_voting:,} ({extreme_heat_pct_voting:.2f}%)")
print(f"    - Mean confidence: {mean_conf_voting:.4f}")

# Save summary report
with open(output_dir / "deployment_summary.txt", 'w') as f:
    f.write("="*70 + "\n")
    f.write("URBAN HEAT ISLAND ML - DEPLOYMENT RESULTS\n")
    f.write("="*70 + "\n\n")
    
    f.write("PREDICTION OVERVIEW\n")
    f.write("-"*70 + "\n")
    f.write(f"Total pixels: {np.prod(bands_shape):,}\n")
    f.write(f"Valid pixels: {np.sum(valid_mask):,}\n")
    f.write(f"Invalid/NoData pixels: {np.prod(bands_shape) - np.sum(valid_mask):,}\n\n")
    
    f.write("RANDOM FOREST PREDICTIONS (BEST MODEL)\n")
    f.write("-"*70 + "\n")
    f.write(f"Extreme heat zones: {extreme_heat_count_rf:,} pixels ({extreme_heat_pct_rf:.2f}%)\n")
    f.write(f"Normal temperature: {np.sum(valid_mask) - extreme_heat_count_rf:,} pixels ({100-extreme_heat_pct_rf:.2f}%)\n")
    f.write(f"Mean confidence score: {mean_conf_rf:.4f}\n")
    f.write(f"Min probability: {np.nanmin(probabilities_rf[valid_mask]):.4f}\n")
    f.write(f"Max probability: {np.nanmax(probabilities_rf[valid_mask]):.4f}\n\n")
    
    f.write("VOTING ENSEMBLE PREDICTIONS\n")
    f.write("-"*70 + "\n")
    f.write(f"Extreme heat zones: {extreme_heat_count_voting:,} pixels ({extreme_heat_pct_voting:.2f}%)\n")
    f.write(f"Normal temperature: {np.sum(valid_mask) - extreme_heat_count_voting:,} pixels ({100-extreme_heat_pct_voting:.2f}%)\n")
    f.write(f"Mean confidence score: {mean_conf_voting:.4f}\n\n")
    
    f.write("OUTPUT FILES\n")
    f.write("-"*70 + "\n")
    f.write(f"predictions_rf.tif - Probability map (Random Forest)\n")
    f.write(f"predictions_voting.tif - Probability map (Voting Ensemble)\n")
    f.write(f"predictions_binary_rf.tif - Binary classifications (1=Extreme, 0=Normal)\n")
    f.write(f"deployment_summary.txt - This summary\n\n")
    
    f.write("FEATURE IMPORTANCE (MODEL INSIGHTS)\n")
    f.write("-"*70 + "\n")
    for feat, imp in sorted(metadata['feature_importance'].items(), 
                           key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 50)
        f.write(f"{feat:<15}: {imp:.4f} {bar}\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("DEPLOYMENT STATUS: ✓ COMPLETE\n")
    f.write("="*70 + "\n")

print(f"\n✓ Summary report saved to deployment_summary.txt")

print("\n" + "="*70)
print("✓ PREDICTIONS COMPLETE - DEPLOYMENT SUCCESSFUL")
print("="*70)
print(f"\nOutput files created in: {output_dir}/")
print("\nUse QGIS or similar tools to visualize the prediction maps:")
print("  - predictions_rf.tif (0-1 confidence scores)")
print("  - predictions_binary_rf.tif (binary heat zones)")
