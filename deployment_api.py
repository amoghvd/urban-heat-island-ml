#!/usr/bin/env python
"""
Urban Heat Island ML - Production Inference API
Provides REST API and batch inference capabilities
"""
import joblib
import pickle
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class UHIPredictor:
    """Production inference wrapper for Urban Heat Island ML models"""
    
    def __init__(self, models_dir="models"):
        """Load all trained models and metadata"""
        self.models_dir = Path(models_dir)
        
        # Load models
        self.rf_model = joblib.load(self.models_dir / "random_forest.pkl")
        self.voting_model = joblib.load(self.models_dir / "voting_ensemble.pkl")
        self.lr_model = joblib.load(self.models_dir / "logistic_regression.pkl")
        self.xgb_model = joblib.load(self.models_dir / "xgboost.pkl")
        self.scaler = joblib.load(self.models_dir / "scaler.pkl")
        
        # Load metadata
        with open(self.models_dir / "metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.feature_names = self.metadata['feature_names']
        self.threshold = self.metadata['threshold']
        
        print("✓ UHI Predictor initialized successfully")
    
    def predict_single(self, features_array):
        """
        Predict for a single sample
        
        Parameters:
        -----------
        features_array : array-like, shape (6,)
            Features: [LST, NDVI, NDBI, Brightness, PopDensity, Bio1]
        
        Returns:
        --------
        dict with predictions from all models
        """
        features = np.array(features_array).reshape(1, -1)
        
        rf_pred = self.rf_model.predict(features)[0]
        rf_prob = self.rf_model.predict_proba(features)[0, 1]
        
        voting_pred = self.voting_model.predict(features)[0]
        voting_prob = self.voting_model.predict_proba(features)[0, 1]
        
        features_scaled = self.scaler.transform(features)
        lr_prob = self.lr_model.predict_proba(features_scaled)[0, 1]
        xgb_prob = self.xgb_model.predict_proba(features)[0, 1]
        
        return {
            "input_features": {f: float(v) for f, v in zip(self.feature_names, features_array)},
            "predictions": {
                "random_forest": {
                    "class": int(rf_pred),
                    "probability": float(rf_prob),
                    "label": "Extreme Heat" if rf_pred == 1 else "Normal Temperature"
                },
                "voting_ensemble": {
                    "probability": float(voting_prob),
                    "label": "Extreme Heat" if voting_pred == 1 else "Normal Temperature"
                },
                "logistic_regression": {
                    "probability": float(lr_prob)
                },
                "xgboost": {
                    "probability": float(xgb_prob)
                }
            },
            "consensus": {
                "votes_for_extreme": sum([rf_pred, voting_pred, 1 if lr_prob > 0.5 else 0, 1 if xgb_prob > 0.5 else 0]),
                "ensemble_agreement": "Strong" if voting_prob > 0.8 else "Moderate" if voting_prob > 0.5 else "Weak"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def predict_batch(self, features_batch):
        """
        Predict for multiple samples
        
        Parameters:
        -----------
        features_batch : array-like, shape (n_samples, 6)
        
        Returns:
        --------
        list of prediction dictionaries
        """
        results = []
        for features in features_batch:
            results.append(self.predict_single(features))
        return results
    
    def get_model_info(self):
        """Return model information and performance metrics"""
        return {
            "project": "Urban Heat Island ML Analysis",
            "models_trained": list(self.metadata['performance'].keys()),
            "best_model": "Random Forest",
            "feature_names": self.feature_names,
            "feature_importance": self.metadata['feature_importance'],
            "model_performance": self.metadata['performance'],
            "decision_threshold_lst": float(self.threshold),
            "training_samples": self.metadata['training_samples'],
            "test_samples": self.metadata['test_samples']
        }
    
    def explain_prediction(self, features_array):
        """Provide interpretation of prediction"""
        lst, ndvi, ndbi, brightness, pop_density, bio1 = features_array
        
        explanation = {
            "lst_analysis": f"Surface temperature: {lst:.2f}°C (threshold: {self.threshold:.2f}°C)",
            "urban_development": f"Built-up index (NDBI): {ndbi:.2f} (negative=vegetation, positive=urban)",
            "vegetation": f"Vegetation index (NDVI): {ndvi:.2f} (higher=more vegetation=cooler)",
            "brightness": f"Spectral brightness: {brightness:.2f}",
            "population": f"Population density: {pop_density:.2f}",
            "climate": f"Annual mean temp (Bio1): {bio1:.2f}°C",
            "interpretation": self._interpret_combination(lst, ndvi, ndbi)
        }
        return explanation
    
    def _interpret_combination(self, lst, ndvi, ndbi):
        """Interpret feature combination"""
        if lst > self.threshold:
            if ndbi > 0:
                return "Urban area with high temperature - likely urban heat island hotspot"
            elif ndvi > 0.5:
                return "Vegetated area with high temperature - possible park with limited cooling"
            else:
                return "Mixed area with high temperature"
        else:
            if ndvi > 0.5:
                return "Vegetated area with normal/cool temperature - likely park or green space"
            else:
                return "Non-vegetated area with normal/cool temperature"

# Example usage and deployment demonstration
if __name__ == "__main__":
    print("="*70)
    print("URBAN HEAT ISLAND ML - PRODUCTION INFERENCE API")
    print("="*70)
    
    # Initialize predictor
    print("\n[1/3] Initializing models...")
    predictor = UHIPredictor(models_dir="models")
    
    # Get model info
    print("\n[2/3] Model Information:")
    info = predictor.get_model_info()
    print(json.dumps(info, indent=2))
    
    # Example predictions
    print("\n[3/3] Example Predictions:")
    print("-"*70)
    
    # Example 1: Hot urban area
    example1 = [35.0, 0.3, 0.2, 0.15, 0.8, 26.0]  # High LST, low vegetation, urban
    print("\nExample 1: Urban Heat Island Hotspot")
    pred1 = predictor.predict_single(example1)
    print(f"Features: LST={example1[0]}, NDVI={example1[1]}, NDBI={example1[2]}")
    print(f"Random Forest Prediction: {pred1['predictions']['random_forest']['label']}")
    print(f"Confidence: {pred1['predictions']['random_forest']['probability']:.4f}")
    print(f"Explanation: {predictor.explain_prediction(example1)['interpretation']}")
    
    # Example 2: Cool vegetated area
    example2 = [25.0, 0.7, -0.2, 0.10, 0.2, 25.0]  # Low LST, high vegetation
    print("\nExample 2: Green Space (Cool Area)")
    pred2 = predictor.predict_single(example2)
    print(f"Features: LST={example2[0]}, NDVI={example2[1]}, NDBI={example2[2]}")
    print(f"Random Forest Prediction: {pred2['predictions']['random_forest']['label']}")
    print(f"Confidence: {pred2['predictions']['random_forest']['probability']:.4f}")
    print(f"Explanation: {predictor.explain_prediction(example2)['interpretation']}")
    
    # Save API demo
    print("\n" + "="*70)
    print("✓ INFERENCE API READY FOR DEPLOYMENT")
    print("="*70)
    print("\nUsage:")
    print("  predictor = UHIPredictor(models_dir='models')")
    print("  result = predictor.predict_single([35.0, 0.3, 0.2, 0.15, 0.8, 26.0])")
    print("  print(result)")
