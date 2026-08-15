#!/usr/bin/env python
"""Quick test to verify Streamlit app loads correctly"""
import sys
print("Python version:", sys.version)

try:
    import streamlit
    print("✓ Streamlit:", streamlit.__version__)
except ImportError as e:
    print("✗ Streamlit not available:", e)
    sys.exit(1)

try:
    import pandas
    print("✓ Pandas:", pandas.__version__)
except ImportError as e:
    print("✗ Pandas not available:", e)
    sys.exit(1)

try:
    import numpy
    print("✓ NumPy:", numpy.__version__)
except ImportError as e:
    print("✗ NumPy not available:", e)
    sys.exit(1)

try:
    from deployment_api import UHIPredictor
    print("✓ deployment_api imported successfully")
    
    # Try to load models
    predictor = UHIPredictor(models_dir="models")
    print("✓ Models loaded successfully")
    
    # Test single prediction
    import numpy as np
    test_features = np.array([32.24, 0.77, -0.30, 0.15, 0.71, 25.88])
    result = predictor.predict_single(test_features)
    print(f"✓ Test prediction: {result['predicted_class']}")
    
except Exception as e:
    print("✗ Error loading models:", e)
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL CHECKS PASSED - Ready to run Streamlit app!")
print("="*60)
print("\nTo start the app, run:")
print("  streamlit run streamlit_app.py")
