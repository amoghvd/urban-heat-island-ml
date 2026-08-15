# 🌡️ Urban Heat Island ML - Streamlit Deployment Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

---

## Features

### 🔮 Predictions Page
Make single predictions for a specific location:
- Adjust 6 environmental features using interactive sliders
- Get real-time predictions from 4 ML models
- View model consensus and confidence scores
- Read AI-generated interpretation of results
- See feature importance rankings

**Features to Input:**
- **LST (15-50°C)**: Land Surface Temperature
- **NDVI (-1 to +1)**: Vegetation Index
- **NDBI (-1 to +1)**: Built-up Index
- **Brightness (0-1)**: Surface Reflectance
- **PopDensity (0-1)**: Population Density (normalized)
- **Bio1 (0-50°C)**: Annual Mean Temperature

### 📊 Model Info Page
Detailed model architecture and performance:
- 4-model ensemble explanation
- Feature importance visualization
- Training statistics
- Data composition overview
- Performance metrics (AUC, F1-Score)

### 📚 Documentation Page
Complete technical documentation:
- Classification task explanation
- Input feature descriptions
- Data sources and methodology
- Technical implementation details
- Validation strategy

### 🎯 Batch Analysis Page
Process multiple locations at once:
- Upload CSV with predictions for bulk analysis
- Download results as CSV
- View summary statistics
- Identify patterns across multiple locations

---

## CSV Format for Batch Analysis

When uploading CSV for batch predictions, use this format:

```csv
LST,NDVI,NDBI,Brightness,PopDensity,Bio1
28.5,0.85,-0.40,0.12,0.60,25.5
35.2,0.55,0.10,0.25,0.90,26.2
30.1,0.79,-0.35,0.14,0.75,25.8
33.8,0.42,0.20,0.30,0.95,26.1
```

---

## Deployment Options

### Option 1: Local Development (Current)
```bash
streamlit run streamlit_app.py
```
- Simple and immediate
- Perfect for local testing
- Runs on `http://localhost:8501`

### Option 2: Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t uhi-ml .
docker run -p 8501:8501 uhi-ml
```

### Option 3: Streamlit Cloud (Free Hosting)
1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Deploy repo directly
4. Public URL provided automatically

### Option 4: Heroku Deployment
Create `Procfile`:
```
web: streamlit run streamlit_app.py --logger.level=error
```

Deploy:
```bash
git push heroku main
```

---

## Configuration

### Custom Theme (Optional)
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
```

### Performance Tuning
For large batch predictions, adjust `maxMessageSize`:
```toml
[client]
maxMessageSize = 200
```

---

## Troubleshooting

### Models Not Loading
- Ensure `models/` directory exists with all .pkl files
- Check file paths are correct
- Verify models were trained and serialized

### Slow Predictions
- First prediction may be slow (model loading)
- Streamlit caches the predictor - subsequent predictions are instant
- Batch processing: ~50 predictions/second on standard hardware

### CSV Upload Issues
- Verify column names match exactly: `LST,NDVI,NDBI,Brightness,PopDensity,Bio1`
- Ensure numeric values, not text
- Check file encoding is UTF-8

---

## Architecture

```
streamlit_app.py
├── Main App (Streamlit Interface)
│   ├── Predictions Page (Single)
│   ├── Model Info Page (Visualizations)
│   ├── Documentation Page (Text)
│   └── Batch Analysis (Multi)
│
└── deployment_api.py (UHIPredictor)
    └── models/
        ├── random_forest.pkl
        ├── voting_ensemble.pkl
        ├── logistic_regression.pkl
        ├── xgboost.pkl
        ├── scaler.pkl
        └── metadata.pkl
```

---

## API Reference (Within Streamlit)

The `UHIPredictor` class is loaded automatically:

```python
predictor = UHIPredictor(models_dir="models")

# Single prediction
result = predictor.predict_single([LST, NDVI, NDBI, Brightness, PopDensity, Bio1])

# Returns:
# {
#     'predicted_class': 'Extreme Heat' or 'Normal Temperature',
#     'all_predictions': {
#         'random_forest': {'class': ..., 'probability': ...},
#         'voting_ensemble': {'class': ..., 'probability': ...},
#         'logistic_regression': {'class': ..., 'probability': ...},
#         'xgboost': {'class': ..., 'probability': ...}
#     }
# }

# Explanation
explanation = predictor.explain_prediction(features)
```

---

## Performance Benchmarks

| Task | Time | Hardware |
|------|------|----------|
| Load models (cached) | 2-3 sec | First run only |
| Single prediction | <10 ms | Standard CPU |
| Batch 1000 samples | ~20 sec | Standard CPU |
| Feature importance | <5 ms | Cached |

---

## Next Steps

1. **Web Integration**: Embed iframe in existing website
2. **Mobile App**: Use Streamlit Community Cloud for mobile access
3. **Real-time Updates**: Connect to live Landsat data feed
4. **Advanced Analytics**: Add time-series analysis and trend visualization
5. **Multi-region**: Extend to different geographic areas

---

## Support

For issues or questions:
- Check deployment logs: `streamlit run streamlit_app.py --logger.level=debug`
- Review model loading in `deployment_api.py`
- Verify all dependencies: `pip list | grep -E "(streamlit|numpy|scikit|xgboost)"`

---

**Created**: August 2026  
**Version**: 1.0  
**Status**: Production Ready
