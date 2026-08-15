"""
Urban Heat Island ML - Streamlit Web Application
Interactive prediction dashboard for urban heat analysis
"""
import streamlit as st
import numpy as np
import pandas as pd
from deployment_api import UHIPredictor
from pathlib import Path
import pickle

# Page configuration
st.set_page_config(
    page_title="Urban Heat Island ML",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .extreme-heat {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff0000;
    }
    .normal-temp {
        background-color: #ccffcc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00aa00;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    """Load predictor once and cache it"""
    return UHIPredictor(models_dir="models")

def get_prediction_color(prediction_class, confidence):
    """Return color based on prediction"""
    if prediction_class == "Extreme Heat":
        return "#ffcccc" if confidence > 0.8 else "#ffe6e6"
    else:
        return "#ccffcc" if confidence > 0.8 else "#e6ffe6"

def main():
    # Header
    st.title("🌡️ Urban Heat Island ML Prediction System")
    st.markdown("**AI-powered analysis of urban heat using satellite data and machine learning**")
    
    # Initialize predictor
    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return
    
    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🔮 Predictions", "📊 Model Info", "📚 Documentation", "🎯 Batch Analysis"]
    )
    
    if page == "🔮 Predictions":
        prediction_page(predictor)
    elif page == "📊 Model Info":
        model_info_page(predictor)
    elif page == "📚 Documentation":
        documentation_page()
    elif page == "🎯 Batch Analysis":
        batch_analysis_page(predictor)

def prediction_page(predictor):
    """Single prediction interface"""
    st.header("🔮 Make a Prediction")
    st.markdown("""
    Enter the environmental features for a location to predict the likelihood of extreme urban heat.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Environmental Features")
        
        lst = st.slider(
            "Land Surface Temperature (°C)",
            min_value=15.0,
            max_value=50.0,
            value=32.24,
            step=0.1,
            help="Temperature of the land surface (LST)"
        )
        
        ndvi = st.slider(
            "NDVI - Vegetation Index",
            min_value=-1.0,
            max_value=1.0,
            value=0.77,
            step=0.01,
            help="Normalized Difference Vegetation Index (-1 to 1)"
        )
        
        ndbi = st.slider(
            "NDBI - Built-up Index",
            min_value=-1.0,
            max_value=1.0,
            value=-0.30,
            step=0.01,
            help="Normalized Difference Built-up Index (-1 to 1)"
        )
    
    with col2:
        st.subheader("Additional Features")
        
        brightness = st.slider(
            "Brightness (Normalized)",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.01,
            help="Surface brightness/reflectance (0 to 1)"
        )
        
        pop_density = st.slider(
            "Population Density (Normalized)",
            min_value=0.0,
            max_value=1.0,
            value=0.71,
            step=0.01,
            help="Normalized population density (0 to 1)"
        )
        
        bio1 = st.slider(
            "Bio1 - Annual Mean Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=25.88,
            step=0.1,
            help="Annual mean temperature from WorldClim"
        )
    
    # Make prediction
    if st.button("🔍 Predict", use_container_width=True, type="primary"):
        features = np.array([lst, ndvi, ndbi, brightness, pop_density, bio1])
        
        with st.spinner("Analyzing..."):
            result = predictor.predict_single(features)
        
        # Display results
        st.divider()
        st.subheader("📈 Prediction Results")
        
        # Main prediction with color coding
        pred_class = result['predictions']['voting_ensemble']['label']
        confidence = result['predictions']['voting_ensemble']['probability']
        
        if pred_class == "Extreme Heat":
            st.error(f"🔴 **{pred_class}** (Confidence: {confidence:.2%})")
            st.markdown(f"<div class='extreme-heat'>This location shows signs of significant urban heat island effect</div>", 
                       unsafe_allow_html=True)
        else:
            st.success(f"🟢 **{pred_class}** (Confidence: {confidence:.2%})")
            st.markdown(f"<div class='normal-temp'>This location has relatively normal temperature patterns</div>", 
                       unsafe_allow_html=True)
        
        # Model consensus
        st.subheader("🤖 Model Consensus")
        col1, col2, col3, col4 = st.columns(4)
        
        models = result['predictions']
        with col1:
            rf_label = models['random_forest']['label']
            st.metric("Random Forest", rf_label, 
                     f"{models['random_forest']['probability']:.1%}")
        
        with col2:
            voting_label = models['voting_ensemble']['label']
            st.metric("Voting Ensemble", voting_label,
                     f"{models['voting_ensemble']['probability']:.1%}")
        
        with col3:
            lr_prob = models['logistic_regression']['probability']
            lr_label = "Extreme Heat" if lr_prob > 0.5 else "Normal Temperature"
            st.metric("Logistic Regression", lr_label,
                     f"{lr_prob:.1%}")
        
        with col4:
            xgb_prob = models['xgboost']['probability']
            xgb_label = "Extreme Heat" if xgb_prob > 0.5 else "Normal Temperature"
            st.metric("XGBoost", xgb_label,
                     f"{xgb_prob:.1%}")
        
        # Interpretation
        st.subheader("💡 Interpretation")
        interpretation = predictor.explain_prediction(features)
        st.info(interpretation)
        
        # Feature analysis
        st.subheader("📊 Feature Analysis")
        feature_df = pd.DataFrame({
            'Feature': predictor.feature_names,
            'Value': [lst, ndvi, ndbi, brightness, pop_density, bio1],
            'Importance (%)': [80.57, 10.23, 5.14, 2.45, 1.21, 0.40]  # From model metadata
        })
        
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
        
        st.caption("**Note:** Feature importance based on Random Forest model training")

def model_info_page(predictor):
    """Display model information"""
    st.header("📊 Model Information")
    
    st.subheader("🎯 Model Architecture")
    st.markdown("""
    The ensemble system combines four complementary ML models:
    
    1. **Random Forest (Best Performer)**
       - 100 decision trees, max_depth=15
       - AUC: 1.0000 | F1-Score: 0.9996
       - Excellent for capturing non-linear relationships
    
    2. **Voting Ensemble (Most Robust)**
       - Combines Random Forest + Logistic Regression + XGBoost
       - AUC: 0.999999 | F1-Score: 0.9997
       - Reduces overfitting through model diversity
    
    3. **Logistic Regression (Interpretable Baseline)**
       - Linear model with feature scaling
       - AUC: 0.999984 | F1-Score: 0.9996
       - Fast inference and explainable predictions
    
    4. **XGBoost (Gradient Boosting)**
       - 50 boosted trees, optimized for class balance
       - AUC: 0.999959 | F1-Score: 0.9995
       - Handles feature interactions well
    """)
    
    st.subheader("🔍 Feature Importance")
    importance_data = {
        'Feature': ['LST', 'NDVI', 'NDBI', 'Brightness', 'PopDensity', 'Bio1'],
        'Importance (%)': [80.57, 10.23, 5.14, 2.45, 1.21, 0.40]
    }
    importance_df = pd.DataFrame(importance_data)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.bar_chart(importance_df.set_index('Feature')['Importance (%)'], use_container_width=True)
    with col2:
        st.dataframe(importance_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **LST (Land Surface Temperature)** is the dominant feature (80.57%) because:
    - Directly measures surface heating
    - Captures urban heat island effect in real-time
    - Satellite-derived from thermal infrared bands
    
    **NDVI (Vegetation)** is second (10.23%) because:
    - Green spaces cool urban areas
    - Inverse relationship with heat
    - Key indicator of urban development
    """)
    
    st.subheader("📈 Training Statistics")
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    with stats_col1:
        st.metric("Training Samples", "24,000", help="Stratified sample from 40.6M pixels")
    with stats_col2:
        st.metric("Features", "6", help="Multi-band satellite + auxiliary data")
    with stats_col3:
        st.metric("Threshold (LST)", "32.24°C", help="90th percentile = Extreme Heat")
    
    st.subheader("🗂️ Data Composition")
    data_info = {
        'Data Source': ['Landsat 8 Collection 2 L2', 'SEDAC Population', 'WorldClim', 'OSM Land Use'],
        'Bands/Vars': ['6 bands (B4, B5, B6, B7, ST_B10, QA)', '1 band (pop density)', '1 band (bio1)', 'Categorical']
    }
    st.dataframe(data_info, use_container_width=True, hide_index=True)

def documentation_page():
    """Display documentation"""
    st.header("📚 Documentation")
    
    st.subheader("🎯 Understanding the Prediction")
    st.markdown("""
    ### Classification Task
    The model predicts whether a location experiences **Extreme Urban Heat** based on satellite observations.
    
    - **Extreme Heat**: LST > 32.24°C (90th percentile)
    - **Normal Temperature**: LST ≤ 32.24°C
    
    ### Input Features
    
    **1. LST (Land Surface Temperature)**
    - Range: 15-50°C (typical 21-44°C)
    - Measured by Landsat 8 Band 10 (thermal infrared)
    - Directly indicates surface heating
    
    **2. NDVI (Normalized Difference Vegetation Index)**
    - Range: -1 to +1 (typical -0.5 to +0.9)
    - Formula: (NIR - Red) / (NIR + Red)
    - Higher values = more vegetation = cooler areas
    
    **3. NDBI (Normalized Difference Built-up Index)**
    - Range: -1 to +1 (typical -0.5 to +0.3)
    - Formula: (SWIR1 - NIR) / (SWIR1 + NIR)
    - Higher values = more built-up = warmer areas
    
    **4. Brightness (Surface Reflectance)**
    - Range: 0 to 1
    - Average reflectance across visible/near-infrared
    - Light surfaces = cooler (reflective)
    
    **5. Population Density**
    - Range: 0 to 1 (normalized)
    - From SEDAC dataset at 30 arc-second resolution
    - More people = more heat generation
    
    **6. Bio1 (Annual Mean Temperature)**
    - Range: 0-50°C (typical 24-27°C)
    - From WorldClim climate dataset
    - Baseline regional temperature
    """)
    
    st.subheader("🌍 Data Sources")
    st.markdown("""
    - **Satellite**: Landsat 8 Collection 2 Level-2 (June 2022, Arizona)
    - **Population**: NASA SEDAC (Socioeconomic Data and Applications Center)
    - **Climate**: WorldClim v2.1 (high-resolution climate data)
    - **Land Use**: OpenStreetMap (voluntary geospatial data)
    """)
    
    st.subheader("📐 Technical Details")
    st.markdown("""
    - **ML Framework**: scikit-learn, XGBoost
    - **Validation**: 5-fold Stratified Cross-Validation
    - **Train/Test Split**: 80/20 with stratification
    - **Class Balance**: Handled via stratified sampling & class weights
    - **Preprocessing**: StandardScaler for logistic regression
    """)

def batch_analysis_page(predictor):
    """Batch prediction interface"""
    st.header("🎯 Batch Analysis")
    st.markdown("Upload CSV with multiple locations for batch prediction")
    
    # Example data
    st.subheader("📝 CSV Format")
    example_data = pd.DataFrame({
        'LST': [28.5, 35.2, 30.1, 33.8],
        'NDVI': [0.85, 0.55, 0.79, 0.42],
        'NDBI': [-0.40, 0.10, -0.35, 0.20],
        'Brightness': [0.12, 0.25, 0.14, 0.30],
        'PopDensity': [0.60, 0.90, 0.75, 0.95],
        'Bio1': [25.5, 26.2, 25.8, 26.1]
    })
    
    st.dataframe(example_data, use_container_width=True, hide_index=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_cols = ['LST', 'NDVI', 'NDBI', 'Brightness', 'PopDensity', 'Bio1']
            if not all(col in df.columns for col in required_cols):
                st.error(f"❌ CSV must contain columns: {', '.join(required_cols)}")
                return
            
            st.info(f"✓ Loaded {len(df)} samples")
            
            if st.button("🔍 Predict All", use_container_width=True, type="primary"):
                progress_bar = st.progress(0)
                predictions = []
                
                for idx, row in df.iterrows():
                    features = np.array([row[col] for col in required_cols])
                    result = predictor.predict_single(features)
                    predictions.append({
                        'Index': idx,
                        'Prediction': result['predictions']['voting_ensemble']['label'],
                        'Confidence': result['predictions']['voting_ensemble']['probability'],
                        'RF_Pred': result['predictions']['random_forest']['label'],
                        'Voting_Pred': result['predictions']['voting_ensemble']['label']
                    })
                    progress_bar.progress((idx + 1) / len(df))
                
                results_df = pd.DataFrame(predictions)
                st.subheader("📊 Results")
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                extreme_count = (results_df['Prediction'] == 'Extreme Heat').sum()
                with col1:
                    st.metric("Extreme Heat Locations", extreme_count, 
                             f"{100*extreme_count/len(results_df):.1f}%")
                with col2:
                    st.metric("Normal Temperature", len(results_df) - extreme_count,
                             f"{100*(len(results_df)-extreme_count)/len(results_df):.1f}%")
                with col3:
                    st.metric("Avg Confidence", 
                             f"{results_df['Confidence'].mean():.2%}")
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="uhi_predictions.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

if __name__ == "__main__":
    main()
