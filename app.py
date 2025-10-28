import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Supabase configuration
SUPABASE_URL = "https://fjfmgndbiespptmsnrff.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZqZm1nbmRiaWVzcHB0bXNucmZmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjEyMzk0NzQsImV4cCI6MjA3NjgxNTQ3NH0.FH9L41cIKXH_mVbl7szkb_CDKoyKdw97gOUhDOYJFnQ"

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_data(ttl=300)
def load_data():
    try:
        supabase = init_supabase()
        response = supabase.table('airquality').select('*').execute()
        df = pd.DataFrame(response.data)
        
        # Convert columns to appropriate data types
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.sort_values('created_at')
        
        # Convert numeric columns
        numeric_cols = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with missing values in target columns
        df = df.dropna(subset=numeric_cols)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def create_features(df, target_columns, n_lags=6):
    """Create lag features for time series prediction"""
    df_eng = df.copy()
    
    for col in target_columns:
        for lag in range(1, n_lags + 1):
            df_eng[f'{col}_lag_{lag}'] = df_eng[col].shift(lag)
    
    # Add time-based features
    df_eng['hour'] = df_eng['created_at'].dt.hour
    df_eng['day_of_week'] = df_eng['created_at'].dt.dayofweek
    df_eng['month'] = df_eng['created_at'].dt.month
    
    # Drop rows with NaN values created by lag features
    df_eng = df_eng.dropna()
    
    return df_eng

def prepare_lstm_data(df, target_columns, sequence_length=10):
    """Prepare data for LSTM model"""
    features = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    X, y = [], []
    
    for i in range(sequence_length, len(df)):
        X.append(df[features].iloc[i-sequence_length:i].values)
        y.append(df[target_columns].iloc[i].values)
    
    return np.array(X), np.array(y)

def train_random_forest(X_train, X_test, y_train, y_test, target_columns):
    """Train Random Forest model"""
    models = {}
    predictions = {}
    scores = {}
    
    for i, col in enumerate(target_columns):
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train[:, i])
        pred = rf.predict(X_test)
        
        models[col] = rf
        predictions[col] = pred
        scores[col] = {
            'rmse': np.sqrt(mean_squared_error(y_test[:, i], pred)),
            'r2': r2_score(y_test[:, i], pred)
        }
    
    return models, predictions, scores

def train_xgboost(X_train, X_test, y_train, y_test, target_columns):
    """Train XGBoost model"""
    models = {}
    predictions = {}
    scores = {}
    
    for i, col in enumerate(target_columns):
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train[:, i])
        pred = xgb_model.predict(X_test)
        
        models[col] = xgb_model
        predictions[col] = pred
        scores[col] = {
            'rmse': np.sqrt(mean_squared_error(y_test[:, i], pred)),
            'r2': r2_score(y_test[:, i], pred)
        }
    
    return models, predictions, scores

def train_svm(X_train, X_test, y_train, y_test, target_columns):
    """Train SVM model"""
    # Scale the data for SVM
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    # Use MultiOutputRegressor for multiple targets
    svm_model = MultiOutputRegressor(SVR(kernel='rbf', C=1.0))
    svm_model.fit(X_train_scaled, y_train_scaled)
    
    pred_scaled = svm_model.predict(X_test_scaled)
    predictions = scaler_y.inverse_transform(pred_scaled)
    
    scores = {}
    for i, col in enumerate(target_columns):
        scores[col] = {
            'rmse': np.sqrt(mean_squared_error(y_test[:, i], predictions[:, i])),
            'r2': r2_score(y_test[:, i], predictions[:, i])
        }
    
    return svm_model, predictions, scores, scaler_X, scaler_y

def train_lstm(X_train, X_test, y_train, y_test, target_columns):
    """Train LSTM model"""
    # Scale the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Reshape for scaling
    X_train_reshaped = X_train.reshape(-1, X_train.shape[2])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[2])
    
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(len(target_columns))
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Make predictions
    pred_scaled = model.predict(X_test_scaled, verbose=0)
    predictions = scaler_y.inverse_transform(pred_scaled)
    
    scores = {}
    for i, col in enumerate(target_columns):
        scores[col] = {
            'rmse': np.sqrt(mean_squared_error(y_test[:, i], predictions[:, i])),
            'r2': r2_score(y_test[:, i], predictions[:, i])
        }
    
    return model, predictions, scores, history, scaler_X, scaler_y

def main():
    st.set_page_config(page_title="Air Quality Prediction", layout="wide")
    
    st.title("🌤️ Air Quality Prediction Dashboard")
    st.markdown("""
    This app predicts future values of PM2.5, PM10, CO2, CO, Temperature, and Humidity using multiple machine learning models.
    """)
    
    # Load data
    with st.spinner('Loading data from Supabase...'):
        df = load_data()
    
    if df.empty:
        st.error("No data loaded. Please check your Supabase connection.")
        return
    
    st.sidebar.header("Configuration")
    
    # Target selection
    target_columns = ['pm25', 'pm10', 'co2', 'co', 'temperature', 'humidity']
    selected_targets = st.sidebar.multiselect(
        "Select targets to predict:",
        target_columns,
        default=target_columns
    )
    
    # Model selection
    models_to_train = st.sidebar.multiselect(
        "Select models to train:",
        ['Random Forest', 'XGBoost', 'SVM', 'LSTM'],
        default=['Random Forest', 'XGBoost']
    )
    
    # Display data overview
    st.header("📊 Data Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df))
        st.metric("Date Range", f"{df['created_at'].min().strftime('%Y-%m-%d')} to {df['created_at'].max().strftime('%Y-%m-%d')}")
    
    with col2:
        st.metric("Features", 6)
        st.metric("Target Variables", len(selected_targets))
    
    with col3:
        st.metric("Data Completeness", f"{(1 - df[target_columns].isna().sum().sum() / (len(df) * len(target_columns))) * 100:.1f}%")
    
    # Show data preview
    if st.checkbox("Show raw data"):
        st.dataframe(df.tail(10))
    
    # Feature engineering
    st.header("🔧 Feature Engineering")
    df_eng = create_features(df, selected_targets)
    
    # Prepare data for traditional ML models
    feature_cols = [col for col in df_eng.columns if col not in ['id', 'created_at'] + selected_targets]
    X = df_eng[feature_cols].values
    y = df_eng[selected_targets].values
    
    # Split data
    test_size = st.sidebar.slider("Test set size:", 0.1, 0.3, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
    
    # Model training section
    st.header("🤖 Model Training & Evaluation")
    
    all_models = {}
    all_predictions = {}
    all_scores = {}
    
    # Train selected models
    for model_name in models_to_train:
        st.subheader(f"{model_name} Model")
        
        with st.spinner(f'Training {model_name}...'):
            if model_name == 'Random Forest':
                models, predictions, scores = train_random_forest(X_train, X_test, y_train, y_test, selected_targets)
                all_models['RF'] = models
                all_predictions['RF'] = predictions
                all_scores['RF'] = scores
                
            elif model_name == 'XGBoost':
                models, predictions, scores = train_xgboost(X_train, X_test, y_train, y_test, selected_targets)
                all_models['XGB'] = models
                all_predictions['XGB'] = predictions
                all_scores['XGB'] = scores
                
            elif model_name == 'SVM':
                model, predictions, scores, scaler_X, scaler_y = train_svm(X_train, X_test, y_train, y_test, selected_targets)
                all_models['SVM'] = (model, scaler_X, scaler_y)
                all_predictions['SVM'] = predictions
                all_scores['SVM'] = scores
                
            elif model_name == 'LSTM':
                # Prepare LSTM data
                sequence_length = 10
                X_lstm, y_lstm = prepare_lstm_data(df_eng, selected_targets, sequence_length)
                
                # Split LSTM data
                split_idx = int(len(X_lstm) * (1 - test_size))
                X_train_lstm, X_test_lstm = X_lstm[:split_idx], X_lstm[split_idx:]
                y_train_lstm, y_test_lstm = y_lstm[:split_idx], y_lstm[split_idx:]
                
                model, predictions, scores, history, scaler_X, scaler_y = train_lstm(
                    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, selected_targets
                )
                all_models['LSTM'] = (model, scaler_X, scaler_y)
                all_predictions['LSTM'] = predictions
                all_scores['LSTM'] = scores
        
        # Display scores for this model
        scores_df = pd.DataFrame(all_scores[model_name[:3]]).T
        scores_df.columns = ['RMSE', 'R² Score']
        st.dataframe(scores_df.style.format({"RMSE": "{:.4f}", "R² Score": "{:.4f}"}))
    
    # Model comparison
    if len(models_to_train) > 1:
        st.header("📈 Model Comparison")
        
        # Create comparison chart
        comparison_data = []
        for model_key in all_scores.keys():
            for target in selected_targets:
                comparison_data.append({
                    'Model': model_key,
                    'Target': target,
                    'RMSE': all_scores[model_key][target]['rmse'],
                    'R2_Score': all_scores[model_key][target]['r2']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # RMSE comparison
        fig_rmse = go.Figure()
        for model in comparison_df['Model'].unique():
            model_data = comparison_df[comparison_df['Model'] == model]
            fig_rmse.add_trace(go.Bar(
                name=model,
                x=model_data['Target'],
                y=model_data['RMSE']
            ))
        
        fig_rmse.update_layout(
            title="RMSE Comparison by Model and Target",
            xaxis_title="Target Variable",
            yaxis_title="RMSE",
            barmode='group'
        )
        
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    # Future prediction
    st.header("🔮 Future Prediction")
    
    if st.button("Predict Next Values"):
        # Use the most recent data to predict next values
        latest_features = df_eng[feature_cols].iloc[-1:].values
        
        st.subheader("Predicted Next Values:")
        
        predictions_data = []
        for model_key in all_models.keys():
            if model_key in ['RF', 'XGB']:
                model_preds = {}
                for target in selected_targets:
                    model_preds[target] = all_models[model_key][target].predict(latest_features)[0]
                predictions_data.append(model_preds)
                
            elif model_key == 'SVM':
                model, scaler_X, scaler_y = all_models['SVM']
                latest_scaled = scaler_X.transform(latest_features)
                pred_scaled = model.predict(latest_scaled)
                pred = scaler_y.inverse_transform(pred_scaled)[0]
                model_preds = {target: pred[i] for i, target in enumerate(selected_targets)}
                predictions_data.append(model_preds)
                
            elif model_key == 'LSTM':
                model, scaler_X, scaler_y = all_models['LSTM']
                # Use last sequence for prediction
                sequence_data = df_eng[['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']].tail(10).values
                sequence_scaled = scaler_X.transform(sequence_data.reshape(-1, 6)).reshape(1, 10, 6)
                pred_scaled = model.predict(sequence_scaled, verbose=0)
                pred = scaler_y.inverse_transform(pred_scaled)[0]
                model_preds = {target: pred[i] for i, target in enumerate(selected_targets)}
                predictions_data.append(model_preds)
        
        # Display predictions
        pred_df = pd.DataFrame(predictions_data, index=list(all_models.keys()))
        st.dataframe(pred_df.style.format("{:.4f}"))
    
    # Feature importance for tree-based models
    if 'RF' in all_models or 'XGB' in all_models:
        st.header("🎯 Feature Importance")
        
        model_for_importance = st.selectbox(
            "Select model for feature importance:",
            [key for key in ['RF', 'XGB'] if key in all_models]
        )
        
        if model_for_importance:
            target_for_importance = st.selectbox("Select target:", selected_targets)
            
            if model_for_importance == 'RF':
                importances = all_models['RF'][target_for_importance].feature_importances_
            else:
                importances = all_models['XGB'][target_for_importance].feature_importances_
            
            feature_importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': importances
            }).sort_values('importance', ascending=True)
            
            fig = go.Figure(go.Bar(
                x=feature_importance_df['importance'],
                y=feature_importance_df['feature'],
                orientation='h'
            ))
            
            fig.update_layout(
                title=f"Feature Importance for {target_for_importance} ({model_for_importance})",
                xaxis_title="Importance",
                yaxis_title="Features"
            )
            
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
