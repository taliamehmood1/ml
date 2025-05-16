import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime, timedelta
import yfinance as yf

def validate_uploaded_data(data):
    """Validate uploaded dataset for required columns and format"""
    # Check if dataframe is empty
    if data.empty:
        return False, "The uploaded file contains no data."
    
    # Check minimum number of rows
    if len(data) < 30:
        return False, "The dataset should contain at least 30 rows for meaningful analysis."
    
    # Check for numeric columns (need at least one for analysis)
    numeric_cols = data.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        return False, "The dataset should contain at least one numeric column for analysis."
    
    return True, "Data validation successful."

def preprocess_data(data, theme="default"):
    """Preprocess the data based on the selected theme"""
    processed_data = data.copy()
    
    # Handle missing values
    processed_data = processed_data.fillna(method='ffill').fillna(method='bfill')
    
    # Convert date columns if any
    date_cols = processed_data.select_dtypes(include=['object']).columns
    for col in date_cols:
        try:
            # Check if the column contains dates
            pd.to_datetime(processed_data[col], errors='raise')
            processed_data[col] = pd.to_datetime(processed_data[col])
        except:
            pass
    
    # Add theme-specific processing
    if theme == "zombie":
        # For zombie theme, we might want to add volatility or "infection" metrics
        numeric_cols = processed_data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            # Add volatility as "infection rate"
            for col in numeric_cols:
                if processed_data[col].std() > 0:  # Only for non-constant columns
                    processed_data[f'{col}_volatility'] = processed_data[col].rolling(window=5).std().fillna(0)
    
    elif theme == "futuristic":
        # For futuristic theme, we might want to add smoothed trends or predictive indicators
        numeric_cols = processed_data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            # Add exponential moving averages as "AI predictions"
            for col in numeric_cols:
                processed_data[f'{col}_EMA'] = processed_data[col].ewm(span=10).mean()
    
    elif theme == "got":
        # For GOT theme, we might want to categorize data into "houses"
        numeric_cols = processed_data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0 and len(processed_data) > 10:
            # Create house allegiance based on value ranges
            main_col = numeric_cols[0]
            quantiles = processed_data[main_col].quantile([0.33, 0.66])
            
            def assign_house(val):
                if val <= quantiles[0.33]:
                    return "House Stark"
                elif val <= quantiles[0.66]:
                    return "House Lannister"
                else:
                    return "House Targaryen"
            
            processed_data['House'] = processed_data[main_col].apply(assign_house)
    
    elif theme == "pixel":
        # For pixel gaming theme, we might want to discretize data into "game levels"
        numeric_cols = processed_data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            # Discretize main column into "power levels"
            main_col = numeric_cols[0]
            processed_data['PowerLevel'] = pd.qcut(processed_data[main_col], 
                                                  q=5, 
                                                  labels=["Novice", "Apprentice", "Adept", "Expert", "Master"])
            
    return processed_data

def get_feature_recommendations(data, theme="default"):
    """Recommend features for analysis based on the data and theme"""
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    date_cols = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]
    
    recommendations = {
        "numeric_features": numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols,
        "date_features": date_cols[:1] if date_cols else [],
        "target_feature": numeric_cols[0] if numeric_cols else None,
    }
    
    # Theme-specific recommendations
    if theme == "zombie":
        # For zombie theme, look for features related to volatility or decline
        volatility_cols = [col for col in numeric_cols if "volatil" in col.lower() or "std" in col.lower() or "var" in col.lower()]
        if volatility_cols:
            recommendations["target_feature"] = volatility_cols[0]
            
    elif theme == "futuristic":
        # For futuristic theme, look for features related to growth or technology
        tech_cols = [col for col in numeric_cols if any(term in col.lower() for term in ["growth", "tech", "ai", "future", "trend"])]
        if tech_cols:
            recommendations["target_feature"] = tech_cols[0]
            
    elif theme == "got":
        # For GOT theme, look for features that could represent power or resources
        power_cols = [col for col in numeric_cols if any(term in col.lower() for term in ["power", "gold", "value", "strength", "price"])]
        if power_cols:
            recommendations["target_feature"] = power_cols[0]
            
    elif theme == "pixel":
        # For pixel theme, look for features that could represent game metrics
        game_cols = [col for col in numeric_cols if any(term in col.lower() for term in ["score", "level", "power", "rating", "points"])]
        if game_cols:
            recommendations["target_feature"] = game_cols[0]
    
    return recommendations

def generate_download_link(df, filename="data.csv", text="Download CSV"):
    """Generate a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href
