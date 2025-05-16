import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import os
import time
import random

# Import local modules
from themes import (
    apply_zombie_theme, apply_futuristic_theme, 
    apply_got_theme, apply_pixel_theme,
    get_theme_description
)
from models import (
    ZombieLinearRegression, FuturisticLogisticRegression,
    GOTKMeansClustering, PixelKMeansClustering
)
from utils import (
    load_sample_data, fetch_stock_data, plot_stock_data
)
from data_handler import (
    validate_uploaded_data, preprocess_data, 
    get_feature_recommendations, generate_download_link
)

# Set page config
st.set_page_config(
    page_title="Multi-Themed Financial ML App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'theme' not in st.session_state:
    st.session_state.theme = "zombie"  # Default theme
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None

# Sidebar for theme selection
with st.sidebar:
    st.title("Theme Selection")
    st.write("Choose your financial adventure:")
    
    # Theme selection buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧟‍♂️ Zombie", use_container_width=True):
            st.session_state.theme = "zombie"
            st.rerun()
        if st.button("🔥 Game of Thrones", use_container_width=True):
            st.session_state.theme = "got"
            st.rerun()
    with col2:
        if st.button("🚀 Futuristic", use_container_width=True):
            st.session_state.theme = "futuristic"
            st.rerun()
        if st.button("🎮 Pixel Gaming", use_container_width=True):
            st.session_state.theme = "pixel"
            st.rerun()
    
    # Display current theme details
    theme_info = get_theme_description(st.session_state.theme)
    st.markdown(f"### {theme_info['title']}")
    st.markdown(theme_info['description'])
    
    # Data options
    st.subheader("Data Options")
    data_source = st.radio(
        "Choose data source:",
        ["Upload Dataset", "Sample Data", "Yahoo Finance"]
    )
    
    # Add theme-specific sidebar elements
    if st.session_state.theme == "zombie":
        st.markdown("### Apocalypse Intensity")
        apocalypse_intensity = st.slider("Set danger level:", 1, 10, 7)
        
    elif st.session_state.theme == "futuristic":
        st.markdown("### AI Confidence Level")
        ai_confidence = st.slider("Quantum accuracy:", 70, 100, 95)
        
    elif st.session_state.theme == "got":
        st.markdown("### House Allegiance")
        house = st.selectbox(
            "Choose your house:",
            ["Stark", "Lannister", "Targaryen", "Baratheon", "Tyrell", "Greyjoy"]
        )
        
    elif st.session_state.theme == "pixel":
        st.markdown("### Game Settings")
        difficulty = st.select_slider(
            "Difficulty level:",
            options=["Easy", "Medium", "Hard", "Expert"]
        )

# Apply the selected theme
if st.session_state.theme == "zombie":
    apply_zombie_theme()
elif st.session_state.theme == "futuristic":
    apply_futuristic_theme()
elif st.session_state.theme == "got":
    apply_got_theme()
elif st.session_state.theme == "pixel":
    apply_pixel_theme()

# Main content area
if st.session_state.theme == "zombie":
    st.write("## Zombie Apocalypse Financial Survival Guide")
    st.write("When the dead rise, your investments shouldn't fall!")
elif st.session_state.theme == "futuristic":
    st.write("## NEURO-FINANCE QUANTUM ANALYZER")
    st.write("Accessing future market probabilities through quantum computing...")
elif st.session_state.theme == "got":
    st.write("## The Iron Bank of Braavos")
    st.write("The Lords of Finance always collect their debts...")
elif st.session_state.theme == "pixel":
    st.write("## STOCK MARKET QUEST")
    st.write("PRESS START TO BEGIN YOUR INVESTMENT ADVENTURE!")

# Data loading section
st.write("---")
st.write("## Data Analysis")

# Process data based on source selection
if data_source == "Upload Dataset":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            is_valid, message = validate_uploaded_data(data)
            
            if is_valid:
                st.session_state.data = data
                st.session_state.processed_data = preprocess_data(data, st.session_state.theme)
                st.success(f"Successfully loaded data with {len(data)} rows and {len(data.columns)} columns")
            else:
                st.error(message)
        except Exception as e:
            st.error(f"Error loading data: {e}")
    
elif data_source == "Sample Data":
    if st.button("Load Sample Data"):
        with st.spinner("Loading sample data..."):
            data = load_sample_data(st.session_state.theme)
            st.session_state.data = data
            st.session_state.processed_data = preprocess_data(data, st.session_state.theme)
            st.success(f"Successfully loaded sample data with {len(data)} rows and {len(data.columns)} columns")

elif data_source == "Yahoo Finance":
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", "AAPL")
    
    with col2:
        period_options = {
            "1 Month": "1mo", 
            "3 Months": "3mo", 
            "6 Months": "6mo",
            "1 Year": "1y", 
            "2 Years": "2y", 
            "5 Years": "5y"
        }
        period = st.selectbox("Select Time Period:", list(period_options.keys()))
    
    with col3:
        if st.button("Fetch Data"):
            with st.spinner(f"Fetching data for {ticker}..."):
                data = fetch_stock_data(ticker, period_options[period], st.session_state.theme)
                if data is not None:
                    st.session_state.data = data
                    st.session_state.processed_data = preprocess_data(data, st.session_state.theme)
                    st.session_state.stock_data = data
                    st.success(f"Successfully fetched data for {ticker}")

# Display loaded data
if st.session_state.processed_data is not None:
    data = st.session_state.processed_data
    
    # Add theme-specific data presentation
    if st.session_state.theme == "zombie":
        st.write("### Survival Resources Inventory")
    elif st.session_state.theme == "futuristic":
        st.write("### QUANTUM DATA MATRIX")
    elif st.session_state.theme == "got":
        st.write("### The Maester's Archives")
    elif st.session_state.theme == "pixel":
        st.write("### INVENTORY ITEMS")
    
    # Display dataframe with first 10 rows
    st.dataframe(data.head(10), use_container_width=True)
    
    # Display stock chart if available
    if st.session_state.stock_data is not None:
        stock_fig = plot_stock_data(st.session_state.stock_data, st.session_state.theme)
        st.plotly_chart(stock_fig, use_container_width=True)
    
    # Display data statistics
    with st.expander("Data Statistics"):
        st.write("#### Summary Statistics")
        st.dataframe(data.describe(), use_container_width=True)
        
        # Column information
        st.write("#### Column Information")
        col_info = pd.DataFrame({
            'Column': data.columns,
            'Type': data.dtypes,
            'Non-Null Count': data.count(),
            'Null Count': data.isna().sum(),
            'Unique Values': [data[col].nunique() for col in data.columns]
        })
        st.dataframe(col_info, use_container_width=True)
    
    # Machine Learning Section
    st.write("---")
    if st.session_state.theme == "zombie":
        st.write("## Zombie Outbreak Prediction Model")
        st.write("Using Linear Regression to forecast which assets will survive the apocalypse.")
    elif st.session_state.theme == "futuristic":
        st.write("## QUANTUM LOGISTIC CLASSIFICATION")
        st.write("Using advanced algorithms to classify market behaviors with unparalleled precision.")
    elif st.session_state.theme == "got":
        st.write("## The Raven's Market Clustering")
        st.write("The Maesters of the Citadel use ancient wisdom to group your assets into powerful houses.")
    elif st.session_state.theme == "pixel":
        st.write("## MARKET ZONE MAPPER")
        st.write("SELECT YOUR ADVENTURE PATH TO DISCOVER HIDDEN TREASURE ZONES!")
    
    # Get feature recommendations
    recommendations = get_feature_recommendations(data, st.session_state.theme)
    
    # Feature selection based on model type
    if st.session_state.theme == "zombie":
        # Linear Regression for Zombie theme
        st.write("### Select Features for Survival Prediction")
        
        target_col = st.selectbox(
            "Select target variable (what you want to predict):",
            options=data.select_dtypes(include=['number']).columns.tolist(),
            index=data.select_dtypes(include=['number']).columns.tolist().index(recommendations["target_feature"]) 
            if recommendations["target_feature"] in data.select_dtypes(include=['number']).columns.tolist() else 0
        )
        
        feature_cols = st.multiselect(
            "Select feature variables (predictors):",
            options=[col for col in data.select_dtypes(include=['number']).columns if col != target_col],
            default=recommendations["numeric_features"][:3]
        )
        
        if st.button("Run Zombie Survival Prediction"):
            if len(feature_cols) > 0:
                with st.spinner("The zombies are processing your data... don't get bitten!"):
                    # Initialize and train model
                    model = ZombieLinearRegression()
                    X, y = model.preprocess_data(data, target_col, feature_cols)
                    results = model.train(X, y)
                    
                    # Store results
                    st.session_state.model_results = results
                    
                    # Display results
                    st.success("Survival prediction complete! Check out your chances below.")
                    
                    # Show accuracy metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Training Error (RMSE)", f"{results['train_rmse']:.4f}")
                    with col2:
                        st.metric("Testing Error (RMSE)", f"{results['test_rmse']:.4f}")
                    
                    # Show prediction plot
                    st.write("### Survival Prediction Results")
                    fig = model.plot_predictions(results['X_test'], results['y_test'], results['y_pred'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show feature importance
                    st.write("### Factors Most Critical for Survival")
                    importance_fig = model.feature_importance_plot(feature_cols)
                    st.plotly_chart(importance_fig, use_container_width=True)
                    
                    # Zombie-themed interpretation
                    st.write("### Survival Strategy Insights")
                    st.markdown("""
                    🧟‍♂️ **Zombie Apocalypse Investment Advice** 🧟‍♀️
                    
                    Based on our undead analysis, your best strategy is to:
                    
                    1. Focus on resources with high stability factors
                    2. Avoid volatile assets - they attract the horde!
                    3. Build a diversified portfolio of survival essentials
                    
                    Remember: In the apocalypse, past performance is still no guarantee of future survival, 
                    but it's better than being caught unprepared!
                    """)
            else:
                st.error("Please select at least one feature for prediction.")
    
    elif st.session_state.theme == "futuristic":
        # Logistic Regression for Futuristic theme
        st.write("### SELECT CLASSIFICATION PARAMETERS")
        
        # For logistic regression, target should be binary/categorical
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # If we have some numeric columns, we can binarize one for classification
        if len(numeric_cols) > 0:
            # Add option to binarize a numeric column
            st.write("#### TARGET CLASSIFICATION VARIABLE")
            
            target_options = categorical_cols + ["[Binarize a numeric column]"]
            target_type = st.selectbox("Select target type:", target_options, index=len(target_options)-1 if len(target_options) > 0 else 0)
            
            if target_type == "[Binarize a numeric column]":
                target_col = st.selectbox(
                    "Select numeric column to binarize:",
                    options=numeric_cols,
                    index=0
                )
                
                # Preview the distribution to help set threshold
                if st.checkbox("Show distribution to help set threshold"):
                    fig = px.histogram(data, x=target_col, title=f"Distribution of {target_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Allow user to set threshold for binarization
                threshold = st.slider(
                    "Set threshold value for classification:",
                    float(data[target_col].min()),
                    float(data[target_col].max()),
                    float(data[target_col].median())
                )
                
                # Create binary target
                binary_target_name = f"{target_col}_binary"
                data[binary_target_name] = (data[target_col] > threshold).astype(int)
                st.write(f"Created binary target: {binary_target_name} (1 if {target_col} > {threshold}, else 0)")
                
                # Set this as the target
                target_col = binary_target_name
            else:
                # Use the selected categorical column
                target_col = target_type
        else:
            st.error("No numeric columns available for classification.")
            target_col = None
            
        # Feature selection
        if target_col is not None:
            feature_cols = st.multiselect(
                "SELECT PREDICTOR VARIABLES:",
                options=[col for col in numeric_cols if col != target_col and col != target_col.replace("_binary", "")],
                default=recommendations["numeric_features"][:3]
            )
            
            if st.button("INITIATE QUANTUM CLASSIFICATION"):
                if len(feature_cols) > 0:
                    with st.spinner("NEURAL NETWORK COMPUTING IN PROGRESS... PLEASE STANDBY..."):
                        # Initialize and train model
                        model = FuturisticLogisticRegression()
                        X, y = model.preprocess_data(data, target_col, feature_cols)
                        results = model.train(X, y)
                        
                        # Store results
                        st.session_state.model_results = results
                        
                        # Display results
                        st.success("CLASSIFICATION COMPLETE! RESULTS AVAILABLE.")
                        
                        # Show accuracy metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("TRAINING ACCURACY", f"{results['train_accuracy']:.4f}")
                        with col2:
                            st.metric("TESTING ACCURACY", f"{results['test_accuracy']:.4f}")
                        
                        # Show decision boundary
                        st.write("### NEURAL DECISION BOUNDARY VISUALIZATION")
                        fig = model.plot_decision_boundary(X, y, feature_cols)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show feature importance
                        st.write("### FEATURE SIGNIFICANCE MATRIX")
                        importance_fig = model.feature_importance_plot(feature_cols)
                        st.plotly_chart(importance_fig, use_container_width=True)
                        
                        # Futuristic-themed interpretation
                        st.write("### PREDICTIVE INSIGHT ANALYSIS")
                        st.markdown("""
                        🚀 **QUANTUM FINANCIAL ADVISORY PROTOCOL** 🌌
                        
                        The AI has analyzed all possible timelines and determined:
                        
                        1. Market classification accuracy is within optimal parameters
                        2. Quantum fluctuations in key indicators suggest emerging patterns
                        3. Suggested action: Deploy capital according to classification model outputs
                        
                        NOTICE: This advisory complies with Temporal Directive 7.3.2 - No explicit future event predictions permitted.
                        """)
                else:
                    st.error("ERROR: MINIMUM ONE FEATURE REQUIRED FOR PREDICTION MATRIX.")
    
    elif st.session_state.theme == "got":
        # K-Means Clustering for Game of Thrones theme
        st.write("### Choose Your Market Kingdoms")
        
        # For clustering, we need numeric features
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) > 0:
            feature_cols = st.multiselect(
                "Select features for the Seven Kingdoms clustering:",
                options=numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
            
            # Number of clusters
            n_clusters = st.slider("Number of Great Houses (clusters):", 2, 7, 3)
            
            if st.button("Summon the Raven's Wisdom"):
                if len(feature_cols) > 0:
                    with st.spinner("The Three-Eyed Raven is examining the markets..."):
                        # Initialize and train model
                        model = GOTKMeansClustering()
                        X, feature_names = model.preprocess_data(data, feature_cols)
                        results = model.train(X, n_clusters)
                        
                        # Store results
                        st.session_state.model_results = results
                        
                        # Display results
                        st.success("The Raven's vision is complete! Behold the houses of the market.")
                        
                        # Show silhouette score
                        st.metric("Kingdom Separation (Silhouette Score)", f"{results['silhouette_score']:.4f}")
                        
                        # Show clusters visualization
                        st.write("### The Seven Kingdoms Market Map")
                        fig = model.plot_clusters(X, feature_cols)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show elbow method
                        st.write("### The Small Council's Recommendation")
                        elbow_fig = model.elbow_method_plot(X)
                        st.plotly_chart(elbow_fig, use_container_width=True)
                        
                        # Add data with cluster labels
                        cluster_data = data.copy()
                        cluster_data['House'] = results['labels']
                        
                        # Map cluster numbers to house names
                        house_names = ['House Lannister', 'House Baratheon', 'House Stark', 
                                      'House Targaryen', 'House Greyjoy', 'House Tyrell', 'House Martell']
                        cluster_data['House'] = cluster_data['House'].apply(lambda x: house_names[x % len(house_names)])
                        
                        # Display clustered data
                        st.write("### Kingdom Census Data")
                        st.dataframe(cluster_data.head(10), use_container_width=True)
                        
                        # GOT-themed interpretation
                        st.write("### The Hand's Market Counsel")
                        st.markdown("""
                        🔥 **From the Royal Treasury Advisory** ❄️
                        
                        The Small Council has reviewed the market kingdoms and advises:
                        
                        1. House Lannister assets show high stability but risk of overvaluation
                        2. House Stark investments demonstrate resilience in downturns
                        3. House Targaryen holdings show volatility but high growth potential
                        
                        Remember: "When you play the game of investments, you win or you lose money - there is no middle ground."
                        """)
                else:
                    st.error("You must select at least one feature for the Raven's vision.")
        else:
            st.error("No numeric features available for clustering.")
    
    elif st.session_state.theme == "pixel":
        # K-Means Clustering for Pixel Gaming theme
        st.write("### SELECT YOUR GAME FEATURES")
        
        # For clustering, we need numeric features
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) > 0:
            feature_cols = st.multiselect(
                "SELECT FEATURES FOR MAP GENERATION:",
                options=numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
            
            # Number of clusters
            n_clusters = st.slider("SELECT NUMBER OF GAME ZONES:", 2, 8, 4)
            
            if st.button("GENERATE MARKET MAP"):
                if len(feature_cols) > 0:
                    with st.spinner("LOADING GAME WORLD... PLEASE WAIT..."):
                        # Initialize and train model
                        model = PixelKMeansClustering()
                        X, feature_names = model.preprocess_data(data, feature_cols)
                        results = model.train(X, n_clusters)
                        
                        # Store results
                        st.session_state.model_results = results
                        
                        # Display results with gaming sounds
                        st.balloons()
                        st.success("MAP GENERATED SUCCESSFULLY! EXPLORE THE MARKET ZONES!")
                        
                        # Show silhouette score as game score
                        score = int(results['silhouette_score'] * 1000)
                        st.metric("EXPLORATION SCORE", f"{score} POINTS")
                        
                        # Show clusters visualization
                        st.write("### MARKET ZONES MAP")
                        fig = model.plot_clusters(X, feature_cols)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show elbow method
                        st.write("### DIFFICULTY SELECTOR")
                        elbow_fig = model.elbow_method_plot(X)
                        st.plotly_chart(elbow_fig, use_container_width=True)
                        
                        # Add data with cluster labels
                        cluster_data = data.copy()
                        cluster_data['GameZone'] = results['labels']
                        
                        # Map cluster numbers to zone names
                        zone_names = ['Fire Zone', 'Water Zone', 'Earth Zone', 'Air Zone', 
                                     'Light Zone', 'Dark Zone', 'Electric Zone', 'Magic Zone']
                        cluster_data['GameZone'] = cluster_data['GameZone'].apply(lambda x: zone_names[x % len(zone_names)])
                        
                        # Display clustered data
                        st.write("### INVENTORY ITEMS BY ZONE")
                        st.dataframe(cluster_data.head(10), use_container_width=True)
                        
                        # Pixel-themed interpretation
                        st.write("### GAME STRATEGY GUIDE")
                        st.markdown("""
                        🎮 **MARKET QUEST STRATEGY TIPS** 👾
                        
                        CONGRATULATIONS PLAYER! YOUR MARKET MAP IS READY:
                        
                        1. FIRE ZONE: HIGH RISK, HIGH REWARD STOCKS
                        2. WATER ZONE: STABLE BLUE CHIP COMPANIES
                        3. EARTH ZONE: RELIABLE DIVIDEND PAYERS
                        4. AIR ZONE: VOLATILE BUT GROWING STARTUPS
                        
                        REMEMBER TO SAVE YOUR PROGRESS REGULARLY!
                        
                        HINT: DIVERSIFY YOUR INVENTORY ACROSS MULTIPLE ZONES FOR MAXIMUM POINTS!
                        """)
                else:
                    st.error("ERROR: YOU MUST SELECT AT LEAST ONE FEATURE TO CREATE THE MAP!")
        else:
            st.error("NO NUMERIC FEATURES AVAILABLE FOR MAPPING.")

# Add footer based on theme
st.write("---")
if st.session_state.theme == "zombie":
    st.markdown("""
    <div style='text-align: center; color: #8B0000; font-family: "Creepster", cursive;'>
        <p>Stay alive. Stay invested. The zombies are watching...</p>
        <p>© 2023 Apocalypse Financial Advisors | Use at your own risk</p>
    </div>
    """, unsafe_allow_html=True)
elif st.session_state.theme == "futuristic":
    st.markdown("""
    <div style='text-align: center; color: #00FFFF; font-family: "Orbitron", sans-serif;'>
        <p>QUANTUM PREDICTION ENGINE v3.42.1</p>
        <p>© 2150 TEMPORAL DYNAMICS CORPORATION | AUTHORIZED USE ONLY</p>
    </div>
    """, unsafe_allow_html=True)
elif st.session_state.theme == "got":
    st.markdown("""
    <div style='text-align: center; color: #FCAF12; font-family: "Cinzel", serif;'>
        <p>"Knowledge is power." - Petyr Baelish, Master of Coin</p>
        <p>© Year 303 AC, Iron Bank of Braavos | All Rights Reserved by Order of the King</p>
    </div>
    """, unsafe_allow_html=True)
elif st.session_state.theme == "pixel":
    st.markdown("""
    <div style='text-align: center; color: white; font-family: "Press Start 2P", monospace;'>
        <p>THANK YOU FOR PLAYING!</p>
        <p>© 20XX PIXELATED PROFITS INC. | INSERT COIN TO CONTINUE</p>
    </div>
    """, unsafe_allow_html=True)
