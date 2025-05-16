import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import io
import base64

def load_sample_data(theme="default"):
    """Load sample data if user doesn't upload their own"""
    
    # Create sample datasets with financial data based on themes
    if theme == "zombie":
        # Sample financial data with a zombie apocalypse twist
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create synthetic data - declining trend with volatility (apocalypse scenario)
        np.random.seed(42)
        base_price = 100
        trend = np.linspace(0, -30, 100)  # Declining trend
        volatility = np.random.normal(0, 5, 100)  # High volatility
        disaster_points = np.zeros(100)
        disaster_points[[20, 50, 80]] = [-15, -20, -25]  # Major "outbreak" events
        
        prices = base_price + trend + volatility + disaster_points
        prices = np.maximum(prices, 1)  # Ensure no negative prices
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'StockPrice': prices,
            'SurvivalIndex': prices / 2 + np.random.normal(0, 3, 100),
            'InfectionRate': 100 - prices / 3 + np.random.normal(0, 2, 100),
            'SupplyShortage': np.random.uniform(0, 100, 100),
            'Volume': np.random.uniform(1000, 10000, 100) * (100 - trend) / 100
        })
        
        # Ensure data makes sense
        df['SurvivalIndex'] = np.maximum(df['SurvivalIndex'], 0)
        df['InfectionRate'] = np.clip(df['InfectionRate'], 0, 100)
        
    elif theme == "futuristic":
        # Sample data with a futuristic high-tech theme
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create synthetic data - various market conditions
        np.random.seed(42)
        base_price = 100
        trend = np.concatenate([
            np.linspace(0, 20, 40),   # Uptrend
            np.linspace(20, 0, 30),   # Downtrend
            np.linspace(0, 10, 30)    # Recovery
        ])
        
        volatility = np.random.normal(0, 3, 100) 
        tech_disruption = np.zeros(100)
        tech_disruption[[25, 55, 85]] = [10, -15, 12]  # Tech breakthrough events
        
        prices = base_price + trend + volatility + tech_disruption
        
        # Market conditions: Bull(1) or Bear(0)
        market_conditions = np.ones(100)
        market_conditions[40:70] = 0  # Bear market in the middle
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'StockPrice': prices,
            'TechIndex': prices * 0.8 + np.random.normal(0, 5, 100),
            'QuantumStability': np.random.uniform(60, 100, 100),
            'MarketCondition': market_conditions,
            'AIConfidence': np.clip(prices / 2 + np.random.normal(10, 5, 100), 0, 100),
            'Volume': np.random.uniform(5000, 20000, 100)
        })
        
    elif theme == "got":
        # Sample data with Game of Thrones theme
        houses = ['Stark', 'Lannister', 'Targaryen', 'Baratheon', 'Tyrell', 'Greyjoy', 'Martell']
        stocks = []
        
        # Create 200 random stocks distributed among houses
        np.random.seed(42)
        for i in range(200):
            house = np.random.choice(houses)
            price = np.random.uniform(10, 500)
            volume = np.random.uniform(1000, 100000)
            volatility = np.random.uniform(0.1, 0.5)
            loyalty = np.random.uniform(0, 100)
            war_impact = np.random.uniform(-50, 50)
            winter_resilience = np.random.uniform(0, 100)
            
            # Adjust characteristics based on houses
            if house == 'Stark':
                winter_resilience += 30
                price *= 0.8
            elif house == 'Lannister':
                price *= 1.5
                loyalty += 20
            elif house == 'Targaryen':
                volatility += 0.3
                war_impact += 20
            
            stocks.append({
                'StockName': f"{house}_{i}",
                'House': house,
                'Price': price,
                'Volume': volume,
                'Volatility': volatility,
                'HouseLoyalty': loyalty,
                'WarImpact': war_impact,
                'WinterResilience': winter_resilience
            })
        
        df = pd.DataFrame(stocks)
        
    elif theme == "pixel":
        # Sample data with pixel gaming theme
        game_zones = ['Fire', 'Water', 'Earth', 'Air', 'Light', 'Dark']
        game_items = []
        
        # Create 200 random game items
        np.random.seed(42)
        for i in range(200):
            zone = np.random.choice(game_zones)
            value = np.random.uniform(1, 100)
            rarity = np.random.uniform(1, 10)
            power = np.random.uniform(1, 50)
            durability = np.random.uniform(1, 100)
            player_rating = np.random.uniform(1, 5)
            
            # Adjust characteristics based on zones
            if zone == 'Fire':
                power += 10
                durability -= 5
            elif zone == 'Water':
                durability += 20
            elif zone == 'Light':
                value += 20
                rarity += 2
            
            game_items.append({
                'ItemName': f"{zone}Item_{i}",
                'GameZone': zone,
                'Value': value,
                'Rarity': rarity,
                'Power': power,
                'Durability': durability,
                'PlayerRating': player_rating
            })
        
        df = pd.DataFrame(game_items)
        
    else:
        # Default dataset - simple stock data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.normal(0, 1, 100))
        df = pd.DataFrame({'Date': dates, 'StockPrice': prices})
    
    return df

def fetch_stock_data(ticker, period="1y", theme="default"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            st.error(f"No data found for ticker {ticker}")
            return None
            
        # Add theme-specific columns based on the real data
        if theme == "zombie":
            hist['SurvivalIndex'] = hist['Close'] / 2 + np.random.normal(0, hist['Close'].std() * 0.1, len(hist))
            hist['InfectionRate'] = 100 - hist['Close'] / 3 + np.random.normal(0, 2, len(hist))
            hist['SupplyShortage'] = np.random.uniform(0, 100, len(hist))
            
            # Ensure data makes sense
            hist['SurvivalIndex'] = np.maximum(hist['SurvivalIndex'], 0)
            hist['InfectionRate'] = np.clip(hist['InfectionRate'], 0, 100)
            
        elif theme == "futuristic":
            price_mean = hist['Close'].mean()
            hist['TechIndex'] = hist['Close'] * 0.8 + np.random.normal(0, hist['Close'].std() * 0.1, len(hist))
            hist['QuantumStability'] = np.random.uniform(60, 100, len(hist))
            hist['AIConfidence'] = np.clip(hist['Close'] / price_mean * 50 + np.random.normal(10, 5, len(hist)), 0, 100)
            
            # Calculate MarketCondition based on price momentum
            hist['PriceChange'] = hist['Close'].pct_change()
            hist['MarketCondition'] = (hist['PriceChange'] > 0).astype(int)  # 1 for positive returns, 0 for negative
            
        elif theme == "got":
            # Add House information
            houses = ['Stark', 'Lannister', 'Targaryen', 'Baratheon', 'Tyrell', 'Greyjoy', 'Martell']
            hist['House'] = random.choice(houses)
            hist['HouseLoyalty'] = np.random.uniform(60, 100, len(hist))
            hist['WarImpact'] = hist['Close'].pct_change() * 100
            hist['WinterResilience'] = np.random.uniform(60, 100, len(hist))
            
        elif theme == "pixel":
            # Add game-like metrics
            zones = ['Fire', 'Water', 'Earth', 'Air', 'Light', 'Dark']
            hist['GameZone'] = random.choice(zones)
            price_normalized = (hist['Close'] - hist['Close'].min()) / (hist['Close'].max() - hist['Close'].min())
            hist['Power'] = price_normalized * 50 + np.random.uniform(1, 10, len(hist))
            hist['Rarity'] = np.clip(price_normalized * 10, 1, 10)
            hist['PlayerRating'] = np.clip(price_normalized * 5, 1, 5)
        
        # Reset index to make Date a column
        hist = hist.reset_index()
        return hist
        
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def plot_stock_data(data, theme="default"):
    """Create a themed plot of stock data"""
    if data is None or len(data) == 0:
        return None
    
    if theme == "zombie":
        # Zombie apocalypse themed stock chart
        fig = go.Figure()
        
        # Add close price line
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['Close'],
            mode='lines',
            name='Stock Price',
            line=dict(color='#8B0000', width=2)
        ))
        
        # Add volume as bars at the bottom
        fig.add_trace(go.Bar(
            x=data['Date'],
            y=data['Volume'] / data['Volume'].max() * data['Close'].min() / 2,
            opacity=0.3,
            name='Trading Volume',
            marker=dict(color='#32CD32')
        ))
        
        # Update layout for zombie theme
        fig.update_layout(
            title="Apocalypse Market Tracker: SURVIVE OR DIE",
            xaxis_title="Days Since Outbreak",
            yaxis_title="Survival Currency Value",
            template="plotly_dark",
            paper_bgcolor='rgba(0, 0, 0, 0.8)',
            plot_bgcolor='rgba(0, 0, 0, 0.8)',
            font=dict(color='#FF5733'),
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color='#FF5733')
            )
        )
        
    elif theme == "futuristic":
        # Futuristic tech themed stock chart
        fig = go.Figure()
        
        # Add close price line with glowing effect
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['Close'],
            mode='lines',
            name='Stock Price',
            line=dict(color='#00FFFF', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 255, 0.1)'
        ))
        
        # Add volume as bars
        fig.add_trace(go.Bar(
            x=data['Date'],
            y=data['Volume'] / data['Volume'].max() * data['Close'].min() / 2,
            opacity=0.3,
            name='Trading Volume',
            marker=dict(color='#FF00FF')
        ))
        
        # Update layout for futuristic theme
        fig.update_layout(
            title="QUANTUM TEMPORAL PRICE ANALYSIS",
            xaxis_title="TEMPORAL COORDINATE",
            yaxis_title="ASSET VALUATION METRIC",
            template="plotly_dark",
            paper_bgcolor='rgba(0, 10, 30, 0.9)',
            plot_bgcolor='rgba(0, 10, 30, 0.9)',
            font=dict(color='#00FFFF'),
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color='#00FFFF')
            )
        )
        
        # Add grid lines
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0, 255, 255, 0.1)',
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0, 255, 255, 0.1)',
        )
        
    elif theme == "got":
        # Game of Thrones themed stock chart
        fig = go.Figure()
        
        # Add close price line
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['Close'],
            mode='lines',
            name='Gold Dragons',
            line=dict(color='#FCAF12', width=3)
        ))
        
        # Add volume as bars
        fig.add_trace(go.Bar(
            x=data['Date'],
            y=data['Volume'] / data['Volume'].max() * data['Close'].min() / 2,
            opacity=0.3,
            name='Trade Volume',
            marker=dict(color='#C7100B')
        ))
        
        # Update layout for GOT theme
        fig.update_layout(
            title="Kingdom Treasury: Value of Gold Dragons",
            xaxis_title="Reign Timeline",
            yaxis_title="Value in Gold Dragons",
            template="plotly_dark",
            paper_bgcolor='rgba(50, 50, 50, 0.9)',
            plot_bgcolor='rgba(50, 50, 50, 0.9)',
            font=dict(color='#FCAF12', family="serif"),
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color='#FCAF12', family="serif")
            )
        )
        
    elif theme == "pixel":
        # Pixel gaming themed stock chart
        fig = go.Figure()
        
        # Convert to pixel-style step chart
        dates = data['Date'].tolist()
        prices = data['Close'].tolist()
        
        # Create step-like data for pixel effect
        step_x = []
        step_y = []
        
        for i in range(len(dates)):
            step_x.append(dates[i])
            if i > 0:
                step_x.append(dates[i])
                step_y.append(prices[i-1])
            step_y.append(prices[i])
        
        # Add pixel-style line
        fig.add_trace(go.Scatter(
            x=step_x,
            y=step_y,
            mode='lines',
            name='Coin Value',
            line=dict(color='#00FF00', width=3, shape='hv'),
        ))
        
        # Add volume as pixel-style bars
        fig.add_trace(go.Bar(
            x=data['Date'],
            y=data['Volume'] / data['Volume'].max() * data['Close'].min() / 2,
            opacity=0.6,
            name='Trades',
            marker=dict(color='#FF00FF')
        ))
        
        # Update layout for pixel theme
        fig.update_layout(
            title="MARKET ADVENTURE - LEVEL 1",
            xaxis_title="GAME TIME",
            yaxis_title="GOLD COINS",
            template="plotly_dark",
            paper_bgcolor='rgba(0, 0, 0, 0.9)',
            plot_bgcolor='rgba(0, 0, 0, 0.9)',
            font=dict(color='white', family="monospace"),
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color='white', family="monospace")
            )
        )
        
        # Add grid lines to make it look more like a pixel game
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255, 255, 255, 0.1)',
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255, 255, 255, 0.1)',
        )
    
    else:
        # Default stock chart
        fig = px.line(
            data, 
            x='Date', 
            y='Close',
            title=f"Stock Price History"
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price",
            margin=dict(l=10, r=10, t=50, b=10),
        )
    
    return fig
