import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

class ZombieLinearRegression:
    """Linear Regression model with Zombie Apocalypse theme"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def preprocess_data(self, data, target_col, feature_cols):
        """Preprocess data for linear regression"""
        if target_col not in data.columns or not all(col in data.columns for col in feature_cols):
            raise ValueError("Target or feature columns not found in data")
            
        # Extract features and target
        X = data[feature_cols]
        y = data[target_col]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train(self, X, y):
        """Train the linear regression model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate metrics
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'coefficients': self.model.coef_,
            'intercept': self.model.intercept_,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': test_pred
        }
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Scale input features if not already scaled
        if isinstance(X, pd.DataFrame):
            X = self.scaler.transform(X)
            
        return self.model.predict(X)
    
    def plot_predictions(self, X_test, y_test, y_pred, theme="zombie"):
        """Create spooky zombie-themed prediction plot"""
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=list(range(len(y_test))),
            y=y_test,
            mode='markers',
            name='Actual Values',
            marker=dict(color='#8B0000', size=8),
        ))
        
        # Add predicted values
        fig.add_trace(go.Scatter(
            x=list(range(len(y_pred))),
            y=y_pred,
            mode='lines',
            name='Predicted Values',
            line=dict(color='#32CD32', width=2)
        ))
        
        # Update layout for zombie theme
        fig.update_layout(
            title="Stock Price Predictions (Survive or Die)",
            xaxis_title="Time (Days until doom)",
            yaxis_title="Stock Price (Survival Currency)",
            paper_bgcolor='rgba(0, 0, 0, 0.8)',
            plot_bgcolor='rgba(0, 0, 0, 0.8)',
            font=dict(color='#FF5733'),
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(
                font=dict(color='#FF5733')
            )
        )
        
        return fig
    
    def feature_importance_plot(self, feature_names):
        """Plot feature importance with zombie theme"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
            
        importance = np.abs(self.model.coef_)
        
        # Create dataframe for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Create bar chart with zombie theme
        fig = px.bar(
            importance_df, 
            x='Feature', 
            y='Importance',
            title='Feature Importance for Survival Prediction',
            color_discrete_sequence=['#8B0000']
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0, 0, 0, 0.8)',
            plot_bgcolor='rgba(0, 0, 0, 0.8)',
            font=dict(color='#FF5733'),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        return fig


class FuturisticLogisticRegression:
    """Logistic Regression model with Futuristic Tech theme"""
    
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def preprocess_data(self, data, target_col, feature_cols):
        """Preprocess data for logistic regression"""
        if target_col not in data.columns or not all(col in data.columns for col in feature_cols):
            raise ValueError("Target or feature columns not found in data")
            
        # Extract features and target
        X = data[feature_cols]
        y = data[target_col]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode target if it's categorical
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        
        return X_scaled, y
    
    def train(self, X, y):
        """Train the logistic regression model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Calculate probabilities
        train_proba = self.model.predict_proba(X_train)
        test_proba = self.model.predict_proba(X_test)
        
        # Calculate accuracy
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'coefficients': self.model.coef_,
            'intercept': self.model.intercept_,
            'classes': self.model.classes_,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': test_pred,
            'y_proba': test_proba
        }
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Scale input features if not already scaled
        if isinstance(X, pd.DataFrame):
            X = self.scaler.transform(X)
            
        return self.model.predict(X)
    
    def plot_decision_boundary(self, X, y, feature_names, theme="futuristic"):
        """Plot decision boundary with futuristic theme"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # For visualization, we'll use PCA to reduce to 2D if more than 2 features
        from sklearn.decomposition import PCA
        
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            x_label = "Component 1"
            y_label = "Component 2"
        else:
            X_2d = X
            x_label = feature_names[0]
            y_label = feature_names[1]
        
        # Create a meshgrid for decision boundary
        h = 0.02  # step size
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Get predictions for each point in the meshgrid
        if X.shape[1] > 2:
            # For PCA transformed data, we need to transform the mesh points as well
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            Z = self.model.predict(pca.inverse_transform(mesh_points))
        else:
            Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        
        Z = Z.reshape(xx.shape)
        
        # Create plot with futuristic theme
        fig = go.Figure()
        
        # Add decision boundary contour
        fig.add_trace(go.Contour(
            z=Z,
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            colorscale='Viridis',
            opacity=0.5,
            showscale=False
        ))
        
        # Add scatter plot of the data points
        for i, label in enumerate(np.unique(y)):
            indices = y == label
            fig.add_trace(go.Scatter(
                x=X_2d[indices, 0],
                y=X_2d[indices, 1],
                mode='markers',
                name=f'Class {label}',
                marker=dict(
                    size=10,
                    color=['#00FFFF', '#FF00FF', '#FFFF00', '#00FF00'][i % 4],
                    line=dict(width=1)
                )
            ))
        
        # Update layout for futuristic theme
        fig.update_layout(
            title="Neural Network Decision Matrix",
            xaxis_title=x_label,
            yaxis_title=y_label,
            paper_bgcolor='rgba(0, 10, 30, 0.9)',
            plot_bgcolor='rgba(0, 10, 30, 0.9)',
            font=dict(color='#00FFFF'),
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(
                font=dict(color='#00FFFF')
            )
        )
        
        return fig
    
    def feature_importance_plot(self, feature_names):
        """Plot feature importance with futuristic theme"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
            
        # For multi-class, we'll use the mean absolute coefficient values
        importance = np.mean(np.abs(self.model.coef_), axis=0)
        
        # Create dataframe for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Create bar chart with futuristic theme
        fig = px.bar(
            importance_df, 
            x='Feature', 
            y='Importance',
            title='Quantum Neural Coefficient Analysis',
            color_discrete_sequence=['#00FFFF']
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0, 10, 30, 0.9)',
            plot_bgcolor='rgba(0, 10, 30, 0.9)',
            font=dict(color='#00FFFF'),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        return fig


class GOTKMeansClustering:
    """K-Means Clustering model with Game of Thrones theme"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.n_clusters = 3  # Default to 3 houses
        
    def preprocess_data(self, data, feature_cols):
        """Preprocess data for clustering"""
        if not all(col in data.columns for col in feature_cols):
            raise ValueError("Feature columns not found in data")
            
        # Extract features
        X = data[feature_cols]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, X.columns.tolist()
    
    def train(self, X, n_clusters=3):
        """Train the K-Means clustering model"""
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.model.fit_predict(X)
        self.is_trained = True
        
        # Calculate silhouette score
        silhouette = silhouette_score(X, clusters) if len(np.unique(clusters)) > 1 else 0
        
        return {
            'silhouette_score': silhouette,
            'cluster_centers': self.model.cluster_centers_,
            'inertia': self.model.inertia_,
            'labels': clusters
        }
    
    def predict(self, X):
        """Assign clusters to new data"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Scale input features if not already scaled
        if isinstance(X, pd.DataFrame):
            X = self.scaler.transform(X)
            
        return self.model.predict(X)
    
    def plot_clusters(self, X, feature_names, theme="got"):
        """Plot clusters with Game of Thrones theme"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # For visualization, we'll use PCA to reduce to 2D if more than 2 features
        from sklearn.decomposition import PCA
        
        # Get cluster labels
        clusters = self.model.labels_
        
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            centers_2d = pca.transform(self.model.cluster_centers_)
            x_label = "First Principle Component"
            y_label = "Second Principle Component"
        else:
            X_2d = X
            centers_2d = self.model.cluster_centers_
            x_label = feature_names[0]
            y_label = feature_names[1]
        
        # Define House colors and names
        got_colors = ['#C7100B', '#FCAF12', '#B2B2B2', '#1A1A1A', '#5D84A3', '#2A632C']
        house_names = ['House Lannister', 'House Baratheon', 'House Stark', 'House Targaryen', 'House Greyjoy', 'House Tyrell']
        
        # Create plot with GOT theme
        fig = go.Figure()
        
        # Add scatter plot for each cluster
        for i in range(self.n_clusters):
            cluster_points = X_2d[clusters == i]
            fig.add_trace(go.Scatter(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                mode='markers',
                name=house_names[i % len(house_names)],
                marker=dict(
                    size=10,
                    color=got_colors[i % len(got_colors)],
                    symbol='circle',
                    line=dict(width=1, color='black')
                )
            ))
        
        # Add cluster centers
        fig.add_trace(go.Scatter(
            x=centers_2d[:, 0],
            y=centers_2d[:, 1],
            mode='markers',
            name='House Capitals',
            marker=dict(
                size=15,
                color='black',
                symbol='star',
                line=dict(width=2, color='white')
            )
        ))
        
        # Update layout for GOT theme
        fig.update_layout(
            title="Seven Kingdoms Market Segmentation",
            xaxis_title=x_label,
            yaxis_title=y_label,
            paper_bgcolor='rgba(50, 50, 50, 0.9)',
            plot_bgcolor='rgba(50, 50, 50, 0.9)',
            font=dict(color='#FCAF12', family="serif"),
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(
                font=dict(color='#FCAF12', family="serif")
            )
        )
        
        return fig
    
    def elbow_method_plot(self, X):
        """Plot elbow method with GOT theme"""
        if not isinstance(self.model, KMeans):
            self.model = KMeans(random_state=42)
            
        # Calculate inertia for different numbers of clusters
        inertia = []
        k_range = range(1, 10)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)
        
        # Create elbow plot with GOT theme
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(k_range),
            y=inertia,
            mode='lines+markers',
            name='Inertia',
            line=dict(color='#C7100B', width=3),
            marker=dict(
                size=10,
                color='#FCAF12',
                symbol='circle',
                line=dict(width=2, color='#C7100B')
            )
        ))
        
        # Update layout for GOT theme
        fig.update_layout(
            title="Battle of the Five Kings: Choosing the Right Number of Houses",
            xaxis_title="Number of Houses (k)",
            yaxis_title="Inertia (Within-Cluster Sum of Squares)",
            paper_bgcolor='rgba(50, 50, 50, 0.9)',
            plot_bgcolor='rgba(50, 50, 50, 0.9)',
            font=dict(color='#FCAF12', family="serif"),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        return fig


class PixelKMeansClustering:
    """K-Means Clustering model with Pixel Gaming theme"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.n_clusters = 4  # Default to 4 game zones
        
    def preprocess_data(self, data, feature_cols):
        """Preprocess data for clustering"""
        if not all(col in data.columns for col in feature_cols):
            raise ValueError("Feature columns not found in data")
            
        # Extract features
        X = data[feature_cols]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, X.columns.tolist()
    
    def train(self, X, n_clusters=4):
        """Train the K-Means clustering model"""
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.model.fit_predict(X)
        self.is_trained = True
        
        # Calculate silhouette score
        silhouette = silhouette_score(X, clusters) if len(np.unique(clusters)) > 1 else 0
        
        return {
            'silhouette_score': silhouette,
            'cluster_centers': self.model.cluster_centers_,
            'inertia': self.model.inertia_,
            'labels': clusters
        }
    
    def predict(self, X):
        """Assign clusters to new data"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Scale input features if not already scaled
        if isinstance(X, pd.DataFrame):
            X = self.scaler.transform(X)
            
        return self.model.predict(X)
    
    def plot_clusters(self, X, feature_names, theme="pixel"):
        """Plot clusters with Pixel Gaming theme"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # For visualization, we'll use PCA to reduce to 2D if more than 2 features
        from sklearn.decomposition import PCA
        
        # Get cluster labels
        clusters = self.model.labels_
        
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            centers_2d = pca.transform(self.model.cluster_centers_)
            x_label = "X Position"
            y_label = "Y Position"
        else:
            X_2d = X
            centers_2d = self.model.cluster_centers_
            x_label = feature_names[0]
            y_label = feature_names[1]
        
        # Define gaming zone colors and names
        pixel_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
        zone_names = ['Fire Zone', 'Forest Zone', 'Water Zone', 'Desert Zone', 'Magic Zone', 'Ice Zone']
        
        # Create plot with Pixel Gaming theme
        fig = go.Figure()
        
        # Add scatter plot for each cluster
        for i in range(self.n_clusters):
            cluster_points = X_2d[clusters == i]
            fig.add_trace(go.Scatter(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                mode='markers',
                name=zone_names[i % len(zone_names)],
                marker=dict(
                    size=12,
                    color=pixel_colors[i % len(pixel_colors)],
                    symbol='square',
                    line=dict(width=1, color='black')
                )
            ))
        
        # Add cluster centers
        fig.add_trace(go.Scatter(
            x=centers_2d[:, 0],
            y=centers_2d[:, 1],
            mode='markers',
            name='Power-Ups',
            marker=dict(
                size=20,
                color='white',
                symbol='star',
                line=dict(width=2, color='black')
            )
        ))
        
        # Update layout for Pixel Gaming theme
        fig.update_layout(
            title="Market Zones Map - Level 1",
            xaxis_title=x_label,
            yaxis_title=y_label,
            paper_bgcolor='rgba(0, 0, 0, 0.9)',
            plot_bgcolor='rgba(0, 0, 0, 0.9)',
            font=dict(color='white', family="monospace"),
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(
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
        
        return fig
    
    def elbow_method_plot(self, X):
        """Plot elbow method with Pixel Gaming theme"""
        if not isinstance(self.model, KMeans):
            self.model = KMeans(random_state=42)
            
        # Calculate inertia for different numbers of clusters
        inertia = []
        k_range = range(1, 10)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)
        
        # Create elbow plot with Pixel Gaming theme
        fig = go.Figure()
        
        # Add line with pixel-like steps
        for i in range(len(k_range)-1):
            fig.add_trace(go.Scatter(
                x=[k_range[i], k_range[i+1]],
                y=[inertia[i], inertia[i+1]],
                mode='lines',
                line=dict(color='#FF00FF', width=4),
                showlegend=False
            ))
            
        # Add markers
        fig.add_trace(go.Scatter(
            x=list(k_range),
            y=inertia,
            mode='markers',
            name='Level Difficulty',
            marker=dict(
                size=15,
                color='#00FFFF',
                symbol='square',
                line=dict(width=2, color='#FFFFFF')
            )
        ))
        
        # Update layout for Pixel Gaming theme
        fig.update_layout(
            title="Choose Your Difficulty Level",
            xaxis_title="Number of Game Zones (k)",
            yaxis_title="Score (Lower is Better)",
            paper_bgcolor='rgba(0, 0, 0, 0.9)',
            plot_bgcolor='rgba(0, 0, 0, 0.9)',
            font=dict(color='white', family="monospace"),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # Add grid lines to make it look more like a pixel game
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickmode='linear',
            tick0=1,
            dtick=1
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255, 255, 255, 0.1)',
        )
        
        return fig
