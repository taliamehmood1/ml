import streamlit as st
import base64
from assets.theme_icons import (
    get_zombie_icon, get_futuristic_icon, 
    get_got_icon, get_pixel_icon
)

def apply_zombie_theme():
    """Apply Zombie Apocalypse theme styling to Streamlit app"""
    
    # Custom CSS for zombie theme
    zombie_css = """
    <style>
    .main {
        background-color: #121212;
        color: #8B0000;
    }
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.85)), 
                          url("https://img.freepik.com/free-vector/zombie-hands-seamless-pattern-halloween-horror-background_107791-11620.jpg");
        background-size: cover;
    }
    .stMarkdown, .stText, p {
        color: #FF5733 !important;
        font-family: "Creepster", cursive;
    }
    h1, h2, h3 {
        color: #8B0000 !important;
        font-family: "Creepster", cursive;
    }
    .stButton button {
        background-color: #8B0000;
        color: #FFFFFF;
        border: 2px solid #FFFFFF;
    }
    .stButton button:hover {
        background-color: #FF5733;
        color: #000000;
    }
    .stTextInput input, .stSelectbox, .stMultiselect, .stNumberInput input {
        background-color: #2D2D2D;
        color: #FF5733;
        border: 1px solid #8B0000;
    }
    .stProgress .st-bo {
        background-color: #8B0000;
    }
    .stProgress .st-bp {
        background-color: #FF5733;
    }
    .stDataFrame {
        background-color: #2D2D2D;
        color: #FF5733;
    }
    .stAlert {
        background-color: rgba(139, 0, 0, 0.2);
        color: #FF5733;
    }
    </style>
    
    <!-- Import Creepster font -->
    <link href="https://fonts.googleapis.com/css2?family=Creepster&display=swap" rel="stylesheet">
    
    <!-- Add some blood dripping animation -->
    <div class="blood-drip" style="
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 20px;
        background: linear-gradient(to right, 
            transparent 5%, #8B0000 6%, #8B0000 8%, transparent 9%,
            transparent 15%, #8B0000 16%, #8B0000 18%, transparent 19%,
            transparent 35%, #8B0000 36%, #8B0000 38%, transparent 39%,
            transparent 45%, #8B0000 46%, #8B0000 48%, transparent 49%,
            transparent 65%, #8B0000 66%, #8B0000 68%, transparent 69%,
            transparent 75%, #8B0000 76%, #8B0000 78%, transparent 79%,
            transparent 85%, #8B0000 86%, #8B0000 88%, transparent 89%);
        z-index: 1000;
    "></div>
    """
    
    # Inject custom CSS
    st.markdown(zombie_css, unsafe_allow_html=True)
    
    # Add a themed header
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; background-color: rgba(0, 0, 0, 0.7); 
                   padding: 10px; border-radius: 5px; border: 2px solid #8B0000; margin-bottom: 20px;">
            <div style="margin-right: 10px;">{get_zombie_icon()}</div>
            <div>
                <h1 style="margin: 0; font-family: 'Creepster', cursive; color: #8B0000;">
                    Zombie Apocalypse Financial Survival Guide
                </h1>
                <p style="margin: 0; font-family: 'Creepster', cursive; color: #FF5733;">
                    When the dead rise, so will your stocks... or will they?
                </p>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )


def apply_futuristic_theme():
    """Apply Futuristic Tech theme styling to Streamlit app"""
    
    # Custom CSS for futuristic theme
    futuristic_css = """
    <style>
    @keyframes neonGlow {
        0% { text-shadow: 0 0 10px #00FFFF, 0 0 20px #00FFFF, 0 0 30px #00FFFF; }
        50% { text-shadow: 0 0 20px #00FFFF, 0 0 30px #00FFFF, 0 0 40px #00FFFF; }
        100% { text-shadow: 0 0 10px #00FFFF, 0 0 20px #00FFFF, 0 0 30px #00FFFF; }
    }
    
    .main {
        background-color: #0A1128;
        color: #00FFFF;
    }
    .stApp {
        background-image: linear-gradient(to bottom, rgba(10, 17, 40, 0.95), rgba(0, 0, 0, 0.99)),
                          url("https://img.freepik.com/free-vector/abstract-technology-particle-background_52683-25766.jpg");
        background-size: cover;
    }
    .stMarkdown, .stText, p {
        color: #00FFFF !important;
        font-family: 'Orbitron', sans-serif;
    }
    h1, h2, h3 {
        color: #7DF9FF !important;
        font-family: 'Orbitron', sans-serif;
        animation: neonGlow 2s infinite alternate;
    }
    .stButton button {
        background-color: transparent;
        color: #00FFFF;
        border: 2px solid #00FFFF;
        border-radius: 5px;
        box-shadow: 0 0 10px #00FFFF;
    }
    .stButton button:hover {
        background-color: rgba(0, 255, 255, 0.2);
        box-shadow: 0 0 20px #00FFFF;
    }
    .stTextInput input, .stSelectbox, .stMultiselect, .stNumberInput input {
        background-color: rgba(10, 17, 40, 0.7);
        color: #00FFFF;
        border: 1px solid #00FFFF;
        border-radius: 5px;
    }
    .stProgress .st-bo {
        background-color: #0A1128;
    }
    .stProgress .st-bp {
        background-color: #00FFFF;
    }
    .stDataFrame {
        background-color: rgba(10, 17, 40, 0.7);
        color: #00FFFF;
    }
    .stAlert {
        background-color: rgba(0, 255, 255, 0.1);
        color: #00FFFF;
        border: 1px solid #00FFFF;
    }
    
    /* Add futuristic grid lines to the background */
    .stApp:before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(to right, rgba(0, 255, 255, 0.1) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(0, 255, 255, 0.1) 1px, transparent 1px);
        background-size: 20px 20px;
        z-index: -1;
    }
    </style>
    
    <!-- Import Orbitron font -->
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
    """
    
    # Inject custom CSS
    st.markdown(futuristic_css, unsafe_allow_html=True)
    
    # Add a themed header
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; background-color: rgba(10, 17, 40, 0.7); 
                   padding: 10px; border-radius: 5px; border: 1px solid #00FFFF; 
                   box-shadow: 0 0 15px #00FFFF; margin-bottom: 20px;">
            <div style="margin-right: 10px;">{get_futuristic_icon()}</div>
            <div>
                <h1 style="margin: 0; font-family: 'Orbitron', sans-serif; color: #00FFFF;">
                    NEURO-FINANCE QUANTUM ANALYZER
                </h1>
                <p style="margin: 0; font-family: 'Orbitron', sans-serif; color: #7DF9FF;">
                    PREDICTIVE MARKET INTELLIGENCE SYSTEM v3.42.1
                </p>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )


def apply_got_theme():
    """Apply Game of Thrones theme styling to Streamlit app"""
    
    # Custom CSS for Game of Thrones theme
    got_css = """
    <style>
    .main {
        background-color: #1A1A1A;
        color: #FCAF12;
    }
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.85)),
                          url("https://img.freepik.com/free-vector/hand-drawn-medieval-castle-illustration_23-2151086002.jpg");
        background-size: cover;
    }
    .stMarkdown, .stText, p {
        color: #FCAF12 !important;
        font-family: 'Cinzel', serif;
    }
    h1, h2, h3 {
        color: #FCAF12 !important;
        font-family: 'Cinzel', serif;
    }
    .stButton button {
        background-color: #2D2D2D;
        color: #FCAF12;
        border: 2px solid #FCAF12;
        font-family: 'Cinzel', serif;
    }
    .stButton button:hover {
        background-color: #FCAF12;
        color: #2D2D2D;
    }
    .stTextInput input, .stSelectbox, .stMultiselect, .stNumberInput input {
        background-color: #2D2D2D;
        color: #FCAF12;
        border: 1px solid #FCAF12;
        font-family: 'Cinzel', serif;
    }
    .stProgress .st-bo {
        background-color: #2D2D2D;
    }
    .stProgress .st-bp {
        background-color: #FCAF12;
    }
    .stDataFrame {
        background-color: #2D2D2D;
        color: #FCAF12;
    }
    .stAlert {
        background-color: rgba(20, 20, 20, 0.7);
        color: #FCAF12;
        border: 1px solid #FCAF12;
    }
    
    /* Add decorative borders to containers */
    .stApp div.stBlock {
        border: 1px solid #FCAF12;
        border-radius: 0;
    }
    </style>
    
    <!-- Import Cinzel font -->
    <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&display=swap" rel="stylesheet">
    """
    
    # Inject custom CSS
    st.markdown(got_css, unsafe_allow_html=True)
    
    # Add a themed header
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; background-color: rgba(20, 20, 20, 0.8); 
                   padding: 10px; border: 2px solid #FCAF12; margin-bottom: 20px;">
            <div style="margin-right: 10px;">{get_got_icon()}</div>
            <div>
                <h1 style="margin: 0; font-family: 'Cinzel', serif; color: #FCAF12;">
                    The Iron Bank of Braavos
                </h1>
                <p style="margin: 0; font-family: 'Cinzel', serif; color: #FCAF12;">
                    "A Lannister Always Pays His Debts... And So Will Your Investments"
                </p>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )


def apply_pixel_theme():
    """Apply Pixel Gaming theme styling to Streamlit app"""
    
    # Custom CSS for pixel gaming theme
    pixel_css = """
    <style>
    @font-face {
        font-family: 'Press Start 2P';
        src: url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
    }
    
    @keyframes rainbow {
        0% { color: #FF0000; }
        17% { color: #FF7F00; }
        33% { color: #FFFF00; }
        50% { color: #00FF00; }
        67% { color: #0000FF; }
        83% { color: #4B0082; }
        100% { color: #9400D3; }
    }
    
    .main {
        background-color: #000000;
        color: #FFFFFF;
    }
    .stApp {
        background-color: #000000;
        background-image: url("data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 20 20' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23333333' fill-opacity='0.4' fill-rule='evenodd'%3E%3Ccircle cx='3' cy='3' r='1'/%3E%3Ccircle cx='13' cy='13' r='1'/%3E%3C/g%3E%3C/svg%3E");
    }
    .stMarkdown, .stText, p {
        color: #FFFFFF !important;
        font-family: 'Press Start 2P', monospace;
        font-size: 0.8rem;
    }
    h1 {
        color: #FFFFFF !important;
        font-family: 'Press Start 2P', monospace;
        text-shadow: 3px 3px 0 #FF00FF;
        font-size: 1.5rem;
        letter-spacing: 2px;
    }
    h2, h3 {
        color: #FFFFFF !important;
        font-family: 'Press Start 2P', monospace;
        animation: rainbow 5s infinite;
        font-size: 1.2rem;
    }
    .stButton button {
        background-color: #333333;
        color: #FFFFFF;
        border: 3px solid #FFFFFF;
        font-family: 'Press Start 2P', monospace;
        box-shadow: 4px 4px 0 #000000;
        border-radius: 0;
    }
    .stButton button:hover {
        background-color: #FF00FF;
        transform: translate(2px, 2px);
        box-shadow: 2px 2px 0 #000000;
    }
    .stTextInput input, .stSelectbox, .stMultiselect, .stNumberInput input {
        background-color: #000000;
        color: #00FF00;
        border: 3px solid #00FF00;
        font-family: 'Press Start 2P', monospace;
        border-radius: 0;
    }
    .stProgress .st-bo {
        background-color: #444444;
    }
    .stProgress .st-bp {
        background-color: #00FF00;
    }
    .stDataFrame {
        background-color: #000000;
        color: #00FF00;
        border: 3px solid #00FF00;
    }
    .stAlert {
        background-color: #000000;
        color: #00FF00;
        border: 3px dashed #00FF00;
    }
    </style>
    
    <!-- Import Press Start 2P font -->
    <link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">
    """
    
    # Inject custom CSS
    st.markdown(pixel_css, unsafe_allow_html=True)
    
    # Add a themed header
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; background-color: #000000; 
                   padding: 10px; border: 3px solid #FFFFFF; margin-bottom: 20px;">
            <div style="margin-right: 10px;">{get_pixel_icon()}</div>
            <div>
                <h1 style="margin: 0; font-family: 'Press Start 2P', monospace; color: #FFFFFF;">
                    STOCK MARKET QUEST
                </h1>
                <p style="margin: 0; font-family: 'Press Start 2P', monospace; color: #00FF00;">
                    PRESS START TO BEGIN TRADING
                </p>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )


def get_theme_description(theme):
    """Get description and details about each theme"""
    
    themes = {
        "zombie": {
            "title": "Zombie Apocalypse Financial Survival Guide",
            "description": """
            The world has fallen. Civilization has collapsed. But your stocks? They might still have a pulse.
            
            In this dark and eerie theme, we use Linear Regression to predict which stocks will survive the apocalypse
            and which will turn... Well, you know what happens when investments go bad in a world full of zombies.
            
            Features:
            - Linear Regression to predict future stock prices
            - Eerie visuals and spooky animations
            - Survival analysis based on market trends
            - Blood-curdling insights that might save your portfolio
            """,
            "icon": get_zombie_icon()
        },
        "futuristic": {
            "title": "NEURO-FINANCE QUANTUM ANALYZER",
            "description": """
            Welcome to the year 2150. Traditional finance is obsolete. Quantum computing has revolutionized market prediction.
            
            This high-tech interface uses advanced Logistic Regression algorithms to classify market trends with
            unprecedented accuracy. Experience the future of financial analysis with sleek visualizations and
            cybernetic insights.
            
            Features:
            - Logistic Regression for binary market classification
            - Neuro-adaptive visualization system
            - Quantum probability distribution analysis
            - High-frequency pattern recognition
            """,
            "icon": get_futuristic_icon()
        },
        "got": {
            "title": "The Iron Bank of Braavos",
            "description": """
            Winter is coming... and so is market volatility. The great houses of the market compete for dominance,
            but only the wisest investors will sit on the Iron Throne of wealth.
            
            Using the ancient wisdom of K-Means Clustering, we divide the market into houses (clusters) to identify 
            allies and enemies in your investment strategy. The Old Gods and the New favor those who understand market segments.
            
            Features:
            - K-Means Clustering for market segmentation
            - House allegiance analysis for stocks
            - The raven's market intelligence reports
            - Epic battle strategies for your portfolio
            """,
            "icon": get_got_icon()
        },
        "pixel": {
            "title": "STOCK MARKET QUEST",
            "description": """
            WELCOME ADVENTURER! Your quest to conquer the Stock Market Dungeon begins now!
            
            Navigate through the treacherous levels of financial data using your trusty K-Means Clustering spell.
            Defeat market monsters, collect golden coins, and level up your portfolio in this retro-style financial adventure.
            
            Features:
            - K-Means Clustering to identify treasure zones
            - Pixel-perfect financial charts
            - Level up system for investment strategies
            - Boss battles against market volatility
            """,
            "icon": get_pixel_icon()
        }
    }
    
    return themes.get(theme, {"title": "Unknown Theme", "description": "No description available", "icon": ""})
