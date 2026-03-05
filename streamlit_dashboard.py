import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import random
import numpy as np

# --- Constants and Configuration ---
CELL_TOWERS = ["cell0001", "cell0012", "cell0015", "cell0020"]
DATA_FILE_PATH = 'datasets/df_daily_sample.csv'

# --- Translation Dictionary for Multilingual Support ---
TRANSLATIONS = {
    "English": {
        "title": "LTE Traffic & Congestion Monitoring App",
        "lang_select": "Select Website Language",
        "max_volume": "Max Volume Alert Threshold (Bytes)",
        "max_congestion": "Max Congestion Alert Threshold (%)",
        "predict_button": "Get Prediction & Alert",
        "pred_date": "Prediction Date",
        "historical_title": "Historical Traffic Volume",
        "volume_alert": "🚨 **HIGH VOLUME ALERT**: Predicted traffic volume ({:.0f}) exceeds the volume threshold of {:.0f}!",
        "volume_safe": "✅ Predicted traffic volume is {:.0f} (below volume threshold).",
        "congestion_alert": "⚠️ **HIGH CONGESTION ALERT**: Predicted congestion ({:.1f}%) exceeds the congestion threshold of {:.0f}%!",
        "congestion_safe": "👍 Predicted congestion is {:.1f}% (below congestion threshold).",
        "pred_volume_label": "Predicted Volume (Bytes):",
        "pred_congestion_label": "Predicted Congestion (%):",
        "tower_code": "Cell Tower Code: {}",
        "load_error": "Error loading data. Check if the CSV file exists and is correctly formatted.",
        "placeholder_date": datetime.date.today() + datetime.timedelta(days=1),
        "alert_volume_info": "Volume alert triggers if prediction > **{:,} Bytes**",
        "alert_congestion_info": "Congestion alert triggers if prediction > **{:,}%**"
    },
    "French": {
        "title": "Application de Surveillance du Trafic et de la Congestion LTE",
        "lang_select": "Sélectionner la Langue du Site",
        "max_volume": "Seuil d'Alerte de Volume Max (Octets)",
        "max_congestion": "Seuil d'Alerte de Congestion Max (%)",
        "predict_button": "Obtenir la Prédiction et l'Alerte",
        "pred_date": "Date de Prédiction",
        "historical_title": "Volume de Trafic Historique",
        "volume_alert": "🚨 **ALERTE VOLUME ÉLEVÉ** : Le volume de trafic prédit ({:.0f}) dépasse le seuil de volume maximum de {:.0f} !",
        "volume_safe": "✅ Le volume de trafic prédit est de {:.0f} (sous le seuil de volume).",
        "congestion_alert": "⚠️ **ALERTE CONGESTION ÉLEVÉE** : La congestion prédite ({:.1f}%) dépasse le seuil de congestion de {:.0f}% !",
        "congestion_safe": "👍 La congestion prédite est de {:.1f}% (sous le seuil de congestion).",
        "pred_volume_label": "Volume Prédit (Octets) :",
        "pred_congestion_label": "Congestion Prédite (%) :",
        "tower_code": "Code de la Tour Cellulaire : {}",
        "load_error": "Erreur lors du chargement des données. Vérifiez si le fichier CSV existe et est correctement formaté.",
        "placeholder_date": datetime.date.today() + datetime.timedelta(days=1),
        "alert_volume_info": "Alerte de volume déclenchée si prédiction > **{:,} Octets**",
        "alert_congestion_info": "Alerte de congestion déclenchée si prédiction > **{:,}%**"
    }
}

# --- Custom CSS for darker blue top bar (Navbar simulation) ---
st.markdown("""
<style>
    /* Target the header/top bar area for the darker blue color */
    header {
        background-color: #87CEEB !important; /* Sky Blue (Darker shade) */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 10px 0;
    }
    /* Style the main header for aesthetics */
    h1 {
        color: #005A9C;
    }
    /* Style the Prediction Metric for clarity */
    .stAlert {
        font-size: 16px;
    }
    /* Ensure the verdict text is clear and readable */
    .verdict-text {
        font-size: 14px;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---

@st.cache_data
def load_data():
    """
    Loads historical traffic data from a CSV file, cleans it, and converts it
    to a dictionary of DataFrames, keyed by cell tower ID.
    """
    try:
        # Load the CSV file as specified, using semicolon separator
        df_all = pd.read_csv(DATA_FILE_PATH, sep=';')
        
        # Rename columns to match internal app logic (date, volume)
        df_all = df_all.rename(columns={
            'Cell': 'cell_id',
            'Date': 'date',
            'Data_Volume': 'volume'
        })
        
        # Convert date column to datetime objects
        df_all['date'] = pd.to_datetime(df_all['date'])
        
        # Convert volume to numeric (just in case)
        df_all['volume'] = pd.to_numeric(df_all['volume'], errors='coerce')
        
        # Convert the single large DataFrame into a dictionary of DataFrames, grouped by cell_id
        df_data = {}
        for tower in CELL_TOWERS:
            # Filter for the current tower and select only date and volume columns
            df_cell = df_all[df_all['cell_id'] == tower][['date', 'volume']].copy()
            if not df_cell.empty:
                # Remove any rows with missing values that may have resulted from conversion
                df_data[tower] = df_cell.dropna()
            
        return df_data
        
    except FileNotFoundError:
        st.error(f"Error: The required data file '{DATA_FILE_PATH}' was not found. Please ensure it exists.")
        return None
    except Exception as e:
        st.error(f"Error processing data from CSV: {e}")
        return None

def predict_traffic_and_congestion(cell_id, prediction_date):
    """
    Placeholder function to simulate model inference for the given cell and date.
    Returns: (predicted_volume, predicted_congestion_percentage)
    """
    # Base volumes for simulation based on the new cell IDs
    CELL_TOWER_BASES = {
        "cell0001": 16000000,
        "cell0012": 10500000,
        "cell0015": 23000000,
        "cell0020": 2200000
    }
    
    # Use hashing for predictable but varying randomness
    seed_value = hash(cell_id + str(prediction_date))
    
    # CRITICAL FIX: Constrain the seed value to be within the [0, 2**32 - 1] range required by NumPy.
    NP_MAX_SEED = 2**32
    np_seed = abs(seed_value) % NP_MAX_SEED
    
    random.seed(np_seed)
    np.random.seed(np_seed) 
    
    base_volume = CELL_TOWER_BASES.get(cell_id, 5000000)
    
    # Simulate Volume: slight randomness around the base
    volume_prediction = base_volume + random.randint(-2000000, 2000000)
    
    # Simulate Congestion (%):
    # Base congestion (e.g., higher volume cells have slightly higher base congestion)
    base_congestion = 20 + (volume_prediction / 500000) * 0.1 
    
    # Add random fluctuation for congestion (e.g., 5-25% variation)
    congestion_prediction = np.clip(base_congestion + np.random.normal(0, 8), 0, 100)
    
    return max(0, volume_prediction), congestion_prediction

def cell_tower_section(tower_id, df, T, max_volume_threshold, max_congestion_threshold):
    """Renders the UI for a single cell tower."""
    
    # Display Cell Tower Code as a subheader
    st.subheader(T.get('tower_code', "Cell Tower Code: {}").format(tower_id))

    # 1. Plotting Historical Data
    st.markdown(f"#### {T.get('historical_title', 'Historical Traffic Volume')}")
    fig = px.line(
        df, 
        x='date', 
        y='volume', 
        title=f"Traffic Volume for {tower_id}",
        labels={'date': 'Date', 'volume': 'Traffic Volume (Bytes)'},
        template="plotly_white"
    )
    fig.update_traces(line_color='#005A9C', line_width=2) # Darker blue for plot line
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 2. Prediction Interface (aligned horizontally)
    
    # Initialize prediction results and new verdict texts if not set
    if f'pred_volume_{tower_id}' not in st.session_state:
        st.session_state[f'pred_volume_{tower_id}'] = "---"
        st.session_state[f'volume_verdict_text_{tower_id}'] = ""
    if f'pred_congestion_{tower_id}' not in st.session_state:
        st.session_state[f'pred_congestion_{tower_id}'] = "---"
        st.session_state[f'congestion_verdict_text_{tower_id}'] = ""

    # Form key to scope prediction logic
    with st.form(key=f'form_{tower_id}'):
        col1, col2, col3 = st.columns([1, 1, 1]) 

        with col1:
            # Date input for prediction
            pred_date = st.date_input(
                T.get('pred_date', 'Prediction Date'), 
                value=T.get('placeholder_date', datetime.date.today() + datetime.timedelta(days=1)), 
                key=f'date_{tower_id}'
            )

        with col2:
            st.caption(T.get('pred_volume_label', 'Predicted Volume (Bytes):'))
            # Display Volume result (the number)
            st.markdown(
                f"<div style='font-size: 24px; font-weight: bold; color: #005A9C;'>{st.session_state[f'pred_volume_{tower_id}']}</div>", 
                unsafe_allow_html=True
            )
            # Display Volume Verdict (the warning/safe text)
            st.markdown(
                f"<div class='verdict-text'>{st.session_state[f'volume_verdict_text_{tower_id}']}</div>",
                unsafe_allow_html=True
            )
            
        with col3:
            st.caption(T.get('pred_congestion_label', 'Predicted Congestion (%):'))
            # Display Congestion result (the number)
            st.markdown(
                f"<div style='font-size: 24px; font-weight: bold; color: #005A9C;'>{st.session_state[f'pred_congestion_{tower_id}']}</div>", 
                unsafe_allow_html=True
            )
            # Display Congestion Verdict (the warning/safe text)
            st.markdown(
                f"<div class='verdict-text'>{st.session_state[f'congestion_verdict_text_{tower_id}']}</div>",
                unsafe_allow_html=True
            )

        # Submit button for the form
        predict_btn = st.form_submit_button(T.get('predict_button', 'Get Prediction & Alert'), type='primary')
    
    # 3. Prediction Logic and Alert (runs after submit)
    if predict_btn:
        # Execute prediction and show alert based on threshold
        with st.spinner(T.get('predict_button', 'Get Prediction & Alert') + "..."):
            predicted_volume, predicted_congestion = predict_traffic_and_congestion(tower_id, pred_date)
            
            # --- Alert Logic ---
            
            # Volume Alert: Use in-line styling to mimic st.error (red) or st.success (green)
            if predicted_volume > max_volume_threshold:
                # Streamlit Error Color: #FF4B4B
                volume_verdict_text = f"<span style='color: #FF4B4B;'>{T.get('volume_alert', '🚨 **HIGH VOLUME ALERT**: Predicted traffic volume ({:.0f}) exceeds the volume threshold of {:.0f}!')}</span>".format(predicted_volume, max_volume_threshold)
            else:
                # Streamlit Success Color: #09AB52
                volume_verdict_text = f"<span style='color: #09AB52;'>{T.get('volume_safe', '✅ Predicted traffic volume is {:.0f} (below volume threshold).')}</span>".format(predicted_volume)

            # Congestion Alert: Use in-line styling to mimic st.warning (orange) or st.info (blue)
            if predicted_congestion > max_congestion_threshold:
                # Streamlit Warning Color: #F9800E
                congestion_verdict_text = f"<span style='color: #F9800E;'>{T.get('congestion_alert', '⚠️ **HIGH CONGESTION ALERT**: Predicted congestion ({:.1f}%) exceeds the congestion threshold of {:.0f}%!')}</span>".format(predicted_congestion, max_congestion_threshold)
            else:
                # Streamlit Info Color: #00A9E0
                congestion_verdict_text = f"<span style='color: #00A9E0;'>{T.get('congestion_safe', '👍 Predicted congestion is {:.1f}% (below congestion threshold).')}</span>".format(predicted_congestion)
            
            # Update the session state with the predicted numbers and the verdict text
            st.session_state[f'pred_volume_{tower_id}'] = f"{predicted_volume:,.0f}"
            st.session_state[f'volume_verdict_text_{tower_id}'] = volume_verdict_text
            st.session_state[f'pred_congestion_{tower_id}'] = f"{predicted_congestion:.1f}%"
            st.session_state[f'congestion_verdict_text_{tower_id}'] = congestion_verdict_text
            
            st.rerun() # Rerun to update the prediction metrics immediately in the columns

def run_app():
    """Main application runner."""
    
    # Initialize session state for language and translation dictionary
    if 'language' not in st.session_state:
        st.session_state.language = 'English'
    if 'T' not in st.session_state:
        st.session_state.T = TRANSLATIONS[st.session_state.language]

    T = st.session_state.T # Current translation dictionary

    # --- Sidebar Configuration ---
    with st.sidebar:
        # 1. Language Slider
        selected_language = st.select_slider(
            T.get("lang_select", 'Select Website Language'), 
            options=['English', 'French'],
            value=st.session_state.language
        )
        
        # Check if language changed and update translations
        if selected_language != st.session_state.language:
            st.session_state.language = selected_language
            st.session_state.T = TRANSLATIONS[selected_language]
            st.rerun() 
            
        # 2. Max Volume Threshold Input
        st.header("Alert Settings")
        max_volume_threshold = st.slider(
            T.get("max_volume", "Max Volume Alert Threshold (Bytes)"),
            min_value=1000000,
            max_value=50000000,
            value=25000000,
            step=1000000,
            format='%i'
        )
        st.info(T.get("alert_volume_info", "Volume alert triggers if prediction > **{:,} Bytes**").format(max_volume_threshold))
        
        st.markdown("---")
        
        # 3. Max Congestion Threshold Input
        max_congestion_threshold = st.slider(
            T.get("max_congestion", "Max Congestion Alert Threshold (%)"),
            min_value=10,
            max_value=100,
            value=70,
            step=5,
            format='%i'
        )
        st.info(T.get("alert_congestion_info", "Congestion alert triggers if prediction > **{:,}%**").format(max_congestion_threshold))

    # --- Main Page Content ---
    
    st.title(T.get("title", "LTE Traffic & Congestion Monitoring App"))
    st.markdown("---")
    
    # Load the data 
    traffic_data = load_data()
    
    if traffic_data is not None:
        # Loop through each cell tower and render its section
        for tower_id in CELL_TOWERS:
            df = traffic_data.get(tower_id)
            if df is not None:
                cell_tower_section(tower_id, df, T, max_volume_threshold, max_congestion_threshold)
                st.markdown("___") # Section separator
            else:
                st.warning(f"No data found for {tower_id}. Please check data source.")

# Run the application
if __name__ == '__main__':
    run_app()