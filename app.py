import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import librosa
import joblib

# Load the trained model and scaler
model = joblib.load("./models/parkinson_detection_model.pkl")
scaler = joblib.load("./models/scaler.pkl")

# Streamlit UI Configuration
st.set_page_config(page_title="Parkinson’s Detector 🎤", page_icon="🧠", layout="wide")

# Styling
st.markdown("""
    <style>
        .main { background-color: #f0f2f6; }
        .big-font { font-size:25px !important; font-weight:bold; color:#ff4b4b; }
        .metric-card {
            border-radius: 10px;
            padding: 10px;
            background: #fff;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 18px;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab"] [aria-selected="true"] {
            border-bottom: 3px solid #ff4b4b;
            color: #ff4b4b;
        }
        .sidebar .block-container { padding-top: 2rem; }
        .sidebar .block-container:first-child { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

# Sidebar Navigation
st.sidebar.title("🔍 App Navigation")
st.sidebar.markdown("*Your Voice, Your Health Insights.*")

# Sidebar Navigation Buttons (Always Visible)
col1, col2, col3 = st.sidebar.columns(3)
if col1.button("🏠 Home"):
    st.session_state.page = "🏠 Home"
if col2.button("📊 Analysis"):
    st.session_state.page = "📊 Analysis"
if col3.button("ℹ️ About"):
    st.session_state.page = "ℹ️ About"

# Voice Recording Guidelines in Sidebar Expander
with st.sidebar.expander("📝 Voice Recording Guidelines"):
    st.markdown("""
    - **Use a quiet place** 🏠🔇
    - **Use a high-quality microphone** 🎙️
    - **Keep the mic 6-12 inches away** 🎧
    - **Record at least 3-5 sec of “ahhh” or “oohhh”** 🗣️
    - **Save as `.wav` format** 🎵
    """)

# Feature Extraction Function
@st.cache_data
def extract_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        nyquist = sr / 2
        f0, voiced_flag, _ = librosa.pyin(y, fmin=75, fmax=min(600, nyquist))

        if not np.any(voiced_flag):
            st.error("No voiced segments detected in the audio.")
            return None

        f0_voiced = f0[voiced_flag > 0]
        f0_diff = np.diff(f0_voiced)

        jitter_percent = np.mean(np.abs(f0_diff)) / np.mean(f0_voiced)
        jitter_abs = np.mean(np.abs(f0_diff))
        rap_ppq = np.abs(f0_diff) / f0_voiced[:-1]
        rap, ppq, ddp = np.mean(rap_ppq), np.mean(rap_ppq), np.var(f0_diff)

        stft = np.abs(librosa.stft(y))
        shimmer = librosa.feature.delta(librosa.amplitude_to_db(stft, ref=np.max))

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, fmin=75, n_bands=6)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        harmonic = librosa.effects.harmonic(y)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        features = {
            'MDVP:Fo(Hz)': np.mean(f0_voiced),
            'MDVP:Fhi(Hz)': np.max(spectral_centroid),
            'MDVP:Flo(Hz)': np.min(spectral_centroid),
            'MDVP:Jitter(%)': jitter_percent,
            'MDVP:Jitter(Abs)': jitter_abs,
            'MDVP:RAP': rap,
            'MDVP:PPQ': ppq,
            'Jitter:DDP': ddp,
            'MDVP:Shimmer': np.mean(shimmer),
            'MDVP:Shimmer(dB)': np.mean(librosa.amplitude_to_db(shimmer)),
            'Shimmer:APQ3': np.max(shimmer),
            'Shimmer:APQ5': np.min(shimmer),
            'MDVP:APQ': np.mean(spectral_flatness),
            'Shimmer:DDA': np.std(spectral_flatness),
            'NHR': np.mean(harmonic),
            'HNR': np.std(harmonic),
            'RPDE': np.mean(spectral_rolloff),
            'DFA': np.std(zero_crossing_rate),
            'spread1': np.mean(spectral_bandwidth),
            'spread2': np.std(spectral_bandwidth),
            'D2': np.mean(spectral_contrast),
            'PPE': np.mean(rms)
        }

        return np.array(list(features.values()))

    except Exception as e:
        st.error(f"❌ Error extracting features: {e}")
        return None

# Prediction Function
@st.cache_data
def predict_parkinson(data):
    feature_names = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
        'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]
    input_data = pd.DataFrame([data], columns=feature_names)
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]
    return prediction[0], probability

# Home Page
def home_page():
    st.title("🎤 Parkinson’s Disease Prediction")
    st.markdown("**Upload a voice sample to analyze Parkinson’s probability.**")

    uploaded_file = st.file_uploader("📂 **Upload a voice recording (.wav)**", type=["wav"], label_visibility="visible")

    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')

        with st.spinner("🔬 Analyzing voice features..."):
            features = extract_features(uploaded_file)

            if features is not None:
                st.session_state["features"] = features
                prediction, probability = predict_parkinson(features)
                st.session_state["prediction"] = prediction
                st.session_state["probability"] = probability

                display_results(prediction, probability)

# Display Results Function
def display_results(prediction, probability):
    col1, col2 = st.columns(2)

    with col1:
        st.metric("🩺 Prediction", "🟢 Healthy" if prediction == 0 else "🔴 Parkinson’s Detected")
        st.progress(probability)

    with col2:
        st.write("### 📌 Probability of Parkinson’s Disease")
        st.write(f"**{probability*100:.2f}%** chance of Parkinson’s.")

    if probability > 0.6:
        st.error("⚠️ **High risk detected. Consult a doctor.**")
    else:
        st.success("✅ **Low risk detected. Stay healthy!**")

# Analysis Page
def analysis_page():
    st.title("📊 Detailed Feature Analysis")
    st.markdown("Explore and visualize the extracted audio features.")

    if "features" in st.session_state and st.session_state.features is not None:
        feature_names = [
            "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
            "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
            "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
            "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
        ]

        feature_values = st.session_state.features.flatten()
        feature_df = pd.DataFrame({"Feature Name": feature_names, "Value": feature_values})

        tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Bar Chart", "📋 Tabular View"])

        with tab1:
            st.markdown("### 🌐 Feature Distribution Overview")
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=feature_values,
                theta=feature_names,
                fill='toself',
                name='Feature Distribution',
                line=dict(color='royalblue')
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max(feature_values) + 0.1])),
                showlegend=False,
                template='plotly_dark',
                title="Radar Chart of Extracted Features"
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("### 📈 Feature Values Bar Chart")
            fig_bar = px.bar(
                feature_df,
                x="Value",
                y="Feature Name",
                orientation='h',
                color="Value",
                color_continuous_scale="RdBu",
                title="Feature Value Comparison",
                template="plotly_white"
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        with tab3:
            st.markdown("### 📋 Extracted Audio Features (Detailed Table View)")
            styled_table = feature_df.style.background_gradient(cmap='coolwarm').format(precision=4)
            st.dataframe(styled_table, use_container_width=True)
            csv = feature_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Feature Data as CSV",
                data=csv,
                file_name='audio_features.csv',
                mime='text/csv'
            )
    else:
        st.warning("⚠️ Please upload an audio file on the Home page first.")

# About Page
def about_page():
    st.title("ℹ️ About This App")
    st.write("""
    This AI-driven application analyzes voice recordings to predict Parkinson’s Disease probability.
    **How It Works:**
    - Upload a `.wav` voice sample.
    - The app extracts features like jitter, shimmer, HNR, etc.
    - A trained ML model predicts Parkinson’s probability.
    - You receive health insights and a risk assessment.

    **🚨 Disclaimer:** This tool is not a medical diagnosis. Consult a doctor for medical advice.
    """)
    st.markdown("🔬 **Developed by AI & Healthcare Researcher: Amio Ghosh**")
    st.markdown("🌐 **Connect with me:** [LinkedIn](https://www.linkedin.com/in/amio-ghosh/) | [GitHub](https://github.com/Amio84)")

# Main Function to Control Page Navigation
def main():
    if st.session_state.page == "🏠 Home":
        home_page()
    elif st.session_state.page == "📊 Analysis":
        analysis_page()
    elif st.session_state.page == "ℹ️ About":
        about_page()

if __name__ == "__main__":
    main()