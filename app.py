import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import librosa
import soundfile as sf  
import joblib

# ---- Load the trained model and scaler ----
model = joblib.load("./models/parkinson_detection_model.pkl")
scaler = joblib.load("./models/scaler.pkl")

# ---- Streamlit UI Configuration ----
st.set_page_config(
    page_title="Parkinson’s Detector 🎤",
    page_icon="🧠",
    layout="wide",
)

# ---- Styling ----
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
    </style>
""", unsafe_allow_html=True)

# ---- Sidebar Navigation ----
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("", ["🏠 Home", "📊 Analysis", "ℹ️ About"])

# ---- Feature Extraction Function ----
def extract_features(audio_file):
    try:
        with sf.SoundFile(audio_file) as audio:
            y = audio.read(dtype="float32")
            sr = audio.samplerate

        n_fft = 2048
        hop_length = 512

        features = {
            "MDVP:Fo(Hz)": np.mean(librosa.feature.zero_crossing_rate(y)[0]),
            "MDVP:Fhi(Hz)": np.max(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)),
            "MDVP:Flo(Hz)": np.min(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)),
            "MDVP:Jitter(%)": np.std(librosa.feature.zero_crossing_rate(y)[0]),
            "MDVP:Jitter(Abs)": np.mean(librosa.feature.zero_crossing_rate(y)[0]),
            "MDVP:RAP": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)),
            "MDVP:PPQ": np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)),
            "Jitter:DDP": np.var(librosa.feature.zero_crossing_rate(y)[0]),
            "MDVP:Shimmer": np.mean(librosa.feature.rms(y=y)),
            "MDVP:Shimmer(dB)": np.std(librosa.feature.rms(y=y)),
            "Shimmer:APQ3": np.max(librosa.feature.rms(y=y)),
            "Shimmer:APQ5": np.min(librosa.feature.rms(y=y)),
            "MDVP:APQ": np.mean(librosa.feature.spectral_flatness(y=y)),
            "Shimmer:DDA": np.std(librosa.feature.spectral_flatness(y=y)),
            "NHR": np.mean(librosa.effects.harmonic(y)),
            "HNR": np.std(librosa.effects.harmonic(y)),
            "RPDE": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)),
            "DFA": np.std(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)),
            "spread1": np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmin=100.0)),
            "spread2": np.std(librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmin=100.0)),
            "D2": np.max(librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmin=100.0)),
            "PPE": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1, n_fft=n_fft, hop_length=hop_length)),
        }

        return np.array(list(features.values()))

    except Exception as e:
        st.error(f"❌ Error extracting features: {e}")
        return None

# ---- Prediction Function ----
def predict_parkinson(data):
    input_data = np.asarray(data).reshape(1, -1)
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]
    
    return prediction[0], probability


# ---- HOME PAGE ----
if page == "🏠 Home":
    st.title("🎤 Parkinson’s Disease Detection")
    st.markdown("**Upload a voice sample to analyze Parkinson’s probability.**")

    with st.expander("📝 **Voice Recording Guidelines**"):
        st.markdown("""
        - **Use a quiet place** 🏠🔇  
        - **Use a high-quality microphone** 🎙️  
        - **Keep the mic 6-12 inches away** 🎧  
        - **Record at least 3-5 sec of “ahhh” or “oohhh”** 🗣️  
        - **Save as `.wav` format** 🎵  
        """)

    uploaded_file = st.file_uploader("📂 **Upload a voice recording (.wav)**", type=["wav"])

    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')

        with st.spinner("🔬 Analyzing voice features..."):
            features = extract_features(uploaded_file)

            if features is not None:
                # ✅ Store extracted features in session state
                st.session_state["features"] = features

                # ✅ Store prediction result
                prediction, probability = predict_parkinson(features)
                st.session_state["prediction"] = prediction
                st.session_state["probability"] = probability

                # Display results
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


# ---- ANALYSIS PAGE ----
elif page == "📊 Analysis":
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
        
        # DataFrame for easier manipulation and visualization
        feature_df = pd.DataFrame({
            "Feature Name": feature_names, 
            "Value": feature_values
        })

        # ---- TABS FOR BETTER NAVIGATION ----
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Bar Chart", "📋 Tabular View"])

        # ---- TAB 1: Overview (Radar Chart) ----
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
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, max(feature_values) + 0.1])
                ),
                showlegend=False,
                template='plotly_dark',
                title="Radar Chart of Extracted Features"
            )

            st.plotly_chart(fig, use_container_width=True)

        # ---- TAB 2: Interactive Bar Chart ----
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

        # ---- TAB 3: Tabular View ----
        with tab3:
            st.markdown("### 📋 Extracted Audio Features (Detailed Table View)")

            # ---- Stylish Data Table ----
            styled_table = feature_df.style.background_gradient(cmap='coolwarm').format(precision=4)
            st.dataframe(styled_table, use_container_width=True)

            # ---- Download Button ----
            csv = feature_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Feature Data as CSV",
                data=csv,
                file_name='audio_features.csv',
                mime='text/csv'
            )

    else:
        st.warning("⚠️ Please upload an audio file on the Home page first.")


# ---- ABOUT PAGE ----
elif page == "ℹ️ About":
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