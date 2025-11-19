import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Optional: only needed for chatbot tab
import google.generativeai as genai
# -------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", None))

if not GEMINI_API_KEY:
    st.error("❗ Gemini API key not found. Please add it to .streamlit/secrets.toml")
else:
    genai.configure(api_key=GEMINI_API_KEY)
# ============================
# 🔧 CONFIG
# ============================
st.set_page_config(page_title="EV Range & Chatbot", layout="wide")

# Try to get Gemini API key


# ============================
# 🚗 LOAD MODEL / DATA
# ============================
@st.cache_resource
def load_ev_assets():
    model = joblib.load(r"E:\sills4future\models\random_forest_model.pkl")
    scaler = joblib.load(r"E:\sills4future\models\scaler.pkl")
    train_columns = joblib.load(r"E:\sills4future\models\train_columns.pkl")
    # cleaned data only for dropdown brands/models (not used in training again)
    df = pd.read_csv(r"E:\sills4future\cleaned_ev_data.csv")
    return model, scaler, train_columns, df

try:
    rf_model, scaler, train_columns, df_ev = load_ev_assets()
except Exception as e:
    st.error(f"❌ Error loading EV model assets: {e}")
    st.stop()

# ============================
# 🚗 PRESET CARS (India + Global)
# ============================
EV_PRESETS = {
    "India": {
        "Tata Nexon EV": {
            "brand": "Tata",
            "battery_kwh": 40.5,
            "eff_wh_km": 160,
            "top_speed": 150,
            "accel_0_100": 9.4,
            "seats": 5,
        },
        "MG ZS EV": {
            "brand": "MG",
            "battery_kwh": 50.3,
            "eff_wh_km": 170,
            "top_speed": 175,
            "accel_0_100": 8.5,
            "seats": 5,
        },
        "Mahindra XUV400": {
            "brand": "Mahindra",
            "battery_kwh": 39.5,
            "eff_wh_km": 165,
            "top_speed": 150,
            "accel_0_100": 8.3,
            "seats": 5,
        },
        "BYD e6": {
            "brand": "BYD",
            "battery_kwh": 71.7,
            "eff_wh_km": 180,
            "top_speed": 130,
            "accel_0_100": 9.0,
            "seats": 5,
        },
        "Hyundai Kona (IN)": {
            "brand": "Hyundai",
            "battery_kwh": 39.2,
            "eff_wh_km": 150,
            "top_speed": 155,
            "accel_0_100": 9.7,
            "seats": 5,
        },
    },
    "Global": {
        "Tesla Model 3": {
            "brand": "Tesla",
            "battery_kwh": 82,
            "eff_wh_km": 150,
            "top_speed": 225,
            "accel_0_100": 5.6,
            "seats": 5,
        },
        "Tesla Model S": {
            "brand": "Tesla",
            "battery_kwh": 100,
            "eff_wh_km": 180,
            "top_speed": 250,
            "accel_0_100": 3.2,
            "seats": 5,
        },
        "Porsche Taycan": {
            "brand": "Porsche",
            "battery_kwh": 93.4,
            "eff_wh_km": 200,
            "top_speed": 260,
            "accel_0_100": 2.8,
            "seats": 4,
        },
        "Kia EV6": {
            "brand": "Kia",
            "battery_kwh": 77.4,
            "eff_wh_km": 170,
            "top_speed": 190,
            "accel_0_100": 5.2,
            "seats": 5,
        },
        "Hyundai Ioniq 5": {
            "brand": "Hyundai",
            "battery_kwh": 77.4,
            "eff_wh_km": 160,
            "top_speed": 185,
            "accel_0_100": 5.1,
            "seats": 5,
        },
    },
}

# Helper: one-hot setter
def set_one_hot(input_df, prefix, value):
    col = f"{prefix}{value}"
    if col in input_df.columns:
        input_df[col] = 1

# Get available brands from training columns
BRAND_OPTIONS = sorted([c.replace("brand_", "") for c in train_columns if c.startswith("brand_")])

# ============================
# 🧠 GEMINI MODEL (for chatbot)
# ============================
def get_gemini_model():
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        return None, "❗ Gemini API key missing. Set GEMINI_API_KEY in .streamlit/secrets.toml or environment."
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        return model, None
    except Exception as e:
        return None, f"Gemini init error: {e}"

# ============================
# 🧩 MAIN TABS
# ============================
tab1, tab2 = st.tabs(["🔋 EV Range Predictor", "🤖 EV Assistant Chatbot"])

# ---------------------------------------------------
# TAB 1: EV RANGE PREDICTOR
# ---------------------------------------------------
with tab1:
    st.markdown("### 🔋 EV Range Predictor")
    st.write("Use real car presets or custom specs to estimate EV driving range.")

    mode = st.radio("Select Input Mode", ["Use Preset Car", "Custom Specs"], horizontal=True)

    # ---- PRESET MODE ----
    if mode == "Use Preset Car":
        col_meta, col_form = st.columns([1.2, 2])

        with col_meta:
            market = st.selectbox("Market", ["India", "Global"])
            model_name = st.selectbox("Choose EV Model", list(EV_PRESETS[market].keys()))
            preset = EV_PRESETS[market][model_name]
            brand = preset["brand"]

            st.markdown(
                f"""
                **Selected Car:** {model_name}  
                **Brand:** {brand}  
                🔋 **Battery:** {preset['battery_kwh']} kWh  
                ⚙ **Efficiency:** {preset['eff_wh_km']} Wh/km  
                🏎 **Top Speed:** {preset['top_speed']} km/h  
                🚀 **0–100:** {preset['accel_0_100']} s  
                🪑 **Seats:** {preset['seats']}
                """
            )

        with col_form:
            st.markdown("Adjust values if needed:")

            col1, col2 = st.columns(2)
            with col1:
                battery = st.number_input(
                    "Battery (kWh)", 20.0, 200.0, float(preset["battery_kwh"]), step=1.0
                )
                efficiency = st.number_input(
                    "Efficiency (Wh/km)", 100.0, 400.0, float(preset["eff_wh_km"]), step=1.0
                )
            with col2:
                top_speed = st.number_input(
                    "Top Speed (km/h)", 100.0, 350.0, float(preset["top_speed"]), step=1.0
                )
                accel = st.number_input(
                    "0–100 km/h (sec)", 2.0, 15.0, float(preset["accel_0_100"]), step=0.1
                )

            seats = st.selectbox("Seats", [2, 4, 5, 6, 7], index=[2, 4, 5, 6, 7].index(preset["seats"]))

    # ---- CUSTOM MODE ----
    else:
        st.markdown("#### ✍️ Custom EV Specs")

        brand = st.selectbox("Brand", BRAND_OPTIONS)

        col1, col2 = st.columns(2)
        with col1:
            battery = st.number_input("Battery (kWh)", 20.0, 200.0, 60.0, step=1.0)
            efficiency = st.number_input("Efficiency (Wh/km)", 100.0, 400.0, 160.0, step=1.0)
        with col2:
            top_speed = st.number_input("Top Speed (km/h)", 100.0, 350.0, 160.0, step=1.0)
            accel = st.number_input("0–100 km/h (sec)", 2.0, 15.0, 8.5, step=0.1)

        seats = st.selectbox("Seats", [2, 4, 5, 6, 7], index=2)

    # ---- BUILD INPUT VECTOR ----
    input_row = pd.DataFrame(np.zeros((1, len(train_columns))), columns=train_columns)

    # numeric
    input_row["battery_capacity_kWh"] = battery
    input_row["efficiency_wh_per_km"] = efficiency
    input_row["top_speed_kmh"] = top_speed
    input_row["acceleration_0_100_s"] = accel
    input_row["seats"] = seats

    # numeric defaults
    defaults = {
        "torque_nm": 300,
        "number_of_cells": 300,
        "fast_charging_power_kw_dc": 150,
        "towing_capacity_kg": 500,
        "cargo_volume_l": 450,
        "length_mm": 4500,
        "width_mm": 1800,
        "height_mm": 1500,
    }
    for c, v in defaults.items():
        if c in input_row.columns:
            input_row[c] = v

    # one-hot brand
    set_one_hot(input_row, "brand_", brand)

    # for preset mode, also try to one-hot model_name
    if mode == "Use Preset Car":
        set_one_hot(input_row, "model_", model_name)

    # ---- PREDICT ----
    if st.button("🔮 Predict Range"):
        try:
            scaled = scaler.transform(input_row)
            pred_range = rf_model.predict(scaled)[0]
            st.success(f"🚗 Estimated Driving Range: **{pred_range:.1f} km**")

            with st.expander("Show model input (debug)"):
                st.write(input_row)

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------------------------------------------------
# TAB 2: GEMINI EV ASSISTANT CHATBOT
# ---------------------------------------------------
with tab2:
    st.markdown("### 🤖 EV Assistant Chatbot")
    st.write(
        "Ask anything about electric vehicles, range, batteries, or your predictions. "
        "This assistant is friendly but also explains the technical side."
    )

    gemini_model, gemini_err = get_gemini_model()
    if gemini_err:
        st.error(gemini_err)
        st.stop()

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Clear chat
    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []

    # Display history
    for msg in st.session_state.chat_history:
        role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
        bg = "#222" if msg["role"] == "assistant" else "#333"
        st.markdown(
            f"<div style='padding:6px 10px;margin:4px 0;border-radius:6px;background:{bg};'>"
            f"<b>{role}:</b> {msg['text']}</div>",
            unsafe_allow_html=True,
        )

    user_msg = st.text_input("Type your question here:")

    if st.button("Send"):
        if not user_msg.strip():
            st.warning("Please type something.")
        else:
            # add user message
            st.session_state.chat_history.append({"role": "user", "text": user_msg})

            # build conversation text
            convo = "You are a helpful EV assistant. You are friendly but give technical explanations.\n"
            for m in st.session_state.chat_history:
                prefix = "User:" if m["role"] == "user" else "Assistant:"
                convo += f"{prefix} {m['text']}\n"
            convo += "Assistant:"

            try:
                response = gemini_model.generate_content(convo)
                reply = response.text
            except Exception as e:
                reply = f"Sorry, I ran into an error: {e}"

            st.session_state.chat_history.append({"role": "assistant", "text": reply})
            st.experimental_rerun()
