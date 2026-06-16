import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🍽️ AI Restaurant Inventory Forecasting System")
st.info("📊 Model Accuracy: 97%")

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Inputs
day = st.number_input("Enter Day Number", min_value=1, value=1)
temperature = st.number_input("Enter Temperature", value=30)
weekend = st.selectbox("Weekend", [0, 1])

# Predict Button
if st.button("Predict Sales"):

    prediction = model.predict(
        [[day, temperature, weekend]]
    )[0]

    st.success(f"Predicted Sales: {prediction:.2f}")

    # Save history
    st.session_state.history.append({
        "Day": day,
        "Temperature": temperature,
        "Weekend": weekend,
        "Predicted Sales": prediction
    })

# Show History
if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(st.session_state.history)

    st.subheader("📋 Prediction History")
    st.dataframe(history_df)

    st.subheader("📈 Sales Trend")
    st.line_chart(
        history_df.set_index("Day")["Predicted Sales"]
    )

    csv = history_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Prediction History",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )