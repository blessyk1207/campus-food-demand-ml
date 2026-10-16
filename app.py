
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Campus Food Demand Forecasting",
    page_icon="🍽️",
    layout="centered",
)

st.title("Campus Food Demand Forecasting")
st.write(
    "Predict meal demand using historical data and machine learning."
)

with st.form("prediction_form"):
    st.subheader("Meal and Canteen Details")

    center_id = st.text_input("Canteen Center ID", value="55")
    meal_id = st.text_input("Meal ID", value="1885")
    week_index = st.number_input(
        "Week Index", min_value=0, value=144, step=1
    )

    col1, col2 = st.columns(2)

    with col1:
        checkout_price = st.number_input(
            "Selling Price", min_value=0.01, value=150.0
        )

    with col2:
        base_price = st.number_input(
            "Base Price", min_value=0.01, value=160.0
        )

    emailer = st.selectbox(
        "Email Promotion", [0, 1],
        format_func=lambda x: "Yes" if x else "No",
    )

    featured = st.selectbox(
        "Featured on Homepage", [0, 1],
        format_func=lambda x: "Yes" if x else "No",
    )

    discount_amount = max(0.0, base_price - checkout_price)
    discount_percentage = (
        discount_amount / base_price * 100
    )

    st.caption(f"Discount: {discount_amount:.2f}")
    st.caption(f"Discount percentage: {discount_percentage:.2f}%")

    submitted = st.form_submit_button(
        "Predict Demand", type="primary", use_container_width=True
    )

if submitted:
    if not center_id.strip() or not meal_id.strip():
        st.error("Please enter both the center ID and meal ID.")
    else:
        payload = {
            "center_id": center_id.strip(),
            "meal_id": meal_id.strip(),
            "week_index": int(week_index),
            "checkout_price": checkout_price,
            "base_price": base_price,
            "emailer_for_promotion": emailer,
            "homepage_featured": featured,
            "discount_amount": discount_amount,
            "discount_percentage": discount_percentage,
        }

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()["predicted_orders"]

            st.success("Prediction completed!")
            st.metric(
                "Estimated Meal Orders",
                f"{result:,.0f}",
            )
            st.info(
                "This is a model estimate, not a guarantee "
                "of actual demand."
            )

        except requests.RequestException:
            st.error(
                "Could not connect to the prediction API. "
                "Make sure the FastAPI server is running."
            )
