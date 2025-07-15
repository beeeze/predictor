import streamlit as st
from algo_core import scan_for_opportunities

st.set_page_config(page_title="📈 Stock Jump Predictor", layout="wide")

st.title("Low-Cap Stock Jump Predictor")
st.markdown("Predicts stocks under $1B market cap likely to jump tomorrow.")

if st.button("Scan Now"):
    with st.spinner("Scanning stocks... This may take 30-60 seconds."):
        df = scan_for_opportunities(threshold=0.55, max_cap=1e9)
        if not df.empty:
            st.success("Scan complete!")
            st.dataframe(df.style.background_gradient(cmap='Blues'), use_container_width=True)
            st.bar_chart(df.set_index('symbol')['probability'])
        else:
            st.warning("No stocks met the prediction threshold.")
