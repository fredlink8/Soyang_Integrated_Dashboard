import streamlit as st

st.set_page_config(
    page_title="Soyang Integrated Decision Support Portal",
    layout="wide"
)

st.title("Soyang Integrated Decision Support Portal")

st.markdown("""
This platform integrates:

1. Reservoir Operation Decision Support
2. Tailrace Floodplain Mapping Tool

Select a module below.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Reservoir Dashboard")

    st.markdown("""
    CNN-LSTM prediction and PPO optimization
    for reservoir operation.
    """)

    if st.button("Open Reservoir Dashboard"):
        st.switch_page("pages/Reservoir_Dashboard.py")

with col2:
    st.subheader("Floodplain Mapping Tool")

    st.markdown("""
    Tailrace flood inundation mapping and
    floodplain visualization.
    """)

    if st.button("Open Floodplain Mapping Tool"):
        st.switch_page("pages/Floodplain_Mapping.py")