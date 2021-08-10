from datetime import datetime

import streamlit as st

st.write(datetime.now())

x = st.slider("Select a value")
st.write(x, "squared is", x * x)
