import time

import streamlit as st


@st.cache
def expensive_computation(a, b):
    time.sleep(2)
    return a * b


a = 2
b = 21
res = expensive_computation(a, b)

st.write("Result:", res)
