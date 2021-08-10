import streamlit as st

# https://docs.streamlit.io/en/stable/api.html#streamlit.radio
genre = st.radio(
    "What's your favorite movie genre", ("Comedy", "Drama", "Documentary")
)
# genreは選択されたラジオボタンの値を指す。ここでは "Comedy", "Drama", "Documentary" のいずれか
if genre == "Comedy":
    st.write("You selected comedy.")
else:
    st.write("You didn't select comedy.")

# https://docs.streamlit.io/en/stable/api.html#streamlit.checkbox
agree = st.checkbox("I agree")
# agreeは True/False 。選択されたときにTrue
if agree:
    st.write("Great!")

# https://docs.streamlit.io/en/stable/api.html#streamlit.file_uploader
uploaded_file = st.file_uploader("Choose a file")
