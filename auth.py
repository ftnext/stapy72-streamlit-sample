import secrets
from datetime import datetime

import streamlit as st

if "is_login" not in st.session_state:
    # st.session_state.is_login という書き方でも同じ
    st.session_state["is_login"] = False

# 認証に成功した後に入力欄が消えるようにする
placeholder = st.empty()

if not st.session_state["is_login"]:
    password = placeholder.text_input("パスワードを入力してください", type="password")
    if not password:
        st.stop()
    if secrets.compare_digest(password, st.secrets["APP_PASSWORD"]):
        st.session_state["is_login"] = True
        placeholder.empty()
    else:
        st.write("パスワードが間違っています")
        st.stop()

# ログイン状態をsession_stateに持つのでパスワード再入力せずにインタラクティブに操作できる
if st.session_state["is_login"]:
    st.title("Welcome")

    st.write(datetime.now())

    x = st.slider("Select a value")
    st.write(x, "squared is", x * x)
