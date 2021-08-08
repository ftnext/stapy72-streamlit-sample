import random

import numpy as np
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, T5Tokenizer


def freeze_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


@st.cache
def initialize(model_name):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


@st.cache
def generate_sentences(
    model,
    text,
    length=100,
    temperature=1.0,
    k=0,
    p=0.9,
    repetition_penalty=1.0,
    num_return_sequences=3,
):
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=length + len(text),
        temperature=temperature,
        top_k=k,
        top_p=p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        num_return_sequences=num_return_sequences,
    )
    return output_sequences


tokenizer, model = initialize("rinna/japanese-gpt2-medium")

st.sidebar.subheader("Parameters")
max_gen_seq_len = st.sidebar.number_input(
    "Generated sequence max length", value=100, step=10
)
temp = st.sidebar.slider("Temperature for decoding", 0.5, 1.5, 1.0, 0.1)
top_k = st.sidebar.slider("Top k", 0, 2, 0, 1)
top_p = st.sidebar.slider("Top p", 0.5, 1.0, 0.9, 0.01)

st.title("日本語文章生成")
num_return_sequences = st.slider("生成する文章の数", 1, 5, 3, 1)

input_text = st.text_input("日本語テキストを入力してください（続きを生成します）")
input_text = input_text.rstrip()
if input_text:
    freeze_seed()
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_sequences = generate_sentences(
        model,
        input_text,
        length=max_gen_seq_len,
        temperature=temp,
        k=top_k,
        p=top_p,
        num_return_sequences=num_return_sequences,
    )

    for idx, sequence in enumerate(output_sequences):
        st.subheader(f"=== GENERATED SEQUENCE {idx + 1} ===")
        sequence = sequence.tolist()

        text = tokenizer.decode(sequence, clean_up_tokenization_spaces=True)

        total_sequence = (
            input_text
            + text[
                len(
                    tokenizer.decode(
                        input_ids[0], clean_up_tokenization_spaces=True
                    )
                ) :
            ]
        )

        st.write(total_sequence)
