import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle 




model = load_model("Next_Word_model.h5")


with open("tokenizer_1.pkl", "rb") as file:
    tokenizer = pickle.load(file)


reverse_index = {
    idx: word
    for word, idx in tokenizer.word_index.items()
}


max_len = 44


def generate_text(seed_text, num_words=10):
    text = seed_text

    for _ in range(num_words):
        seq = tokenizer.texts_to_sequences([text])[0]

        padded = pad_sequences(
            [seq],
            maxlen=max_len,
            padding="pre"
        )

        preds = model.predict(padded, verbose=0)

        pos = np.argmax(preds)

        next_word = reverse_index.get(pos, "")

        text += " " + next_word

    return text


st.title("Next Word Prediction with Deep Learning")

seed = st.text_input(
    "Enter a Starting Text:",
    "Hello"
)

num_words = st.slider(
    "Number of words to generate",
    1,
    20,
    10
)


if st.button("Generate"):
    result = generate_text(seed, num_words)
    st.write(result)
