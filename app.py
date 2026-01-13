import json
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

model = tf.keras.models.load_model("models/translator_model.keras")

with open("models/eng_tokenizer.pkl", "rb") as f:
    eng_tokenizer = pickle.load(f)

with open("models/fr_tokenizer.pkl", "rb") as f:
    fr_tokenizer = pickle.load(f)

with open("models/metadata.json", "r") as f:
    meta = json.load(f)

MAX_ENG = meta["max_eng"]
MAX_FR = meta["max_fr"]

fr_id_to_word = {v: k for k, v in fr_tokenizer.word_index.items()}
fr_id_to_word[0] = ""

def translate(text):
    seq = eng_tokenizer.texts_to_sequences([text.lower()])
    seq = pad_sequences(seq, maxlen=MAX_ENG, padding="post")
    preds = model.predict(seq, verbose=0)[0]
    tokens = np.argmax(preds, axis=1)

    words = []
    for t in tokens:
        w = fr_id_to_word.get(t, "")
        if w == "":
            break
        words.append(w)
    return " ".join(words)

@app.route("/", methods=["GET", "POST"])
def index():
    src, out = "", ""
    if request.method == "POST":
        src = request.form.get("src_text", "")
        out = translate(src)
    return render_template("index.html", src=src, translation=out)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
