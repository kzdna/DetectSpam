from flask import Flask, request, render_template, jsonify, send_file
import joblib
import json
import os
import re
import string
import nltk
from datetime import datetime

# Unduh stopwords kalau belum
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('indonesian'))

app = Flask(__name__)

# Load model & vectorizer
model = joblib.load("svm_spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

HISTORY_FILE = "history.json"

# Fungsi cleaning lengkap seperti saat training
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as file:
            return json.load(file)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as file:
        json.dump(history, file)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    user_input = ""

    if request.method == "POST":
        user_input = request.form["message"]
        cleaned_input = clean_text(user_input)
        input_vectorized = vectorizer.transform([cleaned_input])
        prediction_result = model.predict(input_vectorized)[0]
        prediction = "SPAM" if prediction_result == 1 else "HAM (Bukan Spam)"

        history = load_history()
        history.append({
            "message": user_input,
            "result": prediction,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        save_history(history)

    return render_template("index.html", prediction=prediction, user_input=user_input)

@app.route("/history")
def history():
    sort = request.args.get("sort", "desc")  # default: terbaru
    history_data = load_history()
    try:
        # Ubah timestamp ke objek datetime dulu supaya bisa diurutkan
        history_data.sort(
            key=lambda x: datetime.strptime(x["timestamp"], "%Y-%m-%d %H:%M:%S"),
            reverse=(sort == "desc")
        )
    except:
        pass
    return render_template("history.html", history=history_data, sort=sort)

@app.route("/delete_history", methods=["POST"])
def delete_history():
    save_history([])
    return jsonify({"message": "Histori berhasil dihapus!"})

@app.route("/delete_entry", methods=["POST"])
def delete_entry():
    index = int(request.form["index"])
    history = load_history()

    if 0 <= index < len(history):
        del history[index]
        save_history(history)
        return jsonify({"message": "Entri berhasil dihapus!"})
    return jsonify({"message": "Gagal menghapus entri!"}), 400

@app.route("/export_json")
def export_json():
    history = load_history()
    export_file = "exported_history.json"
    with open(export_file, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)
    return send_file(export_file, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
