import streamlit as st
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

# Load model & vectorizer
model = joblib.load("svm_spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
HISTORY_FILE = "history.json"

# Fungsi cleaning
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

# --- Streamlit UI ---
st.set_page_config(page_title="Deteksi Spam SMS", layout="centered")
st.title("📱 Sistem Deteksi Spam Menggunakan SVM pada Pesan Bahasa Indonesia ")

user_input = st.text_input("Masukkan pesan SMS:")
if st.button("Deteksi"):
    cleaned_input = clean_text(user_input)
    input_vectorized = vectorizer.transform([cleaned_input])
    prediction_result = model.predict(input_vectorized)[0]
    prediction = "SPAM" if prediction_result == 1 else "HAM (Bukan Spam)"

    st.subheader("Hasil Deteksi:")
    st.success(prediction)

    # Simpan ke histori
    history = load_history()
    history.append({
        "message": user_input,
        "result": prediction,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_history(history)

st.markdown("---")
st.subheader("📜 Riwayat Deteksi")

history_data = load_history()
if history_data:
    sort_option = st.radio("Urutkan berdasarkan:", ["Terbaru", "Terlama"])
    reverse = sort_option == "Terbaru"
    history_data.sort(key=lambda x: datetime.strptime(x["timestamp"], "%Y-%m-%d %H:%M:%S"), reverse=reverse)

    for idx, entry in enumerate(history_data):
        with st.expander(f"{entry['timestamp']}"):
            st.write(f"Pesan: {entry['message']}")
            st.write(f"Hasil: {entry['result']}")
            if st.button("❌ Hapus Entri Ini", key=f"del_{idx}"):
                history_data.pop(idx)
                save_history(history_data)
                st.experimental_rerun()

    if st.button("🗑 Hapus Semua Riwayat"):
        save_history([])
        st.experimental_rerun()
else:
    st.info("Belum ada histori deteksi.")

# Tombol untuk ekspor JSON
if st.download_button("📥 Export Riwayat sebagai JSON", json.dumps(history_data, ensure_ascii=False, indent=4), file_name="riwayat_deteksi.json"):
    st.success("Data berhasil diekspor!")
