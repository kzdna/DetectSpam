import joblib

# Load model dan vectorizer
try:
    model = joblib.load("svm_spam_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError:
    print("Error: Model atau vectorizer tidak ditemukan! Pastikan Anda sudah menjalankan train_model.py")
    exit()

# Contoh input pesan untuk diuji
test_messages = [
    "Congratulations! You've won a free iPhone. Click the link to claim your prize.",
    "Hey, are you coming to the meeting later?",
    "URGENT! Your account has been compromised. Please reset your password now."
]

# Pastikan input berupa list string
if not all(isinstance(msg, str) for msg in test_messages):
    print("Error: Semua pesan harus dalam bentuk string.")
    exit()

# Transform teks menjadi TF-IDF vektor
try:
    test_messages_tfidf = vectorizer.transform(test_messages)
except AttributeError:
    print("Error: Vectorizer tidak memiliki metode transform(). Pastikan Anda menggunakan TF-IDF dari scikit-learn.")
    exit()

# Lakukan prediksi
predictions = model.predict(test_messages_tfidf)

# Tampilkan hasilnya
for msg, label in zip(test_messages, predictions):
    print(f"Pesan: {msg} => {'SPAM' if label == 1 else 'HAM (bukan spam)'}")
