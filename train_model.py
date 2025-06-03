import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 1. Load dataset (pastikan dataset tersedia, misalnya spam.csv)
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# 2. Konversi label menjadi angka (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 3. Pisahkan data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 4. Ubah teks menjadi fitur numerik dengan TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train model SVM
model = SVC(kernel='linear', probability=True)
model.fit(X_train_tfidf, y_train)

# 6. Simpan model dan vectorizer
joblib.dump(model, "svm_spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Training selesai! Model disimpan sebagai 'svm_spam_model.pkl'")
    