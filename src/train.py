from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from preprocess import load_and_preprocess

X, y = load_and_preprocess(['joy', 'anger', 'sadness', 'fear', 'neutral', 'surprise', 'love'])

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

joblib.dump(model, 'models/emotion_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
