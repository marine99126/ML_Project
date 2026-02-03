from sklearn.metrics import accuracy_score, classification_report
import joblib
from data_preprocessing import load_and_preprocess_data

model = joblib.load("models/random_forest.pkl")

X, y = load_and_preprocess_data()
preds = model.predict(X)

print("Accuracy:", accuracy_score(y, preds))
print(classification_report(y, preds))
