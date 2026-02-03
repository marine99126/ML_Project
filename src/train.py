from sklearn.model_selection import train_test_split
from data_preprocessing import load_and_preprocess_data
from model import get_random_forest_model
import joblib

X, y = load_and_preprocess_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = get_random_forest_model()
model.fit(X_train, y_train)

joblib.dump(model, "models/random_forest.pkl")
print("Model saved!")
