from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_logistic_model():
    return LogisticRegression(max_iter=1000)

def get_random_forest_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
