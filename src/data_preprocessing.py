import pandas as pd

def load_and_preprocess_data(path=None):
    if path:
        df = pd.read_csv(path)
    else:
        import seaborn as sns
        df = sns.load_dataset("titanic")

    features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    df = df[features + ["survived"]]

    df["age"] = df["age"].fillna(df["age"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True)

    X = df.drop("survived", axis=1)
    y = df["survived"]

    return X, y


