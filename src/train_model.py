import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from preprocess import load_data, split_features_labels, train_val_split, build_preprocessor

def train():
    # Load dataset
    df = load_data("data/nosebleed_dataset.csv")

    # Auto-detect target column
    X, y, target = split_features_labels(df)
    print(f"Detected target column: {target}")

    # Split
    X_train, X_test, y_train, y_test = train_val_split(X, y)

    # Build preprocessing + model pipeline
    preprocessor = build_preprocessor(X)
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "models/nosebleed_model.pkl")
    print("Model training complete. Saved to models/nosebleed_model.pkl")

if __name__ == "__main__":
    train()
