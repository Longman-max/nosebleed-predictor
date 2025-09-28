import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from preprocess import load_data, split_features_labels, train_val_split

def evaluate():
    # Load dataset
    df = load_data("data/nosebleed_dataset.csv")
    X, y, target = split_features_labels(df)
    print(f"Detected target column: {target}")

    X_train, X_test, y_train, y_test = train_val_split(X, y)

    # Load trained model (pipeline includes preprocessing)
    model = joblib.load("models/nosebleed_model.pkl")

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Create docs/plots/ folder if it doesn't exist
    plot_dir = os.path.join("docs", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Plot results
    plt.figure(figsize=(6, 4))
    plt.plot(y_test.values[:50], label="True", linestyle="--", color="black")
    plt.plot(y_pred[:50], label="Predicted", color="red")
    plt.legend()
    plt.title("True vs Predicted (first 50 samples)")
    plt.xlabel("Sample index")
    plt.ylabel("Class")

    # Save plot in docs/plots/
    plot_path = os.path.join(plot_dir, "evaluation_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    plt.show()

if __name__ == "__main__":
    evaluate()
