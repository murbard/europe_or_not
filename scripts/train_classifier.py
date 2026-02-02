"""Label cities and train classifiers on embeddings."""

import json
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score


DB_PATH = Path(__file__).parent.parent / "data" / "cities.db"

# Countries clearly NOT in continental Europe
NOT_EUROPE = {
    # Americas
    "Argentina", "Bolivia", "Brazil", "Canada", "Chile", "Colombia", "Cuba",
    "Dominican Republic", "Ecuador", "El Salvador", "Guatemala", "Haiti",
    "Honduras", "Jamaica", "Mexico", "Nicaragua", "Panama", "Peru",
    "United States", "Uruguay", "Venezuela",
    # Africa
    "Algeria", "Angola", "Benin", "Burkina Faso", "Burundi", "Cameroon",
    "Central African Republic", "Chad", "Congo", "Democratic Republic of the Congo",
    "Djibouti", "Egypt", "Eritrea", "Ethiopia", "Gabon", "Ghana", "Guinea",
    "Guinea-Bissau", "Ivory Coast", "Kenya", "Liberia", "Libya", "Malawi",
    "Mali", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda",
    "Senegal", "Sierra Leone", "Somalia", "South Africa", "Sudan", "Tanzania",
    "The Gambia", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe",
    # Asia (non-ambiguous)
    "Afghanistan", "Bangladesh", "Cambodia", "China", "Hong Kong S.A.R.",
    "India", "Indonesia", "Iran", "Japan", "Malaysia", "Mongolia", "Myanmar",
    "Nepal", "North Korea", "Pakistan", "Philippines", "Singapore",
    "South Korea", "Sri Lanka", "Taiwan", "Thailand", "Vietnam",
    # Oceania
    "Australia", "New Zealand",
    # Arabian Peninsula
    "Kuwait", "Oman", "Saudi Arabia", "United Arab Emirates", "Yemen",
    # Other Middle East
    "Iraq", "Israel", "Jordan", "Lebanon", "Syria",
}

# Countries clearly IN continental Europe (conservative)
EUROPE = {
    # Western Europe
    "Austria", "Belgium", "France", "Germany", "Ireland", "Italy",
    "Luxembourg", "Monaco", "Netherlands", "Portugal", "Spain",
    "Switzerland", "United Kingdom",
    # Northern Europe
    "Denmark", "Finland", "Norway", "Sweden",
    # Southern Europe
    "Greece",
    # Central Europe (excluding Poland)
    "Czech Republic", "Hungary", "Slovakia",
    # Balkans (EU/NATO members)
    "Albania", "Bulgaria", "Croatia", "North Macedonia", "Romania", "Slovenia",
}

# Ambiguous countries (eastern delimitation) - left NULL:
# Russia, Ukraine, Belarus, Moldova, Georgia, Azerbaijan, Armenia, Turkey,
# Kazakhstan, Kyrgyzstan, Tajikistan, Turkmenistan, Uzbekistan, Kosovo,
# Bosnia and Herzegovina, Estonia, Latvia, Lithuania, Poland


def apply_labels(conn: sqlite3.Connection) -> dict[str, int]:
    """Apply labels to cities based on country."""
    # Add label column if not exists
    try:
        conn.execute("ALTER TABLE cities ADD COLUMN label TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    cursor = conn.cursor()
    stats = {"europe": 0, "not_europe": 0, "ambiguous": 0}
    
    # Label European countries
    for country in EUROPE:
        result = cursor.execute(
            "UPDATE cities SET label = 'europe' WHERE country = ? AND label IS NULL",
            (country,)
        )
        stats["europe"] += result.rowcount
    
    # Label non-European countries
    for country in NOT_EUROPE:
        result = cursor.execute(
            "UPDATE cities SET label = 'not_europe' WHERE country = ? AND label IS NULL",
            (country,)
        )
        stats["not_europe"] += result.rowcount
    
    conn.commit()
    
    # Count ambiguous
    stats["ambiguous"] = cursor.execute(
        "SELECT COUNT(*) FROM cities WHERE label IS NULL"
    ).fetchone()[0]
    
    return stats


def load_labeled_data(conn: sqlite3.Connection) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load embeddings and labels for labeled cities."""
    cursor = conn.execute("""
        SELECT city, country, embedding, label 
        FROM cities 
        WHERE label IS NOT NULL AND embedding IS NOT NULL
    """)
    
    X = []
    y = []
    cities = []
    
    for city, country, embedding_blob, label in cursor:
        embedding = json.loads(embedding_blob.decode("utf-8"))
        X.append(embedding)
        y.append(1 if label == "europe" else 0)
        cities.append(f"{city}, {country}")
    
    return np.array(X), np.array(y), cities


def train_and_evaluate():
    """Train classifiers and report performance."""
    conn = sqlite3.connect(DB_PATH)
    
    # Apply labels
    print("Applying labels...")
    stats = apply_labels(conn)
    print(f"  Europe: {stats['europe']}")
    print(f"  Not Europe: {stats['not_europe']}")
    print(f"  Ambiguous (unlabeled): {stats['ambiguous']}")
    
    # Load data
    print("\nLoading embeddings...")
    X, y, cities = load_labeled_data(conn)
    print(f"  Loaded {len(X)} labeled cities")
    print(f"  Europe: {y.sum()}, Not Europe: {len(y) - y.sum()}")
    
    # Train Logistic Regression (linear classifier)
    print("\n" + "=" * 50)
    print("LOGISTIC REGRESSION (Linear Classifier)")
    print("=" * 50)
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X, y, cv=5, scoring="accuracy")
    print(f"5-fold CV Accuracy: {lr_scores.mean():.4f} (+/- {lr_scores.std() * 2:.4f})")
    
    # Fit on all data for classification report
    lr.fit(X, y)
    y_pred_lr = lr.predict(X)
    print(f"Training Accuracy: {accuracy_score(y, y_pred_lr):.4f}")
    print("\nClassification Report (on training data):")
    print(classification_report(y, y_pred_lr, target_names=["Not Europe", "Europe"]))
    
    # Train SVM
    print("\n" + "=" * 50)
    print("SVM (RBF Kernel)")
    print("=" * 50)
    
    svm = SVC(kernel="rbf", random_state=42)
    svm_scores = cross_val_score(svm, X, y, cv=5, scoring="accuracy")
    print(f"5-fold CV Accuracy: {svm_scores.mean():.4f} (+/- {svm_scores.std() * 2:.4f})")
    
    # Fit on all data
    svm.fit(X, y)
    y_pred_svm = svm.predict(X)
    print(f"Training Accuracy: {accuracy_score(y, y_pred_svm):.4f}")
    print("\nClassification Report (on training data):")
    print(classification_report(y, y_pred_svm, target_names=["Not Europe", "Europe"]))
    
    # Linear SVM for comparison
    print("\n" + "=" * 50)
    print("SVM (Linear Kernel)")
    print("=" * 50)
    
    svm_linear = SVC(kernel="linear", random_state=42)
    svm_linear_scores = cross_val_score(svm_linear, X, y, cv=5, scoring="accuracy")
    print(f"5-fold CV Accuracy: {svm_linear_scores.mean():.4f} (+/- {svm_linear_scores.std() * 2:.4f})")
    
    svm_linear.fit(X, y)
    y_pred_svm_linear = svm_linear.predict(X)
    print(f"Training Accuracy: {accuracy_score(y, y_pred_svm_linear):.4f}")
    
    conn.close()
    
    return lr, svm, svm_linear


if __name__ == "__main__":
    train_and_evaluate()
