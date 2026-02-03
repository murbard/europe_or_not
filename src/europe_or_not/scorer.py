"""GPS-based Europe score using adaptive kernel interpolation.

This module provides a scorer that uses adaptive bandwidth (based on k-nearest neighbors)
to score coordinates based on city embeddings, supporting both SVM and Linear classifiers.
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Try multiple possible database locations
def _find_db_path() -> Path:
    base = Path(__file__)
    candidates = [
        base.parent.parent.parent.parent / "data" / "cities.db",  # relative to src (dev)
        base.parent.parent / "data" / "cities.db",              # installed
        Path.cwd() / "data" / "cities.db",                      # current dir
        Path("/home/arthurb/src/europe_or_not/data/cities.db"), # absolute fallback
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    # If not found, look relative to this file
    local_db = Path(__file__).parent.parent.parent / "data" / "cities.db"
    if local_db.exists():
        return local_db
    
    raise FileNotFoundError(f"Could not find cities.db in: {candidates}")


DB_PATH = _find_db_path()
EARTH_RADIUS_KM = 6371.0


def haversine_vectorized(lat1: float, lon1: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance in km."""
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lats_rad, lons_rad = np.radians(lats), np.radians(lons)
    dlat, dlon = lats_rad - lat1_rad, lons_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lats_rad) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


class MultiClassifierScorer:
    """Scorer that supports multiple classifiers (SVM, Linear) with adaptive bandwidth."""
    
    def __init__(self, db_path: Path = DB_PATH, k_neighbors: int = 10, min_bandwidth_km: float = 15.0):
        self.k_neighbors = k_neighbors
        self.min_bandwidth = min_bandwidth_km
        self.min_weight = 1e-6
        
        # Load cities
        print(f"Loading cities from {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("""
            SELECT city, country, latitude, longitude, embedding, label
            FROM cities WHERE embedding IS NOT NULL AND latitude IS NOT NULL
        """)
        
        self.city_lats = []
        self.city_lons = []
        embeddings = []
        labels = []
        self.is_real_city = []  # Track which entries are real cities vs grid points
        
        for name, country, lat, lon, emb_blob, label in cursor:
            self.city_lats.append(lat)
            self.city_lons.append(lon)
            # Read binary embedding
            embeddings.append(np.frombuffer(emb_blob, dtype=np.float32))
            labels.append(label)
            self.is_real_city.append(not name.startswith("Grid_"))
        
        conn.close()
        
        self.city_lats = np.array(self.city_lats)
        self.city_lons = np.array(self.city_lons)
        self.embeddings = np.array(embeddings)
        self.labels = np.array(labels)
        self.is_real_city = np.array(self.is_real_city)
        
        print(f"Loaded {len(self.city_lats)} points ({np.sum(self.is_real_city)} real cities)")
        
        # Train classifiers
        self._train_classifiers()
    
    def _train_classifiers(self):
        """Train both SVM and Linear classifiers."""
        labeled_mask = np.array([l is not None for l in self.labels])
        X = self.embeddings[labeled_mask]
        
        # Handle labels: '1'/'europe' -> 1, else 0
        # NOTE: Using strict comparison to known positives for 1. Everything else (0, '0', 'not_europe') is 0.
        y_labels = self.labels[labeled_mask]
        y = np.array([1 if str(l).lower() in ('1', 'europe') else 0 for l in y_labels])
        
        print(f"Training training set: {len(X)} samples (Europe: {np.sum(y)}, Not Europe: {len(y) - np.sum(y)})")
        
        # SVM classifier
        self.svm = SVC(kernel='rbf', probability=True, random_state=42)
        self.svm.fit(X, y)
        
        # Linear classifier
        self.linear = LogisticRegression(max_iter=1000, random_state=42)
        self.linear.fit(X, y)
        
        # Pre-compute scores for all cities
        self.svm_scores = self.svm.predict_proba(self.embeddings)[:, 1]
        self.linear_scores = self.linear.predict_proba(self.embeddings)[:, 1]
        
        print(f"Trained SVM and Linear classifiers")
    
    def score_both(self, lat: float, lon: float) -> tuple[float, float]:
        """Score with adaptive bandwidth, returning (svm_score, linear_score)."""
        distances = haversine_vectorized(lat, lon, self.city_lats, self.city_lons)
        
        # Adaptive bandwidth
        sorted_dist = np.sort(distances)
        k_idx = min(self.k_neighbors, len(distances) - 1)
        k_dist = sorted_dist[k_idx]
        
        # Adaptive bandwidth: 50% of distance to k-th neighbor, min threshold
        bandwidth = max(k_dist * 0.5, self.min_bandwidth)
        
        weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
        mask = weights > self.min_weight
        
        if not mask.any():
            return 0.5, 0.5
        
        w_sum = np.sum(weights[mask])
        svm_score = np.sum(weights[mask] * self.svm_scores[mask]) / w_sum
        linear_score = np.sum(weights[mask] * self.linear_scores[mask]) / w_sum
        
        return svm_score, linear_score

# Helper for getting the default scorer
_scorer = None
def get_scorer() -> MultiClassifierScorer:
    global _scorer
    if _scorer is None:
        _scorer = MultiClassifierScorer()
    return _scorer
