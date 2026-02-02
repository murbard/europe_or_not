"""GPS-based Europe score using kernel interpolation of city classifier scores.

This module provides a function to score any GPS coordinate from 0-1 based on
how "European" the LLM embedding classifier thinks nearby cities are.
"""

import json
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import numpy as np
from sklearn.linear_model import LogisticRegression

# Try multiple possible database locations
def _find_db_path() -> Path:
    candidates = [
        Path(__file__).parent.parent.parent.parent / "data" / "cities.db",  # src layout
        Path(__file__).parent.parent / "data" / "cities.db",
        Path.cwd() / "data" / "cities.db",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find cities.db in: {candidates}")


DB_PATH = _find_db_path()
EARTH_RADIUS_KM = 6371.0


@dataclass
class City:
    """A city with coordinates and embedding."""
    id: int
    name: str
    country: str
    lat: float
    lon: float
    embedding: np.ndarray
    label: str | None = None


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in km."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return EARTH_RADIUS_KM * c


class EuropeScorer:
    """Scores GPS coordinates for 'Europeanness' using kernel interpolation."""
    
    def __init__(
        self, 
        db_path: Path = DB_PATH,
        kernel_bandwidth_km: float = 500.0,
        min_weight: float = 1e-6,
    ):
        """Initialize scorer.
        
        Args:
            db_path: Path to SQLite database with cities and embeddings.
            kernel_bandwidth_km: Bandwidth for Gaussian kernel in km.
                Smaller = more local influence, larger = smoother interpolation.
            min_weight: Minimum weight threshold to include a city.
        """
        self.kernel_bandwidth_km = kernel_bandwidth_km
        self.min_weight = min_weight
        
        # Load cities and train classifier
        self.cities = self._load_cities(db_path)
        self.classifier = self._train_classifier()
        
        # Pre-compute classifier scores for all cities
        self._precompute_scores()
    
    def _load_cities(self, db_path: Path) -> list[City]:
        """Load all cities with embeddings from database."""
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("""
            SELECT id, city, country, latitude, longitude, embedding, label
            FROM cities
            WHERE embedding IS NOT NULL AND latitude IS NOT NULL
        """)
        
        cities = []
        for row in cursor:
            city_id, name, country, lat, lon, embedding_blob, label = row
            embedding = np.array(json.loads(embedding_blob.decode("utf-8")))
            cities.append(City(
                id=city_id,
                name=name,
                country=country,
                lat=lat,
                lon=lon,
                embedding=embedding,
                label=label,
            ))
        
        conn.close()
        print(f"Loaded {len(cities)} cities with embeddings")
        return cities
    
    def _train_classifier(self) -> LogisticRegression:
        """Train logistic regression classifier on labeled cities."""
        labeled = [c for c in self.cities if c.label is not None]
        
        X = np.array([c.embedding for c in labeled])
        y = np.array([1 if c.label == "europe" else 0 for c in labeled])
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y)
        
        print(f"Trained classifier on {len(labeled)} labeled cities")
        return clf
    
    def _precompute_scores(self) -> None:
        """Compute Europe probability for each city."""
        X = np.array([c.embedding for c in self.cities])
        probs = self.classifier.predict_proba(X)[:, 1]
        
        for city, prob in zip(self.cities, probs):
            city.europe_score = prob
        
        print(f"Computed Europe scores for {len(self.cities)} cities")
    
    def _gaussian_kernel(self, distance_km: float) -> float:
        """Compute Gaussian kernel weight for a distance."""
        return np.exp(-0.5 * (distance_km / self.kernel_bandwidth_km) ** 2)
    
    def score(self, lat: float, lon: float) -> float:
        """Compute Europe score for a GPS coordinate.
        
        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
        
        Returns:
            Score from 0 (not Europe) to 1 (Europe), based on kernel-weighted
            interpolation of nearby city classifier scores.
        """
        total_weight = 0.0
        weighted_score = 0.0
        
        for city in self.cities:
            distance = haversine_distance(lat, lon, city.lat, city.lon)
            weight = self._gaussian_kernel(distance)
            
            if weight > self.min_weight:
                weighted_score += weight * city.europe_score
                total_weight += weight
        
        if total_weight == 0:
            # No cities nearby - return 0.5 (uncertain)
            return 0.5
        
        return weighted_score / total_weight
    
    def score_with_details(self, lat: float, lon: float, top_k: int = 5) -> dict:
        """Score a coordinate and return detailed breakdown.
        
        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            top_k: Number of top contributing cities to return.
        
        Returns:
            Dict with score and top contributing cities.
        """
        contributions = []
        
        for city in self.cities:
            distance = haversine_distance(lat, lon, city.lat, city.lon)
            weight = self._gaussian_kernel(distance)
            
            if weight > self.min_weight:
                contributions.append({
                    "city": city.name,
                    "country": city.country,
                    "distance_km": distance,
                    "weight": weight,
                    "city_score": city.europe_score,
                    "contribution": weight * city.europe_score,
                })
        
        contributions.sort(key=lambda x: x["weight"], reverse=True)
        
        total_weight = sum(c["weight"] for c in contributions)
        weighted_score = sum(c["contribution"] for c in contributions)
        
        score = weighted_score / total_weight if total_weight > 0 else 0.5
        
        return {
            "score": score,
            "lat": lat,
            "lon": lon,
            "total_contributing_cities": len(contributions),
            "top_contributors": contributions[:top_k],
        }


# Global cached scorer instance
_scorer: EuropeScorer | None = None


def get_scorer(bandwidth_km: float = 500.0) -> EuropeScorer:
    """Get or create cached scorer instance."""
    global _scorer
    if _scorer is None or _scorer.kernel_bandwidth_km != bandwidth_km:
        _scorer = EuropeScorer(kernel_bandwidth_km=bandwidth_km)
    return _scorer


def europe_score(lat: float, lon: float, bandwidth_km: float = 500.0) -> float:
    """Convenience function to get Europe score for a coordinate.
    
    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        bandwidth_km: Kernel bandwidth in km (default 500km).
    
    Returns:
        Score from 0 (not Europe) to 1 (definitely Europe).
    
    Example:
        >>> europe_score(48.8566, 2.3522)  # Paris
        0.98...
        >>> europe_score(40.7128, -74.0060)  # New York
        0.01...
    """
    return get_scorer(bandwidth_km).score(lat, lon)


def demo():
    """Demo the scorer with some example locations."""
    scorer = EuropeScorer(kernel_bandwidth_km=500.0)
    
    test_locations = [
        ("Paris, France", 48.8566, 2.3522),
        ("Berlin, Germany", 52.5200, 13.4050),
        ("London, UK", 51.5074, -0.1278),
        ("Moscow, Russia", 55.7558, 37.6173),
        ("Istanbul, Turkey", 41.0082, 28.9784),
        ("New York, USA", 40.7128, -74.0060),
        ("Tokyo, Japan", 35.6762, 139.6503),
        ("Cairo, Egypt", 30.0444, 31.2357),
        ("Kyiv, Ukraine", 50.4501, 30.5234),
        ("Warsaw, Poland", 52.2297, 21.0122),
        ("Tbilisi, Georgia", 41.7151, 44.8271),
        ("Reykjavik, Iceland", 64.1466, -21.9426),
    ]
    
    print(f"\nEurope Score (bandwidth={scorer.kernel_bandwidth_km}km)")
    print("=" * 50)
    
    for name, lat, lon in test_locations:
        result = scorer.score_with_details(lat, lon)
        score = result["score"]
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"{name:25s} [{bar}] {score:.3f}")


if __name__ == "__main__":
    demo()
