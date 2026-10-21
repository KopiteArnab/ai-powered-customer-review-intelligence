"""
ReviewIQ Topic Analysis Module.
Provides modular topic discovery and complaint thematic clustering using
scikit-learn (NMF/LDA), rule-based complaint taxonomy matching, and optional BERTopic support.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

# Domain complaint seed categories for taxonomy guidance and labeling
TAXONOMY_KEYWORDS = {
    "Product Quality & Defects": [
        "broken", "defect", "poor quality", "cheap", "stopped working", "fell apart",
        "faulty", "smell", "scent", "leak", "leaking", "stain", "taste", "terrible",
        "awful", "bad quality", "flimsy", "irritation", "burn", "skin"
    ],
    "Shipping, Delivery & Handling": [
        "delivery", "shipping", "shipped", "arrived", "late", "delayed", "damaged box",
        "broken in transit", "package arrived", "courier", "postal", "tracking", "lost"
    ],
    "Packaging & Dispenser Issues": [
        "package", "packaging", "bottle", "pump", "cap", "lid", "dispenser", "seal",
        "cracked", "unsealed", "container", "spray", "squeeze", "broken seal"
    ],
    "Durability & Longevity": [
        "durability", "durable", "lasting", "broke after", "one day", "one week",
        "month", "wear and tear", "short lived", "ruined", "longevity", "burnout"
    ],
    "Product Description Mismatch": [
        "not as pictured", "misleading", "different", "not as described", "wrong color",
        "wrong size", "expected", "advertised", "picture", "photo", "smaller than expected"
    ],
    "Customer Support & Returns": [
        "customer service", "support", "refund", "return", "warranty", "replacement",
        "unhelpful", "agent", "call center", "email", "policy", "refused"
    ],
    "Pricing & Value for Money": [
        "price", "expensive", "waste of money", "overpriced", "worth", "cost",
        "value", "rip off", "cheaper elsewhere", "not worth"
    ]
}


def rule_based_tag_review(text: str) -> List[str]:
    """Tags a review with matching business complaint categories using keyword heuristics."""
    text_lower = str(text).lower()
    matches = []
    for category, keywords in TAXONOMY_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                matches.append(category)
                break
    return matches if matches else ["General / Uncategorized"]


def discover_topics_nmf(
    texts: List[str],
    n_topics: int = 5,
    top_words: int = 7
) -> Dict[str, Any]:
    """
    Fast, highly interpretable unsupervised topic discovery using TF-IDF + NMF.
    Runs reliably without heavy neural downloads.
    """
    if len(texts) < n_topics:
        n_topics = max(1, len(texts))

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_df=0.85,
        min_df=2 if len(texts) >= 10 else 1,
        ngram_range=(1, 2),
        max_features=5000
    )

    try:
        tfidf = vectorizer.fit_transform(texts)
    except Exception as e:
        return {
            "success": False,
            "error": f"TF-IDF vectorization failed: {e}",
            "topic_assignments": [0] * len(texts),
            "topics": {}
        }

    nmf = NMF(
        n_components=n_topics,
        random_state=42,
        init="nndsvda",
        max_iter=300
    )
    doc_topic_matrix = nmf.fit_transform(tfidf)
    feature_names = np.array(vectorizer.get_feature_names_out())

    topics = {}
    for topic_idx, topic in enumerate(nmf.components_):
        top_features_ind = topic.argsort()[: -top_words - 1 : -1]
        top_terms = feature_names[top_features_ind].tolist()
        
        # Try to map top terms to taxonomy title if applicable
        matched_cat = None
        for cat, kws in TAXONOMY_KEYWORDS.items():
            if any(term in " ".join(kws) for term in top_terms):
                matched_cat = cat
                break

        topic_label = matched_cat if matched_cat else f"Topic {topic_idx + 1}: {', '.join(top_terms[:3])}"
        topics[topic_idx] = {
            "label": topic_label,
            "keywords": top_terms,
            "weight": float(np.sum(doc_topic_matrix[:, topic_idx]))
        }

    # Best topic per review and assignment strength
    best_topics = np.argmax(doc_topic_matrix, axis=1)
    topic_weights = np.max(doc_topic_matrix, axis=1)

    return {
        "success": True,
        "n_topics": n_topics,
        "topics": topics,
        "assignments": best_topics.tolist(),
        "topic_weights": topic_weights.tolist(),
        "method": "TF-IDF + NMF (Non-Negative Matrix Factorization)"
    }


def analyze_topic_breakdown(
    df: pd.DataFrame,
    topic_labels: List[str],
    text_col: str,
    rating_col: Optional[str] = None,
    neg_flag_col: Optional[str] = None,
    date_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes topic-level frequency, negative review rate, representative reviews,
    and temporal trends when date column is available.
    """
    data = df.copy()
    data["topic_label"] = topic_labels

    topic_counts = data["topic_label"].value_counts()
    topic_summary = []

    for label, count in topic_counts.items():
        subset = data[data["topic_label"] == label]
        
        # Calculate negative rate if rating or prediction available
        neg_rate = 0.0
        if neg_flag_col and neg_flag_col in subset.columns:
            neg_rate = float(subset[neg_flag_col].astype(bool).mean() * 100)
        elif rating_col and rating_col in subset.columns:
            valid_ratings = pd.to_numeric(subset[rating_col], errors="coerce").dropna()
            if len(valid_ratings) > 0:
                neg_rate = float((valid_ratings <= 2).mean() * 100)

        # Representative review sample (2 examples)
        sample_texts = subset[text_col].dropna().astype(str).tolist()[:2]

        topic_summary.append({
            "Topic": label,
            "Review Count": int(count),
            "Share (%)": round(count / len(data) * 100, 1),
            "Negative Review Rate (%)": round(neg_rate, 1),
            "Representative Examples": sample_texts
        })

    summary_df = pd.DataFrame(topic_summary)

    # Temporal trends if date column exists
    trend_data = None
    if date_col and date_col in data.columns:
        try:
            data["_date"] = pd.to_datetime(data[date_col], errors="coerce")
            valid_dates = data.dropna(subset=["_date"]).copy()
            if len(valid_dates) > 10:
                valid_dates["_period"] = valid_dates["_date"].dt.to_period("M").dt.to_timestamp()
                trends = (
                    valid_dates.groupby(["_period", "topic_label"])
                    .size()
                    .reset_index(name="review_count")
                )
                trend_data = trends
        except Exception:
            trend_data = None

    return {
        "summary_table": summary_df,
        "trend_data": trend_data
    }


def is_bertopic_available() -> bool:
    """Checks if BERTopic is installed in the current environment."""
    try:
        import bertopic
        return True
    except ImportError:
        return False
