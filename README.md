# ReviewIQ — AI-Powered Customer Review Intelligence System

ReviewIQ is an enterprise-grade NLP and Machine Learning application that transforms raw customer review text into predicted star ratings, flags high-risk negative complaints, uncovers latent product friction themes, and generates prioritized business recommendations.

Built with **Python**, **Streamlit**, **Scikit-learn**, and **Plotly**, ReviewIQ adheres strictly to production ML safeguards: zero duplicate-group train/test leakage, validation-only decision threshold selection, and honest segregation of historical benchmarks from runtime metrics.

---

## 1. System Architecture & Capabilities

```
Raw Review Ingestion (CSV / Single Text)
           │
           ▼
Text Preprocessing & Title Merging ("{Title}. {Body}")
           │
           ▼
Feature Representation (TF-IDF: Unigrams + Bigrams, Sublinear TF, 150K Vocab)
           │
    ┌──────┴───────────────────────────────────────────────────────┐
    ▼                              ▼                               ▼
5-Class Rating Classifier      Binary Negative Detector        Rating Regressor
(Balanced Logistic Regression) (F2-Tuned Threshold: 0.456)   (Ridge Regression: α=2.0)
    │                              │                               │
    ▼                              ▼                               ▼
Discrete Stars (1–5)           Complaint Risk Flag (Yes/No)   Continuous Score (1.0–5.0)
    └──────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
                   Operational Complaint Triage
                   & Evidence-Based Business Recommendations
```

### Dashboard Modules
1. **Executive Dashboard (`pages/1_📊_Executive_Dashboard.py`):** Executive KPIs, actual vs. predicted rating distribution, sentiment share, and temporal review volume trajectories.
2. **Single Review Analysis (`pages/2_🔍_Single_Review_Analysis.py`):** Real-time interactive inference with multi-model agreement, confidence metrics, SLA targets, and support routing.
3. **Batch Review Analysis (`pages/3_📁_Batch_Review_Analysis.py`):** Robust CSV upload, schema auto-detection, batch inference, interactive multi-column filters, and enriched CSV export.
4. **Negative Review & Complaint Triage (`pages/4_🚨_Negative_Review_Triage.py`):** Operational escalation queue featuring a dynamic decision threshold slider and live Precision, Recall, F1, and F2 trade-off curves.
5. **Model Performance & Comparison (`pages/5_📈_Model_Performance.py`):** Transparent evaluation distinguishing 5-class classification, cumulative ordinal modeling, Ridge regression, and 2-star class imbalance diagnostics.
6. **Customer Feedback Topic Analysis (`pages/6_💡_Topic_Analysis.py`):** Unsupervised semantic topic extraction (TF-IDF + NMF) and complaint taxonomy matching (Quality, Logistics, Packaging, Description).
7. **Business Recommendations (`pages/7_🎯_Business_Recommendations.py`):** Quantified, action-oriented recommendations for QA, Fulfillment, Catalog Merchandising, and Customer Support.

---

## 2. Machine Learning Methodology & Safeguards

ReviewIQ strictly preserves the preprocessing and training logic defined in the core project notebook:

| Component | Technical Implementation | Rationale |
|---|---|---|
| **Text Construction** | `f"{title}. {text}".strip(" .")` | Concatenates concise summary headline with the detailed review body. |
| **Data Cleaning** | Drop null bodies, filter valid integer ratings `[1, 5]` | Prevents corrupted records from skewing decision boundaries. |
| **Leakage Safeguard** | `StratifiedGroupKFold` on normalized review bodies | Prevents syndicated/duplicate reviews from spanning train and test sets. |
| **Negative Definition** | Ratings 1–2 = Negative; Ratings 3–5 = Non-Negative | Rating-derived proxy optimized for complaint detection. |
| **Threshold Tuning** | `Threshold = 0.456` selected by max validation $F_2$ | Weights recall more heavily than precision to minimize missed complaints. |
| **Evaluation Honesty** | Segregated historical test set from live uploads | Ground-truth metrics require actual ratings; never fabricated. |

### Reference Historical Test Results (Held-Out Test Set: 6,010 Reviews)
- **TF-IDF + Balanced Logistic Regression:** Accuracy: `69.90%`, Macro F1: `0.5289`, Balanced Acc: `55.42%`
- **TF-IDF + Ridge Regression:** MAE: `0.4654 stars`, RMSE: `0.6865 stars`, $R^2$: `0.5708`
- **Negative-Review Detector:** Recall: `85.90%`, Precision: `60.49%`, $F_1$: `0.7101`, $F_2$: `0.7920`
- **Majority-Class Baseline:** Accuracy: `68.49%`, Macro F1: `0.1626`
- **Cumulative Ordinal Classifier:** Accuracy: `73.88%`, Macro F1: `0.4801`, Two-Star Recall: `10.14%`

---

## 3. Project Structure

```text
reviewiq/
├── app.py                                  # Main application entrypoint
├── pages/
│   ├── 1_📊_Executive_Dashboard.py        # Executive KPI dashboard
│   ├── 2_🔍_Single_Review_Analysis.py      # Interactive single-review prediction
│   ├── 3_📁_Batch_Review_Analysis.py       # Batch CSV upload and processing
│   ├── 4_🚨_Negative_Review_Triage.py       # Negative complaint triage & threshold tuning
│   ├── 5_📈_Model_Performance.py           # Model comparison & performance dashboard
│   ├── 6_💡_Topic_Analysis.py              # Unsupervised NLP topic modeling
│   └── 7_🎯_Business_Recommendations.py    # Evidence-based business recommendations
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                    # Exact text cleaning & combining functions
│   ├── model_loader.py                     # Safe artifact loading & error handling
│   ├── prediction.py                       # Single & batch vectorized inference
│   ├── evaluation.py                       # Live & historical metric calculation
│   ├── topic_analysis.py                   # NMF/BERTopic topic extraction engine
│   ├── recommendations.py                  # Grounded business recommendation generator
│   ├── validation.py                       # File size, schema & rating validation
│   └── ui_helpers.py                       # Plotly theme, layout cards, and sidebar
├── review_rating_artifacts/                # Pre-trained models and evaluation CSVs
│   ├── tfidf_logistic_rating_classifier.joblib
│   ├── tfidf_negative_review_classifier.joblib
│   ├── tfidf_ridge_rating_regressor.joblib
│   ├── ordinal_cumulative_logistic_models.joblib
│   ├── ordinal_tfidf_vectorizer.joblib
│   ├── classification_model_comparison.csv
│   ├── classification_report.csv
│   ├── regression_model_comparison.csv
│   ├── ordinal_classification_report.csv
│   ├── negative_review_metrics.csv
│   └── run_metadata.json
├── data/
│   └── sample_reviews.csv                  # 20 realistic synthetic test reviews
├── tests/                                  # Comprehensive pytest test suite
│   ├── test_preprocessing.py
│   ├── test_model_loader.py
│   ├── test_prediction.py
│   ├── test_validation.py
│   ├── test_evaluation.py
│   └── test_recommendations.py
├── .streamlit/
│   └── config.toml                         # Streamlit theme & server configuration
├── export_artifacts.py                     # Retraining and artifact export script
├── requirements.txt                        # Production dependency specification
├── .gitignore                              # Git exclusion rules
└── README.md                               # Project documentation
```

---

## 4. Installation & Local Execution

### 1. Prerequisites
- Python 3.9 through 3.14
- Git

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-username/reviewiq.git
cd reviewiq

# Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Unit Tests
```bash
python -m pytest tests/
```
All 27 unit tests will execute to verify preprocessing consistency, artifact loading, batch inference, validation guardrails, and metric calculations.

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 5. Artifact Setup & Retraining

The project repository includes pre-trained model artifacts inside `review_rating_artifacts/`. 

To re-run the complete training pipeline and regenerate all models, thresholds, and comparison CSVs from `reviews.csv`:
```bash
python export_artifacts.py
```
This script runs the exact duplicate-group-aware split, validation threshold search ($F_2$ optimization), and serialization.

---

## 6. Deployment to Streamlit Community Cloud

1. **Repository Push:** Push the codebase to a GitHub repository. Ensure `.gitignore` does not exclude `review_rating_artifacts/` if hosting model weights directly in the repo (<25MB total).
2. **Cloud Connection:** Log in to [Streamlit Community Cloud](https://share.streamlit.io), connect your GitHub repository, and specify `app.py` as the main file path.
3. **Environment Secrets:** If you wish to enable the optional Hugging Face zero-shot model with a private token, add `HF_TOKEN = "your_token"` in **Advanced Settings > Secrets**.
4. **Deploy:** Click **Deploy**. The app will start automatically without requiring GPU hardware.

---

## 7. Project Limitations & Ethical Disclosures

- **Text-Only Scope:** Predictions are derived exclusively from linguistic patterns in customer review text. The models cannot verify external physical product failures, carrier transit delays, or subjective customer motivations.
- **Class Imbalance:** Due to standard e-commerce rating distributions (predominantly 5-star reviews), the 2-star category exhibits subtle phrasing that linear models occasionally confuse with 1-star or 3-star reviews.
- **Operational Human-in-the-Loop:** High-priority triage flags should serve as an intake prioritization queue for human support representatives, not as an automated rejection mechanism.
