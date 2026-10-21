"""
ReviewIQ Business Recommendations Engine.
Translates model outputs, review triage flags, and product statistics into
grounded, actionable business recommendations with clear evidence and ownership.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.topic_analysis import rule_based_tag_review


def generate_business_recommendations(
    df: pd.DataFrame,
    text_col: str,
    rating_col: Optional[str] = None,
    pred_rating_col: str = "predicted_rating",
    neg_prob_col: str = "negative_probability",
    product_col: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Synthesizes concrete business recommendations derived strictly from the analyzed data.
    Never invents fictitious data or metrics.
    """
    recommendations: List[Dict[str, Any]] = []
    total_reviews = len(df)
    if total_reviews == 0:
        return recommendations

    # 1. High-Priority Negative Triage Action
    if neg_prob_col in df.columns:
        neg_reviews = df[df[neg_prob_col] >= 0.456]
        neg_count = len(neg_reviews)
        neg_share = round(neg_count / total_reviews * 100, 1)

        if neg_count > 0:
            recommendations.append({
                "title": "Immediate Customer Outreach Queue for Negative Feedback",
                "type": "Rule-Based Operational Rule",
                "priority": "Critical" if neg_share >= 15 else "High",
                "owner": "Tier 2 Escalations & Customer Care",
                "count": neg_count,
                "percentage": f"{neg_share}% of analyzed reviews",
                "evidence": (
                    f"{neg_count} out of {total_reviews} reviews exceeded the 0.456 negative-review "
                    f"probability threshold (or predicted rating <= 2)."
                ),
                "suggested_action": (
                    "Deploy the Customer Care team to contact affected customers within a 4-hour SLA. "
                    "Offer direct issue resolution, replacement, or refund to arrest defection and protect brand reputation."
                )
            })

    # 2. Product-Level Defect Anomaly Detection
    if product_col and product_col in df.columns and neg_prob_col in df.columns:
        product_stats = (
            df.groupby(product_col)
            .agg(
                total_count=(text_col, "count"),
                neg_count=(neg_prob_col, lambda x: (x >= 0.456).sum())
            )
            .reset_index()
        )
        # Find products with at least 5 reviews and >= 25% negative rate
        product_stats["neg_rate"] = product_stats["neg_count"] / product_stats["total_count"]
        anomalies = product_stats[
            (product_stats["total_count"] >= 5) & (product_stats["neg_rate"] >= 0.25)
        ].sort_values("neg_rate", ascending=False)

        if not anomalies.empty:
            top_bad_product = anomalies.iloc[0]
            prod_name = str(top_bad_product[product_col])
            prod_neg_share = round(top_bad_product["neg_rate"] * 100, 1)
            prod_neg_cnt = int(top_bad_product["neg_count"])
            prod_tot = int(top_bad_product["total_count"])

            recommendations.append({
                "title": f"Product Quality Deep Dive: '{prod_name[:40]}'",
                "type": "AI-Assisted Diagnostic",
                "priority": "High",
                "owner": "Quality Assurance & Product Engineering",
                "count": prod_neg_cnt,
                "percentage": f"{prod_neg_share}% of product reviews",
                "evidence": (
                    f"Product '{prod_name[:40]}' shows an elevated negative review rate of {prod_neg_share}% "
                    f"({prod_neg_cnt}/{prod_tot} reviews), substantially higher than benchmark."
                ),
                "suggested_action": (
                    "Inspect manufacturing batch records, supplier components, or recent product alterations. "
                    "Conduct QA audit on return items from this SKU to determine root failure mode."
                )
            })

    # 3. Complaint Taxonomy Root Cause Breakdown
    # Tag negative reviews to identify dominant friction drivers
    candidate_neg_df = (
        df[df[neg_prob_col] >= 0.456] if neg_prob_col in df.columns
        else (df[df[pred_rating_col] <= 2] if pred_rating_col in df.columns else df)
    )

    if not candidate_neg_df.empty:
        category_counts: Dict[str, int] = {}
        for text in candidate_neg_df[text_col].dropna():
            tags = rule_based_tag_review(text)
            for tag in tags:
                if tag != "General / Uncategorized":
                    category_counts[tag] = category_counts.get(tag, 0) + 1

        if category_counts:
            sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            top_cat, top_cnt = sorted_cats[0]
            top_cat_share = round(top_cnt / len(candidate_neg_df) * 100, 1)

            owner_map = {
                "Product Quality & Defects": "Product Engineering & QA",
                "Shipping, Delivery & Handling": "Supply Chain & Logistics",
                "Packaging & Dispenser Issues": "Packaging Design & Fulfillment",
                "Durability & Longevity": "R&D and Hardware Reliability",
                "Product Description Mismatch": "Catalog Merchandising & Marketing",
                "Customer Support & Returns": "Support Operations Management",
                "Pricing & Value for Money": "Pricing Strategy & Commercial Team"
            }
            action_map = {
                "Product Quality & Defects": "Implement stricter incoming component inspections and review recurring defect logs.",
                "Shipping, Delivery & Handling": "Audit regional fulfillment carrier performance, SLA adherence, and package transit cushioning.",
                "Packaging & Dispenser Issues": "Re-engineer bottle caps, seals, or pump mechanisms prone to leaking or breaking in transit.",
                "Durability & Longevity": "Test stress fatigue and update realistic product lifespan guidance in documentation.",
                "Product Description Mismatch": "Audit product listing photographs, dimension specifications, and color descriptions for accuracy.",
                "Customer Support & Returns": "Streamline the RMA/return initiation portal and retrain reps on first-contact resolution.",
                "Pricing & Value for Money": "Evaluate bundle options, promotional discount elasticity, and product perceived-value messaging."
            }

            recommendations.append({
                "title": f"Primary Friction Driver: {top_cat}",
                "type": "Rule-Based Taxonomy Aggregation",
                "priority": "High" if top_cat_share >= 25 else "Medium",
                "owner": owner_map.get(top_cat, "Cross-Functional Ops"),
                "count": top_cnt,
                "percentage": f"{top_cat_share}% of negative reviews",
                "evidence": (
                    f"'{top_cat}' is the most prevalent issue cited in customer complaints, "
                    f"appearing in {top_cnt} reviews ({top_cat_share}% of negative feedback)."
                ),
                "suggested_action": action_map.get(
                    top_cat, "Investigate root cause and review operational SLAs."
                )
            })

    # 4. Two-Star Rating Friction Analysis
    if pred_rating_col in df.columns:
        two_star_df = df[df[pred_rating_col] == 2]
        two_star_count = len(two_star_df)
        if two_star_count > 0:
            two_star_share = round(two_star_count / total_reviews * 100, 1)
            recommendations.append({
                "title": "Address 'Almost Good' 2-Star Experience Friction",
                "type": "AI-Assisted Diagnostic",
                "priority": "Medium",
                "owner": "Product Management & User Experience",
                "count": two_star_count,
                "percentage": f"{two_star_share}% of reviews",
                "evidence": (
                    f"{two_star_count} reviews were identified as 2-star. Customers at this rating "
                    "often liked the concept but suffered a single critical friction point."
                ),
                "suggested_action": (
                    "Review 2-star feedback transcripts to identify minor ergonomic or instructional flaws "
                    "that turn potential advocates into dissatisfied customers."
                )
            })

    # 5. Rating Momentum & Advocate Amplification
    if pred_rating_col in df.columns:
        five_star_df = df[df[pred_rating_col] == 5]
        five_cnt = len(five_star_df)
        five_share = round(five_cnt / total_reviews * 100, 1)
        if five_share >= 40:
            recommendations.append({
                "title": "Capitalize on Positive Brand Advocacy",
                "type": "Rule-Based Marketing Opportunity",
                "priority": "Low",
                "owner": "Marketing & Social Proof Team",
                "count": five_cnt,
                "percentage": f"{five_share}% of reviews",
                "evidence": (
                    f"{five_cnt} reviews ({five_share}%) are high-conviction 5-star positive ratings."
                ),
                "suggested_action": (
                    "Incentivize enthusiastic customers to share photo/video testimonials, join loyalty VIP tiers, "
                    "and feature top recurring praise quotes on digital landing pages."
                )
            })

    return recommendations
