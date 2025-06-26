## Credit Scoring Business Understanding

### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord requires banks and financial institutions to hold regulatory capital in proportion to the credit risk they carry. It promotes the use of internal risk-based models, provided they are transparent, auditable, and interpretable. Therefore, any credit scoring model used must be:

- Well-documented: so that regulators can understand the assumptions and decisions made.
- Interpretable: so that business users, auditors, and compliance teams can justify the output of the model, especially when denying or approving credit.

This pushes us toward explainable models such as Logistic Regression with Weight of Evidence (WoE), as these models can provide reasoning behind predictions — unlike “black box” models.

---

### 2. Why is creating a proxy variable necessary, and what are the potential business risks?

In our case, the dataset lacks a direct **"default" label** — we don’t explicitly know who repaid loans vs. who didn’t. To address this, we engineer a **proxy target** using behavioral data (RFM - Recency, Frequency, Monetary value), clustering customers into “likely high-risk” or “low-risk” groups.

This is necessary to:
- Enable supervised learning
- Simulate a risk scoring model in the absence of labeled outcomes

**Business Risks include:**
- **Mislabeling**: Customers could be misclassified, leading to denial of credit to good customers (false positives).
- **Model Bias**: Proxy labels may reflect patterns in historical data that don't translate well to future behavior.
- **Regulatory Non-compliance**: If proxy logic is too opaque, it may violate financial regulations or anti-discrimination laws.

---

### 3. What are the key trade-offs between simple vs. complex models?

| Model Type                     | Advantages                                   | Limitations                                 |
|-------------------------------|----------------------------------------------|---------------------------------------------|
| **Logistic Regression + WoE** | High interpretability, regulator-friendly    | May miss complex patterns, lower accuracy   |
| **Gradient Boosting (GBM)**   | High accuracy, non-linear feature interactions | Hard to explain, less transparent           |

In a regulated financial context:
- **Interpretability** is essential to gain approval from compliance and regulatory bodies.
- **Accuracy** is essential to minimize financial risk.

Thus, the trade-off involves choosing an interpretable model for deployment and using complex models for internal decision support — or using techniques like **SHAP** to explain complex models.

---
