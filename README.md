
# 🧠 Employee Attrition Analysis & Prediction
### From Exploratory Epidemiology to Production-Ready Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

---

## 📌 Project Overview

> *"Employee attrition costs organisations 50–200% of an employee's annual salary in recruiting, onboarding, and lost productivity."*

This project presents a **rigorous, end-to-end analysis** of voluntary employee attrition using the IBM HR Analytics dataset. It bridges **inferential epidemiology** (identifying causal risk factors via adjusted logistic regression) with **predictive machine learning** (flagging at-risk employees before they leave).

The pipeline is designed to reflect how this analysis would actually be delivered to a **People Analytics or HR leadership team** — with statistical rigour, explainable predictions, and actionable insights.

---

## 🔬 Research Questions

| # | Question | Method |
|---|----------|--------|
| 1 | What individual, job, and organisational factors are **independently associated** with attrition after adjustment for confounders? | Adjusted Logistic Regression (Odds Ratios + 95% CIs) |
| 2 | Which employee profile has the **highest predicted probability** of leaving? | ML ensemble scoring |
| 3 | What is the **minimum feature set** that retains 80% of predictive gain? | SHAP feature importance |
| 4 | Does **overtime moderate** the relationship between job satisfaction and attrition? | Interaction analysis + stratified visualisation |

---

## 🗂️ Repository Structure

```
hr-attrition-analysis/
│
├── HR_Attrition_notebook_.ipynb     # Main analysis notebook
├── data/
│   └── HR_Employee_Attrition.csv    # IBM HR Analytics dataset
├── results/
│   └── figures/                     # All saved plots (300 DPI)
│       ├── Correlation matrix.png
│       ├── Confusion Matrices Across Models.png
│       └── ROC Curves and Precision recall curves.png
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

**Source:** [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — Kaggle

| Property | Value |
|----------|-------|
| Observations | 1,470 employees |
| Features | 35 (after leakage removal: 31) |
| Target | `Attrition` (binary: Yes/No) |
| Class prevalence | ~16% attrition (imbalanced) |
| Missing values | None |

**Variable categories covered:**
- **Demographic:** Age, Gender, MaritalStatus, Education
- **Job:** JobRole, Department, JobLevel, BusinessTravel, OverTime
- **Compensation:** MonthlyIncome, StockOptionLevel, PercentSalaryHike
- **Satisfaction:** JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance, RelationshipSatisfaction
- **Tenure:** YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, TotalWorkingYears

---

## 🔧 Methodology

### 1. Data Quality & Preprocessing
- Zero-variance and data leakage audit (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber` removed)
- Outlier inspection via boxplots across all continuous features
- Encoding: binary mapping for `Attrition` and `OverTime`

### 2. Feature Engineering
| Feature | Rationale |
|---------|-----------|
| `TenureGroup` | Captures non-linear attrition risk across career stages |
| `IncomeBand` | Quantile-based salary tiers to model non-linear effects |
| `SatisfactionIndex` | Composite psychosocial measure (Job + Environment + Relationship) |
| `PromotionStagnation` | Years since promotion relative to tenure — career ceiling signal |
| `OverTimeTenureInteraction` | Long-term overtime may affect employees differently by career stage |
| `Income_JobLevel` | Tests if income effects vary across the organisational hierarchy |

### 3. Statistical Inference (Epidemiological Lens)
- **Chi-square tests** + **Cramér's V** for categorical predictors
- **Independent samples t-tests** + **Cohen's d** for continuous predictors
- **Multicollinearity check** via Variance Inflation Factor (VIF) — Age, TotalWorkingYears, YearsAtCompany correlation handled explicitly
- **Adjusted Logistic Regression** (statsmodels) → Odds Ratios with 95% Confidence Intervals
- Overtime as the primary exposure of interest; confounders: Age, MonthlyIncome, JobSatisfaction, YearsAtCompany, WorkLifeBalance

### 4. Machine Learning Pipeline
```
Raw data
  → Train/Test Split (80/20, stratified)
  → SMOTE oversampling on training set only (prevents data leakage)
  → StandardScaler (fit on train, transform test)
  → 5-Fold Cross-Validation (AUC-ROC)
  → Model comparison & selection
  → SHAP explainability (TreeExplainer)
```

### 5. Models Compared

| Model | Notes |
|-------|-------|
| Logistic Regression | Interpretable baseline; scaled inputs |
| Random Forest | Best balance of accuracy + SHAP explainability |
| Gradient Boosting | High performance; less interpretable |
| SVM (RBF) | Included for completeness; probability calibrated |

### 6. Evaluation Metrics
Given class imbalance (~16% attrition), accuracy is misleading. Primary metrics:
- **AUC-ROC** — discrimination across all thresholds
- **Recall (Sensitivity)** — minimising missed at-risk employees
- **F1-Score** — harmonic mean of precision and recall
- **Precision-Recall curves** — better than ROC under severe imbalance
- **Confusion matrices** — operational false negative cost framing

---

## 📈 Key Results

### Inferential Findings (Logistic Regression)
> Odds Ratios adjusted for Age, MonthlyIncome, JobSatisfaction, YearsAtCompany, WorkLifeBalance

| Predictor | OR | Interpretation |
|-----------|-----|----------------|
| OverTime (Yes vs No) | >2.0 | Employees working overtime have 2× higher odds of attrition |
| MonthlyIncome | <1.0 | Higher income is protective against leaving |
| JobSatisfaction | <1.0 | Each unit increase in satisfaction reduces attrition odds |
| YearsAtCompany | <1.0 | Longer tenure reduces attrition risk |
| WorkLifeBalance | <1.0 | Better work-life balance is associated with retention |

### Predictive Performance (held-out test set)

| Model | AUC-ROC | F1 | Recall |
|-------|---------|-----|--------|
| Random Forest | Best overall | 0.2903 | 0.383 |
| Gradient Boosting | High | 0.3538 | 0.4894 |
| Logistic Regression | Interpretable baseline | 0.4 | 0.617 |
| SVM | Competitive | 0.3673 | 0.5745 |

*Exact values are reported in the notebook output tables.*

### SHAP Feature Importance
Top drivers of attrition probability (via SHAP TreeExplainer on Random Forest):
- `OverTime`
- `MonthlyIncome`
- `Age`
- `TotalWorkingYears`
- `SatisfactionIndex`

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Requirements
```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.2
imbalanced-learn>=0.10
statsmodels>=0.13
shap>=0.41
scipy>=1.9
```

### Run the Analysis
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/hr-attrition-analysis.git
cd hr-attrition-analysis

# Launch notebook
jupyter notebook HR_Attrition_notebook_.ipynb
```


> ## Design Decisions & Lessons Learned

**Why SMOTE only on training data?**  
Applying SMOTE before splitting causes data leakage — synthetic samples derived from test observations inflate evaluation metrics. All oversampling is strictly confined to the training fold.

**Why logistic regression alongside ML models?**  
Logistic regression is used as an *inferential* tool (Odds Ratios, confidence intervals, hypothesis testing), not just a classifier. It answers *why* attrition happens; ML models answer *who* is at risk.

**Why VIF before modelling?**  
Age, TotalWorkingYears, and YearsAtCompany are highly collinear. Including all three inflates standard errors and makes OR interpretation unreliable. VIF-informed variable selection prevents this.

**Why Recall over Accuracy as the primary metric?**  
The cost of missing an at-risk employee (false negative) far exceeds the cost of a false positive intervention. Recall directly optimises for this business priority.

---

## 🎯 Business Applications

This analysis framework is directly transferable to:

- **People Analytics teams** building retention early-warning systems
- **HRIS platforms** (Workday, SAP SuccessFactors) needing explainable risk scores
- **Healthcare workforce planning** — a critical application given global nursing/physician shortages
- **County government HR departments** and public institutions managing civil service retention

---

## 👤 Author

**Ezra Komen Kipyegon**  
BSc Biostatistics | JKUAT, School of Mathematical Sciences  
📧 ezrakomen693@gmail.com  
🔗 [GitHub](https://github.com/ezrakomen693) | [LinkedIn](https://www.linkedin.com/in/YOUR_LINKEDIN)

*Interested in public health data science, workforce analytics, and applied machine learning for real-world decision-making.*

---

## 📄 License



---

## 🙏 Acknowledgements

- IBM for releasing the synthetic HR dataset for educational purposes
- The SHAP library authors (Lundberg & Lee, 2017) for making model explainability accessible
- JKUAT Department of Statistics and Actuarial Science
