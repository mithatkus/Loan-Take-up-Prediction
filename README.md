# Loan Take-Up Prediction

This project analyzes a randomized control trial (RCT) dataset of 53,000 microcredit customers to predict whether a customer will apply for a loan after receiving a marketing offer. Conducted as a Columbia University Data Analytics group project, the study tests seven machine learning models and finds that the best achievable ROC-AUC is approximately 0.70, which reflects the fundamental unpredictability of human financial decision-making rather than any deficiency in modeling approach.

## Repository Structure

| File | Description |
|---|---|
| `DA_Project.ipynb` | Main Jupyter notebook containing all data preprocessing, model training, evaluation, and analysis |
| `adcontentworth.csv` | Raw dataset — RCT data from a microcredit lender with 53,194 observations |
| `DA Project Report.pdf` | Full written report with detailed methodology, results, and discussion |

## Dataset

The dataset contains 53,194 observations and 27 features after preprocessing. It originates from a field experiment conducted by a microcredit lender in which customers were randomly assigned different marketing offer configurations.

**Feature categories:**
- **Demographics:** race, gender, education level, risk category
- **Behavioral:** account dormancy, transaction count, previous borrowing history
- **Experimental treatments:** interest rate, prize incentive, deadline length, photo types, demographic matching in marketing materials

**Target variable:** `applied` — binary indicator of whether the customer submitted a loan application (0 = no, 1 = yes).

The dataset exhibits severe class imbalance: only 8.5% of customers applied, with 91.5% non-applicants.

**Source:** Bertrand, M., Karlan, D., Mullainathan, S., Shafir, E., & Zinman, J. (2010). *What's Advertising Content Worth? Evidence from a Consumer Credit Marketing Field Experiment.* Quarterly Journal of Economics, 125(1), 263–306.

## Methodology

**Preprocessing:**
- Removed post-application leakage variables that would not be available at prediction time
- Removed rows with null values in the `prize` column
- Dropped `deadlinemed` due to high multicollinearity (VIF = 18.81)
- Applied label encoding for tree-based models and one-hot encoding for logistic regression

**Models trained:**
1. Logistic Regression (standard)
2. Logistic Regression (class-balanced)
3. K-Nearest Neighbors (KNN)
4. Random Forest
5. XGBoost
6. LightGBM
7. Neural Network (MLP)

Class imbalance was addressed through class weighting, threshold tuning, and SMOTE where applicable.

## Results

| Model | AUC | Precision | Recall |
|---|---|---|---|
| Logistic Regression | 0.703 | 0.15 | 0.64 |
| Logistic Regression (Balanced) | 0.688 | 0.65 | 0.63 |
| KNN | 0.670 | 0.33 | 0.00 |
| Random Forest | 0.683 | 0.20 | 0.21 |
| XGBoost | 0.701 | 0.16 | 0.62 |
| LightGBM | 0.701 | 0.16 | 0.62 |
| Neural Network | 0.692 | 0.23 | 0.11 |

Standard Logistic Regression achieves the highest AUC (0.703) and is recommended for deployment due to its interpretability and competitive performance. XGBoost and LightGBM are preferred for customer ranking tasks. All models converge near AUC = 0.70, suggesting a data-imposed performance ceiling rather than a modeling limitation.

## Key Findings

- **Dormancy is the strongest predictor:** Customers with longer periods of account inactivity are significantly less likely to apply, regardless of offer type.
- **Interest rate deters applications:** Higher interest rates (captured by `offer4`) consistently reduce application probability across all models.
- **Risk category and education matter:** Lower-risk and more educated customers apply at higher rates, pointing to financial confidence as a driver of take-up.
- **Gradient boosting captures non-linear interactions** between features but yields only marginal AUC gains over logistic regression, indicating that feature quality limits performance more than model complexity.
- **Neural networks offer no advantage here:** The problem is data-constrained, not model-constrained — additional model complexity does not overcome weak signal in the features.

## 👥 Authors

Nuha Pagarkar, Mithat Kus, Nandini Sistla, Anbo Wang, Xinrong Hao

Columbia University

## 📚 References

Bertrand, M., Karlan, D., Mullainathan, S., Shafir, E., & Zinman, J. (2010). What's Advertising Content Worth? Evidence from a Consumer Credit Marketing Field Experiment. *Quarterly Journal of Economics*, 125(1), 263–306.
