# ────────────────────────────────────────────────
#  Galaxy Count Modeling Project
#  Poisson → Negative Binomial → Collinearity checks → Centering
# ────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from scipy.stats import f_oneway, shapiro, kruskal

# ─── 1. Data loading and basic renaming ────────────────────────────────
data = pd.read_csv("melted_galaxies.csv")

data = data.rename(columns={
    "Redshift (r)": "r",
    "Magnitude (m)": "m",
    "R^2": "r2",
    "r*m": "rm",
    "m^2": "m2",
    "Galaxy count": "Count"
})

print(data)

# Quick look at the response variable distribution
plt.figure(figsize=(7, 4))
plt.hist(data['Count'], bins=40, edgecolor='black')
plt.title("Distribution of Galaxy Count")
plt.xlabel("Count")
plt.ylabel("Frequency")
plt.show()

# ─── 2. Poisson GLM (first attempt) ─────────────────────────────────────
print("\n" + "="*60)
print("Poisson GLM")
print("="*60)

model = smf.glm(
    formula="Count ~ r + r2 + m + rm + m2",
    data=data,
    family=sm.families.Poisson()
).fit()

print(model.summary())

# Residual diagnostics – Poisson
residual_pearson = model.resid_pearson
residual_deviance = model.resid_deviance

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(model.fittedvalues, residual_pearson, alpha=0.6)
plt.axhline(0, color='red', linestyle='--', lw=1.2)
plt.xlabel("Fitted values")
plt.ylabel("Pearson residuals")
plt.title("Poisson – Pearson residuals vs fitted")

plt.subplot(1, 2, 2)
plt.scatter(model.fittedvalues, residual_deviance, alpha=0.6)
plt.axhline(0, color='red', linestyle='--', lw=1.2)
plt.xlabel("Fitted values")
plt.ylabel("Deviance residuals")
plt.title("Poisson – Deviance residuals vs fitted")

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(data["m"], residual_deviance, alpha=0.6)
plt.axhline(0, color='red', linestyle='--', lw=1.2)
plt.xlabel("Magnitude (m)")
plt.ylabel("Deviance residuals")
plt.title("Poisson – Deviance residuals vs m")
plt.show()

# ─── 3. Negative Binomial (main model) ──────────────────────────────────
print("\n" + "="*60)
print("Negative Binomial GLM")
print("="*60)

nb_model = smf.negativebinomial(
    formula="Count ~ r + r2 + m + rm + m2",
    data=data
).fit(maxiter=500)

print(nb_model.summary())

# Residual diagnostics – NB
residual_pearson_nb = nb_model.resid_pearson

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.scatter(nb_model.fittedvalues, residual_pearson_nb, alpha=0.6)
plt.axhline(0, color='red', linestyle='--', lw=1.2)
plt.xlabel("Fitted values")
plt.ylabel("Pearson residuals")
plt.title("NB – Pearson residuals vs fitted")

plt.subplot(2, 2, 2)
plt.scatter(data["m"], residual_pearson_nb, alpha=0.6)
plt.axhline(0, color='red', linestyle='--', lw=1.2)
plt.xlabel("Magnitude (m)")
plt.ylabel("Pearson residuals")
plt.title("NB – Pearson residuals vs m")

plt.subplot(2, 2, 3)
plt.scatter(data["r"], residual_pearson_nb, alpha=0.6)
plt.axhline(0, color='red', linestyle='--', lw=1.2)
plt.xlabel("Redshift (r)")
plt.ylabel("Pearson residuals")
plt.title("NB – Pearson residuals vs r")

plt.tight_layout()
plt.show()

# ─── 4. Non-parametric & parametric group comparisons ───────────────────
print("\n" + "="*60)
print("Group comparisons (Redshift & Magnitude bins)")
print("="*60)

# Create binned categorical variables
data["Redshift_ANOVA"] = pd.cut(
    data["r"], bins=3,
    labels=["Low Redshift", "Medium Redshift", "High Redshift"]
)

data["Magnitude_ANOVA"] = pd.cut(
    data["m"], bins=3,
    labels=["Low Magnitude", "Medium Magnitude", "High Magnitude"]
)

# Shapiro-Wilk normality test per group
print("\nGroups with p > 0.05 (Shapiro-Wilk – arguably normal):")
for category in data['Redshift_ANOVA'].cat.categories:
    group = data.loc[data['Redshift_ANOVA'] == category, "Count"]
    stat, p = shapiro(group)
    if p > 0.05:
        print(f"  {category:20}  p = {p:.4f}")

for category in data['Magnitude_ANOVA'].cat.categories:
    group = data.loc[data['Magnitude_ANOVA'] == category, "Count"]
    stat, p = shapiro(group)
    if p > 0.05:
        print(f"  {category:20}  p = {p:.4f}")

# Kruskal-Wallis (non-parametric)
stat_r, p_r = kruskal(
    data.loc[data["Redshift_ANOVA"]  == "Low Redshift", "Count"],
    data.loc[data["Redshift_ANOVA"]  == "Medium Redshift", "Count"],
    data.loc[data["Redshift_ANOVA"]  == "High Redshift", "Count"]
)

stat_m, p_m = kruskal(
    data.loc[data["Magnitude_ANOVA"] == "Low Magnitude", "Count"],
    data.loc[data["Magnitude_ANOVA"] == "Medium Magnitude", "Count"],
    data.loc[data["Magnitude_ANOVA"] == "High Magnitude", "Count"]
)

print("\nKruskal-Wallis results:")
print(f"Redshift   H = {stat_r:.3f}   p = {p_r:.5f}")
print(f"Magnitude  H = {stat_m:.3f}   p = {p_m:.5f}")

# ANOVA (for comparison – even though normality is questionable)
f_r, pval_r = f_oneway(
    data.loc[data["Redshift_ANOVA"]  == "Low Redshift", "Count"],
    data.loc[data["Redshift_ANOVA"]  == "Medium Redshift", "Count"],
    data.loc[data["Redshift_ANOVA"]  == "High Redshift", "Count"]
)

f_m, pval_m = f_oneway(
    data.loc[data["Magnitude_ANOVA"] == "Low Magnitude", "Count"],
    data.loc[data["Magnitude_ANOVA"] == "Medium Magnitude", "Count"],
    data.loc[data["Magnitude_ANOVA"] == "High Magnitude", "Count"]
)

print("\nANOVA results (for reference):")
print(f"Redshift   F = {f_r:.3f}   p = {pval_r:.5f}")
print(f"Magnitude  F = {f_m:.3f}   p = {pval_m:.5f}")

# ─── 5. Multicollinearity investigation ─────────────────────────────────
print("\n" + "="*60)
print("Multicollinearity checks")
print("="*60)

# Original predictors
predictors = data[["r", "m", "r2", "rm", "m2"]]
corr_matrix = predictors.corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", square=True)
plt.title("Correlation Matrix – Original predictors")
plt.show()

# Centered version
data["r_c"]  = data["r"] - data["r"].mean()
data["r2_c"] = data["r_c"] ** 2
data["m_c"]  = data["m"] - data["m"].mean()
data["m2_c"] = data["m_c"] ** 2
data["rm_c"] = data["r_c"] * data["m_c"]


new_predictors = data[["r_c", "r2_c", "m_c", "rm_c", "m2_c"]]
new_corr_matrix = new_predictors.corr()

plt.figure(figsize=(6, 5))
sns.heatmap(new_corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", square=True)
plt.title("Correlation Matrix – Centered predictors")
plt.show()

# VIF on centered model
X = sm.add_constant(data[["r_c", "r2_c", "m_c", "rm_c", "m2_c"]])

vif = pd.DataFrame()
vif["variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nVariance Inflation Factors (centered):")
print(vif.round(3))

# ─── 6. Final NB model with centered predictors ─────────────────────────
print("\n" + "="*60)
print("Negative Binomial – centered predictors")
print("="*60)

nb_model_centered = smf.negativebinomial(
    "Count ~ r_c + r2_c + m_c + rm_c + m2_c",
    data=data
).fit(method="lbfgs", maxiter=500)

print(nb_model_centered.summary())

# Quick residual check
res_pearson_c = nb_model_centered.resid_pearson

plt.figure(figsize=(6, 4))
plt.scatter(nb_model_centered.fittedvalues, res_pearson_c, alpha=0.6)
plt.axhline(0, color='red', linestyle='--', lw=1.2)
plt.xlabel("Fitted values")
plt.ylabel("Pearson residuals")
plt.title("NB centered – Pearson residuals vs fitted")
plt.show()

nb_model_centered_exc_r2 = smf.negativebinomial(
    "Count ~ r_c + m_c + rm_c + m2_c",
    data=data
).fit(method="lbfgs", maxiter=500)

print(nb_model_centered_exc_r2.summary())

res_pearson_c_exc_r2 = nb_model_centered_exc_r2.resid_pearson

plt.figure(figsize=(6, 4))
plt.scatter(nb_model_centered_exc_r2.fittedvalues, res_pearson_c_exc_r2, alpha=0.6)
plt.axhline(0, color='red', linestyle='--', lw=1.2)
plt.xlabel("Fitted values")
plt.ylabel("Pearson residuals")
plt.title("NB centered – Pearson residuals vs fitted")
plt.show()

# Dispersion statistic
pearson_chi2 = np.sum(res_pearson_c ** 2)
df_resid = nb_model_centered.df_resid
dispersion = pearson_chi2 / df_resid
print(f"\nPearson chi² / df (dispersion) = {dispersion:.4f}")
