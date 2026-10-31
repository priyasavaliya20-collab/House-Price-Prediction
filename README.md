# 🏠 House Price Prediction — Supervised Learning

<img width="1600" height="560" alt="download" src="https://github.com/user-attachments/assets/1bf3991a-126c-4d79-ab57-4073b55470d8" />


---

## 📄 Problem Statement

A PropTech startup wants an **automatic house price estimator**. Using the **Ames Housing dataset** (1,460 houses, 79 features), this project builds a **regression model** that predicts `SalePrice`.

The work covers theory, EDA, preprocessing, feature engineering, **Linear / Ridge / Lasso**, **Random Forest**, **XGBoost** (with tuning), cross-validation, residual analysis and a **deployable pipeline** saved as `house_price_model.pkl`.

---










## 📂 Project Files

| 📄 File | 📌 Description |
|---------|----------------|
| 📓 `HousePrice_SupervisedLearning.ipynb` | Main notebook — theory, EDA, preprocessing, modelling, evaluation, residuals, pipeline export |
| 📊 `train.csv` | Ames Housing dataset (1,460 rows × 81 columns) |
| 🤖 `house_price_model.pkl` | Final saved pipeline (preprocessing + tuned XGBoost) |
| 📋 `summary_report.md` | One-page business summary |
| 📦 `requirements.txt` | Python dependencies |
| 📘 `README.md` | Project documentation |

---

## 🛠 Tools Used

<div>

<img src="https://img.shields.io/badge/Python-7440A8?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter%20Notebook-8B5AAE?style=for-the-badge&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-F05A8A?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-D94F7A?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-7440A8?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-F05A8A?style=for-the-badge&logo=xgboost&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-D94F7A?style=for-the-badge&logo=plotly&logoColor=white"/>
<img src="https://img.shields.io/badge/Seaborn-8B5AAE?style=for-the-badge&logoColor=white"/>
<img src="https://img.shields.io/badge/SciPy-7440A8?style=for-the-badge&logo=scipy&logoColor=white"/>
<img src="https://img.shields.io/badge/Joblib-F7A1BB?style=for-the-badge&logoColor=1D1020"/>

</div>

---


## 🎬 Project Demo

<a href="https://drive.google.com/file/d/1oD4nGN8-yN_oPSTUD4Knz48nX6V0lk5o/view?usp=sharing">
  <img src="https://img.shields.io/badge/WATCH%20DEMO-GOOGLE%20DRIVE-F05A8A?style=for-the-badge&logo=googledrive&logoColor=white&labelColor=7440A8"/>
</a>


---



## 🧬 Step 2: Dataset Loading & Exploratory Data Analysis

<img width="1600" height="560" alt="download" src="https://github.com/user-attachments/assets/c66e4b82-0b06-4dad-a16b-7293a138d1ac" />




### 2.1 Load & Understand

```python
df = pd.read_csv('train.csv')      # (1460, 81)
```

🎯 Insight: 38 numerical and 43 categorical columns, and 19 columns have missing values.

### 2.2 Target Distribution

```python
print(df['SalePrice'].skew(), np.log1p(df['SalePrice']).skew())   # 1.88 -> 0.12
```

🎯 Insight: `SalePrice` has a long right tail (skew 1.88). After `log1p` the skew is 0.12, so the model is trained on `log1p(SalePrice)`.

### 2.3 Univariate Analysis

```python
df[num_features].skew().sort_values(ascending=False).head(5)
```

🎯 Insight: Most skewed columns are `MiscVal`, `PoolArea`, `LotArea`, `3SsnPorch` and `LowQualFinSF`. Most houses have 0 in these and only a few have very big values.

### 2.4 Bivariate Analysis

```python
corr = df.corr(numeric_only=True)['SalePrice'].abs().sort_values(ascending=False)
```

🎯 Insight: Strongest correlation with price is `OverallQual` (0.79), then `GrLivArea` (0.71), `GarageCars` (0.64), `GarageArea` (0.62) and `TotalBsmtSF` (0.61). Price rises with every `OverallQual` level, and `NridgHt` is the priciest neighborhood (median $315,000), followed by `NoRidge` and `StoneBr`.

### 2.5 Outlier Detection

```python
df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)]
```

🎯 Insight: Id 524 (4,676 sq.ft., $184,750) and Id 1299 (5,642 sq.ft., $160,000) have a very large area but a low price.

---

## 🔧 Step 3:  Data Preprocessing & Feature Engineering


<img width="1600" height="560" alt="download" src="https://github.com/user-attachments/assets/9daa65f8-b175-4b28-a4f6-693b50805f41" />



### 3.1 Missing Values

```python
drop_cols = list(missing_pct[missing_pct > 80].index)
df[none_cols] = df[none_cols].fillna('None')      # not present -> 'None'
df[zero_cols] = df[zero_cols].fillna(0)           # not present -> 0
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
```

🎯 Insight: Dropped 4 columns with >80% missing (`Alley`, `PoolQC`, `Fence`, `MiscFeature`) plus `Id` and `GarageYrBlt`. Missing values left: **0**.

### 3.2 Outlier Removal

```python
df = df[~((df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000))]   # 1460 -> 1458
```

🎯 Insight: Linear models minimise squared error, so these 2 far-away houses would pull the line down and teach a wrong area-price relation.

### 3.3 Feature Engineering

```python
df['TotalSF']   = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['HouseAge']  = df['YrSold'] - df['YearBuilt']
df['RemodAge']  = df['YrSold'] - df['YearRemodAdd']
df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
df['HasPool']   = (df['PoolArea'] > 0).astype(int)
```

🎯 Insight: Five new features capture total size, age, renovation age and the presence of a garage or pool.

### 3.4 Encoding & Scaling

```python
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), low_cat),
    ('label', high_cat_pipe, high_cat),
    ('flags', 'passthrough', flag_cols)])
```

🎯 Insight: Ordinal encoding for quality columns (`ExterQual`, `KitchenQual`, `BsmtQual`, `FireplaceQu`), `log1p` on 21 skewed columns, one-hot for 32 columns (≤10 categories) and label encoding for `Neighborhood`, `Exterior1st`, `Exterior2nd`.

### 3.5 Train-Test Split

```python
X = df.drop('SalePrice', axis=1)
y = np.log1p(df['SalePrice'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_p = preprocessor.fit_transform(X_train)     # fit on train only
```

🎯 Insight: 1,166 train and 292 test rows with 79 features, which become **217 features** after preprocessing. The preprocessor is fit on train only, so there is no data leakage.

---

## 📐 Step 4: Regression Models — Linear, Ridge & Lasso


<img width="1600" height="560" alt="download" src="https://github.com/user-attachments/assets/b1c3ef3b-6cba-46c4-97f1-0aee2079b552" />



*All metrics are calculated after `expm1()`, so they are in USD.*

### 4.1 Linear Regression (Baseline)

```python
LinearRegression().fit(X_train_p, y_train)
```

🎯 Insight: RMSE $22,646, MAE $15,800, R² 0.9072. Most points are near the diagonal, but the spread is bigger for expensive houses.

### 4.2 Ridge Regression (L2)

```python
RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])       # best alpha = 10
```

🎯 Insight: RMSE $20,442, MAE $14,661, R² 0.9243. Better than the baseline.

### 4.3 Lasso Regression (L1)

```python
LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1], max_iter=10000)   # best alpha = 0.001
```

🎯 Insight: RMSE $19,910, MAE $14,614, R² 0.9282. Lasso made **148 of 217** coefficients exactly zero, so it removed noisy features and beat Ridge.

---

## 🌳 Step 5: Advanced Models — Random Forest & XGBoost Regressor


<img width="1600" height="560" alt="download" src="https://github.com/user-attachments/assets/39e4721a-0b73-4b2e-a7df-cd5c289f1d96" />



### 5.1 Random Forest Regressor

```python
RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
```

🎯 Insight: RMSE $23,168, MAE $16,326, R² 0.9028. This is the weakest model here.

### 5.2 XGBoost Regressor

```python
XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
             subsample=0.8, colsample_bytree=0.8, random_state=42)
```

🎯 Insight: RMSE $19,871, MAE $14,016, R² 0.9285. Much better than Random Forest.

### 5.3 Cross-Validation

```python
cross_val_score(xgb, X_train_p, y_train, cv=5, scoring='neg_root_mean_squared_error')
```

🎯 Insight: Mean CV RMSE is 0.1224 ± 0.0067 and test RMSE is 0.1197 (log scale). They are very close, so the model is stable.

### 5.4 Hyperparameter Tuning

```python
RandomizedSearchCV(XGBRegressor(random_state=42), params, n_iter=20, cv=3,
                   scoring='neg_root_mean_squared_error', random_state=42)
```

🎯 Insight: Best parameters are `n_estimators=500`, `max_depth=3`, `learning_rate=0.03`, `subsample=0.7`, `colsample_bytree=0.6`. The tuned model gives RMSE **$18,411**, MAE $13,539 and R² **0.9386**.

---

## 📈 Step 6: Model Evaluation & Comparison


<img width="1600" height="560" alt="download" src="https://github.com/user-attachments/assets/6fb27343-3259-43e2-af46-0ff885a28649" />



| Model | RMSE ($) | MAE ($) | R² | CV RMSE (log) | Train Time (s) |
|-------|---------:|--------:|---:|--------------:|---------------:|
| Linear Regression | 22,646 | 15,800 | 0.9072 | 0.1336 | 0.11 |
| Ridge | 20,442 | 14,661 | 0.9243 | 0.1177 | 0.08 |
| Lasso | 19,910 | 14,614 | 0.9282 | 0.1176 | 0.54 |
| Random Forest | 23,168 | 16,326 | 0.9028 | 0.1387 | 1.32 |
| XGBoost | 19,871 | 14,016 | 0.9285 | 0.1224 | 2.36 |
| **XGBoost (Tuned)** | **18,411** | **13,539** | **0.9386** | **0.1195** | 2.42 |

🎯 Insight: **Tuned XGBoost** is the best model with the lowest error and highest R². Lasso is very close and easier to explain, so it works as a simple backup model.

---

## 🔍 Step 7: Residual Analysis & Model Interpretation


<img width="1600" height="560" alt="download" src="https://github.com/user-attachments/assets/85c9c4cf-4075-4c07-afac-d8b86e8c9b87" />


### 7.1 Residual Plots

```python
residuals = y_test - best_model.predict(X_test_p)
```

| Check | Result |
|-------|--------|
| Residuals vs Fitted | Randomly scattered around 0, no clear pattern |
| Mean / Std | -0.0071 / 0.113 (log scale) |
| Skewness / Kurtosis | -1.27 / 5.82 |

🎯 Insight: The left tail is a few cheap houses whose price is over-predicted. No big pattern is left in the errors.

### 7.2 Business Interpretation

```python
pd.Series(best_model.feature_importances_, index=feature_names).nlargest(5)
```

| Rank | Feature | Importance | Meaning |
|:---:|---------|-----------:|---------|
| 1 | `ExterQual` | 0.229 | Quality of outside material |
| 2 | `OverallQual` | 0.127 | Overall material and finish quality |
| 3 | `TotalSF` | 0.082 | Total area of the house |
| 4 | `KitchenQual` | 0.047 | Kitchen quality |
| 5 | `GarageCond_TA` | 0.030 | Garage condition is typical |

🎯 Insight: Quality and total area decide the price the most. Median price is $139,450 for typical `ExterQual`, $220,000 for good and $369,086 for excellent. At `OverallQual` 5 the median is $133,000 and at 10 it is $445,365. Houses with central air have a median of $168,000 against $98,000 without it.

---

## 🚀 Step 8: Pipeline & Deployment


<img width="1600" height="560" alt="download" src="https://github.com/user-attachments/assets/6c521343-7e6f-4656-b85e-8f3f88c81606" />



```python
final_pipeline = Pipeline([('preprocessing', preprocessor), ('model', best_model)])
joblib.dump(final_pipeline, 'house_price_model.pkl')

loaded = joblib.load('house_price_model.pkl')
price = np.expm1(loaded.predict(X_test.head(5)))      # back to USD
```

| Actual ($) | Predicted ($) |
|-----------:|--------------:|
| 190,000 | 206,984 |
| 100,000 | 95,076 |
| 115,000 | 112,845 |
| 159,000 | 159,703 |
| 315,500 | 328,948 |

🎯 Insight: One pickle holds preprocessing and model together, so raw house data can be passed directly and predictions are close to actual prices.

---

## 📂 Project Workflow

1. Theory → Regression, Ridge/Lasso, over/underfitting, metrics, k-fold CV
2. EDA → Target skew, correlations, outliers
3. Preprocessing → Missing values, outlier removal, feature engineering, encoding, scaling
4. Linear Models → Linear, Ridge, Lasso
5. Tree Models → Random Forest, XGBoost, CV and tuning
6. Evaluation → RMSE, MAE, R², CV RMSE, training time
7. Residual Analysis → Plots, feature importance, business meaning
8. Deployment → Full pipeline saved with `joblib`

---

## ⚙️ Installation & Setup

```bash
pip install -r requirements.txt
jupyter notebook HousePrice_SupervisedLearning.ipynb
```

> 📌 `house_price_model.pkl` was saved with scikit-learn 1.9.0. Use the same version when loading it to avoid warnings.

---

## ✅ Next Steps

- [ ] Add recent sales and Indian market data with location features
- [ ] Try ensembling (XGBoost + Lasso) and Bayesian optimisation
- [ ] Monitor errors after deployment and retrain regularly

---

## 🙏 Thank You

Thank you for exploring this project! ⭐ Star the repository if you found it helpful.
