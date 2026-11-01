# Summary Report - House Price Prediction

## Business Problem & Dataset





<img width="1600" height="560" alt="download" src="https://github.com/user-attachments/assets/c500fb29-8539-40e4-96bd-ab8c938a4651" />








A PropTech startup (like NoBroker / MagicBricks) wants an automatic house price estimator. I used the Ames Housing dataset (`train.csv`, 1,460 houses, 79 features) and built a regression model to predict `SalePrice`.

## Preprocessing & Feature Engineering
- **Target:** `SalePrice` was right skewed (1.88), so I used `log1p` (skew 0.12). Metrics are shown in USD after `expm1`.
- **Missing values:** dropped 4 columns with more than 80% missing, filled "not present" columns with `None` / 0, and `LotFrontage` with the median.
- **Outliers:** removed 2 houses with very large area but low price, because they pull the regression line down.
- **New features:** `TotalSF`, `HouseAge`, `RemodAge`, `HasGarage`, `HasPool`.
- **Encoding & scaling:** ordinal encoding for quality columns, one-hot for columns with up to 10 categories, label encoding for the rest, `log1p` on skewed columns and `StandardScaler`, all in a `ColumnTransformer`.
- 80/20 train-test split with `random_state=42`.

## Best Model
| Model | RMSE ($) | MAE ($) | R² |
|---|---|---|---|
| Linear Regression | 22,646 | 15,800 | 0.907 |
| Ridge | 20,442 | 14,661 | 0.924 |
| Lasso | 19,910 | 14,614 | 0.928 |
| Random Forest | 23,039 | 16,213 | 0.904 |
| XGBoost | 19,470 | 14,144 | 0.931 |
| **XGBoost (Tuned)** | **19,048** | **13,902** | **0.934** |

Tuned XGBoost is the best because it has the lowest error and highest R². Lasso is very close and easy to explain. The CV RMSE (0.12 on log scale) is close to the test RMSE, so the model is stable. The residuals are randomly scattered around zero, with only a few cheap houses that are over-predicted, so the model has no big pattern left in its errors.

## Top 3 Predictive Features
1. **ExterQual** - quality of the outside material.
2. **OverallQual** - overall quality of material and finish.
3. **TotalSF** - total area of the house.

Other important features are `CentralAir` and `KitchenQual`. For a real estate professional, this means quality of the house and total area are the main things that decide the price, so a better finish (for example a better kitchen) and a bigger usable area can raise the price, and listings should always show them clearly.

## Next Steps
- Collect more data (recent sales, Indian market data) and add location features.
- Try ensembling (XGBoost + Lasso) and Bayesian optimisation.
- Monitor the error after deployment and retrain regularly.
