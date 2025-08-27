**Marketing Spend Optimization** 



## 1) Problem Type

**Binary classification**: predict whether a lead is **Interested (1)** or **Not Interested (0)**.

Target column: `Interest Level` (transformed to 0/1 as described below).

> The notebook does **not** build a regression model, and it does **not** implement a budget reallocation/optimization algorithm. It focuses on predicting lead interest and interpreting model drivers.

---

## 2) Data Ingestion

* Data is loaded directly from an S3 CSV URL:

  * `csv_file_path = "https://s3.amazonaws.com/.../Marketing_Data.csv"` (exact path present in notebook cell).
  * `df = pd.read_csv(csv_file_path)`

---

## 3) Date/Time Parsing

All of the following columns are parsed to `datetime` using the **exact format** `%d-%m-%Y %H:%M`:

* `Lead created`
* `Lead Last Update time`
* `Next activity`
* `Demo Date`

```python
pd.to_datetime(df['Lead created'], format="%d-%m-%Y %H:%M")
```

From `Lead created`, two time features are created:

* `hour_of_day = df['Lead created'].dt.hour`
* `day_of_week = df['Lead created'].dt.weekday`

---

## 4) Target Cleaning & Binarization

Rows with the following `Interest Level` values are **dropped**:

* `"Not called"`, `"Closed"`, `"Invalid Number"`

`NaN` in `Interest Level` is removed. The remaining values are **mapped to binary** according to the notebook’s rule:

* **Positive (1)**: `"Slightly Interested"`, `"Fairly Interested"`, `"Very Interested"`
* **Negative (0)**: `"Not Interested"`, `"No Answer"`

This yields `y = df["Interest Level"]` with values in `{0, 1}`.

---

## 5) Feature Cleaning & Engineering

Columns explicitly **dropped** (as in the notebook):

```
["Lead Id", "Lead Location(Auto)", "Next activity",
 "What are you looking for in Product ?", "Lead Last Update time",
 "Lead Location(Manual)", "Demo Date", "Demo Status", "Closure date"]
```

`Marketing Source` missing values are filled as:

```python
df['Marketing Source'].fillna("Unknown", inplace=True)
```

`What do you do currently ?` is transformed into a **binary flag** for whether the value contains the word "student" (case‑ and space‑normalized):

```python
df['What do you do currently ?'] = (
    df['What do you do currently ?']
      .apply(lambda x: 1 if 'student' in str(x).strip().lower() else 0)
)
```

**Label encoding** (sklearn `LabelEncoder`) is applied to the following categoricals:

* `Marketing Source`
* `Lead Owner`
* `Creation Source`

---

## 6) Final Feature Set (X) and Target (y)

The notebook defines **X** using these **six** features:

```
["Lead Owner", "What do you do currently ?",
 "Marketing Source", "Creation Source", "hour_of_day", "day_of_week"]
```

The **target** is:

```
y = df["Interest Level"]  # 0/1 after mapping
```

---

## 7) Train/Test Split

A simple random split is used (no time‑based split):

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 8) Models Trained (Supervised, Classification)

Three models are **trained** with the **exact** hyperparameters shown below:

1. **RandomForestClassifier**

   ```python
   rf = RandomForestClassifier(n_estimators=300)
   rf.fit(X_train, y_train)
   ```

2. **XGBClassifier** (XGBoost)

   ```python
   xgb = XGBClassifier(
       n_estimators=300,
       objective='binary:logistic',
       tree_method='hist',
       eta=0.1,
       max_depth=3
   )
   xgb.fit(X_train, y_train)
   ```

3. **LGBMClassifier** (LightGBM)

   ```python
   lgb = LGBMClassifier(n_estimators=300)
   lgb.fit(X_train, y_train)
   ```

> Note: The notebook imports `GradientBoostingClassifier` and `AdaBoostClassifier` but **does not** fit/evaluate them.

There is **no cross‑validation** or hyperparameter tuning beyond the values above.

---

## 9) Metrics & Plots

**Printed metric:**

* `accuracy_score` on the test set for each of the three models via a helper:

  ```python
  def get_evaluation_metrics(model_name, model, pred, actual):
      print("Accuracy of %s: " % model_name, accuracy_score(pred, actual))
  ```

  Called as:

  ```python
  get_evaluation_metrics("Random Forest", rf, rf.predict(X_test), y_test)
  get_evaluation_metrics("XGBoost",       xgb, xgb.predict(X_test), y_test)
  get_evaluation_metrics("Light GBM",     lgb, lgb.predict(X_test), y_test)
  ```

### Test‑set results 

* **Random Forest** — Accuracy: **0.7925869049281027**
* **XGBoost** — Accuracy: **0.831257852854949**  *(highest accuracy)*
* **LightGBM** — Accuracy: **0.8400013960631021**

**Curves plotted:**

* **Random Forest**

  * Precision–Recall curve: `PrecisionRecallDisplay.from_estimator(rf, X_test, y_test)`
  * ROC curve: `RocCurveDisplay.from_estimator(rf, X_test, y_test)`
* **LightGBM**

  * Precision–Recall curve: `PrecisionRecallDisplay.from_estimator(lgb, X_test, y_test)`
  * ROC curve: `RocCurveDisplay.from_estimator(lgb, X_test, y_test)`



**Interpretation in the notebook:** Although XGBoost achieved the **highest test accuracy** (\~0.7313), the author notes—based on PR/ROC **plots**—that **LightGBM** provided the “best possible results.” (No numerical AUC/PR‑AUC values are computed; only the plots are shown.)

---

## 10) Explainability (SHAP)

The notebook defines a `shap_analysis(...)` helper and runs SHAP for all three models:

```python
shap_out_rf  = shap_analysis(rf,  X_train, X_test)
shap_out_xgb = shap_analysis(xgb, X_train, X_test)
shap_out_lgb = shap_analysis(lgb, X_train, X_test)
```

It then prints the **global importance (mean |SHAP|)** series for each model.

### SHAP results (Random Forest, positive class)

```
Marketing Source              0.138415
Lead Owner                    0.135276
hour_of_day                   0.048382
day_of_week                   0.041308
What do you do currently ?    0.040668
Creation Source               0.011176
dtype: float64
```

### SHAP results (XGBoost, positive class)

```
Lead Owner                    0.147387
Marketing Source              0.124046
What do you do currently ?    0.039312
hour_of_day                   0.014375
Creation Source               0.007119
day_of_week                   0.005376
dtype: float64
```

### SHAP results (LightGBM, positive class)

```
Lead Owner                    0.139894
Marketing Source              0.137092
What do you do currently ?    0.041990
hour_of_day                   0.024167
day_of_week                   0.015996
Creation Source               0.007599
dtype: float64
```

**Takeaway across models:** `Lead Owner` and `Marketing Source` are consistently the top drivers; temporal signals (`hour_of_day`, `day_of_week`) and the `student` flag contribute moderately; `Creation Source` is lowest among the six features.

---
**Business Impacts of Marketing Budget Optimization**
---

Increase product conversions: Marketing Budget Optimization leads to righ user targeting through right channels/assets which leads to better conversions.

Increase revenue: Increased conversions(as mentioned in point above) will lead to more revenue or buyer engagement. For example, if the company is able to target a user who is more active on Instagram, chances are more that he/she will click on the Ad and add the product to cart. So overall probability of an order increases and hence the revenue.

Improve budget allocation: Over budgeting on non-efficient channels lead to waste of marketing money without getting enough revenue.

Improve Customer Acquisition Cost: Customer Acquisition Cost(CAC) improves if right targeting channels are used for a customer often leading to better repeat rates as well.



---


