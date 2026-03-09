# regCalibRSW

Measurement error correction using regression calibration with the **deattenuation factor method (RC‑DF)**.

---

## Description

`regCalibRSW()` implements **Rosner’s deattenuation factor method** for correcting measurement error in exposures or covariates. The method is an extension of regression calibration under additional assumptions and provides corrected regression coefficients together with corresponding standard errors and p‑values.

Compared with the imputation‑based regression calibration approach (`regCalibCRS()`), the deattenuation factor method is **computationally simpler** and the corrected variance can be obtained using a **closed‑form delta‑method formula**.

The function can be applied to several outcome models, including

- Linear regression (`lm`)
- Generalized linear models (`glm`)
- Cox proportional hazards models (`cox`)

depending on the study design and available information.

Both **external validation studies** and **internal validation studies** are supported.

---

## Usage

```r
regCalibRSW(
  supplyEstimates = FALSE,
  ms,
  vs,
  sur,
  exp,
  covCalib = NULL,
  covOutcome = NULL,
  outcome,
  event = NULL,
  time = NULL,
  method = c("lm","glm","cox"),
  family = NULL,
  link = NULL,
  external = TRUE,
  pointEstimates = NA,
  vcovEstimates = NA
)
```

---

## Arguments

### `supplyEstimates`

Logical indicator specifying whether the **uncorrected regression estimates** will be supplied by the user.

- `FALSE` (default): the function fits the outcome model internally.
- `TRUE`: the user provides uncorrected coefficient estimates and their variance‑covariance matrix.

If `supplyEstimates = TRUE`, the main study dataset `ms` is optional.

---

### `ms`

A data frame containing the **main study dataset**.

The dataset should minimally include variables specified in

- `sur`
- `covCalib`
- `covOutcome` (if applicable)

as well as the outcome variable.

---

### `vs`

A data frame containing the **validation dataset**.

This dataset is used to estimate the calibration relationship between the surrogate variables and the corresponding true variables.

The validation dataset should include:

- the surrogate variables listed in `sur`
- the true variables listed in `exp`
- calibration covariates listed in `covCalib`

---

### `sur`

Character vector specifying **mismeasured exposure variables or covariates** (surrogates) observed in the main study dataset.

Example

```r
sur = c("fqtfatinc","fqcalinc","fqalcinc")
```

---

### `exp`

Character vector specifying the **correctly measured variables** in the validation dataset corresponding to the surrogate variables in `sur`.

The variables in `exp` must have a **one‑to‑one correspondence** with those in `sur`.

Example

```r
exp = c("drtfatinc","drcalinc","dralcinc")
```

---

### `covCalib`

Character vector of **correctly measured covariates included in the calibration model**.

These variables are used when modeling the relationship between the true variables (`exp`) and the surrogate variables (`sur`).

Use `NULL` if no such covariates are included.

Example

```r
covCalib = c("agec")
```

---

### `covOutcome`

Character vector specifying **correctly measured covariates included in the outcome model**.

These covariates are assumed not to be measured with error.

They should **not overlap with variables listed in `covCalib`**.

Use `NULL` if no additional covariates are included.

---

### `outcome`

Name of the **outcome variable** in the main study dataset.

Example

```r
outcome = "case"
```

---

### `event`

Event indicator variable used when `method = "cox"`.

Typically coded as

- `0` = censored
- `1` = event

---

### `time`

Follow‑up time variable required when `method = "cox"`.

---

### `method`

Outcome modeling method.

Available options

- `"lm"` — linear regression
- `"glm"` — generalized linear model
- `"cox"` — Cox proportional hazards model

---

### `family`

Distribution family used in `glm()`.

Example

```r
family = binomial
```

---

### `link`

Link function used in `glm()`.

Example

```r
link = "logit"
```

---

### `external`

Logical indicator specifying whether the validation dataset is **external**.

- `TRUE`: external validation dataset
- `FALSE`: internal validation subset within the main study

---

### `pointEstimates`

Numeric vector of **uncorrected regression coefficient estimates** from the original outcome model.

The intercept estimate must be removed.

---

### `vcovEstimates`

Variance‑covariance matrix of the uncorrected regression coefficient estimates.

The intercept row and column must be removed.

---

## Details

The **deattenuation factor method (RC‑DF)** adjusts regression coefficients for measurement error by estimating a correction factor derived from the relationship between surrogate and true measurements in the validation dataset.

The corrected coefficient is obtained by dividing the naive coefficient estimate by the estimated attenuation factor. The variance of the corrected estimate is obtained using the **delta method**.

Compared with imputation‑based regression calibration, this approach is computationally simpler but relies on stronger modeling assumptions.

---

## Value

`regCalibRSW()` returns regression results corrected for measurement error, including

- corrected regression coefficients
- standard errors
- p‑values

---

## Example

```r
source("regCalibRSW.R")

rcdf <- regCalibRSW(
  supplyEstimates = FALSE,
  ms = main,
  vs = valid,
  sur = c("fqtfatinc","fqcalinc","fqalcinc"),
  exp = c("drtfatinc","drcalinc","dralcinc"),
  covCalib = c("agec"),
  covOutcome = NULL,
  outcome = "case",
  method = "glm",
  family = binomial,
  link = "logit",
  external = TRUE
)
```

---

## See Also

- [`regCalibCRS()`](docs/regCalibCRS.md) for regression calibration using the imputation method.

---

## References

Rosner B, Spiegelman D, Willett WC (1992).  
Correction of logistic regression relative risk estimates and confidence intervals for random within‑person measurement error.  
*American Journal of Epidemiology*, 136(11), 1400‑1413.
