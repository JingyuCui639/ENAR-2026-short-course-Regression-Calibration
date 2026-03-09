# regCalibCRS

Measurement error correction by regression calibration using the imputation method.

---

## Description

`regCalibCRS()` implements **regression calibration via imputation** to correct for measurement error in exposures or covariates. The function uses information from a validation dataset to estimate the relationship between the error-prone surrogate measurements and their corresponding true values, and then uses this calibration model to obtain corrected regression estimates in the outcome model.

The function is designed for analyses in which one or more covariates are measured with error and a validation dataset is available. It supports both **external validation** and **internal validation** settings, and can be used with either a **linear model** or a **generalized linear model** for the outcome.

---

## Usage

```r
regCalibCRS(
  ms,
  vs,
  sur,
  exp,
  covCalib = NULL,
  covOutcome = NULL,
  outcome,
  method = c("lm", "glm"),
  external = TRUE,
  vsIndicator = NULL,
  family = NULL,
  link = NULL
)
```
---
## Arguments
`ms`

A data frame containing the **main study dataset**.

This is the primary dataset used for fitting the outcome model after measurement error correction. It should contain:

* the outcome variable specified in `outcome`,

* the error-prone surrogate variables specified in `sur`,

* and any correctly measured covariates included in `covOutcome`.

If an internal validation design is used (`external = FALSE`), then `ms` should also contain the validation-sample indicator specified by `vsIndicator`.

---

`vs`

A data frame containing the **validation dataset**.

This dataset is used to fit the calibration model that links the surrogate variables to the corresponding true variables. In general, `vs` should contain:

* the surrogate variables listed in `sur`,

* the corresponding true variables listed in `exp`,

* and any correctly measured covariates listed in `covCalib`.

If `external = FALSE`, then the validation data are internal to the main study, so the structure of `vs` should be consistent with the internal validation setting used by the function.

---

`sur`

A character vector specifying the names of the **surrogate variables** measured with error.

These are the variables observed in the main study and treated as error-prone versions of the true variables. The order of variables in `sur` must match the order of the corresponding variables in `exp`.

Example
```r
sur = c("fqtfatinc", "fqcalinc", "fqalcinc")
```

In this example, the variables in `sur` are the error-prone measurements used in the main dataset.

---

`exp`

A character vector specifying the names of the **true variables** in the validation dataset.

These are the reference or gold-standard measurements corresponding to the surrogate variables in `sur`. The variables in `exp` must appear in the same order as their matched surrogate variables in `sur`.

Example
```r
exp = c("drtfatinc", "drcalinc", "dralcinc")
```

Here, each variable in `exp` is treated as the true counterpart of the corresponding variable in sur.

---

`covCalib`

A character vector specifying the names of **correctly measured covariates** included in the calibration model.

These variables are used when modeling the relationship between the true variables (`exp`) and the surrogate variables (`sur`). They may improve calibration accuracy when the measurement error structure depends on additional subject characteristics.

Use `NULL` if no additional covariates are included in the calibration model.

Example
```r
covCalib = c("agec")
```
This means that age is included as a correctly measured predictor in the calibration model.

---

`covOutcome`

A character vector specifying the names of **correctly measured covariates** included in the outcome model.

These are covariates that enter the regression model for the outcome together with the corrected exposure variables. They should be measured without error, or treated as correctly measured for the purpose of the analysis.

Use `NULL` if no additional covariates are included in the outcome model.

Example
```r
covOutcome = c("agec")
```

This means that age is adjusted for in the outcome model.

---

`outcome`

A character string giving the name of the **outcome variable** in the main study dataset.

This variable is modeled as the response in the corrected outcome regression.

Example
```r
outcome = "case"
```

---

`method`

A character string specifying the type of outcome model to be fitted.

Currently supported options are:

* `"lm"` for linear regression,

* `"glm"` for generalized linear models.

Example
```r
method = "glm"
```

Choose `"glm"` when the outcome is binary, count, or otherwise modeled through a generalized linear model.

---

`external`

A logical value indicating whether the validation dataset is **external**.

* `TRUE`: the validation dataset is external to the main study.

* `FALSE`: the validation data come from an internal validation subset of the main study.

This argument determines how the calibration information is incorporated into the analysis.

Example
```r
external = TRUE
```

---

`vsIndicator`

A character string specifying the name of the **indicator variable for validation-sample membership**.

This argument is used only when `external = FALSE`, that is, under an internal validation design. It identifies which observations belong to the internal validation subset.

Use `NULL` when `external = TRUE`.

Example
```r
vsIndicator = "valid.ind"
```

---

`family`

The family object passed to `glm()` when `method = "glm"`.

This argument specifies the distributional family of the outcome model, such as `binomial` or `gaussian`.

It is not needed when `method = "lm"`.

Example
```r
family = binomial
```

---

`link`

A character string specifying the link function used in the generalized linear model when `method = "glm"`.

Typical choices include `"logit"`, `"identity"`, and `"log"` depending on the outcome type and model specification.
It is not needed when `method = "lm"`.

Example
```r
link = "logit"
```

---

## Details

The function applies an imputation-based regression calibration approach. A calibration model is first fitted using the validation data to describe the relationship between the true variables and their error-prone surrogates, possibly adjusting for additional correctly measured covariates. The fitted calibration model is then used to replace or correct the error-prone covariates in the main study, after which the outcome model is fitted using the corrected covariates.

This approach is useful when direct use of the surrogate variables would lead to biased estimation because of measurement error.

## Value

`regCalibCRS()` returns measurement error–corrected regression results for the specified outcome model. The returned object typically contains:

* corrected regression coefficient estimates,

* covariance matrix of the parameters.

Depending on the implementation of the function, additional model components may also be returned.

## Notes

* The vectors `sur` and `exp` must have the same length.

* Each element of `sur` must correspond to the matching element of `exp` in the same position.

* Variables listed in `covCalib` and `covOutcome` should be correctly measured.

* When `method = "glm"`, both `family` and `link` should be specified.

* When `external = FALSE`, vsIndicator should be provided.

## Example
source("regCalibCRS.R")

```r
rcim <- regCalibCRS(
  ms = main,
  vs = valid,
  sur = c("fqtfatinc", "fqcalinc", "fqalcinc"),
  exp = c("drtfatinc", "drcalinc", "dralcinc"),
  covCalib = c("agec"),
  covOutcome = c("agec"),
  outcome = "case",
  method = "glm",
  family = binomial,
  link = "logit",
  external = TRUE
)
```

## See Also

* regCalibRSW() for regression calibration using the deattenuation-factor method.

## References
