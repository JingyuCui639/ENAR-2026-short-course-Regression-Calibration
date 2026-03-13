# Measurement Error Models in Action  
## The Latest Methods and Their Applications in Nutrition and Environmental Health


![ENAR](https://img.shields.io/badge/event-ENAR%20Short%20Course-orange)
![Maintained](https://img.shields.io/badge/maintained-yes-brightgreen)

# 👨‍🏫 Welcome to the repository for the short course!


This repository contains the **demo code and datasets** used in the short course. The materials provide hands-on examples demonstrating how to implement modern measurement error correction methods in **R**.


## Course Instructors 

<img src="images/Raymond_Dusty_Rose.jpg" width="150">

**Raymond J. Carroll** is Distinguished Professor of Statistics, Nutrition, and Toxicology at Texas A&M University and one of the leading contributors to the statistical theory and practice of measurement error modeling. He has served as editor of *Biometrics* and *Journal of the American Statistical Association* and is the author of the widely used textbook *Measurement Error in Nonlinear Models*. His research spans measurement error methodology, semiparametric modeling, and applications in epidemiology and biomedical science.


![Donna Spiegelman](images/SC5-Donna-Spiegelman.png)

**Donna Spiegelman** is Professor of Biostatistics at the Yale School of Public Health and founding director of the Center for Methods in Implementation and Prevention Science. She previously served on the faculty at the Harvard T.H. Chan School of Public Health for nearly three decades. Her research focuses on statistical methods for epidemiology, including measurement error correction, causal inference, and methods for complex public health studies. 

![Molin Wang](images/SC5-Molin-Wang.png)

**Molin Wang** is Associate Professor of Epidemiology and Biostatistics at the Harvard T.H. Chan School of Public Health. Her research addresses statistical challenges arising in large epidemiologic cohort studies, with particular emphasis on measurement error methods and their applications in nutritional and environmental epidemiology. She serves as lead statistician for several major cohort studies including the Nurses’ Health Study II and the Health Professionals Follow-up Study. 

![Jingyu Cui](images/SC5-Jingyu-Cui.jpg)

**Jingyu Cui** is a Postdoctoral Associate in the Department of Biostatistics at the Yale School of Public Health. His research focuses on developing statistical methods for complex and high-dimensional data, including measurement error models, adaptive study designs for clinical trials, and statistical inference under imperfect data. 

---

# 📚 What You Will Learn

In this short course, participants will gain practical experience applying measurement error correction methods in epidemiologic and nutritional studies.

### 1️⃣ Apply measurement error correction methods in R

Participants will learn how to correct measurement error using two approaches:

- **Imputation-based regression calibration** by R function [regCalibCRS](docs/regCalibCRS_help.md) (Wenze Tang, Molin Wang)
- **Deattenuation factor method** by R function [regCalibRSW](docs/regCalibRSW_help.md) (Wenze Tang, Molin Wang)

Both methods will be implemented using R functions provided in this repository.

---

### 2️⃣ Handle different validation study designs

Participants will also learn how to apply measurement error correction in different scenarios, including:

- Correcting measurement error in **exposure variables**
- Consider **nonlinearity** in the measurement error model.

These methods will be illustrated through practical examples.


---

# ⚙️ Setup Instructions

## Step 1: Install R and RStudio

Please ensure that the following software is installed on your computer.

**R**

https://cran.r-project.org/

**RStudio**

https://posit.co/download/rstudio-desktop/

---

## Step 2: ⬇️ Download Course Materials

Click the button below to download all files required for the demo.

[![Download Materials](https://img.shields.io/badge/Download-Course%20Materials-blue?style=for-the-badge)](course_material/ENAR_short_course.zip)

After downloading, unzip the file to access the course materials.

## Step 3: Unzip the folder

After unzipping, the folder will contain the following files:

| File | Description |
|-----|-------------|
| `main_data_external.csv` | Main study dataset |
| `valid_data_external.csv` | External validation dataset |
| `regCalibCRS.R` | R function implementing regression calibration using the **imputation method** |
| `regCalibRSW.R` | R function implementing regression calibration using the **deattenuation factor method** |
| 'testLinear.R' | R function testing the linearity of measurement error and outcome models |
| `Regression Calibration demo2.qmd` | Quarto document containing the demonstration code |

---

## Step 4: Run the demo

1. Open the file: Regression Calibration demo.qmd in **RStudio**

---

## Solution of demo Code

Please see here for the solution of the demo code: [solution](https://jingyucui639.github.io/ENAR-2026-short-course-Regression-Calibration/).


# 📌 Notes

- [Course Slides](/course_material) are available in the folder `/course_material`.
- The documentation for [regCalibCRS](docs/regCalibCRS_help.md) to implement imputation-based regression calibration method;
- The documentation for  [regCalibRSW](docs/regCalibRSW_help.md) to implement deattenuation factor method.




