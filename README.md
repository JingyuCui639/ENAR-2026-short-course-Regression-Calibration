# Measurement Error Models in Action  
### The Latest Methods and Their Applications in Nutrition and Environmental Health


![ENAR](https://img.shields.io/badge/event-ENAR%20Short%20Course-orange)
![Maintained](https://img.shields.io/badge/maintained-yes-brightgreen)

Welcome to the repository for the short course:

**Measurement Error Models in Action: The Latest Methods and Their Applications in Nutrition and Environmental Health**

This repository contains the **demo code and datasets** used in the short course. The materials provide hands-on examples demonstrating how to implement modern measurement error correction methods in **R**.

---

# 📚 What You Will Learn

In this short course, participants will gain practical experience applying measurement error correction methods in epidemiologic and nutritional studies.

### 1️⃣ Apply measurement error correction methods in R

Participants will learn how to correct measurement error using two approaches:

- **Imputation-based regression calibration** by R function [regCalibCRS()](docs/refCalibCRS.md)
- **Deattenuation factor method** by R function [regCalibRSW](docs/regCalibRSW.md)

Both methods will be implemented using R functions provided in this repository.

---

### 2️⃣ Handle different validation study designs

Participants will also learn how to apply measurement error correction in different scenarios, including:

- Correcting measurement error in **exposure variables or covariates**
- Working with **validation datasets**, including:
  - Internal validation data
  - External validation data

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

[![Download Materials](https://img.shields.io/badge/Download-Course%20Materials-blue?style=for-the-badge)](data/ENAR_short_course.zip)

After downloading, unzip the file to access the course materials.

## Step 3: Unzip the folder

After unzipping, the folder will contain the following files:

| File | Description |
|-----|-------------|
| `part1_main.dat` | Main study dataset |
| `part1_valid.dat` | External validation dataset |
| `regCalibCRS.R` | R function implementing regression calibration using the **imputation method** |
| `regCalibRSW.R` | R function implementing regression calibration using the **deattenuation factor method** |
| `Regression Calibration demo.qmd` | Quarto document containing the demonstration code |

---

## Step 4: Run the demo

1. Open the file: Regression Calibration demo.qmd in **RStudio**

---

# 📌 Notes

- All datasets and functions required for the demo are included.
- No additional downloads are required.
- The demo code is designed to be run interactively during the course.

---

# 👨‍🏫 Short Course

**Measurement Error Models in Action: The Latest Methods and Their Applications in Nutrition and Environmental Health**

ENAR Short Course

---
