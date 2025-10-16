# Distributed-ML-for-Marine-Debris-Detection-from-Satellite-Data

![Project Status: In Progress](https://img.shields.io/badge/status-in%20progress-yellow)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![Main Libraries](https://img.shields.io/badge/libraries-Spark%20%7C%20Scikit--learn%20%7C%20Pandas-orange)

This repository contains the work for a university project for the "Distributed Data Analysis and Mining" course. The project focuses on applying distributed computing and machine learning techniques to detect, classify, and analyze floating marine debris from satellite and sensor data.

---

### ⚠️ Project Status

**This is an active university project currently in its initial planning phase.** The structure and code will be developed and pushed to this repository starting from **[Inserisci mese e anno di inizio, es. October 2025]**. The pipeline and goals described below represent the planned workflow, which may evolve as the project progresses.

---

## Project Pipeline

This project follows a structured data analysis pipeline as required by the course syllabus.

### 1. Dataset Description

* **Data Source:** The project utilizes the "Floating Marine Debris Data" dataset, curated by Miguel Mendes Duarte. It is publicly available on [this GitHub repository](https://github.com/miguelmendesduarte/Floating-Marine-Debris-Data).
* **Content:** The dataset is composed of sensor measurements simulating data that could be acquired from Earth Observation platforms. It likely includes spectral bands, GPS coordinates, and other sensor readings associated with different classes of marine debris (e.g., plastic, wood, seaweed) and surrounding water.
* **Structure:** The data is provided in CSV format, suitable for analysis with frameworks like Pandas and Spark.

### 2. Project Goal & Research Questions

The primary goal is to build robust machine learning models capable of automatically identifying and classifying marine debris from sensor data, leveraging distributed computing to handle potentially large datasets efficiently.

Our main research questions are:
1.  **Classification Accuracy:** Can we develop a supervised learning model that accurately classifies different types of marine debris (e.g
