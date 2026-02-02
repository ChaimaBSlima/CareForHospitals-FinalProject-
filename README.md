<img width="2048" height="1152" alt="cover" src="https://github.com/user-attachments/assets/cd37a919-9cb5-4022-a54c-c7e6ea9eeadc" />

# Overview :page_facing_up:

Early Warning System for Hospital Overload (U.S. State-Level)

`CareForHospitals` is an end-to-end machine learning project that predicts next-week hospital stress at the U.S. state level.
It provides early warnings, risk classification, and actionable recommendations to support healthcare planning and regional coordination.

This project was built as a full `ML pipeline`, from raw public health data to predictive models and a live web dashboard.

---

# Project Objectives 📌:

This project predicts **four critical outcomes for the next week** (per U.S. state):

1. **ICU Bed Occupancy (%)**
2. **Inpatient Bed Occupancy (%)**   
3. **Critical Stress Risk (Yes / No)** 
4. **Respiratory Disease Burden (COVID-19, Influenza, RSV)**

These predictions are designed to support **capacity planning, staffing decisions, and regional coordination**.

---

#  Data Source 📊:

- **Dataset:** [Weekly Hospital Respiratory Data (HRD)](https://healthdata.gov/CDC/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/8jcp-h4am/about_data)
- **Provider:** CDC National Healthcare Safety Network (NHSN).
- **Platform:** HealthData.gov
- **Granularity:** State-level, weekly
- **Update Frequency:** Weekly
 
All modeling is performed at the **state-week level**, aggregated across reporting hospitals.

---

# 📁 Project Structure:

```
├── data/
│ ├── raw/
│ ├── cleaned/
│
├── notebooks/
│ ├── 01_cleaning.ipynb
│ ├── 02_modeling.ipynb
│
├── src/
│ ├── preprocess.py
│ ├── train.py
│ ├── predict_next_week.py
│
├── website/
│ ├── app/
│ ├── templates/
│   ├── images/
│ ├── static/
│
├── models/
│ ├── *.joblib
│
├── README.md
└── requirements.txt
```
---
#  Data Cleaning & Feature Engineering 🔧:

**Look at the notebook `cleaning_data.ipynb' for detailed explanation**

- Filtering to 50 U.S. states only
- Converting weekly dates and sorting time-series
- Normalizing percentage columns (0–1 → 0–100)
- Creating lag features:
    - `icu_pct_last_week`
    - `inpatient_pct_last_week`
- Rolling averages (4-week trends)
- Aggregating disease burden signals
- Handling missing values with state-level medians
- All preprocessing logic is fully reproducible in `src/cleaningdata.py`.

---

# Model Experiments & Selection  📈:
**Look at the notebook `training_models.ipynb' for detailed explanation**

###  ICU Bed Occupancy (Regression)
Models tested:
- Baseline (Persistence)
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

 **Selected Model:** Random Forest Regressor  
**Reason:** Lowest MAE/RMSE and strong non-linear pattern capture.


###  Inpatient Bed Occupancy (Regression)
Models tested:
- Baseline (Persistence)
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

 **Selected Model:** Random Forest Regressor  
**Reason:** Best balance of accuracy and robustness.


###  Critical Stress Risk (Classification)
Target:
- Binary indicator derived from ICU + inpatient stress thresholds

Models tested:
- Baseline Classifier
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Initial results showed **class imbalance** (few high-risk states).

** Improvements Applied** 
- Probability threshold tuning
- F1-score optimization
- Confusion matrix analysis

 **Final Model:** Logistic Regression (tuned threshold)  
**Reason:** Best recall–precision tradeoff for early warning use.


### Respiratory Disease Burden (Regression)
Target:
- Combined COVID-19 + Influenza + RSV hospitalizations

Models tested:
- Baseline (Persistence)
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

**Selected Model:** Linear Regression  
**Reason:** Stable performance and better interpretability for large-scale counts.

---

#  Next-Week Forecasting Logic 🤖:

The system automatically:
1. Identifies the **latest available week** in the dataset
2. Treats it as the **current week**
3. Generates predictions for **current week + 7 days**
4. Outputs one forecast row per state

Predictions include:
- ICU %
- Inpatient %
- Critical risk probability & flag
- Disease burden
- Recommended neighboring state for overflow coordination

---

#  Web Application 🌐:

A **Flask web application** was built to demonstrate real-world usability.

### Features
- State selection dashboard
- Top-risk states overview
- Full state names (NY → New York)
- Neighbor-state recommendations
- Clear risk messaging

> This app is a **demonstration prototype**.

<img width="1235" height="643" alt="image" src="https://github.com/user-attachments/assets/10e49791-92dd-4291-b9f0-7747f92393dd" />

---
<img width="1247" height="937" alt="image" src="https://github.com/user-attachments/assets/5a992adc-fdb7-4ea9-a8ec-893c50467ef0" />

---
<img width="1249" height="938" alt="image" src="https://github.com/user-attachments/assets/8df3e580-c175-4b89-ba3b-fddd798f5526" />

---
#  How to Run ▶️:

- Install the Raw data from the website and put it on the `data/raw` directory
**Dataset:** [Weekly Hospital Respiratory Data (HRD)](https://healthdata.gov/CDC/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/8jcp-h4am/about_data)

- Install dependencies
```bash
pip install -r requirements.txt
```

- Clean the data
```bash
python src/cleaningdata.py
```

- Train models
```bash
python src/train.py
```

- Generate forecasts
```bash
python src/predict_next_week.py
```

- Run the web app
```bash
python website/run_app.py
```

- open:
```bash
http://127.0.0.1:5000
```
---

# Technologies 🧰:

### Programming & Libraries
- `Python 3`
- `Pandas` — data manipulation
- `NumPy` — numerical operations
- `Scikit-learn` — machine learning algorithms
- `Joblib` — model serialization

### Visualization & Analysis
- `Matplotlib`
- `Seaborn`

### Web Application
- `Flask` — backend web framework
- `HTML / CSS` — frontend rendering

### Data Source
- [Weekly Hospital Respiratory Data (HRD)](https://healthdata.gov/CDC/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/8jcp-h4am/about_data)

> All models were trained from scratch on the project dataset.

---
# Report of Developments 📈 :

### ✅ Successes

- Built a **fully functional forecasting pipeline** from raw data to predictions
- Successfully predicted **four different healthcare stress indicators**
- Identified **critical-risk states** with improved recall through threshold tuning
- Designed a **clear, user-friendly web interface**
- Implemented **neighbor-state recommendations** to support regional coordination
- Documented **failed models and reasoning**, improving transparency



### ❌ Failures & Challenges

- Initial model performance suffered from **class imbalance** in critical stress prediction
- Some advanced models (e.g., Gradient Boosting) did not outperform simpler approaches
- Disease burden prediction showed **large numerical error values** due to wide value ranges
- Dataset limitations prevented hospital-level predictions

> These failures were intentionally documented and analyzed in the notebook.

---

#  What Can Be Improved 🔧:

- Introduce **class imbalance handling** (SMOTE, class weights)
- Apply **log-transformation** or scaling to disease burden targets
- Automate weekly data ingestion via API
- Add confidence intervals to predictions
- Improve frontend with interactive charts and filters
- Expand beyond state-level if finer-grained data becomes available

---
#  What I Learned 📚:

- How to design **end-to-end ML systems**.
- The importance of **data quality over model complexity**.
- How to evaluate models **based on real-world objectives**, not just metrics.
- Why interpretability matters in healthcare applications.
- How to turn ML outputs into **actionable recommendations**.
- How to communicate failures professionally and transparently.

---
# Next Steps for the Project 🚀 :

- Deploy the application using a production-ready server
- Add role-based access for healthcare professionals
- Integrate real-time or near-real-time updates
- Explore time-series deep learning approaches (LSTM, Temporal CNN)
- Validate predictions with historical backtesting
- Extend recommendations using optimization logic

---

#  Ethical Implications ⚖️:

This project raises important ethical considerations:

- Predictions must **not replace clinical judgment**
- Risk of misinterpretation without proper context
- Data represents aggregated trends, not individual patients
- Transparency and explainability are essential in healthcare ML
- Access to predictions should be **restricted to qualified users**

> Ethical responsibility was prioritized through conservative modeling choices, clear disclaimers, and explainable methods.
---

# Authors :woman_technologist:
- [@ChaimaBSlima](https://github.com/ChaimaBSlima)
- [@ChaimaLinkin](https://www.linkedin.com/in/chaima-ben-slima-35477120a/)
  ### 🚀 About me 
      I'm a Machine Learning developer...
