#  Electric Vehicle Market Growth Analysis & Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-Classification-orange?style=for-the-badge&logo=scikit-learn"/>
  <img src="https://img.shields.io/badge/Gradio-Interactive%20App-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SciPy-Forecasting-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  <strong>EDA + Forecasting + ML Classification + Interactive Gradio App</strong><br/>
  Analysing the US Electric Vehicle population dataset to understand adoption trends,
  geographic distribution, manufacturer dominance, and predict EV type using ML.
</p>

---

##  Project Overview

This project performs a **comprehensive analysis of the US Electric Vehicle (EV) population dataset**, covering:

1. **Exploratory Data Analysis (EDA)** — adoption trends, geographic distribution, manufacturer popularity, electric range analysis
2. **Market Forecasting** — exponential growth model to project EV registrations from 2024–2029
3. **ML Classification** — Logistic Regression and KNN to predict Electric Vehicle Type (BEV vs PHEV)
4. **Interactive App** — Gradio web interface for real-time EV type prediction

### Key Findings
- EV adoption follows an **exponential growth curve** — registrations are forecasted to grow significantly through 2029
- **Tesla, Chevrolet, and Nissan** dominate the top 3 manufacturer spots
- **King County** leads all Washington State counties in EV registrations
- **BEV (Battery Electric Vehicle)** significantly outnumbers PHEV in the dataset
- Logistic Regression and KNN both achieve strong classification accuracy on EV type prediction

---

##  Dataset

| Field | Detail |
|---|---|
| **Name** | Electric Vehicle Population Data |
| **Source** | US Electric Vehicle Registration Records |
| **Format** | CSV (zipped) |
| **Key Columns** | Model Year, Make, Model, Electric Vehicle Type, County, City, Electric Range, State |
| **Target** | Electric Vehicle Type (BEV / PHEV) |

---

##  Project Workflow

```
Electric Vehicle Population CSV
          │
          ▼
1. Data Loading & Cleaning
   ├── Load CSV, check shape, dtypes
   ├── Handle missing values (dropna)
   └── Descriptive statistics
          │
          ▼
2. Exploratory Data Analysis (EDA)
   ├── EV adoption by model year (bar chart)
   ├── Geographic distribution — top counties & cities
   ├── EV type distribution (BEV vs PHEV)
   ├── Top 10 manufacturers by registrations
   ├── Top models within top 3 manufacturers
   ├── Electric range distribution (histogram + KDE)
   └── Average electric range by model year (trend line)
          │
          ▼
3. Market Forecasting (SciPy curve_fit)
   ├── Fit exponential growth model on 2024 data
   └── Forecast EV registrations: 2024 → 2029
          │
          ▼
4. ML Classification
   ├── Label encoding (Make, Model, Model Year)
   ├── 70/30 train-test split
   ├── Logistic Regression
   ├── K-Nearest Neighbours (KNN)
   └── Classification report (precision, recall, F1)
          │
          ▼
5. Interactive Gradio App
   └── Dropdown-based EV type predictor
       (Select Make → Model → Year → Get Prediction)
```

---

##  Exploratory Data Analysis

### EV Adoption Over Time
Tracks the number of EVs registered per model year — reveals exponential growth post-2018 with a sharp surge from 2020 onward as EV adoption accelerated in the US market.

### Geographic Distribution
Identifies the **top 3 counties** by EV registrations and drills into the **top 10 cities** within those counties. King County (Seattle area) leads significantly, followed by Snohomish and Pierce counties.

### Electric Vehicle Types
Compares **BEV (Battery Electric Vehicle)** vs **PHEV (Plug-in Hybrid Electric Vehicle)** — BEVs dominate the dataset, reflecting the market shift toward fully electric vehicles.

### Top 10 EV Manufacturers
Ranks manufacturers by total registrations. Tesla leads by a substantial margin, followed by Chevrolet and Nissan, with Ford, BMW, and Kia in the mid-tier.

### Electric Range Analysis
- **Distribution:** Right-skewed; most EVs cluster at lower ranges (0–100 miles) with a secondary cluster around 200–300 miles (Tesla models)
- **Trend over years:** Average electric range has increased steadily year-on-year, reflecting battery technology improvement

---

##  Forecasting — Exponential Growth Model

Using **SciPy `curve_fit`** with the exponential growth function:

```python
def exp_growth(x, a, b):
    return a * np.exp(b * x)
```

- Fitted on actual registration data up to 2023
- Projected forward to **2029**
- Results plotted as actual (blue) vs forecasted (red dashed)

| Year | Forecasted EV Registrations |
|---|---|
| 2024 | Projected |
| 2025 | Projected |
| 2026 | Projected |
| 2027 | Projected |
| 2028 | Projected |
| 2029 | Projected |

> Exact forecasted values printed in notebook output.

---

##  Machine Learning Models

### Preprocessing
- **Label Encoding** applied to `Make`, `Model`, and `Model Year` columns
- **Target:** Electric Vehicle Type encoded as binary category codes

### Models

| Model | Algorithm | Use Case |
|---|---|---|
| **Logistic Regression** | Linear classifier | Baseline EV type classification |
| **K-Nearest Neighbours** | Distance-based classifier | Non-linear EV type classification |

### Evaluation
Both models evaluated with a full **classification report** (precision, recall, F1-score, support) on the 30% test split.

---

##  Interactive Gradio App

A **Gradio web interface** was built for real-time EV type prediction:

```
┌─────────────────────────────────────┐
│     EV Vehicle Type Prediction      │
├─────────────────────────────────────┤
│  Select Make:    [Dropdown ▼]       │
│  Select Model:   [Dropdown ▼]       │
│  Select Year:    [Dropdown ▼]       │
├─────────────────────────────────────┤
│  Output: Predicted EV Type: BEV   │
└─────────────────────────────────────┘
```

**How it works:**
1. User selects Make, Model, and Year from dropdowns
2. Inputs are label-encoded using stored encoders
3. Logistic Regression model predicts BEV or PHEV
4. Result displayed instantly in the output box

**To launch:**
```python
# Already included in the notebook — just run the Gradio cell
gr.Interface(...).launch()
```

---

##  Visualizations

### 1. EV Adoption Over Time
> Bar chart — number of vehicles registered per model year. Exponential growth visible post-2018.

### 2. Top Cities in Top Counties
> Horizontal grouped bar chart — top 10 cities across top 3 counties (King, Snohomish, Pierce) by EV registrations.

### 3. EV Type Distribution
> Horizontal bar chart — BEV vs PHEV registration counts. BEV dominates significantly.

### 4. Top 10 EV Makes
> Horizontal bar chart — Tesla leads, followed by Chevrolet, Nissan, Ford, BMW, Kia, Volkswagen, Toyota, Jeep, Hyundai.

### 5. Top Models in Top 3 Makes
> Grouped bar chart — best-selling models within Tesla, Chevrolet, Nissan. Tesla Model Y and Model 3 dominate.

### 6. Electric Range Distribution
> Histogram + KDE — bimodal distribution (short-range PHEVs + long-range Tesla BEVs). Mean range marked with red dashed line.

### 7. Average Electric Range by Year
> Line chart with markers — clear upward trend from 2010 to 2023 showing battery improvement over time.

### 8. Top 10 Models by Average Range
> Grouped bar chart — Tesla Model S and Model X lead on range; top 3 makes compared.

### 9. Current & Estimated EV Market (Forecast)
> Combined line chart — actual registrations (blue dots) + exponential forecast 2024–2029 (red dashed).

---

##  How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/MUQADAS-03/EV-Market-Growth-Analysis.git
cd EV-Market-Growth-Analysis
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy gradio joblib
```

### 3. Add Dataset
Place the dataset file in the root directory and update the path:
```python
# Change this line:
ev_data = pd.read_csv("C:/Users/muqad/Downloads/Electric Vehicle Datasets/Electric_Vehicle_Population_Data.csv.zip")

# To:
ev_data = pd.read_csv("Electric_Vehicle_Population_Data.csv")
```

### 4. Run the Notebook
```bash
jupyter notebook EV_Project.ipynb
```

### 5. Launch the Gradio App
Run all cells in order — the Gradio interface launches automatically in the final cell. A local URL will be provided:
```
Running on local URL:  http://127.0.0.1:7860
```

---

##  Libraries Used

```python
pandas          # Data loading and manipulation
numpy           # Numerical operations and array handling
matplotlib      # Core plotting library
seaborn         # Statistical data visualization
scikit-learn    # ML models, LabelEncoder, train_test_split, metrics
scipy           # curve_fit for exponential growth forecasting
gradio          # Interactive web app for EV type prediction
joblib          # Model serialization (save/load)
```

---

##  Author

**Muqadas Yasin**
---

<p align="center">
  <em>From gas stations to charging stations — the data tells the story. </em>
</p>
