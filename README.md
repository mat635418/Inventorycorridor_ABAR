MEIO for Raw Materials (v1.05 — Jan 2026)
Executive Summary

The MEIO for Raw Materials tool is a Multi‑Echelon Inventory Optimization application designed to calculate statistically sound safety stock levels across a network of locations (plants, warehouses, hubs) for raw materials.

It combines:

Historical consumption variability

Forward-looking demand forecasts

Network topology (upstream/downstream relationships)

Lead time uncertainty

Service‑level policies by network tier

The output is a time‑phased inventory corridor (Forecast + Safety Stock) per material and location, fully explainable and auditable.

1. Business Problem Addressed

Raw material inventory decisions are often made:

Locally (ignoring downstream demand)

Using fixed rules of thumb (days of stock)

Without accounting for demand & lead time variability

This leads to:

Excess upstream buffers

Stock‑outs downstream

Poor capital allocation

MEIO solves this by optimizing safety stock at every node based on its true network exposure.

2. High‑Level Architecture
Inputs (CSV)

Sales / History (sales.csv)

Historical consumption

Historical forecast

Future Demand (demand.csv)

Forward forecast by Product / Location / Month

Lead Time Network (leadtime.csv)

From → To relationships

Mean lead time and variability

Outputs

Safety Stock per Product / Location / Month

Inventory Corridor (SS + Forecast)

Efficiency metrics and diagnostics

Full exportable plan

3. Core Concept: Network‑Aware Demand

For each Product – Location – Month, the tool computes:

Aggregated Future Demand

Agg_Future_Demand = Local Forecast + All Downstream Forecasts

This is done by:

Building a directed graph from lead‑time routes

Traversing all reachable downstream nodes (transitive)

Summing their forecast demand

This ensures upstream nodes protect their own demand plus everything they feed.

4. Demand Variability Modeling
Local Variability

Historical standard deviation of consumption per Product / Location

Missing or zero values are filled using:

Product median

Global median fallback

Network Variability Aggregation

Variance is aggregated as:

σ²_network = σ²_local + Σ σ²_downstream

(Optionally scaled with a correlation factor ρ)

Result:

Agg_Std_Hist = true network‑level demand uncertainty

5. Lead Time Uncertainty Integration

Safety stock accounts for two independent risk drivers:

Demand variability during lead time

Lead time variability itself

The implemented formula:

Var(total) = (σ_d² × LT_mean) + (σ_LT² × D̄²)

Where:

σ_d = daily demand std dev

D̄ = average daily demand

This avoids under‑buffering long and unstable supply routes.

6. Tier‑Based Service Level Policy
Network Tiering (Hop Distance)

Each node is classified by distance to final consumption nodes:

Hop Distance	Meaning	Service Level
0	End node	99%
1	1 step upstream	95%
2	2 steps upstream	90%
≥3	Deep upstream	85%

This reflects risk pooling: upstream nodes require lower SL.

Z‑Score Assignment

Z = Φ⁻¹(Service Level)

Used directly in SS calculation.

7. Safety Stock Calculation
Base Statistical Formula

SS = Z × √( total variance )

Where total variance includes:

Demand variance

Lead time variance

Additional Rules

Minimum floor

≥ 1% of mean demand during lead time

Zero‑Demand Rule

If aggregated demand ≤ 0 → SS forced to 0

Capping Rule

SS constrained within configurable % of total demand

Each adjustment is tracked via Adjustment_Status.

8. Inventory Corridor Concept

For each node and month:

Max Corridor = Forecast + Safety Stock

Displayed visually as:

Base demand (internal)

External network demand

Safety buffer

This corridor defines:

Replenishment upper bound

Stock‑holding target

9. Application Tabs Explained
1️⃣ Inventory Corridor

Time‑phased view of SS and demand

Internal vs downstream demand split

Days of coverage KPI

2️⃣ Network Topology

Visual graph of material flow

Immediate understanding of structure

3️⃣ Full Plan

Complete exportable MEIO output

Filters by product, location, period

4️⃣ Efficiency Analysis

SS vs Forecast ratios

Capital intensity metrics

Top SS contributors

5️⃣ Forecast Accuracy

Historical forecast vs actuals

WAPE, Bias, Accuracy

Network‑level aggregation

6️⃣ Calculation Trace

Full transparency of formulas

Auditable SS computation per node

7️⃣ By Material

Material‑centric drilldown

8️⃣ All Materials View

Portfolio‑level overview

10. Governance & Explainability

The tool is designed to be:

Fully deterministic (no black box)

Auditable (every SS has a trace)

Policy‑driven (SL, caps, floors configurable)

This makes it suitable for:

S&OP

IBP

Audit & finance discussions

11. Strategic Value
What MEIO Enables

Lower total inventory with same service

Correct upstream vs downstream buffering

Fact‑based discussions with plants & buyers

Scalable deployment across materials

Why It Matters

MEIO shifts inventory from:

"Local safety stock guessing"

To:

Network‑optimized, risk‑based buffering

12. Final Takeaway

MEIO for Raw Materials is not just a calculator.

It is a:

Decision support system

Inventory policy engine

Transparency tool

Designed to industrialize best‑practice multi‑echelon inventory logic in a practical, explainable way.







# MEIO for Raw Materials — Inventory Corridor (v1.05 — Jan 2026)

This application implements a Multi‑Echelon Inventory Optimization (MEIO) workflow for raw materials, presented as a Streamlit app. It ingests historical sales and consumption, future demand forecasts, and network lead‑time data to compute safety stock, inventory corridors, service level tiering across network hops, and diagnostic views. It also provides visualizations of the network topology and exportable tables for analysis.

Core technologies: Python, Streamlit, Plotly, NumPy/Pandas, SciPy (normal distribution), PyVis (network graph).

- Main app file: [MEIO.py](https://github.com/mat635418/Inventorycorridor_ABAR/blob/main/MEIO.py)
- Logo: `GY_logo.jpg`
- CSV inputs: `sales.csv`, `demand.csv`, `leadtime.csv`
- Requirements: [requirements.txt](https://github.com/mat635418/Inventorycorridor_ABAR/blob/main/requirements.txt)

---

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run MEIO.py
   ```
   The app opens in your browser with a wide layout and custom styling.

3. Upload the three CSV files via the sidebar (or rely on repository defaults if present):
   - Sales History: `sales.csv`
   - Demand Forecast: `demand.csv`
   - Lead Times: `leadtime.csv`

The app validates the presence of required columns and parses dates to monthly timestamps.

---

## Data Inputs and Required Schemas

The app expects the following columns. Incoming values are cleaned (spaces/commas removed, parentheses treated as negatives like `(1,234)` → `-1234`, and strings like `na`, `n/a`, `-`, `—` converted to NaN). Periods are coerced to month start timestamps.

- `sales.csv` (historical, network-level view is supported)
  - `Product` (string)
  - `Location` (string)
  - `Period` (date-like; parsed to month start)
  - `Consumption` (numeric)
  - `Forecast` (numeric)

- `demand.csv` (future forecasts)
  - `Product`
  - `Location`
  - `Period` (date-like; parsed to month start)
  - `Forecast` (numeric)

- `leadtime.csv` (routes and variability)
  - `Product`
  - `From_Location`
  - `To_Location`
  - `Lead_Time_Days` (numeric)
  - `Lead_Time_Std_Dev` (numeric)

Example minimal rows:

```csv
# sales.csv
Product,Location,Period,Consumption,Forecast
NOKANDO2,DEW1,2025-11-01,1200,1300

# demand.csv
Product,Location,Period,Forecast
NOKANDO2,DEW1,2026-01-01,1400

# leadtime.csv
Product,From_Location,To_Location,Lead_Time_Days,Lead_Time_Std_Dev
NOKANDO2,B616,DEW1,8,1.5
```

---

## Core Logic Overview

The app computes safety stock (SS) and the inventory corridor per Product–Location–Period using demand and lead‑time variability, service levels per tier/hop, and business rules like floors and caps.

Key elements:

- Date normalization: `Period` is parsed to monthly (`to_period("M").to_timestamp()`).
- Numeric cleaning: Custom logic converts user-friendly formats to reliable numeric values.
- Demand/Lead‑time variability:
  - Safety stock uses a statistical term:
    ```
    SS_stat = Z_node * sqrt( demand_component + lt_component )
    ```
    where `Z_node = norm.ppf(Service_Level_Node)` from SciPy’s normal distribution.
  - A floor is applied to avoid very small SS:
    ```
    SS_floor = Mean_Demand_LT * 0.01
    Mean_Demand_LT = D_day * LT_Mean
    Safety_Stock = max(SS_stat, SS_floor)
    ```
- Business rules:
  - If aggregated future demand (`Agg_Future_Demand`) is ≤ 0 and the rule is enabled, SS is forced to zero (“Forced to Zero”).
  - Optional capping: SS can be constrained to a percentage band of `Agg_Future_Demand` via lower/upper caps. Exceeding/under limits labels rows as “Capped (High)” or “Capped (Low)”.
  - Rounding: SS is rounded to integer units for clarity.
  - Known exception: SS at location `B616` is set to 0.
- Corridor computation:
  - `Max_Corridor = Safety_Stock + Forecast`
  - `Days_Covered_by_SS = Safety_Stock / D_day` (where `D_day` is daily demand)

Network aggregation:
- Historical network view aggregates consumption and forecast by Product and Period:
  ```
  Network_Consumption = sum(Consumption)
  Network_Forecast_Hist = sum(Forecast)
  ```

Tiering and service levels:
- The app maintains per‑product tier params (e.g., `SL_hop_0_pct`, `SL_hop_1_pct`, … up to `max_tier_hops`) and summarizes them for analysis and display.

---

## App Structure and Tabs

The UI is divided into eight tabs, each targeting a specific aspect of MEIO.

1. 📈 Inventory Corridor
   - Purpose: Visualize the corridor for a selected Material (Product) and Location, anchored to the current month.
   - Controls:
     - Material selector (default from meaningful results, e.g., `NOKANDO2`)
     - Location selector (prioritizes current month presence, gracefully falls back to historic or any location)
     - Period selector (defaults to current month when available)
   - Views:
     - Corridor visualization (Plotly) comparing Forecast, Safety Stock, and implied maximum corridor
     - Metrics such as `Days_Covered_by_SS`, `Pre_Rule_SS`, `Agg_Future_Demand`
     - Styled tables with numeric formatting and zero‑row hiding to focus on meaningful entries
   - Exports: Buttons to download CSVs for the selection (if present)

2. 🕸️ Network Topology
   - Purpose: Show the multi‑echelon network routes for the selected Product using `leadtime.csv`.
   - Logic:
     - Nodes represent locations; directed edges represent routes (`From_Location → To_Location`)
     - Edge annotations include `Lead_Time_Days` and `Lead_Time_Std_Dev`
   - Visualization: PyVis interactive graph embedded in Streamlit, enabling hover, pan/zoom, and drill-down exploration.

3. 📋 Full Plan
   - Purpose: Present a full table of computed results across Product–Location–Period.
   - Contents commonly include:
     - `Safety_Stock`, `Forecast`, `Agg_Future_Demand`, `Pre_Rule_SS`, `Pre_Cap_SS`, `Max_Corridor`, `Days_Covered_by_SS`
     - Status labels: “Optimal (Statistical)”, “Forced to Zero”, “Capped (High/Low)”
   - Utilities:
     - Numeric formatting: thousands separators and sign handling
     - Zero‑row hiding for clarity
     - CSV export button for further analysis

4. ⚖️ Efficiency Analysis
   - Purpose: Evaluate planning efficiency and parameter effects.
   - Typical metrics:
     - Ratio analyses (e.g., SS vs. future demand)
     - Service level tiering summaries by hop (`SL_hop_0_pct` … `SL_hop_3_pct`) and `max_tier_hops`
     - Days of coverage distribution and cap impacts
   - Aids sensitivity/efficiency reviews across tiers and corridors.

5. 📉 Forecast Accuracy
   - Purpose: Compare historical forecasts against actual consumption to measure accuracy.
   - Logic:
     - Uses network/historic aggregation (`Network_Consumption`, `Network_Forecast_Hist`) by Product and Period
     - Displays accuracy visualizations and/or tables (e.g., error trends)
   - Helps calibrate forecast quality feeding into SS and corridor computations.

6. 🧮 Calculation Trace & Sim
   - Purpose: Reveal the calculation pipeline for SS and corridor components and provide simulated views.
   - Contents:
     - Intermediate variables such as `Z_node`, `D_day`, `LT_Mean`, `lt_component`, `demand_component`
     - Switches and modes (e.g., lead‑time variance application) with explanatory notes
     - Scenario tables summarizing per‑product tiering parameters and their planning implications

7. 📦 By Material
   - Purpose: Focused view of one material across all locations.
   - Views:
     - Tables of SS, forecasts, corridor metrics for the chosen Product
     - Filters and selectors improve discoverability of hotspots (e.g., low SL tiers or capped SS)

8. 📊 All Materials View
   - Purpose: Portfolio‑level overview across all products and locations.
   - Views:
     - Summaries and sortable tables to identify outliers and prioritize actions
     - Export mechanisms to share the plan broadly

---

## UI/Styling Notes

- The app employs custom CSS for tags, select boxes, download buttons, table headers, and tab captions to enhance readability and consistency.
- The header displays:
  ```
  MEIO for Raw Materials — v0.998 — Jan 2026
  ```
- A logo (`GY_logo.jpg`) can be rendered above parameter panels with configurable size.
- Numeric display uses friendly thousands formatting and hides zero‑only rows in certain views.

---

## Defaults and Assumptions

- Default Material: `NOKANDO2`
- Default Location: `DEW1`
- Current month: `pd.Timestamp.now().to_period("M").to_timestamp()`
- Lead‑time floor rule: minimum SS is 1% of mean demand during lead time
- Rounding: Safety Stock rounded to whole units
- Special case: SS at location `B616` is set to 0
- Corridor: `Max_Corridor = Safety_Stock + Forecast`

---

## Key Functions and Helpers (conceptual)

- Numeric cleaning:
  - Converts human‑entered strings to numeric, handling negatives in parentheses and common NA tokens.
- Display formatting:
  - Applies thousand‑separators and hides zeros; two‑decimal formatting used where appropriate.
- SS calculation:
  - Combines demand and lead‑time variability, uses Normal quantile for service level, applies floors/caps and rounding.
- Aggregations:
  - Historic network metrics (`Network_Consumption`, `Network_Forecast_Hist`) by Product–Period support accuracy and trend views.

---

## Export and Sharing

- Many tables include a CSV export button (styled as pill buttons). Use these exports to share the full plan, corridor data, or material views with stakeholders or to feed other planning tools.

---

## Troubleshooting

- File validation:
  - If columns are missing, the app halts with a clear error listing missing fields.
- Date parsing:
  - Ensure `Period` values are parseable; the app converts to monthly timestamps. Use ISO formats like `YYYY-MM-DD`.
- Numeric cleaning:
  - Verify numbers do not contain unparseable characters. Parentheses for negatives and commas are handled automatically.

---

## Version and License

- Version: v1.05 — Jan 2026
- License: Not specified in the repository; add one if distribution is intended.

---
