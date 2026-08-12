# Pitcher Degradation Index

An unsupervised machine learning pipeline designed to detect pitcher fatigue and kinetic chain collapse using 3D Hawk-Eye tracking data.

## Project Overview

The baseball industry traditionally relies on radar gun velocity drops to determine when a pitcher is fatigued. However, evaluating recent Statcast data (2023–2026) reveals that pulling a pitcher strictly for a 1.5 mph velocity drop triggers a false alarm 41.2% of the time. 

Because supervised machine learning models fail in this domain due to noisy, biased, and non-binary injury labels, this project utilizes an unsupervised anomaly detection approach. The Degradation Index (DI) evaluates structural biomechanics—such as release extension and vertical arm slot—to flag when a pitcher loses proprioception and the ability to repeat their delivery.

By prioritizing specificity, this pipeline reduces the false alarm rate for pitcher fatigue to a surgical 14.0% (a 66% reduction compared to traditional velocity tracking), preserving bullpen arms while optimizing run prevention.

## Machine Learning Architecture

The pipeline consists of a two-phase machine learning approach:

### Phase 1: The Contextual Filter (XGBoost)
Pitchers strategically alter their mechanics based on game context (e.g., shortening extension with runners on base). To prevent the model from flagging these intentional adjustments as fatigue, the pipeline utilizes a `MultiOutputRegressor(xgb.XGBRegressor)`. 
* A highly personalized model is trained for every individual pitcher.
* The model predicts expected mechanics based on pitch type, count, and base-out state.
* The predicted values are subtracted from the actual Hawk-Eye measurements to isolate the **residual**—the pure physical execution stripped of strategic noise.

### Phase 2: Unsupervised Anomaly Detection (Isolation Forest)
The model tracks the rolling variance of the residuals to identify structural failure. 
* To eliminate magnitude bias (as extension varies by inches while arm slot varies by fractions of an inch), all rolling variances are standardized into Z-scores.
* The standardized variances are fed into an unsupervised `IsolationForest` algorithm to detect multi-dimensional outliers.
* The output anomaly score is inverted, scaled from 0 to 100, and localized to the pitcher's specific outing (requiring the variance to hit the 90th percentile of that exact game) to generate the final **Degradation Index**.

## Results & Trade-offs

The model was evaluated against late-inning, sustained workloads (innings 4+, minimum 15 pitches in the inning) to measure genuine fatigue rather than early-game command issues.

* **Baseline (1.5 mph Velocity Drop):** Caught 54.8% of run-scoring breakdowns, but generated a **41.2% False Alarm Rate**.
* **Degradation Index:** Caught 41.9% of run-scoring breakdowns while generating only a **14.0% False Alarm Rate**.

## Repository Structure

* `data/`: Contains the processed `.parquet` files and Statcast datasets (Note: Raw data files are tracked in `.gitignore`).
* `notebooks/`: Jupyter notebooks detailing the exploratory data analysis and model training steps.
* `src/`: Python scripts for data processing, the XGBoost contextual filter, and the Isolation Forest pipeline.
* `requirements.txt`: List of dependencies required to run the pipeline.

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/degradation-index.git](https://github.com/yourusername/degradation-index.git)
