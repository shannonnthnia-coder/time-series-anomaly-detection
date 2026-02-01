# Time Series Anomaly Detection

A comprehensive study comparing statistical and machine learning approaches for anomaly detection in time series data using the Yahoo Webscope S5 A1Benchmark dataset.

## Project Overview
This project implements and compares multiple anomaly detection techniques:
- **Statistical Baseline**: Z-score on STL residuals
- **Machine Learning**: Isolation Forest with feature engineering
- **Deep Learning**: LSTM Autoencoder (planned)

### Key Results
- **Isolation Forest F1-Score**: 0.93 (93% accuracy)
- **Improvement over Z-score**: +116% in F1-score
- **Detection Rate**: 97.1% (catches 66/68 anomalies)
- **False Alarm Rate**: 2.2% (only 8 false positives)

## 🗂️ Project Structure
```
time-series-anomaly-detection/
├── data/                          # Data directory (not committed)
│   └── raw/A1Benchmark/          # Yahoo S5 dataset
├── notebook/                      # Jupyter notebooks (analysis pipeline)
│   ├── 01_exploration.ipynb      # Data exploration & visualization
│   ├── 02_preprocessing_stl.ipynb # STL decomposition & normalization
│   ├── 03_zscore_baseline.ipynb  # Statistical baseline (Z-score)
│   └── 04_isolation_forest.ipynb # Isolation Forest implementation
├── report/                        # Project reports
│   └── proposal.md               # Project proposal
├── .gitignore                    # Git ignore rules
├── LICENSE                       # MIT License
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```
### Prerequisites
- Python 3.8+
- pip or conda

### Installation
1. **Clone the repository**
```
git clone https://github.com/yourusername/time-series-anomaly-detection.git
cd time-series-anomaly-detection
```
2. **Create virtual environment**
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. **Install dependencies**
```
pip install -r requirements.txt
```
4. **Download the dataset**
   - Download Yahoo Webscope S5 A1Benchmark dataset
   - Extract to `data/raw/A1Benchmark/`
   - Dataset structure should be:
```
     data/raw/A1Benchmark/
     ├── real_1.csv
     ├── real_2.csv
     └── ...
```
### Usage
Run notebooks in order:
```bash
jupyter notebook
```
1. `01_exploration.ipynb` - Understand the data
2. `02_preprocessing_stl.ipynb` - Preprocess and decompose
3. `03_zscore_baseline.ipynb` - Statistical baseline
4. `04_isolation_forest.ipynb` - ML model training
5. 

## Results Summary

### Model Comparison

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Z-Score (Residual) | 0.65 | 0.32 | 0.43 |
| **Isolation Forest** | **0.89** | **0.97** | **0.93** |

### Key Findings

1. **Feature Engineering is Crucial**: Statistical features (rolling windows, lags) outperformed raw values
2. **Temporal Context Matters**: Rolling min/max features were most important (SHAP analysis)
3. **STL Decomposition Helps**: Residual component captures anomalies effectively
4. **Isolation Forest > Z-Score**: 116% improvement in F1-score

## Technologies Used

- **Python 3.13**
- **Data Processing**: pandas, numpy
- **Time Series**: statsmodels (STL decomposition)
- **Machine Learning**: scikit-learn (Isolation Forest)
- **Visualization**: matplotlib, seaborn
- **Interpretability**: SHAP
- **Notebooks**: Jupyter

## Dataset

**Yahoo Webscope S5 - A1Benchmark**
- real_1 and real_17
- Real-world time series with labeled anomalies
- 1,424 data points
- 227 anomalies (15.94%)
- Used for evaluating anomaly detection algorithms

## Methodology

### 1. Data Preprocessing
- Normalization using StandardScaler
- STL decomposition (trend, seasonal, residual)
### 2. Feature Engineering
- Rolling statistics (mean, std, min, max) with windows [5, 10, 20]
- Lag features [1, 2, 3, 5, 10]
- Difference features
### 3. Model Training
- Stratified train-test split (70/30)
- Hyperparameter tuning via grid search
- Evaluation on held-out test set
### 4. Evaluation
- Precision, Recall, F1-Score
- ROC AUC (0.990)
- Confusion Matrix
- SHAP feature importance

## Visualizations

The project includes comprehensive visualizations:
- Time series with labeled anomalies
- STL decomposition plots
- Feature importance (SHAP)
- ROC and Precision-Recall curves
- Confusion matrices

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Shannon Nathania**
- GitHub: [@shannonnthnia-coder](https://github.com/shannonnthnia-coder)
- LinkedIn: [---](---)

## Acknowledgments
- Yahoo Research for the S5 A1Benchmark dataset
- Anthropic for Claude AI assistance
- scikit-learn and statsmodels communities

## References

1. 
## Roadmap

- [x] Data exploration
- [x] STL decomposition
- [x] Z-score baseline
- [x] Isolation Forest implementation
- [ ] LSTM Autoencoder
- [ ] Ensemble methods
- [ ] Real-time deployment
