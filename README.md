# Time Series Anomaly Detection

A comprehensive study comparing statistical and machine learning approaches for anomaly detection in time series data using the Yahoo Webscope S5 A1Benchmark dataset.

## Project Overview
This project implements and compares multiple anomaly detection techniques:
- **Statistical Baseline**: Z-score on STL residuals
- **Machine Learning**: Isolation Forest with feature engineering
- **Deep Learning**: LSTM Autoencoder
- **Real time Deployment** : COMING SOON !!

## Project Structure
```
time-series-anomaly-detection/
├── data/                          # Data directory
│   └── README.md
├── notebook/                      # Jupyter notebooks (analysis pipeline)
│   ├── 01_exploration.ipynb      # Data exploration & visualization
│   ├── 02_preprocessing_stl.ipynb # STL decomposition & normalization
│   ├── 03_zscore_baseline.ipynb  # Statistical baseline (Z-score)
│   └── 04_isolation_forest.ipynb # Isolation Forest implementation
|   └── 05_lstm.ipynb             # LSTM Autoencoder Implementation
├── report/                        # Project reports
│   └── REPORT.md              
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
git clone https://github.com/shannonnthnia-coder/time-series-anomaly-detection.git
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
5. `05_lstm.ipynb` - ML model training

## Results Summary
### Model Comparison
| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Z-Score (Residual) | 0.65 | 0.32 | 0.43 |
| Isolation Forest | 0.89 | 0.97 | 0.93 |
| LSTM Autoencoder | 0.94 | 0.98 | 0.96 |

### Key Findings
1. **Feature Engineering is Crucial**: Statistical features (rolling windows, lags) outperformed raw values
2. **Temporal Context Matters**: Rolling min/max features were most important (SHAP analysis)
3. **STL Decomposition Helps**: These residual features improved both Isolation Forest and LSTM reconstruction-based detection.
4. **Isolation Forest > Z-Score**: 116% improvement in F1-score
5. **LSTM Autoencoder excels at capturing Sequential Dependencies** : Unlike Isolation Forest, the LSTM model implicitly captured long-range temporal dependencies without requiring extensive manual feature engineering.
6. **Model Strengths differ based on Temporal Complexity** : Isolation Forest performed exceptionally well on feature-engineered data and the LSTM Autoencoder showed advantages in scenarios where anomalies were defined by subtle temporal dynamics rather than point-wise deviations.

## Technologies Used

- **Python 3.13**
- **Data Processing**: pandas, numpy
- **Time Series**: statsmodels (STL decomposition)
- **Machine Learning**: scikit-learn, TensorFlow / Keras
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
### 3. Baseline Method (Z-Score)
- Anomalies are flagged when: ∣z∣>𝜏 , where τ is a predefined threshold.
- Even though its computationally efficient, it lacks adaptability to non-stationary patterns and complex temporal structures.
### 4. Isolation Forest Model
- Isolation Forest is used as an unsupervised tree-based anomaly detection method.
- Operates on engineered features
- No anomaly labels are used during training
- Anomalies are detected based on isolation depth
- Isolation Forest significantly outperforms the Z-score baseline This demonstrates its robustness to complex anomaly patterns.
### 5. LSTM Autoencoder Model
- LSTM Encoder compresses sequences into a latent representation
- LSTM Decoder reconstructs the original input sequence
- Reconstruction error is used as the anomaly score
- Samples with reconstruction error exceeding a certain threshold are flagged as anomalies
- This approach is good at detecting contextual and temporal anomalies that cannot be identified using static features alone.
### 6. Model Training
- Stratified train-test split (70/30)
- Hyperparameter tuning via grid search for Isolation Forest
- Manual threshold selection for Z-score and LSTM reconstruction error.
- Evaluation on test set
### 4. Evaluation
- Precision, Recall, F1-Score
- ROC AUC (0.990)
- Confusion Matrix
- Precision-Recall Curve
- SHAP feature importance
### 5. Model Comparison
- Z-Score: Simple baseline, high false positives
- Isolation Forest: Best balance of precision and recall
- LSTM Autoencoder: Strong temporal awareness, effective for complex anomalies
  
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
- LinkedIn: www.linkedin.com/in/shannon-nathania-susilo-169857368 

## Acknowledgments
- Yahoo Research for the S5 A1Benchmark dataset
- Anthropic for Claude AI assistance
- scikit-learn and statsmodels communities

## Roadmap

- [x] Data exploration
- [x] STL decomposition
- [x] Z-score baseline
- [x] Isolation Forest implementation
- [X] LSTM Autoencoder
- [ ] Real-time deployment



