# Time Series Anomaly Detection

**Author:** Shannon Nathania

**Date:** February 04, 2026

---

## Table of Contents

1. [Introduction](#introduction)
2. [Methodology](#methodology)
   * [Pre-Researches](#21-pre-researches)
   * [Data Pre-processing](#22-data-pre-processing)
   * [Model Architecture](#23-model-architecture)
3. [Training and Validation](#training-and-validation)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Predicting with Test Dataset](#predicting-with-test-dataset)
6. [Results](#results)
7. [Limitations](#limitations)
8. [What I Learned](#what-i-learned)
9. [Citations](#citations)

---

## 1. Introduction

Time series data are data points collected or sequenced over time. They have become familiar across different fields such as industrial, finance, healthcare, network security, and environmental science. Time series datasets often contain patterns that shows normal system behavior.  However, they may also include anomalies, which are data points or outliers that deviate from the expected patterns. These anomalies indicate critical events such as system malfunctions, fraud transactions, or sensor failures. The timing for anomaly detection is essential for decision making and risk mitigation in real-world systems.

Anomaly detection in time series refers to the finding of these unusual patterns. Traditional statistical methods such as sliding window techniques, ARIMA models, and threshold based approaches have been used to detect outliers for a long time. These are proven to be effective in certain contexts, but these traditional models often struggle with complex, high dimensional, or non linear data patterns commonly found in modern applications.

This project focuses on the empirical evaluation of multiple anomaly detection approaches applied to time series data. The goal is not only to compare model performance but also to understand the assumptions, strengths, and weaknesses of each method. The implemented approaches range from classical statistical techniques to machine learning and deep learning models, providing a comprehensive comparison across different levels of complexity.

---

## 2. Methodology

### 2.1 Pre-Researches

Before implementing the anomaly detection models, I did my research and study from academic articles and technical resources to build a solid theoretical foundation of understanding. Time series anomaly detection is different from traditional classification tasks because anomalies are typically undefined or sparsely labeled. According to a survey by Varun Chandola, anomaly detection systems often focuses on learning normal behaviors and identify deviations rather than directly classify known anomaly classes. This point-of-view directed the overall approach of this project.

Christopher Olah’s article “Understanding LSTM Networks” was studied in detail for conceptual understanding of sequence modeling. This resource explains the internal mechanics of Long Short-Term Memory (LSTM) networks, specifically in the role of gating mechanisms like input, forget, and output gates in preserving long-term dependencies. This understanding was critical for applying LSTM based models to time series anomaly detection where temporal context determines whether a data point is normal or anomalous.

For deep learning–based anomaly detection, the Medium article “Time Series Anomaly Detection with LSTM Autoencoder” provided practical insights to how LSTM Autoencoders can be applied to time series data. The model is trained to reconstruct normal time series patterns with anomalies identified through elevated reconstruction errors. This approach differs from prediction-based methods and influenced the design of the LSTM model in this project by emphasizing reconstruction loss as an anomaly signal.

Classical statistical approaches such as Z-score–based detection rely on distributional assumptions and are highly interpretable. However, these methods are sensitive to seasonality and trends. Seasonal-Trend Decomposition using Loess (STL) was studied to solve this issue. STL decomposes a time series into trend, seasonal, and residual components, allowing anomalies to be detected more reliably in the residual signal.

“Discover Unusual Patterns in Time Series Data with Unsupervised Anomaly Detection and Isolation Forests” clarified how Isolation Forests can be adapted to time series data. Isolation Forest works by randomly partitioning the feature space and isolating observations that require fewer splits. Because anomalies are easier to isolate, this method is suitable for unlabeled data and computationally efficient.Isolation. 

Hence, these resources shaped the project’s methodology by combining theoretical foundations from anomaly detection literature, intuitive explanations of sequence modeling, and practical implementations of both machine learning and deep learning based anomaly detection techniques.

---

### 2.2 Data Pre-processing

Effective preprocessing is very essential for a reliable result in anomaly detection, especially for time series data that contains trend, seasonality, and noise. The preprocessing pipeline consists of the following steps:

#### Exploratory Data Analysis

The raw time series was visualized to identify global trends, seasonal cycles, and potential irregular spikes. This step informed model selection and preprocessing decisions.

#### STL Decomposition

STL was applied to separate the time series into trend, seasonal, and residual components. The residual component was used as the primary input for anomaly detection models.

#### Normalization and Windowing

For machine learning and deep learning models, the data was normalized to stabilize training. Sliding window techniques were applied for sequence-based models such as LSTM.

---

### 2.3 Model Architecture

#### Z-Score Baseline Model

The Z-Score method is a classical statistical baseline for anomaly detection in time-series data. It assumes that normal observations comes with a stable distribution from mean and standard deviation. Under this assumption, anomalies are defined as data points whose standardized distance from the mean exceeds a certain threshold.

In this project, the Z-Score model computes the mean (μ) and standard deviation (σ) of the training time series under normal operating conditions. It provides a strong baseline to understand anomaly behavior in stationary time series and acts as a reference point for evaluating more complex models. But because it relies on global statistics, the Z-Score method is limited in its ability to capture non linear patterns, temporal dependencies, or changes in data distribution over time.

#### Isolation Forest

Isolation Forest isolates anomalies by randomly partitioning data points. Observations that require fewer splits to isolate are assigned higher anomaly scores. In this project, Isolation Forest is applied to engineered time-series features, allowing the model to detect unusual patterns without requiring labeled anomaly data. This makes it particularly effective in scenarios where anomalies are rare or unknown beforehand.

Isolation Forest offers several advantages like computationally efficient, scalable to large datasets, and capable of detecting non-linear anomalies. It does not explicitly model temporal dependencies. As a result, while it performs well at detecting global and contextual anomalies, it might miss anomalies that are defined primarily by long-term sequential behavior.

#### LSTM-Based Model

Long Short-Term Memory (LSTM) networks are designed to model long-range dependencies in sequential data through gated mechanisms that regulate information flow over time. The architecture follows an encoder–decoder structure.

The encoder consists of stacked LSTM layers that process the input sequence and compress it into a fixed length latent representation. In this implementation, multiple LSTM layers with decreasing units are used and then followed by batch normalization and dropout to make a stable training and reduce overfitting. The final encoder LSTM outputs a latent vector that summarizes the normal temporal patterns of the sequence.

The decoder mirrors the encoder by first repeating the latent representation across the original sequence length using a RepeatVector layer. This repeated latent sequence is then passed through stacked LSTM layers to reconstruct the original time series. A TimeDistributed dense layer produces the final reconstructed output at each timestep.

During training, the LSTM Autoencoder is optimized using mean squared error (MSE) loss where it gives accurate reconstruction of normal sequences. During inference, reconstruction error is used as an anomaly score. These are sequences that deviate significantly from learned temporal patterns produce higher reconstruction errors and are flagged as anomalous. LSTM Autoencoder provides a flexible and powerful unsupervised anomaly detection framework.



---

## 3. Training and Validation

In statistical method, models were evaluated directly on the residual signal. Machine learning and deep learning models used time aware train-validation-test splits to prevent data leakage. For the LSTM model, training involved minimizing reconstruction or prediction error using backpropagation through time.

---

## 4. Hyperparameter Tuning

To optimize model performance and ensure fair comparison across methods, hyperparameter tuning was conducted for both the Isolation Forest and the LSTM Autoencoder models.

### Isolation Forest 

For the Isolation Forest model, a grid search was performed over hyperparameters that influence model complexity and anomaly sensitive. The number of estimators (n_estimators), the maximum number of samples used to build each tree (max_samples), and the contamination ratio were explored.

The results show that the best performing configuration achieved an F1 score of 0.9296 on the test set. From the parameter combinations, max_samples = 512 consistently appeared, means that using a larger subset of data for each tree improved the model’s ability to isolate anomalous patterns. In contrast, variations in n_estimators (50, 100, and 200) resulted in almost identical performance. This shows that increasing the number of trees across a certain point outputs great returns.

The contamination parameter remained stable across top configurations. Precision values remained above 0.86, while recall consistently stays around 0.9 or reached 1.0, indicates that the model was effective at identifying true anomalies with minimal false negatives.

The tuning results demonstrate that Isolation Forest is robust to changes in ensemble size but sensitive to sampling strategy. Larger sample sizes leads to improved and more stable anomaly detection performance.

### LSTM Autoencoder

The tuning process explored various sequence length (timesteps), latent dimension size, LSTM hidden unit configurations, learning rate, and batch size. Among the evaluated configurations, the best-performing model achieved an F1-score of 0.9650 and high recall and precision. The top results shows model prefers short input sequences (5–10 timesteps), moderate latent dimensions (4–8), and compact LSTM architectures such as (16, 8) or (32, 16). This suggests that over deep or wide architectures are unnecessary in capturing the temporal patterns in the dataset and could have unnecessary complexity.

A learning rate of 0.0005 came out as the most stable and effective choice, outperforming higher learning rates. Larger batch sizes (64) also appeared more often ahowed smoother gradient updates and more stable convergence during training.

The tuning results indicate that the LSTM Autoencoder benefits most from balanced architectural but not aggressive scaling. Compact latent representations are more sufficient to model normal temporal behavior and alsp reconstruction error remained as a reliable signal for anomaly detection.


---

## 5. Predicting with Test Dataset Results

After selecting the best-performing configurations, the trained models were applied to unseen test data. Anomaly scores were generated for each time step and thresholds were applied to classify points as normal or anomalous.

### Isolation Forest

The Isolation Forest model was evaluated on a test set consisting of 428 observations with 360 normal instances and 68 anomalies. Performance was recorded using precision, recall, F1-score, and confusion matrix analysis. The model achieved an overall accuracy of 97.66%. It means a strong discrimination between normal and anomalous patterns.

- Normal class:
Precision = 0.9944, Recall = 0.9778, F1-score = 0.9860

- Anomaly class:
Precision = 0.8919, Recall = 0.9706, F1-score = 0.9296

--

#### Confusion Matrix
The confusion matrix with :
1. True Positives: 66
2. False Negatives: 2
3. False Positives: 8
4. True Negatives: 352
   
These results highlight a low false negative rate. The model is robust in capturing anomalous behavior while maintaining a low false positive count.

--

#### ROC and Precision-Recall Curve
The strong ROC AUC of 0.99 suggests that the Isolation Forest successfully captures the structural differences between normal behavior and rare deviations in the feature space. By isolating observations that require fewer random partitions to separate, the model identifies anomalies as sparse or irregular points that differ significantly from the majority of the data distribution.

The Precision-Recall Curve results on the test set shows more the behavior of the Isolation Forest model under class imbalance. For the anomaly class, the model achieves a recall of 0.9706, indicating that it successfully detects nearly all true anomalous events. This high recall is critical in anomaly detection.  Missed anomalies can carry significant risk. At the same time, the precision of 0.8919 suggests that the majority of detected anomalies are correct  with a limited number of false positives.

--

#### SHAP Analysis
The SHAP analysis gives insights into the model's decision-making process. It shows that extreme values within rolling windows combined with dependencies from lag features create a robust multi-dimensional print of anomalous behavior. This model essentially learns that normal data outputs certain patterns across different time scales and when multiple features indicate deviation from these patterns, it confidently flags an anomaly.

--

Overall,  Isolation Forest implementation demonstrates that successful anomaly detection in time series fundamentally depends on capturing context. The success of statistical features particularly rolling minimums, maximums, and lag values reveals that anomalies are best identified by how they deviate from their local neighborhood patterns rather than global statistics. This explains why the Isolation Forest achieved improvement over the Z-score baseline

### LSTM Autoencoder

The test set consists of 425 sequences, including both normal and anomalous samples that was not used during training or validation. 

The quantitative evaluation results on the test set are summarized below:
1. Overall accuracy: 0.9882
2. Weighted F1-score: 0.9883
3. Anomaly class F1-score: 0.9640

For the anomaly class specifically, the model achieved:
1. Precision: 0.9437
2. Recall: 0.9853

The high recall indicates that the model successfully detected nearly all anomalous sequences while maintaining strong precision. This results in a low false positive rate. This balance is important in anomaly detection tasks especially where missing anomalies can be more costly than occasional false alarms.

-- 

#### Confusion Matrix
The confusion matrix confirms the effectiveness of the model. Out of 68 anomalous sequences in the test set, 67 were correctly identified with only one false negative. For normal sequences, 353 were correctly classified with four false positives.

This distribution shows that the LSTM Autoencoder is highly sensitive to abnormal temporal patterns but still preserving the difference between normal and anomalous behavior.

--

#### Reconstruction Error
The reconstruction error plot illustrates a clear separation between normal and anomalous samples. Most normal sequences produce low reconstruction error values clustered near zero. Anomalous sequences outputs significantly higher reconstruction errors. The selected threshold successfully separates the majority of anomalies from normal behavior.

This visualization supports the assumption of the LSTM Autoencoder where sequences that deviate from learned normal temporal dynamics are harder to reconstruct and therefore outputs higher reconstruction error.

--

#### ROC and Precision-Recall Curve
The ROC curve demonstrates excellent separation capability and achieved an AUC of approximately 0.99. This indicates that the model performs well across a wide range of thresholds.

The Precision–Recall curve further confirms robust performance in the presence of class imbalance. Precision remains high across most recall values. The model maintains reliability even when aggressively detecting anomalies.

--

#### Threshold
Multiple thresholding strategies were evaluated, including mean plus standard deviation, percentile based thresholds, and robust statistics such as median plus MAD. Among these, the selected threshold achieved the highest F1-score, an optimal balance between precision and recall for this dataset.

--

Overall, the LSTM Autoencoder achieved strong and consistent performance on the test set. It outperforms classical baselines by effectively modeling temporal dependencies in the data. Its ability to detect anomalies with high recall and precision confirms it is suitable to deep sequence models for time-series anomaly detection especially in scenarios where anomalies are defined by deviations in temporal structure and not in individual point values.

---

## Limitations

Despite achieving strong experimental results, this project has several limitations that arise from both project constraints and my current level of experience in time-series anomaly detection and deep learning.

As this project was undertaken during an early stage of my learning in time-series models, the models implemented were intentionally kept relatively simple and interpretable. While established methods such as Z-score, Isolation Forest, and LSTM Autoencoders were successfully implemented, more advanced architectures ; such as attention-based models, Transformer based time-series models, or hybrid deep statistical approaches were not explored yet. This limits the potential upper bound of any performance and model expressiveness, particularly for complex temporal patterns. 

Although the core theoretical concepts behind anomaly detection and LSTM networks were studied and applied, my current level of expertise limited deeper experimentation with advanced optimization strategies. For example, alternative loss functions, dynamic thresholding methods, probabilistic reconstruction models, or ensemble-based deep learning approaches were not implemented. Some architectural and hyperparameter decisions were therefore guided by empirical intuition and available references. Extensive hyperparameter sweeps, longer training durations, and repeated cross-validation runs were not implemented as well.

The evaluation was performed on a single dataset, which may limit the generalizability of the findings. While the models performed well under the given conditions, their robustness across different domains, noise levels, and anomaly types was not tested. Additionally, the anomaly labels and distributions in the dataset may not fully reflect real-world anomaly scenarios, where anomalies can be more subtle or evolve over time.

As the project was developed as part of a learning process, there may exist minor implementation inefficiencies or non-optimal design choices within the codebase. Although care was taken to ensure correctness and reproducibility, further refactoring, optimization, and formal code review could improve robustness and maintainability.

Despite these limitations, this project represents a meaningful step in my learning journey. It provided my first hands on experience in implementing anomaly detection models from scratch, interpreting evaluation metrics, and critically analyzing model behavior. The limitations identified here highlight clear directions for future improvement, both in terms of technical depth and model building.

---

## What I Learned

Working on this time-series anomaly detection project has been one meaningful learning experiences in my machine learning journey so far. Prior to this project, my understanding of time-series data and anomaly detection was largely theoretical. Implementing multiple models from scratch and evaluating them systematically allowed me to develop a much deeper and more practical understanding of how these methods behave in real scenarios.

The most significant learning outcome came from developing and analyzing the LSTM Autoencoder. This process helped me understand how temporal dependencies are modeled in sequential data, why reconstruction error can serve as a reliable anomaly signal, and how architectural choices such as sequence length, latent dimensionality, and learning rate directly affect model behavior. I also gained hands on experience with evaluation techniques such as ROC curves, Precision–Recall analysis, and threshold selection, which are especially critical in imbalanced anomaly detection tasks.

Beyond modeling, this project strengthened my understanding of the full machine learning workflow, including data preprocessing, hyperparameter tuning, performance interpretation, and critical reflection on model limitations. I also learned that strong results do not only come from complex models, but from careful experimentation, thoughtful evaluation, and an awareness of trade offs between precision, recall, and practical usability.

Most importantly, this project motivated my continued interest in time-series analysis. Moving forward, I plan to work on additional projects involving time-series data, including real-time anomaly detection systems, streaming deployment pipelines, and time-series forecasting models for predictive analysis. I see this project as a foundation for exploring more advanced sequence models and real world applications and I intend to progressively build upon these concepts as my technical skills and experience grow.

---

## Citations

[1] C. Olah, “Understanding LSTM Networks,” 2015. [Online]. Available: https://colah.github.io/posts/2015-08-Understanding-LSTMs/.
[2] P. Wang, “Discover Unusual Patterns in Time Series Data with Unsupervised Anomaly Detection and Isolation Forests,” Medium, 2021. [Online]. Available: https://medium.com/@pw33392/discover-unusual-patterns-in-time-series-data-with-unsupervised-anomaly-detection-and-isolation-78db408caaed.
[3] C. Hui, “Anomaly Detection Analysis — Isolation Forest,” Deepnote, 2023. [Online]. Available: https://deepnote.com/app/christopher-hui/Anomaly-Detection-Analysis-Isolation-Forest-c012da68-8081-4e2e-9bc8-8bc59a1c2d6c.
[4] L. Wang, O. Bergmeir, and T. Hyndman, “RobustSTL: A Robust Seasonal-Trend Decomposition Algorithm for Long Time Series,” *ResearchGate*, 2019. [Online]. Available: https://www.researchgate.net/publication/335496645_RobustSTL_A_Robust_Seasonal-Trend_Decomposition_Algorithm_for_Long_Time_Series.
