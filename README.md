# Genetic_Forecasting_of_Huntington_Disease
## 📖 Overview
Huntington’s Disease (HD) is a hereditary neurodegenerative disorder caused by abnormal expansions of **CAG trinucleotide repeats** in the **HTT gene**.  
This project proposes a **Deep Neural Network (DNN) based forecasting system** that predicts the likelihood of HD using genetic sequence data.  
The model analyzes DNA sequences, identifies abnormal repeat patterns, and classifies individuals as either **healthy** or **at risk**.

---

## 🎯 Objectives
- To build a computational model capable of **detecting abnormal CAG expansions** in the HTT gene.  
- To leverage **Deep Learning** for early risk prediction of Huntington’s Disease.  
- To evaluate the model using **accuracy, precision, and AUC**.  
- To provide a framework for **genomic-based clinical decision support**.  

---

## 🧪 Methodology

### 1. Data Collection
- DNA sequences focusing on **CAG repeat regions** of the **HTT gene**.  
- Public sources: NCBI GenBank, Ensembl Genome Browser, GEO datasets.  
- Example dataset (`sample_data.csv`) is provided with synthetic sequences.  

### 2. Preprocessing
- **One-hot encoding** of DNA nucleotides (A, T, G, C).  
- Normalization of sequence lengths (padding/truncation).  
- Splitting into **training (80%)** and **testing (20%)** datasets.  

### 3. Model Architecture
The DNN is based on a **1D Convolutional Neural Network (CNN)**:
- `Conv1D` – learns sequence motifs.  
- `MaxPooling1D` – reduces noise.  
- `Dropout` – prevents overfitting.  
- `Dense layers` – fully connected classification.  
- `Sigmoid output` – binary prediction (0: healthy, 1: at-risk).  

### 4. Training
- Loss: **Binary Crossentropy**  
- Optimizer: **Adam**  
- Metrics: **Accuracy, Precision, AUC**  

### 5. Evaluation
- Performance is measured on unseen test data.  
- Metrics include:  
  - **Accuracy**: Overall correctness.  
  - **Precision**: Correct positive predictions (HD risk).  
  - **AUC**: Distinguishing power between classes.  

---
