# Genetic Forecasting of Huntington Disease

## Overview
Huntington's Disease (HD) is a hereditary neurodegenerative disorder caused by abnormal CAG repeat expansions in the HTT gene. Early diagnosis is essential for effective disease management and clinical intervention.

This project presents an **Explainable Multimodal Deep Learning Framework** that combines **brain MRI images** and **genetic data** to improve Huntington's Disease prediction. The model integrates **EfficientNet-B3** for MRI feature extraction, **Graph Neural Networks (GNN)** for genetic relationship modeling, and a **Transformer-based Cross-Attention Fusion** mechanism for multimodal learning.

## Key Features
- Multimodal analysis using MRI and genetic data
- EfficientNet-B3 for brain image feature extraction
- Graph Neural Network (GNN) for genetic data analysis
- Cross-Attention Fusion for combining multimodal features
- Explainable AI using SHAP and Grad-CAM
- Early and accurate Huntington's Disease prediction

## Methodology

### 1. Data Collection
- Brain MRI images
- HTT gene DNA sequences

### 2. Data Preprocessing
- MRI resizing, normalization, and denoising
- Genetic sequence encoding

### 3. Feature Extraction
- EfficientNet-B3 for MRI features
- GNN for genetic feature learning

### 4. Multimodal Fusion
- Transformer-based Cross-Attention Fusion

### 5. Classification
- Deep Neural Network for disease prediction

### 6. Explainability
- SHAP for genomic feature interpretation
- Grad-CAM for MRI visualization

## Technologies Used
- Python
- TensorFlow / Keras
- EfficientNet-B3
- Graph Neural Networks (GNN)
- Transformer Networks
- SHAP
- Grad-CAM
- NumPy
- Pandas
- OpenCV
- Scikit-learn

## Future Scope
- Integration of additional biomarkers such as EEG and clinical data
- Real-time clinical decision support systems
- Edge and mobile deployment for remote healthcare monitoring
- Enhanced transformer-based multimodal architectures

## Applications
- Early Huntington's Disease prediction
- Clinical decision support systems
- AI-assisted healthcare diagnostics
- Precision medicine research

