# BridgeNet: A High-Efficiency Framework Integrating Sequence and Structure for Protein and Enzyme Function Prediction

Welcome to the official repository of **BridgeNet**, a novel pre-trained model designed to integrate protein sequence and structural information for accurate and biologically meaningful protein representation learning. BridgeNet provides a robust and scalable framework for tackling diverse protein property prediction tasks, such as enzyme function classification, coenzyme specificity prediction, and peptide toxicity prediction.

---

## Overview

Proteins play critical roles in biological systems, and understanding their properties is essential for advancements in fields like biotechnology, synthetic biology, and drug discovery. Computational methods, particularly those leveraging deep learning, have revolutionized protein characterization. However, existing approaches often rely solely on sequence or structure information, limiting their predictive capabilities.

**BridgeNet** overcomes this limitation by integrating sequence and structure representations during pretraining. This novel design enables the model to capture the complementary strengths of both data modalities, resulting in state-of-the-art performance across multiple protein-related tasks. 

### Key Features

- **Integration of Sequence and Structure**: Combines the strengths of sequence-based and structure-based models for comprehensive protein characterization.
- **Pretrained and Fine-Tunable**: Allows for fine-tuning on multiple downstream tasks, including enzyme classification and toxicity prediction.
- **Scalable and Efficient**: Designed to address large-scale protein datasets with a focus on both accuracy and computational efficiency.
- **Biologically Meaningful Representations**: Embeds environmental factors and protein dynamics into the learned representations for enhanced interpretability.

---

## Usage Instructions

After running `main.py`, the script will automatically execute the following six methods to perform different downstream prediction tasks and output the corresponding performance metrics:

- `tAMPer()`
- `DeepFRI(task='EC')`
- `DeepFRI(task='BP')`
- `DeepFRI(task='MF')`
- `DeepFRI(task='CC')`
- `CoEnzyme()`

### Required Third-Party Libraries

Ensure the following libraries are installed before running the script:

1. **Python**: The base programming language required to run the script.
2. **Torch**: A machine learning library used for building and training deep learning models.
3. **Torch_geometric**: Extends PyTorch to handle graph-based data, enabling graph neural network operations.
4. **Numpy**: Used for numerical computations, such as handling arrays and matrices.
5. **Sklearn**: Provides tools for machine learning tasks like performance evaluation and data preprocessing.
6. **Tqdm**: Displays progress bars for loops, making it easier to monitor runtime progress.
7. **Math**: Provides basic mathematical functions and operations.

### File Organization

- **Encoded Data**: Preprocessed `.pkl` files should be placed in the `tmp` directory.
- **Model Files**: Downstream model files should be placed in the `models` directory.

---

