# Neural Networks – Perceptron from Scratch and MLPs on KMNIST

This project is part of the Deep Learning (94691) course for the Master of Data Science and Innovation at the University of Technology Sydney (UTS).  
It is divided into two parts: building a Perceptron from scratch using NumPy and training multiple Multi-Layer Perceptrons (MLPs) on the KMNIST dataset using PyTorch.

---

## Part A – Perceptron from Scratch

A basic multi-class Perceptron was implemented using only NumPy.  
Key tasks included:

- Forward pass computation
- Backpropagation and weight updates
- Manual accuracy calculation

Constraints:
- Only NumPy and Pandas were allowed (no PyTorch or Scikit-learn)

---

## Part B – Neural Network Experiments on KMNIST

The KMNIST dataset contains 70,000 grayscale images (28x28) of handwritten Hiragana characters across 10 classes.  
Images were flattened into 784-dimensional vectors.

### Preprocessing
- Normalized pixel values
- Encoded class labels
- Flattened image arrays

### Architectures Tested

1. **Experiment 1** – Shallow 2-layer MLP (256 → 128), dropout: 20%  
2. **Experiment 2** – Deep 3-layer MLP (512 → 256 → 128), dropout: 30%  
3. **Experiment 3** – Small 2-layer MLP (128 → 64), dropout: 50%  

### Training Setup

- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Learning Rate: 0.001
- Batch Size: 128
- Epochs: 500
- Layers used: Fully Connected (Linear), Dropout

---

## Results

- **Best Accuracy**: 92.51% (Experiment 2)
- Deeper networks showed better generalization
- Over-regularization led to underfitting in smaller models
- Training was stable with no overfitting spikes

---

## Key Insights

- Model depth improves performance on handwriting datasets, even without CNNs
- Moderate dropout helps regularize effectively
- High dropout in small networks reduces learning capacity

---

## Technologies Used

- Python
- NumPy (Part A)
- PyTorch (Part B)
- Matplotlib, TQDM for visualizations and progress tracking

---

## Future Work

- Extend models using Convolutional Neural Networks (CNNs)
- Add early stopping and learning rate scheduling
- Apply data augmentation techniques to increase generalization

---

## Academic Integrity Notice

This repository is shared publicly for learning and demonstration purposes only.  
Unauthorized copying or direct submission of this work for academic credit is strictly prohibited and may be considered academic misconduct.

---

## Author

Kunal Gurung  
Master of Data Science and Innovation  
University of Technology Sydney
