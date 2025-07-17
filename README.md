# Fine-Grained Vehicle Classification

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodologies](#methodologies)
  - [Transfer Learning](#transfer-learning)
  - [Image Retrieval](#image-retrieval)
  - [End-to-End CNN](#end-to-end-cnn)
- [Results](#results)
- [Installation & Execution](#installation--execution)
- [Authors](#authors)

## Project Overview

This project implements and compares three different deep learning approaches for fine-grained vehicle classification across 196 car models. The challenge requires distinguishing between very similar vehicle types, making it a complex computer vision task. We explore the effectiveness of transfer learning, image retrieval through embeddings, and custom CNN architectures built from scratch.

## Dataset

The Stanford Cars dataset consists of 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where classes are typically at the level of Make, Model, Year (e.g., 2012 Tesla Model S).

Dataset characteristics:

- 196 car classes (makes and models)
- High intra-class variance (viewpoints, colors, backgrounds)
- High inter-class similarity (visually similar models from different manufacturers)
- Resolution variations across images

## Project Structure

The project is organized into four main Jupyter notebooks:

1. **TRANSFER_LEARNING.ipynb**: Implements transfer learning using a pre-trained ResNet50 model.
2. **Image_retrieval_emb.ipynb**: Uses embeddings from the fine-tuned model for image retrieval with KNN.
3. **End_to_end_cnn.ipynb**: Designs and trains custom CNN architectures from scratch.
4. **inference_best_configs.ipynb**: Provides an inference pipeline for all trained models.

## Methodologies

### Transfer Learning

We implemented three configurations of a ResNet50-based classifier:

1. **Base Model**:

   - Pre-trained ResNet50 with frozen weights
   - New classification head with Global Average Pooling, FC layers (512, 256 neurons)
   - Dropout layers (0.5, 0.3) for regularization
   - Adam optimizer with learning rate 0.001

2. **Augmented Model**:

   - Same architecture as the base model
   - Extensive data augmentation: rotation, flipping, brightness adjustments, shear and zoom

3. **Fine-Tuned Model**:
   - Last 20 layers of ResNet50 unfrozen for fine-tuning
   - Deeper classification head with FC layers (1024, 512, 256 neurons)
   - Lower learning rate (0.0005) for stable fine-tuning
   - Enhanced regularization with dropout rates (0.5, 0.4, 0.2)

### Image Retrieval

We leveraged the fine-tuned model from the transfer learning experiment as a feature extractor:

1. Feature embeddings extracted from the penultimate layer
2. KNN-based retrieval with Euclidean distance metric
3. Evaluated with different k values (1, 5, 10)
4. No additional training required, exploiting the learned representations

This approach allows for efficient image similarity search without classification overhead.

### End-to-End CNN

We developed three custom CNN architectures trained from scratch:

1. **Custom CNN**:

   - Four convolutional layers with batch normalization
   - FC layer with 1024 neurons and dropout (0.5)
   - Adam optimizer with learning rate 0.001
   - Limited data augmentation

2. **Augmented CNN**:

   - Five convolutional layers with higher dropout (0.25-0.5)
   - FC layers with 512 and 256 neurons
   - Extreme data augmentations for improved generalization

3. **ResNet-Inspired Model**:
   - Identity and projection blocks for residual connections
   - Global Average Pooling followed by FC layers
   - Lower learning rate (0.0001) for stable training
   - Moderate data augmentation

## Results

### Transfer Learning Performance

| Model            | Top-1 Accuracy | Top-5 Accuracy |
| ---------------- | -------------- | -------------- |
| Base Model       | 66.36%         | 92.33%         |
| Augmented Model  | 52.12%         | 82.07%         |
| Fine-Tuned Model | 56.85%         | 84.87%         |

### Image Retrieval Performance

| k   | Top-1 Accuracy | Top-5 Accuracy |
| --- | -------------- | -------------- |
| 1   | 58%            | 59%            |
| 5   | 59%            | 79%            |
| 10  | 60%            | 81%            |

### End-to-End CNN Performance

| Model           | Top-1 Accuracy | Top-5 Accuracy |
| --------------- | -------------- | -------------- |
| Custom CNN      | 10.55%         | 28.39%         |
| Augmented CNN   | 6.09%          | 18.85%         |
| ResNet-Inspired | 14.36%         | 37.45%         |

### Key Findings

1. Transfer learning significantly outperforms end-to-end training from scratch for this complex classification task
2. Fine-tuning pre-trained models yields the best overall performance (56.85% Top-1 accuracy)
3. Excessive data augmentation can degrade performance if not carefully calibrated
4. Image retrieval with embeddings provides an efficient alternative classification approach with competitive accuracy (60% Top-1 accuracy with k=10)
5. ResNet's architecture principles (skip connections, deeper networks) prove valuable even in custom implementations

## Installation & Execution

### Requirements

- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- PIL

### Running the Notebooks

1. **Clone the repository**:

2. **Install dependencies**:

   ```
   pip install -r requirements.txt
   ```

3. **Download the dataset**:

   - Each notebook contains code to download and prepare the Stanford Cars dataset
   - Alternatively, download from [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

4. **Run the notebooks in sequence**:

   - Start with `TRANSFER_LEARNING.ipynb` to train the transfer learning models
   - Run `Image_retrieval_emb.ipynb` to implement the embedding-based retrieval
   - Execute `End_to_end_cnn.ipynb` to train the custom CNN architectures
   - Use `inference_best_configs.ipynb` to test the trained models on new images

5. **GPU Acceleration**:
   - All notebooks are configured to use GPU acceleration when available
   - For optimal performance, we recommend using a system with CUDA-compatible GPU

## Authors

- Ori Daniel
- Gal Levi
