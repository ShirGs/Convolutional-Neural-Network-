# Convolutional Neural Network Project

## Project Overview

This project involves developing and evaluating Convolutional Neural Network (CNN) models to classify images using the CIFAR-10 dataset. The project is divided into three parts, each focusing on different aspects of CNN model development, optimization, and handling class imbalance. The work is conducted as part of the Large-Scale Computing for Data Analytics course at the University of Glasgow.

## Project Description

### Part 1: CNN for Six-Class Dataset
In this part, a CNN model was developed to classify images into one of six classes from the CIFAR-10 dataset. The model architecture includes:
- Three convolutional layers with ReLU activation and max-pooling layers.
- A dense layer with 256 units and a dropout layer for regularization.
- An output layer with softmax activation for classification.

Key experiments explored the impact of hyperparameters such as batch size, learning rate, and the number of epochs on model performance. The best configuration was found to be a batch size of 32, learning rate of 0.0001, and 34 epochs, achieving 90% training accuracy and 83% validation accuracy.

### Part 2: CNN for Four-Class and Full Imbalanced Datasets
This part extends the analysis to a four-class dataset and a full imbalanced dataset. Three different architectures were tested:
1. **Baseline Model**: Two convolutional layers with max-pooling and two fully connected layers.
2. **Deeper Model**: An additional convolutional layer to increase complexity.
3. **Wider and Deeper Model**: Increased filter sizes along with three convolutional layers.

Results indicated that smaller batch sizes generally provided better generalization, and that handling class imbalance posed significant challenges, particularly for underrepresented classes.

### Part 3: Handling Imbalanced Data with Transfer Learning and Class Weighting
In the final part, transfer learning using a pre-trained VGG16 model was applied to the imbalanced dataset. The model was fine-tuned for both the four-class and full imbalanced datasets, and class weighting was employed to address the imbalance.
- **Transfer Learning**: Showed high accuracy for the four-class dataset but struggled with the imbalanced dataset.
- **Class Weighting**: Helped improve accuracy for underrepresented classes, but challenges remained.

### Key Findings
- **Hyperparameter Tuning**: Smaller batch sizes and lower learning rates provided better generalization, especially in scenarios with imbalanced data.
- **Class Imbalance**: Addressing class imbalance is critical for improving per-class accuracy, with strategies such as class weighting and transfer learning providing partial solutions.
- **Model Complexity**: Increasing model depth and width can help capture more complex features, but also increases the risk of overfitting, particularly in imbalanced datasets.

## Repository Structure

- `part1_cnn_six_class.ipynb`: Jupyter notebook containing the code and analysis for Part 1 of the project.
- `part2_cnn_four_and_imbalanced.ipynb`: Jupyter notebook with code for developing CNN models for the four-class and imbalanced datasets.
- `part3_transfer_learning_class_weighting.ipynb`: Jupyter notebook focusing on transfer learning and class weighting to address class imbalance.
- `report.pdf`: The project report providing detailed analysis, results, and conclusions.
- `README.md`: This file, providing an overview of the project.

## How to Run the Project

1. **Install Required Packages**:
   - Ensure you have the necessary Python packages installed:
     ```bash
     pip install tensorflow keras numpy matplotlib
     ```

2. **Run the Jupyter Notebooks**:
   - Open each notebook (`.ipynb` file) in Jupyter and run the cells sequentially to reproduce the analysis and results.

3. **Review the Report**:
   - The `report.pdf` file contains a comprehensive summary of the project, including discussions on model performance, hyperparameter selection, and strategies for handling imbalanced data.

## Conclusion

This project demonstrates the development and evaluation of CNN models for image classification, highlighting the importance of hyperparameter tuning and strategies to handle class imbalance. The findings provide insights into the challenges and potential solutions for training deep learning models on real-world, imbalanced datasets.

