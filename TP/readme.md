
# Random Forests in Machine Learning: Two Practical Exercises

## General Idea

This repository hosts exercises that explore the application of random forests in machine learning. The primary objective is to gain insights into how to use random forests effectively for classification tasks. We work with two distinct datasets and scenarios to understand their practical implementation.

## Exercise 1: Flower Classification

**Objective:** In this exercise, our focus is on classifying flower data into different categories using random forests.

**Data:** We employ the well-known Iris dataset, which contains measurements of various features from three distinct flower species.

**Details:** The first exercise guides you through the fundamentals. We start by importing essential libraries, including scikit-learn. The Iris dataset is then loaded and divided into a training set and a test set. We create a random forest model with default parameters, train it, and make predictions on the test set. The model's performance is assessed through metrics such as accuracy, recall, and F1-score. As an extra step, we delve into hyperparameter tuning, experimenting with settings like the number of trees and depth.

## Exercise 2: Random Forests Optimization

**Objective:** In this exercise, we take a deeper dive into optimizing random forests to enhance model performance.

**Data:** Our dataset of choice is CIFAR-10, featuring 60,000 images distributed across ten different classes. These images can be sourced from the PyTorch torchvision library.

**Details:** The second exercise is all about refinement. We initiate the process by importing crucial libraries, including scikit-learn and PyTorch. Next, we load the CIFAR-10 dataset and perform data preprocessing, which includes normalization and the creation of training, validation, and test sets. We then construct a random forest model for image classification using the RandomForestClassifier class from scikit-learn. The model is trained on the training set and evaluated for performance on the validation set, using a chosen metric (such as accuracy). Here, hyperparameter exploration becomes the key, where we tinker with settings like the number of trees, maximum tree depth, and other relevant parameters. We utilize grid search or random search to pinpoint the most effective combinations of hyperparameters for optimal model performance on the validation set. Finally, once we've identified the best hyperparameters, we test the model's capabilities on the test set, providing a realistic estimation of its real-world potential.

