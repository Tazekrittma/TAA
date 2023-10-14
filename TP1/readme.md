#TP1


This repository contains a Python script for performing credit scoring using machine learning models. It demonstrates the process of loading and preparing data, learning and evaluating machine learning models, and performing various data transformations.

## Requirements

Before running the code, make sure you have the following Python libraries installed:

- `numpy`
- `pandas`
- `scikit-learn` (for machine learning models)
- `matplotlib` (for visualizations)
- `seaborn` (for enhanced visualization)

You can install these libraries using `pip` if you haven't already:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn

## Usage
Loading and Preparing Data: The script loads credit scoring data from a CSV file, transforms it into NumPy arrays, and splits it into training and testing sets.

Learning and Evaluating Models: It trains two classifiers: a Decision Tree (CART) and k-Nearest Neighbors (k=5). The accuracy of these models is evaluated using the testing data.

Normalization of Continuous Variables: The script normalizes continuous variables using the StandardScaler from scikit-learn. The models are re-evaluated with the normalized data.

Creating New Features via Linear Combinations (PCA): Principal Component Analysis (PCA) is applied to create new features. The script then re-evaluates the models with these PCA features.

## Visualizing Results
You can enhance the analysis by visualizing the results using Python libraries. Here are some common visualization techniques you can use:

### Confusion Matrix
Visualize the performance of your classification models using a confusion matrix. The confusion_matrix function from scikit-learn can be used to compute the matrix, and libraries like matplotlib and seaborn can help you visualize it.

'''python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute confusion matrix
cm = confusion_matrix(y_test, dt_classifier.predict(X_test))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix for Decision Tree Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
'''

###ROC Curve
For binary classification, the ROC (Receiver Operating Characteristic) curve is a valuable visualization. You can use the roc_curve function from scikit-learn and matplotlib for plotting.

'''python
from sklearn.metrics import roc_curve, roc_auc_score

y_prob = dt_classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr, label="ROC Curve (Decision Tree)")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
'''
###Feature Importance
Visualize feature importance when using decision tree-based models. Most of these models provide a feature_importances_ attribute that you can use to plot the importance of each feature.

'''python
import matplotlib.pyplot as plt

feature_importance = dt_classifier.feature_importances_
feature_names = df.drop(columns=['Status']).columns

plt.barh(feature_names, feature_importance)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Decision Tree Classifier - Feature Importance')
plt.show()
'''
By incorporating these visualizations into your analysis, you can gain deeper insights into the performance of your credit scoring models.

##Files
credit_scoring.csv: The CSV data file containing credit scoring data.

##Running the Script
You can run the script by executing python credit_scoring.py. Make sure the required libraries are installed, and the credit_scoring.csv file is in the same directory as the script.

##Author
M.TAZEKRITT

##License
MIT License





