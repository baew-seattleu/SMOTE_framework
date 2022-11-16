REU2022 README

REU2022/data:
  This folder contains the datasets to run through the framework. Six synthetic datasets of varying sizes and overlap degrees between the minority and majority classes
  are provided. To use your own datasets in the framework, upload them here. 
  
  The data you use MUST be a binary classification dataset that has numbers as the class labels and not strings (i.e., 1 and 0 are OK, "benign" and "malignant" are not).

REU2022/Classification/classification_framework.ipynb:
  Upload your dataset by entering the filename in section #2.

  If you need to drop any unnecessary columns from your data, do so in section 3, and set DROP_ATTRIBUTES to the list of attributes you would like to drop.
  
  Run the classifiers on the balanced data by running the cells with the following titles: 1. Tuning DecisionTree, 2. K-NN, 3. Logistic Regression, 4. Naive Bayes.

REU2022/Classification/classification_results:
  Each subfolder in this folder corresponds to a classifier. If you run the same classifier again and do not change the filename, 
  the results of the last run will be overwritten by the results of the new run, so make sure to save or rename the file containing the old results 
  if you want to keep them.
  
REU2022/utils/balancing_algorithms/Balance.py:
  This file contains the implementation of all the balancing algorithms. These are used in Classificationv3.py.

REU2022/Classification/ClassificationV3.py:
CHANGING SAMPLING METHODS: 
  Within the split_data function there are lines of sampling algorithm calls that can be uncommented to be used to augment the datasets. 
  To change the sampling algorithm used, uncomment the line for the algorithm you would like to use and comment the 
  rest of the list. It is best to only use one sampling algorithm at a time to get the most accurate results.
