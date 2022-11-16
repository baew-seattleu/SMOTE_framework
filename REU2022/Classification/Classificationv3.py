import matplotlib.pyplot as plt
from numpy import *
from collections import Counter
import numpy as np
import pylab as pl
from matplotlib import colors
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets, cluster, preprocessing, decomposition, svm, datasets
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF,ConstantKernel

from sklearn.decomposition import PCA
import pandas as pd
import numpy.random as random
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import time
from IPython.display import display
from IPython.display import Image

import scipy;
from scipy	import stats;
import sklearn;
import sklearn.ensemble;
import sklearn.neighbors
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import *
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV 

import os
import subprocess
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn import datasets
from IPython.display import Image
#import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pydotplus
from six import StringIO

import importlib.util
ROOT_DIR='/content/gdrive/MyDrive/REU2022/utils/balancing_algorithms'
spec = importlib.util.spec_from_file_location('balance', ROOT_DIR+'/Balance.py')
balance_method = importlib.util.module_from_spec(spec)
spec.loader.exec_module(balance_method)

def evaluate(y_hat_class, Y): #Y = real value, Yhat = expected

    cm = np.array([[0, 0], [0, 0]])
    if ((y_hat_class == Y.flatten()).all()):
      cm[0][0] += sum(y_hat_class == 0)
      cm[1][1] += sum(y_hat_class == 1)
    else:
      cm = confusion_matrix(Y.reshape(-1,1), y_hat_class) #BUG HERE
      
      t0, f1, f0, t1 = cm.ravel()
      assert (t1 + t0 + f1 + f0) != 0.0

      a = (t1 + t0) / (t1 + t0 + f1 + f0)
      wa0 = (t0 / (2*(t0+f1))) if (t0+f1) != 0.0 else 0.0
      wa1 = (t1 / (2*(t1+f0))) if (t1+f0) != 0.0 else 0.0
      wa = wa0 + wa1
      s = t1 / (t1 + f0) if (t1 + f0) != 0.0 else 0.0
      p = t0 / (t0 + f0) if (t0 + f0) != 0.0 else 0.0
      p1 = t1/(t1 + f1) if (t1 + f1) != 0.0 else 0.0
      r = t0 / (t0 + f1) if (t0 + f1) != 0.0 else 0.0
      fscore = 2 * p * r / (p + r) if p + r != 0.0 else 0.0
      fscore1 = 2 * p1 * s / (p1 + s) if p1 + s !=0.0 else 0.0


      pavg = (p + p1)/2.0
      f1avg = (fscore + fscore1)/2.0
      return np.array([wa, r, pavg, f1avg, s, p, p1, fscore, fscore1]), cm

def split_data(data, class_var, minority_var, test_frac=0.2, balance=False):
    # to hold dataset
    dataset = pd.DataFrame()
    
    dataset = data.copy()

    X = dataset.drop([class_var], axis=1)
    Y = dataset.loc[:,class_var:class_var] 
    dataset_list = []
    kf = KFold(n_splits=3, shuffle=True)
    for train_index, test_index in kf.split(X.to_numpy()):
        X_train, X_test = X.to_numpy()[train_index], X.to_numpy()[test_index]
        Y_train, Y_test = Y.to_numpy()[train_index], Y.to_numpy()[test_index]

        if balance == True:
            # create train dataset as DataFrame
            df_X_train = pd.DataFrame(X_train, columns=X.columns.values)
            df_Y_train = pd.DataFrame(Y_train, columns=Y.columns.values)
            train_dataset = df_X_train.join(df_Y_train)
            
            print("~~~~~~~~~~ Number of instances before balancing: ", train_dataset.shape[0], " ~~~~~~~~~~")

            # Sampling Algorithms (uncomment algorithm line to use)
            train_dataset = balance_method.ROS(train_dataset, class_var, printDebug=False)
            #train_dataset = balance_method.SMOTE(train_dataset, class_var=class_var, minor_class=minority_var, printDebug=False)
            #train_dataset = balance_method.GAMMA(train_dataset, class_var=class_var, minor_class=minority_var, printDebug=False)
            #train_dataset = balance_method.SDDSMOTE(train_dataset, class_var=class_var, minor_class=minority_var, printDebug=False)
            #train_dataset = balance_method.ACOR(train_dataset, class_var=class_var, minor_class=minority_var)
            #train_dataset = balance_method.GaussianSMOTE(train_dataset, classIndex=class_var, minorityLabel=minority_var, printDebug=False, sigma=0.05)
            #train_dataset = balance_method.GAN(train_dataset, class_var=class_var, minor_class=minority_var, codingSizeScaler=2, batchSize=32, printDebug=True)
            #train_dataset = balance_method.IKC(train_dataset, class_var=class_var, minor_class=minority_var, printDebug=False)
            #train_dataset = balance_method.ADKNN(train_dataset, classIndex=class_var, minorityLabel=minority_var, printDebug=False)
            #train_dataset = balance_method.SMOTEBoost(train_dataset, class_var=class_var, minor_class=minority_var, printDebug = False)
            #train_dataset = balance_method.SMOTEBoostCC(train_dataset, class_var=class_var, minor_class=minority_var, printDebug = False)


            print("~~~~~~~~~~ Number of instances after balancing: ", train_dataset.shape[0], " ~~~~~~~~~~")
            
            # convert train dataset into X_train and Y_train
            X_train = train_dataset.drop(class_var, axis=1).to_numpy()
            Y_train = train_dataset.loc[:,class_var:class_var].to_numpy() 
        dataset_list.append([X_train, X_test, Y_train, Y_test])
    
    return dataset_list
      
# -*- coding: utf-8 -*-
def run_cv(df, clf_class, class_variable=None, minority_variable=None, printDebug = False, clf=None, clfnm=None, after_split=False):
    # load target patient data
    dataset_list = split_data(data=df, 
                              class_var=class_variable, #explanatory_var
                              minority_var=minority_variable,
                              balance=True)
    print('~~~~~~~~~~', clfnm, '~~~~~~~~~~~~~~~')    
    # Iterate through folds\
    i = 0;
    best_score = 0
    
    # setup for k-fold cross-validation
    kfold_evaluation_results_list = []
    kfold_confusion_matrix_list = []
    roc_features = []
    important_features = []
    best_models = []
    
    for i in range(3):
        print('-----------kfold---------------', i, '------------')
        # load the target data in the current round
        X_train, X_test, y_train, y_test = dataset_list[i]
        
        if clfnm == 'DecisionTree':
          parameters = {
                'max_depth':(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22), 
                'criterion':('entropy', 'gini'), 
          }

          Grid_DT_depth = GridSearchCV(sklearn.tree.DecisionTreeClassifier(),parameters, cv=5)
          Grid_DT_depth.fit(X_train, y_train)
          best_parameters = Grid_DT_depth.best_params_
          
          clf = Grid_DT_depth.best_estimator_ #cl
          
          if Grid_DT_depth.best_score_>best_score:
            best_score = Grid_DT_depth.best_score_
            best_clf = Grid_DT_depth.best_estimator_ #cl
            print('----DecisionTree kfold----------', best_score, best_clf)
            
        if clfnm == 'SVM':
          parameters={'kernel':('linear', 'poly', 'rbf', 'sigmoid'),
              'C':np.arange(1,42,10),
              'degree':np.arange(3,6),   
              'coef0':np.arange(0.001,3,0.5),
              'gamma': ('auto', 'scale')
          }
          SVModel = SVC()
          GridS = GridSearchCV(SVModel, parameters, cv=5)
          GridS.fit(X_train, y_train.ravel())
          best_parameters = GridS.best_params_
          clf = GridS.best_estimator_ #cl
          if GridS.best_score_>best_score:
            best_score = GridS.best_score_
            best_clf = GridS.best_estimator_ #cl
            print('----SVM kfold----------', best_score, best_clf)
            
        if clfnm == 'Random Forest':
          parameters = { 
            'n_estimators': [100, 200,300,400, 500,600],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth' : [4,5,6,7,8],
            'criterion' :['gini', 'entropy']
          	}

          rfc=sklearn.ensemble.RandomForestClassifier(random_state=42)
          GridRF = GridSearchCV(estimator=rfc, param_grid=parameters, cv= 5)
          GridRF.fit(X_train, y_train.ravel())
          best_parameters = GridRF.best_params_
          clf = GridRF.best_estimator_ #cl
          if GridRF.best_score_>best_score:
            best_score = GridRF.best_score_
            best_clf = GridRF.best_estimator_ #cl
            print("---Random Forest---", best_score, best_clf)
            
        if clfnm == 'K-NN':
          parameters = { 
            'n_neighbors': [3,5,11,9],
            'weights': ['uniform', 'distance'],
            'metric' :['euclidean', 'manhattan']
        	}

          GridKNN = GridSearchCV(sklearn.neighbors.KNeighborsClassifier(), param_grid=parameters, cv= 5)
          GridKNN.fit(X_train, y_train.ravel())
          best_parameters = GridKNN.best_params_
          clf = GridKNN.best_estimator_ #cl
          if GridKNN.best_score_>best_score:
              best_score = GridKNN.best_score_
              best_clf = GridKNN.best_estimator_ #cl
              print("---K-NN---", best_score, clf)
        
        if clfnm == 'Gradient Boosting':
          parameters = { 
            "loss":["deviance"],
            "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
            "min_samples_split": np.linspace(0.1, 0.5, 12),
            "min_samples_leaf": np.linspace(0.1, 0.5, 12),
            "max_depth":[3,5,8],
            "max_features":["log2","sqrt"],
            #"criterion": ["friedman_mse",  "squared_error"],  #"mae"
            "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
            "n_estimators":[10]
        	}

          GridGB = GridSearchCV(sklearn.ensemble.GradientBoostingClassifier(), param_grid=parameters, cv= 5)
          GridGB.fit(X_train, y_train.ravel())
          best_parameters = GridGB.best_params_
          clf = GridGB.best_estimator_ #cl
          if GridGB.best_score_>best_score:
            best_score = GridGB.best_score_
            best_clf = GridGB.best_estimator_ #cl
            print("---Gradient Boosting---", best_score, best_clf)
        
        if clfnm == 'Logit Regression':
          parameters = {"penalty":["l1","l2"], 'C': [0.001,0.01,0.1,1,10,100,1000], "solver":["liblinear", "lbfgs"]} #"C":np.logspace(-3,3,7)
          GridLR = GridSearchCV(sklearn.linear_model.LogisticRegression(), param_grid=parameters, cv= 5)
          GridLR.fit(X_train, y_train.ravel())
          best_parameters = GridLR.best_params_
          clf = GridLR.best_estimator_ #cl
          if GridLR.best_score_>best_score:
            best_score = GridLR.best_score_
            best_clf = GridLR.best_estimator_ #cl
            print("---Logit Regression---", best_score, best_clf)
        
        if clfnm == 'Neural NW':
          parameters = { 
            'solver': ['sgd', 'adam'],
          	'alpha': [1e-5, 0.0001, 0.05],
          	'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,),(32, 16)],
          	'activation': ['tanh', 'relu'],
          	'learning_rate': ['constant','adaptive']
          }

          nnwc=sklearn.neural_network.MLPClassifier(max_iter=10000, random_state=1)

          GridNNW = GridSearchCV(estimator=nnwc, param_grid=parameters, cv= 5)
          GridNNW.fit(X_train, y_train.ravel())
          best_parameters = GridNNW.best_params_
          clf = GridNNW.best_estimator_ #cl
          if GridNNW.best_score_>best_score:
            best_score = GridNNW.best_score_
            best_clf = GridNNW.best_estimator_ #cl
            print("---Neural NW---", best_score, best_clf)
        
        if clfnm == 'Naive Bayes':
          parameters = { 
            'kernel': [ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed"), 1.0 * RBF(1.0), DotProduct() + WhiteKernel()]
            #'alpha': [1e-10, 1e-4, 1e-1, 1]
            }

          nb=sklearn.gaussian_process.GaussianProcessClassifier(random_state=42)

          GridNB = GridSearchCV(estimator=nb, param_grid=parameters, cv= 5)
          GridNB.fit(X_train, y_train.ravel())
          best_parameters = GridNB.best_params_
          clf = GridNB.best_estimator_ #cl
          if GridNB.best_score_>best_score:
            best_score = GridNB.best_score_
            best_clf = GridNB.best_estimator_ #cl
            print("---Naive Bayes---", best_score, best_clf)
            
        clf.fit(X_train,y_train)
        
        if ( hasattr(clf,'feature_importances_') ):
          print ("There is Feature Importance matrix for this classifier:", clf)
          feature_importance = best_clf.feature_importances_
          important_features.append(feature_importance)
          
        y_predicted = clf.predict(X_test)

        e_rsults, c_matrix = evaluate(y_predicted,y_test)
        print('-----',e_rsults, c_matrix)
        kfold_evaluation_results_list.append(e_rsults)
        kfold_confusion_matrix_list.append(c_matrix)
        roc_features.append((y_test, y_predicted))
        
        if (printDebug): print ("*",i, end ="");
        i = i +1;
    avg_kfold_evaluation_results = np.array(kfold_evaluation_results_list).mean(0)
    if (printDebug): print ("*");
    print(clf.get_params())
      
    print("------run_cv runs successfully------------")

    return avg_kfold_evaluation_results, kfold_confusion_matrix_list, roc_features, important_features, best_clf
    
    
##########################################################################

#Classification Problems 
# Usually in case of classifcation, it is best to draw scatter plot of 
# Target Varible using Scatter plot
# df, t,m = encodeCategorical(dfL, "FiveTile1", "Target" );
# scatter_matrix(dfL, alpha=1, figsize=(10,10), s=100, c=df.Target);
# print "categorical Plot {}".format(m)
#
#
# Df - Data Frame that does not have a Predict Column
# y  - Predict Column Series 
def Classify(df,
             printDebug = True ,
             class_var=None,
             minority_var=None,
             drawConfusionMatrix = True,
             classifiers = None,
             scale =True,
             save_path = False,
             after_split = False,
             txt_file = False
             ):

    cls = classifiers

    if save_path is not False and txt_file is not False:
        metrics_file = open(save_path+'metrics.txt', 'w')
        metrics_file.write('========== ACCURACY AND R^2 SCORE ==========\n\n')
        
    y_preds = {}
    ret_accuracy = [];
    cms = [];
    acc = [];
    r2_score = [];
    clfs = {}
    for i in arange( int (len(cls)/2) ):
        nm = cls[i*2];
        cl = cls[i*2 +1]
        
        avg_kfold_evaluation_results, kfold_confusion_matrix_list, roc_features, important_features, clfi = run_cv(df, None, class_variable=class_var, minority_variable=minority_var, clf=cl,clfnm=nm, printDebug=printDebug, after_split=after_split)
        clfs[nm] = clfi
        cls[i*2 +1] = clfi
        
    return (avg_kfold_evaluation_results, kfold_confusion_matrix_list, roc_features, important_features, clfi)

def save_confusion_matrix(cms, filename):
  print("++++++++++++save_confusion_matrix++++++++++++")
  fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20,5))
  for i in range(len(cms)):
    cm = cms[i]
    cax = axs[i].matshow(cm, cmap='binary', interpolation='nearest')
    axs[i].set_title('Confusion Matrix Round #{}'.format(i+1), 
                     fontsize='xx-large')
    axs[i].set_xlabel('Predicted', fontsize='xx-large')
    axs[i].set_ylabel('True', fontsize='xx-large')
    axs[i].set_xticklabels([''] + ['0', '1'], fontsize='xx-large')
    axs[i].set_yticklabels([''] + ['0', '1'], fontsize='xx-large')
    axs[i].xaxis.set_ticks_position('bottom')
    for (ii, jj), z in np.ndenumerate(cm):
      axs[i].text(jj, ii, '{:0.1f}'.format(z),
                  bbox=dict(facecolor='white', edgecolor='0.3'),
                  ha='center', va='center', fontsize='xx-large')
  plt.savefig('{}.png'.format(filename), bbox_inches='tight')
  plt.savefig('{}.pdf'.format(filename))
  plt.show()
  plt.close()
  print("++++++++++++save_confusion_matrix end++++++++++++")

from itertools import cycle

def save_Draw_Roc(roc_features, save_path=False, printDebug=True):

  fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20,5))
  m=15
  print("(((((((((((((((((((((( draw droc ))))))))))))))))))))))))))))))" )
  for i in range(len(roc_features)):

    class_names = np.unique(roc_features[i][1])
    class_names[::-1].sort()
    n_classes = len(class_names)

    roc_y = sklearn.preprocessing.label_binarize(roc_features[i][0], classes = [0, 1, 2])
    roc_y_pred = sklearn.preprocessing.label_binarize(roc_features[i][1], classes = [0, 1, 2])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for j in range(n_classes):
        fpr[j], tpr[j], _ = sklearn.metrics.roc_curve(roc_y[:, j], roc_y_pred[:, j])
        roc_auc[j] = sklearn.metrics.auc(fpr[j], tpr[j])

      #plt.subplot(len(cls)/3, 3, j+1)
    colors = cycle(['r', 'y', 'g'])
    lw = 2
    for j, color in zip(range(n_classes), colors):
        axs[i].plot(fpr[j], tpr[j], color=color, lw=lw,
                  label='ROC curve of class {0} (area = {1:0.2f})'
                  ''.format(j, roc_auc[j]))

    axs[i].plot([0, 1], [0, 1], 'k--', lw=lw)
    axs[i].set_title('Roc Curve')
    axs[i].set_xlabel('False Positive Rate')
    axs[i].set_ylabel('True Positive Rate')
    axs[i].set_xlim([0.0, 1.0])
    axs[i].set_ylim([0.0, 1.0])
  
  if save_path is not False :
      plt.savefig(save_path+'.png', bbox_inches='tight')
  if printDebug :
      plt.show()
  plt.close()
  print("(((((((( draw droc end ))))))))")

  
def save_Draw_FeatureImportanceMatrix(dft,feature_importances, classifierName, save_path=False, printDebug=True):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(24,3))
    m=15
    
    print("---------------DrawFeatureImportanceMatrix------------")
    for i in range(len(feature_importances)):
      feature_importance=feature_importances[i]

      if ( len(feature_importances)==0 ):
        print ("No Feature Importance matrix for this classifier:", clf)
        return;

      feature_importance = 100.0 * (feature_importance / feature_importance.max())
      sorted_idx = np.argsort(feature_importance)
      sorted_idx10 = sorted_idx[-m:]; #[0:5]  # TODO: Print Top 10 only ??
      pos = np.arange(sorted_idx10.shape[0]) + .5
      fc10=np.asanyarray(dft.columns.tolist())[sorted_idx10];

      axs[i].barh(pos, feature_importance[sorted_idx10], align='center', color='#7A68A6')
      axs[i].set_yticks(pos)
      axs[i].set_yticklabels(tuple(fc10))
      axs[i].set_xlabel('Relative: '+ classifierName)
      axs[i].set_title('Variable Importance')
            
    if save_path is not False :
        plt.savefig(save_path+'.png', bbox_inches='tight')
    if printDebug :
        plt.show()
    plt.close()
    print("------------DrawFeatureImportanceMatrix end----------------")  
