#ROS ALGORITHM
import pandas as pd

def ROS(dataFrame, index, printDebug=True):
    newDF = dataFrame
    classes = dataFrame[index].unique()
    numberOfClasses = len(dataFrame[index].unique())
    if printDebug:
        print("classes: [" + ''.join(str(e)+',' for e in classes)+']')
        print("Number of Classes:" + str(numberOfClasses))
    
    sizeOfClasses=[]
    for aClass in classes:
        sizeOfClasses.append(len(dataFrame.values[dataFrame[index].values == aClass]))
    sizeOfLargestClass = max(sizeOfClasses)
    
    if printDebug:
        print("Size of Classes: [" + ''.join(str(e)+', ' for e in sizeOfClasses)+']') 
        print("Size of Largest Class: " + str(sizeOfLargestClass)) 
    
    amountToAddToEachClass = []
    for classSize in sizeOfClasses:
        amountToAddToEachClass.append(sizeOfLargestClass - classSize)
    
    if printDebug:
        print("Amount to Add to Each Class: [" + ''.join(str(e)+', ' for e in amountToAddToEachClass)+']') 
    
    k=0
    
    for amount in amountToAddToEachClass:
        for i in range(0,amount):
           newDF = newDF.append(dataFrame[dataFrame[index].values == classes[k]].sample(n=1, replace=True))
        k += 1

    return newDF;

#SMOTE ALGORITHM

#Import Necessary Libraries
import random
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from random import randrange

#For data analysis
import seaborn as sn
import matplotlib.pyplot as plt

#Main Program
#dataFrame - imbalanced dataset
#class_var - column name containing class identifier
#minor_class - minority identifier
#KNN - number of k-nearest neighbors to search for 
def SMOTE(dataFrame, class_var, minor_class, KNN=5, printDebug = False):  
    #Calculate number of attributes in dataFrame
    numAttributes = dataFrame.shape[1]
    
    #Divide data set into Majority (MA) and Minority (MI) Classes
    MA = dataFrame[dataFrame[class_var] != minor_class]
    MI = dataFrame[dataFrame[class_var] == minor_class]
    
    #Record number of instances for each class
    MA_num = MA.shape[0]
    MI_num = MI.shape[0]
    
    #Calculate how much data to generate
    numToSynthesize = MA_num - MI_num
    
    #Calculate k neighbors to search for
    if (MI_num - 1) < KNN:
        k = MI_num - 1 
    else:
        k = KNN
    
    if printDebug == True:
        print("~~~~~~~~~~~ MI_num = ", MI_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ k = ", k, " ~~~~~~~~~~")
    
    #Convert minority class to NumPy array for synthesizing
    mi = MI.to_numpy()
    
    #Create NumPy array for synthetic data    
    syntheticArray = np.empty((0, numAttributes))
    
    for i in range(numToSynthesize):
        #Select and instance x in minority class randomly
        x = random.choice(mi)
        x = np.reshape(x, (-1, numAttributes))

        #Find indices of k nearest neighbors of x
        _, knn = findNeighbors(x, mi, k)

        #Select one knn of sample and record it as y
        y = mi[knn[0,randrange(1, k+1)]]
        
        #Generate new minority instance w/ equation
        diff = y - x
        gap = random.uniform(0,1)
        xnew = x + gap * diff
        syntheticArray = np.concatenate((syntheticArray, xnew))
    
    #Convert synthetic array into DataFrame
    syntheticData = pd.DataFrame(syntheticArray, columns=MI.columns.values)
    
        
    #Join imbalance data with synthetic data set
    newDF = pd.concat([dataFrame, syntheticData], ignore_index=True) 
    
    if printDebug == True:
        display(dataFrame)
        display(newDF)

    return newDF

#Find Neighbors
#value - sample to query
#allValues - values to search for neighbors
#k - number of neighbors to search for
def findNeighbors(value, allValues, k):
    nn = NearestNeighbors(n_neighbors = k+1, metric = "euclidean").fit(allValues)
    #using k+1 since first nearest neighbor is itself
    dist, indices = nn.kneighbors(value, return_distance = True)
    return dist, indices
    

### GAMMA ALGORITHM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pandas.core.frame import DataFrame
from scipy.stats import gamma
from sklearn.neighbors import NearestNeighbors
from random import randrange
import random
from math import *
import math
import seaborn as sn

#Main Program
#dataFrame - imbalanced dataset
#class_var - column name containing class identifier
#minor_class - minority identifier
def GAMMA(dataFrame, class_var, minor_class, printDebug = False):
    #Calculate number of attributes in dataFrame
    numAttributes = dataFrame.shape[1]
    
    #Divide data set into Majority (MA) and Minority (MI) Classes
    MA = dataFrame[dataFrame[class_var] != minor_class]
    MI = dataFrame[dataFrame[class_var] == minor_class]

    #Record number of instances for each class
    MA_num = MA.shape[0]
    MI_num = MI.shape[0]

    #Calculate how much data to generate
    numToSynthesize = MA_num - MI_num

    #Convert minority class to NumPy array for synthesizing
    mi = MI.to_numpy()

    #Create NumPy array for synthetic data
    syntheticIndex = 0
    syntheticArray = np.empty((numToSynthesize, numAttributes))
    
    #Calculate k neighbors to search for
    if (MI_num - 1) < 5:
        k = MI_num - 1 
    else:
        k = 5
    
    if printDebug == True:
        print("~~~~~~~~~~~ MI_num = ", MI_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ k = ", k, " ~~~~~~~~~~")
    
    #default alpha and beta value
    alpha = 0.5
    beta = 0.0125
    maxCD = beta*(alpha-1) #coordinate of the max value

    for i in range(numToSynthesize):
        #randomly choosing x in the minority class
        x = random.choice(mi) 
        
        x = np.reshape(x, (-1, numAttributes))
        knn = NearestNeighbors(n_neighbors = k+1).fit(mi)
        ind = knn.kneighbors(x, return_distance = False)

        #Select one knn of sample and record it as y
        y = randrange(1, k+1)
        x_prime = mi[ind[0, y]]
        
        # define vector v
        v = x_prime - x
        
        #generate t using the gamma distribution
        t = stats.gamma.rvs(beta, alpha, random_state=None)
        
        p = x + (t - maxCD) * v

        syntheticArray[syntheticIndex] = p
        syntheticIndex += 1

    #convert syntheticArray into DataFrame
    syntheticData = pd.DataFrame(syntheticArray, columns=MI.columns.values)

    #concatenate synthetic dataset with imbalanced dataset
    newDF = pd.concat([dataFrame, syntheticData], ignore_index=True)

    return newDF


#SDD-SMOTE ALGORITHM

#Import Necessary Libraries
import random
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from random import randrange
from scipy.spatial import cKDTree

#For Data Analysis
import seaborn as sn
import matplotlib.pyplot as plt

#Main Program
#dataFrame - dataset to balance
#class_var - column name containing class identifier
#minor_class - minority identifier
#KNN - number of k-nearest neighbors to search for 
def SDDSMOTE(dataFrame, class_var, minor_class, KNN=5, printDebug = False):
    #Calculate number of attributes in dataFrame
    numAttributes = dataFrame.shape[1]
    
    #Divide data set into Majority (MA) and Minority (MI) Classes
    MA = dataFrame[dataFrame[class_var] != minor_class]
    MI = dataFrame[dataFrame[class_var] == minor_class]
    
    #Record number of instances for data set and each class
    DF_num = dataFrame.shape[0]
    MA_num = MA.shape[0]
    MI_num = MI.shape[0]
    
    #Calculate number of data samples to generate
    numToSynthesize = MA_num - MI_num
    
    #Calculate k-nearest neighbors to search for
    if (MI_num - 1) < KNN:
        k = MI_num - 1 
    else:
        k = KNN
    
    if printDebug == True:
        print("~~~~~~~~~~~ DF_num = ", DF_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ MI_num = ", MI_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ MA_num = ", MA_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ k = ", k, " ~~~~~~~~~~")
    
    #Convert dataframes to NumPy array for synthesizing and density distribution calculations
    df = dataFrame.to_numpy()
    mi = MI.to_numpy()
    ma = MA.to_numpy()
    
    #Create NumPy array for synthetic data    
    syntheticIndex = 0
    syntheticArray = np.empty((numToSynthesize, numAttributes))
    
    #Calculate radius to delimit circular area for density calculations
    radius = calculateRadius(mi, MI_num, ma, MA_num, numAttributes, mu = 2)
    
    #Create Density List of all minority samples
    densityList = []
    for i in range(MI_num):
        sample = mi[i]
        sample = np.reshape(sample, (-1, numAttributes))
        
        #Calculate density D of each minority class sample
        D = calculateSampleDensity(sample, radius, mi, ma)
        
        densityList.append((i, D))
    densityList.sort(key = lambda x: x[1], reverse = True)
    
    #Calculate average Euclidean distance Dpos of the total minority class sample set
    Dpos = calculateAverageDistance(mi, MI_num, mi, numAttributes, k)

    #Calculate average Euclidean distance Dneg of the total majority class sample set
    Dneg = calculateAverageDistance(mi, MI_num, ma, numAttributes, k)
    
    if printDebug == True:
        print("Radius = ", radius)
        print(densityList)
        print("Dpos = ", Dpos)
        print("Dneg = ", Dneg)
    
    densityIndex = 0
    for i in range(numToSynthesize):
        if densityIndex >= len(densityList):
            densityIndex = 0

        #Based chose sample based on density from densityList
        instance = densityList[densityIndex]
        index = instance[0]
        x = mi[index]
        x = np.reshape(x, (-1, numAttributes))

        #Calculate Control Coefficient
        cc = calculateControlCoefficient(x, k, mi, ma, Dneg, Dpos)
        
        if printDebug == True:
            print("cc = ", cc)

        #Find indices of k nearest neighbors of x; NOTE: Disregard distance output
        _, knn = findNeighbors(x, mi, k)

        #Select one knn of sample and record it as y
        y = randrange(1, k+1)
  
        #Generate new minority instance w/ equation
        diff = mi[knn[0, y]] - x
        syntheticArray[syntheticIndex] = x + cc * diff
        syntheticIndex += 1
        densityIndex += 1
    
    #Convert synthetic array into DataFrame
    syntheticData = pd.DataFrame(syntheticArray, columns=MI.columns.values)
    
    #Join imbalance data with synthetic data set
    newDF = pd.concat([dataFrame, syntheticData], ignore_index=True) 
    
    #Print Scatter Plots of Original and Balanced Data
    if printDebug == True:
        display(dataFrame)
        display(newDF)

    return newDF

#Calculate Radius
#mi - minority NumPy
#MI_NUM - number of minority samples
#ma - majority NumPy
#MA_NUM - number of majority samples
#numAttr - number of attributes in each sample
#mu - scaling coefficient to control radius
def calculateRadius(mi, MI_num, ma, MA_num, numAttr, mu = 1):
    E = 0
    for i in range(MI_num):
        #Select an instance x in minority class
        x = mi[i]
        x = np.reshape(x, (-1, numAttr))
        
        #Find distances from minority sample to EVERY majority sample; NOTE: Disregard index output
        dist, _ = findNeighbors(x, ma, MA_num-1)
        #Calculate sum of distances between minority instance x and ALL majority instances as Ei
        Ei = dist.sum()
        #Accumulate and sum all Ei to get E
        E += Ei
    
    #Calculate Emean of the sum of distances between all minority samples and majority samples
    Emean = mu * (E / (MI_num * MA_num))
    return Emean

#Calculate Sample Density
#value - sample to query
#r - radius to delimit circular area
#mi - minority NumPy
#ma - majority NumPy
def calculateSampleDensity(value, r, mi, ma):
    mi_nn = NearestNeighbors(radius = r).fit(mi)
    ma_nn = NearestNeighbors(radius = r).fit(ma)
    mi_rng = mi_nn.radius_neighbors(value)
    ma_rng = ma_nn.radius_neighbors(value)
    #Default weight ratio of MA to MI is 8:2
    density = 0.8 * len(np.asarray(ma_rng[1][0])) + 0.2 * len(np.asarray(mi_rng[1][0]))
    return density

#Calculate Average Euclidean Distance
#c1 - class 1 NumPy
#c1_num - number of class 1 samples
#c2 - class 2 NumPy
#numAttr - number of attributes in each sample
#k - number of neighbors to search for
def calculateAverageDistance(c1, c1_num, c2, numAttr, k):
    D = 0
    for i in range(c1_num):
        #Select an instance x in minority class
        x = c1[i]
        x = np.reshape(x, (-1, numAttr))
        
        #Find distances from minority sample to k minority class neighbors; NOTE: Disregard index output
        dist, _ = findNeighbors(x, c2, k)
        Di = dist.sum() / k
        D += Di
   
    D = D/c1_num
    return D

#Calculate Control Coefficient
#value - sample to query
#k - number of neighbors to search for
#mi - minority NumPy
#ma - majority NumPy
#Dn - average majority density
#Dp - average minority density
def calculateControlCoefficient(value, k, mi, ma, Dn, Dp, printDebug = False):
    #Calculate average distance D1 between sample and k minority class neighbors
    dist, _ = findNeighbors(value, mi, k)
    D1 = dist.sum() / k

    #Calculate average distance D2 between sample and k majority class neighbors
    dist, _ = findNeighbors(value, ma, k)
    D2 = dist.sum() / k

    #Calculate the relative distance ratio u according to D1, D2, Dneg, and Dpos

    u = (D1 * Dn)/(D2 * Dp)

    #According to u, calculate control coefficient cc
    if u < 1:
        cc = random.uniform(0, 1)
    elif (u >= 1) & (u <= 2):
        cc = 0.5 + 0.5 * random.uniform(0, 1)
    elif u > 2:
        cc = 0.8 + 0.2 * random.uniform(0, 1)
    if printDebug == True:
        print("D1 = ", D1)
        print("D2 = ", D2)
        print("u = ", u)
    return cc
    

# ACOR-SMOTE ALGORITHM
import random
from random import randint, choice
from random import randrange
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy import stats
from pandas.core.frame import DataFrame
from scipy.stats import gamma
from math import *
import math
import seaborn as sn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def ACOR(dataFrame, class_var, minor_class, printDebug=False):
    best_OPS = 0
    best_AUC = 0
    prob_of_sel = [] #Pij = Tij/sum(Tij)
    selection_set = []
    sample_set = []
    optimized_set = []
    D_star = []

    balanced_data = SMOTE(dataFrame, class_var, minor_class)
    iteration = 100
    num_ants = 10
    numSample = balanced_data.shape[0]
    balanced_set = balanced_data.to_numpy()

    line_pheromone=[[1 for i in range(0,2)] for j in range(0, numSample)]
    
    for i in range(iteration):
        for j in range(num_ants):
            prob_of_sel = get_prob_of_selection(line_pheromone)
            selection_set = get_selection_set(prob_of_sel)
            sample_set = get_sample_set(selection_set, balanced_set)
            df = pd.DataFrame(sample_set, columns=dataFrame.columns.values)
            if printDebug == True:
                print("prob_of_sel:" , prob_of_sel)
                print("selection_set: ", selection_set)
                print("sample set: ", df)
            AUC = LogisticRegressionCLF(df, dataFrame)
            if (AUC > best_AUC):
                best_AUC = AUC
                optimized_set = df
        OPS = best_AUC
        if (best_OPS == 0 or OPS > best_OPS):
            best_OPS = OPS
            newDF = optimized_set

        line_pheromone = update_pheromone(selection_set, line_pheromone, best_OPS)
        
    return newDF

### ACOR Helper funtions

def get_prob_of_selection(line_pheromone):
    prob_of_sel = []
    for i in range(len(line_pheromone)):
        sum = line_pheromone[i][0] + line_pheromone[i][1]
        if (sum == 0):
            prob_of_sel.append([0,0])
        else:
            prob_of_sel.append([line_pheromone[i][0]/sum,line_pheromone[i][1]/sum])
    return prob_of_sel

def get_selection_set(prob_of_sel):
    sel_set = []
    for k in range(len(prob_of_sel)):
        if (prob_of_sel[k][0] == prob_of_sel[k][1]):
            path_choice = randint(0, 1)
            sel_set.append(path_choice)
        elif (prob_of_sel[k][0] > prob_of_sel[k][1]):
            sel_set.append(0)
        elif (prob_of_sel[k][0] < prob_of_sel[k][1]):
            sel_set.append(1)
    return sel_set

def get_sample_set(sel_set, balanced_set):
    sample_set = []
    for k in range(len(sel_set)):
        if (sel_set[k] == 1):
            sample_set.append(balanced_set[k])
    return sample_set

def LogisticRegressionCLF(sample_set, org_set):
    from sklearn.model_selection import train_test_split

    x_train = sample_set.values
    y_train = sample_set['class'].values
    
    x_test = org_set.values
    y_test = org_set['class'].values

    model = LogisticRegression()
    clf = model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    
    AUC = roc_auc_score(y_test, prediction)

    return AUC

def update_pheromone(selection_set, line_pheromone, OPS):
    for c in range(len(selection_set)):
        if (selection_set[c] == 0):
            p = randint(0,1)
            prev_pheromone = line_pheromone[c][0]
            if (prev_pheromone > 0 and prev_pheromone < OPS):
                change_in_pheromone = (1-p)*(prev_pheromone/1)
            else:
                change_in_pheromone = 0
            # Tij(t+1)= p * Tij(t)+ change in Tij(t)
            line_pheromone[c][0] = p * prev_pheromone + change_in_pheromone
        elif(selection_set[c] == 1):
            p = randint(0,1)
            prev_pheromone = line_pheromone[c][1]
            if (prev_pheromone > 0 and prev_pheromone < OPS):
                change_in_pheromone = (1-p)*(prev_pheromone/1)
            else:
                change_in_pheromone = 0
            # Tij(t+1)= p * Tij(t)+ change in Tij(t)
            line_pheromone[c][1] = p * prev_pheromone + change_in_pheromone
    return line_pheromone

# GAUSSIAN SMOTE ALGORITHM
# Balances an imbalanced binary classification dataset using Gaussian SMOTE.
# Precondition: The class labels must be numerical.
# dataFrame - the dataset to balance
# classIndex - the name of the column containing the class labels
# minorityLabel - the class label corresponding to the minority class, such as 0
# sigma - standard deviation for the Gaussian distribution used to generate new data
def GaussianSMOTE(dataFrame, classIndex, minorityLabel, printDebug = True, sigma = 0.05):  
                                 
    # Divide data set into Majority (MA) and Minority (MI) Classes
    MA = dataFrame[dataFrame[classIndex] != minorityLabel]
    MI = dataFrame[dataFrame[classIndex] == minorityLabel]
    
    #Record number of instances for each class
    MA_num = MA.shape[0]
    MI_num = MI.shape[0]
    
    # Calculate how much data to generate
    numToSynthesize = MA_num - MI_num
    
    # Calculate k-nearest neighbors to search for
    if (MI_num - 1) < 5:
        k = MI_num - 1 
    else:
        k = 5

    # Convert minority class to NumPy array for synthesizing
    mi = MI.to_numpy()
    
    # Number of attributes in the dataset
    numAttributes = mi.shape[1]

    # Create NumPy array for synthetic data    
    syntheticIndex = 0
    syntheticArray = np.empty((numToSynthesize, numAttributes)) 
    
    # For storing a vector of random values from the uniform distribution
    gap = np.empty(numAttributes)
    
    # For storing a vector of random values from the Gaussian distribution
    gaussianRange = np.empty(numAttributes)
    
    # Numbered position of the column where the class label is
    classColNum = dataFrame.columns.get_loc(key=classIndex)
    
    if printDebug:
        print("~~~~~~~~~~~ MI_num = ", MI_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ MA_num = ", MA_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ sigma =", sigma, "~~~~~~~~~~~")

    for i in range(numToSynthesize):
        # Select an instance x in minority class randomly
        x = random.choice(mi)
        x = np.reshape(x, (-1, numAttributes)) 
        
        # Find indices of all the k nearest neighbors of x, discarding the distances from x to the neighbors
        _, knn = findNeighbors(x, mi, k)
        
        # Select one knn of sample randomly
        neighbor = mi[knn[0, randrange(1, k+1)]]
        
        x = x.flatten()

        # Difference between the chosen neighbor and x (a vector)
        diff = neighbor - x
        
        # Make sure the class label of diff is the minority label
        diff[classColNum] = minorityLabel

        # gap is the point between x and its neighbor where the Gaussian distribution 
        # will be centered; generate gap using a different value from the uniform distribution 
        # for each component/attribute of diff.
        
        for j in range(len(gap)):
            # Check the value of diff[j] to get appropriate upper and lower bounds for the uniform distribution

            if diff[j] > 0:
                gap[j] = np.random.uniform(0, diff[j])
            elif diff[j] < 0:
                gap[j] = np.random.uniform(diff[j], 0)
            else: 
                gap[j] = 0 # uniform distribution between 0 and 0 will return 0
        
        # Make sure the class label of gap is the minority label
        gap[classColNum] = minorityLabel
        
        # gaussianRange is a vector of random values from the Gaussian distribution with mean 
        # x[j] + gap[j] and standard deviation sigma
        for j in range(len(gaussianRange)):
            if j != classColNum:
                # If the column contains a feature, generate a random value with Gaussian distribution
                gaussianRange[j] = random.gauss(x[j] + gap[j], sigma)
            else:
                # If the column contains the class labels, make sure gaussianRange has the minority class label
                gaussianRange[j] = minorityLabel 

        # Generate new data point using x + diff * gaussianRange, where diff and gaussianRange
        # are multiplied using element-wise multiplication (np.multiply performs element-wise multiplication)
        syntheticArray[syntheticIndex] = x + np.multiply(diff, gaussianRange) 

        # Make sure the class label of the new data point is the same as the minority class label
        syntheticArray[syntheticIndex][classColNum] = minorityLabel
       
        syntheticIndex += 1
    
    # Convert synthetic array into DataFrame
    syntheticData = pd.DataFrame(syntheticArray, columns=MI.columns.values)
    
    # Join imbalanced data with synthetic data set
    # Add the synthetic minority data to the entire training set
    newDF = pd.concat([dataFrame, syntheticData], ignore_index=True)

    if printDebug:
        print("~~~~~~~~~~~ Number of minority instances in balanced data:", len(newDF[ newDF[classIndex] == minorityLabel]), "~~~~~~~~~~~")
        print("~~~~~~~~~~~ Number of majority instances in balanced data:", len(newDF[ newDF[classIndex] != minorityLabel]), "~~~~~~~~~~~")
        
    return newDF


# GAN ALGORITHM 
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')


#Main Program
# dataFrame - dataset to balance
# class_var - column name containing class identifier
# minor_class - minority identifier
# codingSizeScalar - scalar value for coding size
# batchSize - number of batches
def GAN(dataFrame, class_var, minor_class, codingSizeScaler = 2, batchSize = 32, printDebug=False):
    
    #Find unique columns
    uniqueColumns = dataFrame.columns[dataFrame.nunique() <= 1]
    
    #Random seed
    random_seed = 42
    # Set random seed in tensorflow
    tf.random.set_seed(random_seed)
    # Set random seed in numpy
    np.random.seed(random_seed)
    
    #Divide data set into Majority (MA) and Minority (MI) Classes
    MA = dataFrame[dataFrame[class_var] != minor_class]
    MI = dataFrame[dataFrame[class_var] == minor_class]
    
    #Record number of instances for each class
    MA_num = MA.shape[0]
    MI_num = MI.shape[0]
    
    #Calculate number of data samples to generate
    numToGenerate = MA_num - MI_num
    
    #Split data for GAN training
    X = dataFrame.drop([class_var], axis=1)
    Y = dataFrame.loc[:,class_var:class_var] 

    X_train = X.to_numpy()
    y_train = Y.to_numpy()
    
    # Get the training feature matrix of the minority class
    X_minor = MI.drop([class_var], axis = 1)
    X_minor_train = X_minor.to_numpy()

    # Get the training target vector of the minority class
    y_minor_train = y_train[np.where(y_train == minor_class)]
    
    # ~~~~~~~~~~ 1 - Building GAN ~~~~~~~~~~
    #Set the number of features
    n_features = X_train.shape[1]

    #Set the coding size, which is the dimension of the noise used as input for the generator
    #GANS takes in noise as input and outputs realistic features
    coding_size = n_features // codingSizeScaler #Change denominator to change performance

    if printDebug == True:
        print("n_features: ", n_features)
        print("coding_size: ",coding_size)

    # Build the generator using Kera Sequential API to create fully connected feed forward Neural Network
    # Generator gets wider and wider as you move forward
    # Number of perceptions on each layer can be hyper parameter
    generator = keras.models.Sequential([
        keras.layers.Dense(100, activation='selu', input_shape=[coding_size]),
        keras.layers.Dense(200, activation='selu'),
        keras.layers.Dense(300, activation='selu'),
        keras.layers.Dense(400, activation='selu'),
        keras.layers.Dense(500, activation='selu'),
        keras.layers.Dense(n_features, activation='sigmoid')
    ])

    # Build the discriminator using Kera Sequential API to create fully connected feed forward Neural Network
    # Discriminator gets more narrow as you move forward
    discriminator = keras.models.Sequential([
        keras.layers.Dense(n_features),
        keras.layers.Dense(500, activation='selu'),
        keras.layers.Dense(400, activation='selu'),
        keras.layers.Dense(300, activation='selu'),
        keras.layers.Dense(200, activation='selu'),
        keras.layers.Dense(100, activation='selu'),
        keras.layers.Dense(1, activation='sigmoid') 
        # will work on binary classification problem, real or fake, 1 or 0
        # Applies the sigmoid activation function
        # For small values (<-5) returns value close to 0
        # For large values (>5) the result of the function gets close to 1
    ])

    # Build GAN using Sequential API
    gan = keras.models.Sequential([generator, discriminator])

    if printDebug == True:
        # Get the summary
        gan.summary()
    
    # ~~~~~~~~~~ 2 - Compiling GAN ~~~~~~~~~~
    # Compile the discriminator
    discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=10 ** -4))
    #binary_crossentropy computes the cross-entropy loss between true labels and predicted labels
    #Adam optimization is a stochastic gradient descent method
    #based on adaptive estimation of first-order and second-order moments

    # Freeze the discriminator
    discriminator.trainable = False
    # dont want to discrimnate and generate at the same time

    # Compile the generator
    gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=10 ** -4))
    
    
    # ~~~~~~~~~~ 3 - Training GAN ~~~~~~~~~~
    # Get the generator and discriminator
    generator, discriminator = gan.layers

    # Get the indices of the training data of the minority class
    idxs_minor_train = np.array(range(X_minor_train.shape[0]))

    # The batch size, can be fine tuned
    batch_size = batchSize

    # Get the number of minibatches
    n_batch = len(idxs_minor_train) // batch_size

    # The number of maximum epoch
    max_iter = 10

    # For each epoch
    for _ in range(max_iter):
        # Shuffle the data
        np.random.RandomState(seed=random_seed).shuffle(idxs_minor_train)

        # For each minibatch
        for i in range(n_batch):
            # Get the first and last index (exclusive) of the minibatch
            first_idx = i * batch_size
            last_idx = min((i + 1) * batch_size, len(idxs_minor_train))

            # Get the minibatch
            mb = idxs_minor_train[first_idx : last_idx]

            # Get the real feature matrix
            real_features = X_minor_train[mb, :]

            # Get the noise
            noise = tf.random.normal(shape=[len(mb), coding_size], seed=random_seed)

            # Get the gen feature matrix
            gen_features = generator(noise)

            # Combine the generated and real feature matrix
            gen_real_features = tf.concat([gen_features, real_features], axis=0)

            # Get the target vector
            y = tf.constant([[0.]] * len(mb) + [[1.]] * len(mb))

            # Unfreeze the discriminator
            discriminator.trainable = True

            # Train the discriminator
            discriminator.train_on_batch(gen_real_features, y)

            # Get the noise
            noise = tf.random.normal(shape=[len(mb), coding_size], seed=random_seed)

            # Get the target
            y = tf.constant([[1.]] * len(mb))

            # Freeze the discriminator
            discriminator.trainable = False

            # Train the generator
            gan.train_on_batch(noise, y)

        # Save the gan
        gan.save('model.h5')
        
    # Load the model
    model = keras.models.load_model('model.h5')

    # Get the generator
    generator = gan.layers[0]

    # Initialize the generated data
    gen_data = np.zeros((numToGenerate, X_minor_train.shape[1] + 1))

    for i in range(numToGenerate):
        # Get the noise
        noise = tf.random.normal(shape=[1, coding_size], seed=random_seed)

        # Get the generated features
        gen_features = generator(noise)

        # Update the generated data
        gen_data[i, :-1], gen_data[i, -1] = gen_features, minor_class

    syntheticData = pd.DataFrame(gen_data, columns=dataFrame.columns.values)
    

    for column in uniqueColumns:
        syntheticData[column] = dataFrame[column].iat[0]

    newDF = pd.concat([dataFrame, syntheticData], ignore_index=True)
    
    if printDebug == True:
        display(dataFrame)
        display(newDF)

    return newDF

### Incremental K-Means Clustering (IKC)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pandas.core.frame import DataFrame
from math import *
import math
import seaborn as sn
from sklearn.cluster import KMeans

def IKC(dataFrame, class_var, minor_class, printDebug = False):
    #Divide data set into Majority (MA) and Minority (MI) Classes
    MA = dataFrame[dataFrame[class_var] != minor_class]
    MI = dataFrame[dataFrame[class_var] == minor_class]
    
    #Record number of instances for each class
    MA_num = MA.shape[0]
    MI_num = MI.shape[0]
    
    mi = MI.to_numpy()
    numAttributes = mi.shape[1]
    
    #Calculate how much data to generate
    numToSynthesize = MA_num - MI_num
    syntheticArray = np.empty((0, numAttributes))
    X = mi
    
    clusterNum = 2
    rows, columns = X.shape

    #let X be a copy of the minority class, initial clusterNum = 2, majority number = 160
    #while rows of X and the new generated data do't exceed the majority number
    #run cluster through the new X
    #for each cluster do:
    #   take mean -> new data point
    #   add new data point into X
    #   add new data point into syntheticArray -> to form a syntheticArray with only synthetic data
    #   increament synthethicIndex by 1
    #increase clusterNum by 1
    iterVal = 1
    while (rows + clusterNum < MA_num):
        kmeans = KMeans(n_clusters=clusterNum, init='random')
        kmeans.fit(X)
        label = kmeans.labels_
        for i in range(clusterNum):
            cluster = X[np.where(label==i)]
            average = np.mean(cluster, axis = 0)
            average = np.reshape(average, (1, numAttributes))
            syntheticArray = np.concatenate((syntheticArray, average))
            X = np.vstack((X,average))
        clusterNum+=1
        iterVal+=1
        rows, columns = X.shape
        
    #convert syntheticArray into DataFrame
    syntheticData = pd.DataFrame(syntheticArray, columns=MI.columns.values)

    #concatenate synthetic dataset with imbalanced dataset
    newDF = pd.concat([dataFrame, syntheticData], ignore_index=True)

    return newDF

# AD-KNN Algorithm

# Balances an imbalanced binary classification dataset using Average Difference KNN.
#  dataFrame - the binary classification dataset to balance (dataFrame)
#  classIndex - the name of the column containing class labels
#  minorityLabel - the value in the classIndex column that indicates the instance is a minority, such as 0
#  kMin - minimum k-value to use when generating a random k-value for the KNN search
#  kMax - maximum k-value to use when generating a random k-value for the KNN search
#  epsilon - when the average difference is 0 (i.e., the generated data point is a duplicate of the existing 
#                  data point), epsilon is how far away from the existing point to place the new point

def ADKNN(dataFrame, classIndex, minorityLabel, kMin=2, kMax=6, printDebug=False, epsilon=0.0001):
    
    #Divide data set into Majority (MA) and Minority (MI) Classes
    MA = dataFrame[dataFrame[classIndex] != minorityLabel]
    MI = dataFrame[dataFrame[classIndex] == minorityLabel]
    
    #Record number of instances for each class
    MA_num = MA.shape[0]
    MI_num = MI.shape[0]
    
    #Calculate how much data to generate
    numToSynthesize = MA_num - MI_num
    
    # Make sure kMax has a valid value (does not exceed the number of minority instances)
    if (MI_num - 1) < kMax:
        kMax = MI_num - 1 
        
    # Make sure kMin and kMax are not the same in order not to generate duplicates.
    # If so, set kMin to a default value (kMax / 2).
    if kMin >= kMax:
        kMin = int(np.ceil(kMax / 2))
        
    if printDebug:
        print("~~~~~~~~~~~ MI_num = ", MI_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ MA_num = ", MA_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ kMin = ", kMin, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ kMax = ", kMax, " ~~~~~~~~~~")

    #Convert minority class to NumPy array for synthesizing
    mi = MI.to_numpy()
    
    # Number of attributes for each data instance
    numAttributes = mi.shape[1]
    
    # Create NumPy array that holds the synthetic data    
    syntheticIndex = 0
    syntheticArray = np.empty((numToSynthesize, numAttributes))
    
    # Stores the average of the differences between a query point and its nearest neighbors
    averageDifference = np.zeros(numAttributes)
    
    # The position/index number of the class column label
    classColNum = dataFrame.columns.get_loc(key=classIndex)
    
    # Dictionary that stores the list of k-values that have already been used by each minority
    # point to generate data, in order to avoid generating duplicates
    pointDict = {}
    
    # Initialize each point to have the value of an empty list. k-values will be added to the list
    # Convert each point (a NumPy array) to a tuple to avoid an error, since a NumPy array is unhashable.
    for point in mi:
        pointDict[tuple(point)] = []
        
    for i in range(numToSynthesize):
        # Flag that stops the loop if all possible data points have been generated 
        done = True
        
        # Check if all possible data points have been generated (if all the minority points have used all the k-values)
        for val in pointDict.values():
            if len(val) < kMax - kMin + 1:
                done = False
        
        # If there are still data points left to generate, generate one
        if not done:
            #Select an instance in the minority class randomly
            point = random.choice(mi)
            
            # Convert point to a tuple - this is the dictionary key corresponding to point
            t = tuple(point)
            
            # Make sure this minority point has not used all the possible k-values.
            # If so, pick a new one.
            while len(pointDict[t]) == kMax - kMin + 1:
                point = random.choice(mi)
                t = tuple(point)
                
            point = np.reshape(point, (-1, numAttributes))
            
            # Generate the random k-value used for the KNN search in this iteration.
            k = random.randint(kMin, kMax)
            
            # If this k-value has already been used for this point, then generate a 
            # new k-value that hasn't been used
            while k in pointDict[t]:
                k = random.randint(kMin, kMax)
                
            # Append the newly generated k-value to the list of used k-values for this point
            pointDict[t].append(k)
            
            # Find indices of k nearest neighbors of point. "_" is used to discard the distances to the neighbors.
            _, neighborIndices = findNeighbors(point, mi, k)
            
            point = point.flatten()
            
            # Discard the first neighbor returned, because the first neighbor is always 
            # the query point itself, and flatten neighborIndices into a 1D array
            neighborIndices = neighborIndices[0][1::]
    
            # Sum all the differences between the query point and its neighbors, to find their average
            for j in range(len(neighborIndices)):
                
                # The data point that is the jth neighbor of point
                neighbor = mi[neighborIndices[j]]
                
                # Difference/displacement between the query point and neighbor
                diff = neighbor - point
                
                # Add diff to the sum of all the differences
                averageDifference = np.add(averageDifference, diff)
                
            # Find the average by dividing the sum of the differences by k
            averageDifference = np.divide(averageDifference, float(k))
            
            # New point is the query point plus the average difference between the query point and its neighbors
            newPoint = np.add(point, averageDifference)
            
            # If averageDifference is 0, a duplicate will be generated, so add epsilon to 
            # all attributes of averageDifference
            if np.array_equal(averageDifference, np.zeros(numAttributes)):
                newPoint = np.add(newPoint, np.full(numAttributes, epsilon))

            syntheticArray[syntheticIndex] = newPoint
            
            # Make sure the class label of the new data point is the minority class label
            syntheticArray[syntheticIndex][classColNum] = minorityLabel
            
            averageDifference = np.zeros(numAttributes) # reset all elements to 0 for next iteration
            syntheticIndex += 1
    
    # If the number of points generated is less than numToSynthesize, resize syntheticArray
    # to that number of points, to avoid garbage values.
    if syntheticIndex < numToSynthesize:
        syntheticArray = syntheticArray[0:syntheticIndex]
        
    # Convert synthetic array into DataFrame
    syntheticData = pd.DataFrame(syntheticArray, columns=MI.columns.values)
        
    # Join imbalanced data with synthetic data set
    newDF = pd.concat([dataFrame, syntheticData], ignore_index=True) 
    
    if printDebug:
        print("~~~~~~~ Number of points generated:", syntheticArray.shape[0], " ~~~~~~~")
        print("~~~~~~~ Length of balanced dataset:", newDF.shape[0], " ~~~~~~~")
        
    return newDF


#SMOTEBoost
import random
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from random import randrange
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

#For data analysis
import seaborn as sn
import matplotlib.pyplot as plt

#Main Program
#dataFrame - imbalanced dataset
#class_var - column name containing class identifier
#minor_class - minority identifier
#KNN - number of k-nearest neighbors to search for
#numIterations - number of iterations to generate 'best' data
def SMOTEBoost(dataFrame, class_var, minor_class, KNN=5, numIterations = 5, printDebug = False):  

    newDF = dataFrame
    
    #Calculate number of attributes in dataFrame
    numAttributes = dataFrame.shape[1]
    
    #Divide data set into Majority (MA) and Minority (MI) Classes
    MA = dataFrame[dataFrame[class_var] != minor_class]
    MI = dataFrame[dataFrame[class_var] == minor_class]
    
    #Record number of instances for each class
    MA_num = MA.shape[0]
    MI_num = MI.shape[0]
    
    #Calculate how much data to generate
    numToSynthesize = MA_num - MI_num
    
    #Calculate k neighbors to search for
    if (MI_num - 1) < KNN:
        k = MI_num - 1 
    else:
        k = KNN
    
    # Split numToSynthesize into numIterations for synthesizing
    GENERATE = split(numToSynthesize, numIterations)

    if printDebug == True:
        print("~~~~~~~~~~~ MA_num = ", MA_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ MI_num = ", MI_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ k = ", k, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ numToSynthesize = ", numToSynthesize, " ~~~~~~~~~~")
        print("GENERATE: ", GENERATE)

    # make copy of MI
    newMI = MI
    newDF = dataFrame

    for synth_num in GENERATE:
        best_recall = 0
        best_set = pd.DataFrame()

        #Convert minority class to NumPy array for synthesizing
        mi = newMI.to_numpy()

        for interation in range(5):
            #Create NumPy array for synthetic data
            syntheticArray = np.empty((0, numAttributes))

            for i in range(synth_num):
                #Select and instance x in minority class randomly
                x = random.choice(mi)
                #x = mi[np.random.choice(len(mi),replace=False)]
                x = np.reshape(x, (-1, numAttributes))

                #Find indices of k nearest neighbors of x
                _, knn = findNeighbors(x, mi, k)

                #Select one knn of sample and record it as y
                y = randrange(1, k+1)

                #Generate new minority instance w/ equation
                diff = mi[knn[0, y]] - x
                gap = random.uniform(0,1)
                xnew = x + gap * diff
                syntheticArray = np.concatenate((syntheticArray, xnew))
            
            if printDebug == True:
                print("Iteration Done!")
            #Convert synthetic array into DataFrame
            syntheticData = pd.DataFrame(syntheticArray, columns=MI.columns.values)
            tempDF = pd.concat([newDF, syntheticData], ignore_index=True)

            #Run Logistic Regression on new dataset and original to get recall
            recall = LogisticRegressionCLF(tempDF, dataFrame)
            if (recall > best_recall):
                best_recall = recall
                best_set = syntheticData
        if printDebug == True:
            print("Recall: ", best_recall)
            display(best_set)
        newMI = pd.concat([newMI, best_set], ignore_index=True)
        newDF = pd.concat([newDF, best_set], ignore_index=True)
      
    if printDebug == True:
        display(dataFrame)
        display(newDF)
      
    return newDF

def split(x, n):
    arr = []
    if(x < n): # If we cannot split the number into exactly 'N' parts
        print("ERROR")
        return arr
    elif (x % n == 0): # If x % n == 0 then the minimum difference is 0 and all numbers are x/n
        for i in range(n):
            arr.append(x//n)
    else:
        # upto n-(x % n) the values will be x / n
        # values after will be x / n + 1
        zp = n - (x % n)
        pp = x//n
        for i in range(n):
            if(i >= zp):
                arr.append(pp+1)
            else:
                arr.append(pp)
    return arr

def LogisticRegressionCLF(sample_set, original):
    from sklearn.model_selection import train_test_split
    X = original.drop(['class'], axis=1)
    Y = original.loc[:,'class':'class']

    _, X_test, _, Y_test = train_test_split(X.to_numpy(), Y.to_numpy(), test_size=0.2, stratify = Y.to_numpy(), shuffle = True)

    X_train = sample_set.drop(['class'], axis=1).values
    Y_train = sample_set['class'].values

    model = LogisticRegression()
    clf = model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    
    recall = recall_score(Y_test, prediction)

    return recall
    
#SMOTEBoostCC
import random
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from random import randrange
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

#For data analysis
import seaborn as sn
import matplotlib.pyplot as plt

#Main Program
#dataFrame - imbalanced dataset
#class_var - column name containing class identifier
#minor_class - minority identifier
#KNN - number of k-nearest neighbors to search for
#numIterations - number of iterations to generate 'best' data
def SMOTEBoostCC(dataFrame, class_var, minor_class, KNN=5, numIterations = 5, printDebug = False): 

    newDF = dataFrame
    
    #Calculate number of attributes in dataFrame
    numAttributes = dataFrame.shape[1]
    
    #Divide data set into Majority (MA) and Minority (MI) Classes
    MA = dataFrame[dataFrame[class_var] != minor_class]
    MI = dataFrame[dataFrame[class_var] == minor_class]
    
    #Record number of instances for each class
    MA_num = MA.shape[0]
    MI_num = MI.shape[0]
    
    #Calculate how much data to generate
    numToSynthesize = MA_num - MI_num
    
    #Calculate k neighbors to search for
    if (MI_num - 1) < KNN:
        k = MI_num - 1 
    else:
        k = KNN
    
    # Split numToSynthesize into numIterations for synthesizing
    GENERATE = split(numToSynthesize, numIterations)

    if printDebug == True:
        print("~~~~~~~~~~~ MA_num = ", MA_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ MI_num = ", MI_num, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ k = ", k, " ~~~~~~~~~~")
        print("~~~~~~~~~~~ numToSynthesize = ", numToSynthesize, " ~~~~~~~~~~")
        print("GENERATE: ", GENERATE)

    # make copy of MI
    newMI = MI
    newDF = dataFrame

    #Convert majority to NumPy array for synthesizing and density distribution calculations

    ma = MA.to_numpy()

    for synth_num in GENERATE:
        best_recall = 0
        best_set = pd.DataFrame()

        #Convert minority class to NumPy array for synthesizing
        mi = newMI.to_numpy()

        #Calculate average Euclidean distance Dpos of the total minority class sample set
        Dpos = calculateAverageDistance(mi, MI_num, mi, numAttributes, k)

        #Calculate average Euclidean distance Dneg of the total majority class sample set
        Dneg = calculateAverageDistance(mi, MI_num, ma, numAttributes, k)

        if printDebug == True:
            print("Dpos = ", Dpos)
            print("Dneg = ", Dneg)

        for interation in range(5):
            #Create NumPy array for synthetic data
            syntheticArray = np.empty((0, numAttributes))

            for i in range(synth_num):
                #Select and instance x in minority class randomly
                x = random.choice(mi)
                #x = mi[np.random.choice(len(mi),replace=False)]
                x = np.reshape(x, (-1, numAttributes))

                controlCoeff = calculateControlCoefficient(x, k, mi, ma, Dneg, Dpos)
            
                if printDebug == True:
                    print("cc = ", controlCoeff)

                #Find indices of k nearest neighbors of x
                _, knn = findNeighbors(x, mi, k)

                #Select one knn of sample and record it as y
                y = randrange(1, k+1)

                #Generate new minority instance w/ equation
                diff = mi[knn[0, y]] - x
                gap = controlCoeff
                xnew = x + gap * diff
                syntheticArray = np.concatenate((syntheticArray, xnew))
            
            if printDebug == True:
                print("Iteration Done!")
            #Convert synthetic array into DataFrame
            syntheticData = pd.DataFrame(syntheticArray, columns=MI.columns.values)
            tempDF = pd.concat([newDF, syntheticData], ignore_index=True)

            #Run Logistic Regression on new dataset and original to get recall
            recall = LogisticRegressionCLF(tempDF, dataFrame)
            if (recall > best_recall):
                best_recall = recall
                best_set = syntheticData
        if printDebug == True:
            print("Recall: ", best_recall)
            display(best_set)
        newMI = pd.concat([newMI, best_set], ignore_index=True)
        newDF = pd.concat([newDF, best_set], ignore_index=True)
      
    if printDebug == True:
        display(dataFrame)
        display(newDF)
      
    return newDF
