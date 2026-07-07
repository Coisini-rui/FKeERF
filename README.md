# PseU-FKeERF
PseU-FKeERF demonstrates high accuracy in identifying pseudouridine sites from RNA sequencing data.<br>
FKeERF method has three parts, including constructing fuzzy feature set, constructing evidence decision tree and category prediction. a. By clustering the original feature set through fuzzy means, several clusters and the mean and variance of each cluster are obtained. Then, multiple fuzzy feature subsets are obtained by using Gaussian membership function, and fuzzy feature sets are obtained by fusing multiple fuzzy feature subsets. b. Use fuzzy feature set to construct evidence random forest. c. Input the training set and test set samples into the evidence random forest respectively, count the number of the two falling on the same node, use the kernel function and prediction function to obtain the prediction result, and combine the symbol function to obtain the final prediction label.

## Requirements
python=3.9.13

numpy=1.21.5<br>
pandas=1.4.4<br>
scikit-learn=1.0.2<br>
scikit-fuzzy=0.4.2<br>


## Usage
### decision_tree_imperfect.py
Constructing evidence decision tree. EDT for Evidential Decision Tree is used to predict labels when input data are imperfectly labeled.
### evidential_random_forest.py
Constructing evidence random forest. ERF for Evidential Random Forest, it is used to predict labels when input data are imperfectly labeled.
### fuzzy_feature_construct.py
Constructing fuzzy feature set. By clustering the original feature set through fuzzy means, several clusters and the mean and variance of each cluster are obtained. Then, multiple fuzzy feature subsets are obtained by using Gaussian membership function, and fuzzy feature sets are obtained by fusing multiple fuzzy feature subsets.
### ibelief.py
The file provides combination rules for multiple masses, different rules for decision making in the framework of belief functions and so on. In addition, the Jousselme distance between two mass functions mass1 and mass2 can be calculated.
### utils.py
The file provides tool methods, including logging runtime.
### demo.py
Read the csv file after feature extraction and processing, and input it into the FKeERF model to calculate the relevant indicators.<br>

First, read the original feature set xx.csv after feature selection processing.<br>
```java
pd.read_csv("xxx.csv")
```
Then, cross-validation tests are performed.<br>
```java
acc, mcc, sn, sp = cross_validation(np.array(data))
```
When constructing fuzzy feature set, the number of clusters can be customized. <br>
```java
def gene_ante_fcm(data):
    k = 3
```
When constructing a random forest of evidence, you can customize the tree size and minimum sample leaf number of the tree.<br>
```java
def train_test(train_set, test_set):
    clf = ERF(n_estimators=100, min_samples_leaf=4, criterion="conflict", rf_max_features="sqrt", n_jobs=1)
```
Use the fit() method to train the model.<br>
```java
clf.fit(train_set[:, 1:], y_train_belief)
```
Finally, the kernel method is used to predict the labels of unknown samples.<br>

## Parameters description
**k** is the number of clusters in the cluster, and the default value is 3. <br>
**n_estimators** is the number of estimators (evidence decision trees), and the default is 100. <br>
**min_samples_leaf** is the minimum number of sample leaves. The default value is 4. <br>





