# Bank Subscription Prediction Machine Learning Project

## About
This project focuses on whether customer will subscribe for the bank schemes or not. It is a binary classification problem which will help bank to get the outcome from customer.

### Dataset
* [Dataset](https://www.kaggle.com/datasets/berkayalan/bank-marketing-data-set) link from kaggle.


### Models Report:
![model_report](https://user-images.githubusercontent.com/34678255/225872690-ec48c454-6777-486c-9686-adb57db39d29.png)


* Information about numerical features

![describe_df](https://user-images.githubusercontent.com/34678255/225874846-d445353c-5227-4684-9a8e-3d8ee38d3477.png)


### Imbalance target feature was balanced

![target_col_unbalanced](https://user-images.githubusercontent.com/34678255/225873228-6b3dd665-4da2-4ace-9dab-4e00489ffa8f.png)

### After applying SMOTE 
![target_col_balanced](https://user-images.githubusercontent.com/34678255/225873291-26b5ecf0-35ed-43ca-b440-b5ccaad39a79.png)


* Correlation Plot

![heatmap](https://user-images.githubusercontent.com/34678255/225873757-b5e8965b-84f2-4557-893f-171d7c6b60cf.png)

* Outliers 

![outliers](https://user-images.githubusercontent.com/34678255/225874250-ecd60cad-93ce-47e3-af98-cd881323648a.png)

* Bivariate Analysis

![crosstab](https://user-images.githubusercontent.com/34678255/225874471-adbd096e-7e6e-4d93-89fd-c2b796d16059.png)

![outcomewithsubscription](https://user-images.githubusercontent.com/34678255/225874522-303e86de-d5cb-46bd-bad3-c8a9a2c51228.png)

KDEPlot to do scaling

![distribution](https://user-images.githubusercontent.com/34678255/225874722-833660a5-a703-4718-a697-7b365b85200e.png)


### ML Models Used:
* Classification Models
     * Logistic Regression
     * Decision Tree Classifier
     * KNN Classifier
     * RandomForest Classifier
     * XGBoost Classifier


### Python Dependencies:
* pandas
* numpy
* pickle
* matplotlib
* statsmodels
* seaborn
* sklearn
* django

### Install Dependencies (requirements.txt)
1. pip install -r requirements.txt
