# 3day-machine-learning-starting
code demos and notes for a starting course on machine learning, strenghten my foundation on ML, learning from following link: https://www.bilibili.com/video/BV1nt411r7tj?t=98&amp;amp;p=23  

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)

### General info
This project includes all the code demos of on-class exercise and instance trainning, as well as notes for the 3-day learning.    
Codes are in English version.  
Notes are in Chinese version.    
User can find some basic machine learning models and deployment inside this project, but if you want to find more deep knowledge in machine learning filed, this project is totally too simple, it is more suitable for a starting learning user.  
1. day1_demos includes each on-class small exercises for day1 to let user know the basic concept and deployment of ML models  
2. day2_demos includes each on-class small exercises for day2 to let user know the basic concept and deployment of ML models  
3. day3_demos includes each on-class small exercises for day3 to let user know the basic concept and deployment of ML models   
4. other instance file includes more comprehensive deployment of thaat whole day's knowledge  

### Technologies
Project is created with:
* Pycharm version: 2020.2.5(community edition)
* Jupyter Notebook
	
### Setup
To run this project, install followings locally using pip:  
Inner-built libraries are time, math and os. User needs to install some additional libraries, they are all included inside requirements.txt, can just use pip to install.
```
pip install -r requirements.txt
```

### Features
#### Day 1 demos include following content:
  1. sklaern dataset usage 
  2. dictionary feature extraction (DicVectorizer)
  3. text feature extraction (CountVectorizer)
  4. cut word usage (jieba)
  5. use TF-IDF to process text feature extraction
  6. Normalization
  7. Standardization
  8. filter features with low variance (pearsonr \ VarianceThreshold)
  9. PCA dimension reduction (PCA)
#### Day 2 demos include following content:
  1. use KNN algorithm to classify the iris category
  2. use KNN algorithm to classify the iris category, add grid search and cross validation
  3. use Naive Bayes to classify the data
  4. use Decision Tree to classify the iris dataset 
#### Day 3 demos include following content:
  1. use Normal Equation to predict the house price in boston with Linear Regression model   
  2. use Gradient Descent to predict the house price in boston with Linear Regression model   
  3. use Ridge Regression to predict the house price in boston
