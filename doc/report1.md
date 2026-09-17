# Machine Learning Project

## Introduction

This project aims to predict wine quality based on objective physicochemical characteristics, such as acidity, density, pH, and residual sugar and alcohol content. To achieve this, we will develop a machine learning model trained on a dataset of red and white variants of the Portuguese "Vinho Verde" wine. Each wine gets a quality score from 1 to 10, 10 being the best.

## Problem Formulation

The application receives an array of a wine's chemical properties and predicts its quality score. This is a supervised machine learning problem where each data point represents a specific wine sample with the following 12 features:

| # | Feature Name | Data Type |
| --- | --- | --- |
| 1 | fixed acidity | Continuous |
| 2 | volatile acidity | Continuous |
| 3 | citric acid | Continuous |
| 4 | residual sugar | Continuous |
| 5 | chlorides | Continuous |
| 6 | free sulfur dioxide | Continuous |
| 7 | total sulfur dioxide | Continuous |
| 8 | density | Continuous |
| 9 | pH | Continuous |
| 10 | sulphates | Continuous |
| 11 | alcohol | Continuous |
| 12 | is white | Categorical (red or white) |

The label is an integer score from 1 to 10, where a higher value indicates superior quality. Because the label is an ordinal integer, this problem can be approached as a regression task.
The dataset is sourced from the UC Irvine Machine Learning Repository (Wine Quality - <https://archive.ics.uci.edu/dataset/186/wine+quality>).

## Methods

- number of data points: 4898
- data preprocessing: correlations?
- random forest?
- splitting the data: 60 - 20 - 20
- sizes of the sets: 2939, 980, 979 (?)
- design choice?

### Preparing the data

The data consists of a total of 6497 samples. It is split into two parts: red wine and white whine with 1599 and 4898 samples respectively (figure?). To analyze the data as a whole, we combined these two tables into one. As part of our preparation we dropped the score column from our data table since it will be used as the label. In addition we introduced a new binary feature called `is white` which indicates whether a certain wine is white (1) or red (0). This increased the dimension of our feature vector to 12.

With this data we did some preliminary analysis. First of all, as shown in figure ? it is obvious that our data contains more white wines than red wines. In addition we can see from figure ? that most of the wines are of average quality, meaning that the data isn't heavily skewed in either direction. 

### Data correlation

A Pearson correlation matrix (figure ?) shows that there are some redundant features. The strongest correlation (r = 0.72) is shown between `free sulfur dioxide` and `total sulfur dioxide`, which can be expected as the free sulfur dioxide is a subset of the total sulfur dioxide. This is something we considered combining into one single `sulfur dioxide ratio` which could be calculated as follows: $x_{\text{ratio}} = \frac{\text{free\_sulfur\_dioxide}}{\text{total\_sulfur\_dioxide}}$. This might be explored in Stage 2 of our report. There are also two other correlating pairs of `density` and `alcohol` (r = -0.69) and `density` and `residual sugar` (r = 0.55). Both of these are to be expected due to the physical properties of alcohol and sugar. In addition our engineered feature of `is white` correlates with both of the sulfur dioxide values as well as well as both of the acidity values.

These values are, however, still left in our model to give us a starting baseline for our model. This gives us all the relevant chemical information, even if it is redundant at times. In addition, should we combine all of the features mention above, we would be left with a rather low-dimensional feature vector that could make our model inaccurate due to lack of data. This could be seen in the resulting model as underfitting.

### Model

To help us analyze our correlated data, without dropping too many features, we have chosen to use Ridge Regression.

## Use of AI

- ideas on what to predict with the model
- ideas on which method to use for the training

## Appendices and References
