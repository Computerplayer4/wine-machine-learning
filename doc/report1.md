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
| 12 | color | Categorical (red or white) |

The label is an integer score from 1 to 10, where a higher value indicates superior quality. Because the label is an ordinal integer, this problem can be approached as a regression task.
The dataset is sourced from the UC Irvine Machine Learning Repository (Wine Quality - <https://archive.ics.uci.edu/dataset/186/wine+quality>).

## Methods

- number of data points: 4898
- data preprocessing: correlations?
- random forest?
- splitting the data: 60 - 20 - 20
- sizes of the sets: 2939, 980, 979 (?)
- design choice?

## Use of AI

- ideas on what to predict with the model
- ideas on which method to use for the training

## Appendices and References
