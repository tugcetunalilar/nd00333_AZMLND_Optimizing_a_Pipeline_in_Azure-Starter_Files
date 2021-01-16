# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset is on the direct marketing campaign of bank. Each record represents a potential target customer for deposit placement with the bank and its result as to whether effort led to placement of deposit. What we tried to do with the ML model was to train a binary classification model to predict whether a potential target of marketing campaign would lead to a deposit placement. 

I used two ML pipelines, defining the primary metric as "Accuracy". They are:

1) Scikit-learn pipeline with Logistic Regression Classifier and hyperdrive for hyperparameter tuning.

2) Microsoft Azure proprietary AutoML pipeline for automated machine learning including selection of classifiers and their hyper parameters.

The selected best models were:

i) Scikit-learn: Logistic Regression model with 91.43% Accuracy and with parameter values: C1: 1,max_iter: 100.

ii) Azure AutoML: VotingEnsemble classifier with 91.639 % Accuracy. Therefore, best pipeline was Azure AutoML and best model was VotingEnsemble classifier with 91.639 % Accuracy.

Scikit-learn Pipeline
Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.

Scikit-learn pipeline was based on prescribed Logistic Regression classifier, which is suitable since we applied binary classification based on yes and no.

The overall data included 32,950 records. 29258 records were no, and 3692 records were yes. It was split into training and test set with 80/20 split.

Two hyperparameter tuned are

i) C: Inverse of regularization strength

ii) max_iter: Maximum number of iterations taken for the solvers to converge.

In order to define sample space for hyperparameter tuning we used RandomParameterSampling class. In this sampling algorithm, parameter values are chosen from a set of discrete or continuous values

What are the benefits of the parameter sampler you chose?

For Hyperdrive package other sampling class available are

i) GridParameter Sampling: Defines a grid of samples over hyperparameter search space.

ii) BayesianParameter Sampling: It tries to intelligently pick the next sample of hyper parameters based on how the previous sample performed.

As compared to GridParameter sampling Random Parameter is much faster as it is not required to search over entire grid of sample space. However as compared to Bayesian method over selected approach lacks the element of learning from prior sample performance.

What are the benefits of the early stopping policy you chose?

Early stopping policies are useful to stop as run when specified policy criteria are met.

We used Bandit policy that defines early termination based on slack criteria and frequency and delay interval for evaluation. Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated.

Use of termination policy like Bandit is more time and resource efficient as compared to NoTerminationPolicy option, as it ensures that all runs that are certain% worst then the best current run are killed.

AutoML
AutoML pipeline evaluated 26 classifiers. These range from simple classifiers such as Logistic Regression and Random forest to complex ensemble classifiers such XGBoostClassiers, GradientBoosting etc. It was able to tune across a large no of hyperparameter (10 main for best classifier)

Pipeline comparison

Scikit-Learn pipeline only evaluated one classifier i.e. Logistic Regression and that only on 2 hyperparameter. Whereas AutoML tested over 26 classifiers’ and that too across a very large number of hyper parameters. It is interesting to note that as per our observation AutoML also tested LogisticRegression classifier and came up with around same accuracy for Logistic Regression best model (91.12% vs 91.53% ) However as AutoML was able to test over a very large range of classifier many of which were ensemble classifiers along with a very large no hyperparameter, it was able to select a marginally better VotingEnsemble classifier with 91.639 % Accuracy.

Scikit learn pipeline was limited but was still able to compete well and produced a highly accurate classier.

The main difference between two pipelines was thus their ability to select and score on number of classifier’s and parameter samples. However, AutoML run was much faster as in within 30 minutes it tested over 26 classifiers. Hyperdrive run in comparison took same amount of time for testing one classifier with limited hyperparameter space.

Future work
What are some areas of improvement for future experiments? Why might these improvements help the model?

Some of areas of improvements and their impact of model improvement are as follows:

i) In Hyperdrive pipeline test more classifiers including some of ensemble classifiers as identified by AutoML. This might improve model performance by identifying a faster and more accurate model. ii) In Hyperdrive pipeline use Bayesian Parameter sampling: This might make experiment run faster and be able to quickly identify best hyperparameter.

iii) In Hyperdrive pipeline test more hyperparameter for tuning such as penalty, solver, class_weight etc. They might improve model performance by testing a hyperparameter combination that is able to yield more accurate model.

iv) In Hyperdrive pipeline to address class imbalance in data (29258 records of "No" class and 3692 records of "Yes" class) by either using SMOTE resampling technique or using class_weight parameter.

v) Allow more time for AutoML run. It might be able to cover more classifiers and their hyperparameter and may identify a more accurate model.

Proof of cluster clean up
We used AmlCompute command delete() (documented in our python notebook) to clean up cluster when both pipeline runs have finished.
