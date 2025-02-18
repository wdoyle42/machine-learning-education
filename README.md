---
output:
  pdf_document: default
  html_document: default
---
# Machine Learning in Education

# LPO 7500

# Spring 2025

## Will Doyle


Machine learning constitutes a different field from statistics given its focus on 
out-of-sample prediction as opposed to statistics' emphasis on 
inference and estimation. While many of the same algorithms are used,
the purpose of the two are quite distinct. In education, machine learning
supports a variety of predictive tasks, but education researchers are 
rarely trained in the development of machine learning models. 

This course will introduce the learner to machine learning in education settings
The focus will be pragmatic, with an emphasis on how various models
are tuned in order to best provide out-of-sample predictions with the lowest
loss rates. Learners will know how to train and deploy some widely-used models
after taking the course. 

## Evaluation

*Problem Sets* 

There will be 10 problem sets, each of which will be worth 10 points. Assignments 
will constitute 50% of the final grade. 

*Semester-Long Assignment*

Students will create a deployable model trained on an existing dataset. 
Students will write a manuscript describing their process and the nature of the model
(predictive results and variable importance, as appropriate).This manuscript should be 
a good first draft for a manuscript suitable for publication.  Students will also create a 
github repository containing the code to create the model and the deployable model. Students may work either 
alone or in a group of two on this assignment.

There will be four check-in assignments, a draft of the final paper, and the paper. These will be
referred to as "progress reports" draft and final version. Combined these will constitute 50% of the final grade, the final paper will be 50% of this portion, and each
check-in and the draft will be 10%. 

## Expectations

Attendance: you have to show up. If you can't make it, please let me know. 

Honor code: assignments are to be completed individually, group assignments only by members of the group. Generative AI
may be used as a learning tool, but copying and pasting the results of generative AI and describing it as your own work
is a clear violation of the honor code. 

Technology: please do not use your mobile device at any time except for two-factor authentication. Do not use your laptop for
anything other than class purposes. Using technology for other purposes distracts your classmates, and for this reason if 
you make a habit of doing so I'll ask you to leave class for the day. 


## Required Texts

James et al, An Introduction to Statistical Learning 

https://www.statlearning.com/

Referred to as *ISLR*

Kuhn and Silge Tidy Modeling with R

https://www.tmwr.org/

Referred to a *TMWR*

## Optional References:

Wickham et al, R for Data Science

https://r4ds.hadley.nz/

Referred to as *R4DS*

Chang, R Graphics Cookbook

https://r-graphics.org/

Referred to as *R Graphics Cookbook*

CS 229 Lecture notes

https://cs229.stanford.edu/main_notes.pdf

https://cs229.stanford.edu/cs229-notes-decision_trees.pdf

Referred to as *CS 229*


## Schedule of Topics

### L1 Regularization, the Lasso 

- Intro to tidymodels

- Feature engineering basics

ISLR Chs. 1,2,3 and 6.1-6.2  (Skip 2.3 and 3.6, no exercises)

TMWR Chs. 1,2, 3

### Machine Learning Workflow

- Testing and Training

- Validation

- Hyperparameter tuning

- Choosing best fit

- A complete workflow


TMWR Chs. 4-8

ISLR Ch. 5

### L1 and L2 Regularization, Elastic Net

- Understanding hyperparameters

- Measures of model fit: regression and classification

- Tuning approaches: grid search, simulated annealing, genetic algorithms 

ISLR ch. 10

TMWR Ch. 14

### Tree-Based Methods

- Classification and Regression Trees (CART)

- Random Forests

- Understanding hyperparameters in Random Forests

- Fitting and Tuning Parameters

- Prediction from trained models

ISLR 8

CS 229 Notes on Decision Trees


TMWR 10.1

### Support Vector Machines

- Support Vector Machine Concepts: support vectors, kernel "trick"

- Hyperparameters

- Fitting and Tuning Hyperparameters

- Prediction from trained models

ISLR 9

CS 229 6

TMWR 14.1

### Neural Networks

ISLR  10

CS 229 7

- Neural Network Concepts

- Backpropagation

- Activation functions

- Measuring Model fit via cross entropy

- Cross validation and choosing hyperparameters

- Applications


## Assignments

Weekly-ish problem sets

Four Check-Ins for Final Assignment

Draft of Final Paper

Final Paper



