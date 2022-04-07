import numpy as np

# pasted from DecisionTreeFun
header = ["level", "lang", "tweets", "phd"]
attribute_domains = {"level": ["Senior", "Mid", "Junior"], 
    "lang": ["R", "Python", "Java"],
    "tweets": ["yes", "no"], 
    "phd": ["yes", "no"]}
X = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]

y = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
# stitch X and y together to make one table
table = [X[i] + [y[i]] for i in range(len(X))]

# TODO: begin notes on ensemble learning here
# ensemble learning: a collection of "weak" learners
# that work together to make predictions
# together, the learners from a "stronger" learner
# though there is no guarantee that the ensemble
# is "better" than any one learner

# ensemble classification: a collection of "weak" classifiers
# that work together to make classifications 
# via voting strategy (simple majority voting for the project;
# there is also weighted majority voting; track record voting
# is bonus for the project)

# let N be the number of "weak" learners in our ensemble
# example N = 100, 100 weak learners that work together (vote)
# to make classifications
# homogeneous ensemble: all the learners are of the same type
# example N = 100 decision trees (called a random forest;
# what we will implement for project)
# heterogenous ensemble: a mix of types

# let M be the number of "better" learners from the N
# "weak" learners that we retain to form our final ensemble
# M < N

# what are some ways to generate "weak" learners
# goal is to have diversity
# 1. generate a classifier (tree) using "different" training data
# 2. generate a classifier (tree) using "different" attribute
# selection techniques (random, entropy, gini, etc...)
# 3. generate a classifier (tree) using "different" attribute
# subsets
# 4. others??? creative

# recall: what are the techniques for train and test set
# 1. holdout method**
# 2. random subsampling
# 3. k fold cross validation (and variants)
# 4. bootstrap method**

# TODO: begin notes on bagging here
# bagging: bootstrap aggregating
# an ensemble method for creating N weak learners
# and retaining the M best learners to form an ensemble
# basic approach
# 1. divide the dataset into a test set and "remainder" set
# 2. using the remainder set, sample N bootstrap samples
# one for each of N trees (note this is a diff N than the sample size)
# for each sample (used to build 1 tree):
#   ~63.2% of instances will be sampled into training set
#   ~36.8% of instances will not (form VALIDATION SET)
# 3. measure the performance of each tree on its validation set
# using some performance measure. retain the best M trees based
# on the performance scores
# 4. using majority voting, make M predictions from the M trees
# for each unseen instance in the test set (the majority vote is the instance's
# prediction)

# advantages of bagging
# 1. simple idea, simple to implement
# 2. reduces overfitting
# 3. typically improves accuracy
# (reduces the variance across classifications)


# pasted from ClassificationFun (note this N is different from ensemble learning's use of N):
# 4. bootstrap method
# like random subsampling but with replacement
# create a training set by sampling N instances
# with replacement 
# N is the number instances in the dataset
# the instances not sampled form your test set
# ~63.2% of instances will be sampled into training set
# ~36.8% of instances will not (form test set)
# see github for math intuition
# repeat the bootstrap sampling k times
# accuracy is the weighted average accuracy
# over the k runs
# (weighted because test set size varies over k runs)

# (done on PA5) Ensemble Lab Task 1: 
# Write a bootstrap function to return a random sample of rows with replacement
# (test your function with the interview dataset)
def compute_bootstrapped_sample(table):
    n = len(table)
    sample = []
    for _ in range(n):
        rand_index = np.random.randint(0, n) # Return random integers from low (inclusive) to high (exclusive)
        sample.append(table[rand_index])
    return sample 

sample = compute_bootstrapped_sample(table)
for row in sample:
    print(row)

# TODO: begin notes on random attribute subsets here
# let's return to #3
# let F be the size of random attribute subsets
# F >= 2


# TODO: Ensemble Lab Task 2:
# Define a python function that selects F random attributes
# from an attribute list
# (test your function with att_indexes (or header))
def compute_random_subset(values, num_values):
    # you can use np.random.choice()
    values_copy = values[:] # shallow copy
    np.random.shuffle(values_copy) # in place
    return values_copy[:num_values]
F = 2
print(compute_random_subset(header, F))
att_indexes = list(range(len(header)))
print(compute_random_subset(att_indexes, F))

# project notes
# implement a random forest (all learners are trees)
# with bagging and with random attribute subsets
# will need to modify tree generation: for each node 
# in our tree, we use random attribute subsets.
# call compute_random_subset() right before a call to
# select_attribute() in tdidt pass the return subset
# (size F) into select_attribute()

# TODO: Ensemble Lab Task 3 (For Extra Practice):
# see https://github.com/GonzagaCPSC322/U7-Ensemble-Learning/blob/master/A%20Ensemble%20Learning.ipynb