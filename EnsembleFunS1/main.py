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
# together they form a "stronger" learner
# though there is no guarantee the ensemble is
# any better than a single learner

# ensemble classification: a collection of "weak"
# classifiers that work together to make classifications
# using some voting policy (simple majority voting for the project;
# track record voting for the project bonus)

# let N be the number of "weak" learners in our ensemble
# example: N = 100, then we have 100 learners that work (vote)
# together to make predictions
# homegenous ensemble: all the learners are of the same type
# example: N = 100 decision trees (called a random forest;
# classification algorithm for the project)
# heterogeneous ensemble: a mix of types

# let M be the number of "better" learners from the N learners
# that we retain to form our ensemble
# M < N

# what are some ways to generate N learners?
# goal is to have some diversity amongst the learner
# 1. generate a classifier (tree) using "different"
# attribute subsets
# 2. generate a classifier (tree) using "different"
# training sets
# 3. generate a classifier (tree) using "different"
# attribute selection techniques
# 4. others??? creative

# lets start with #2.)
# recall: we talked about 4 different ways to generate
# train and test sets
# 1. holdout method
# 2. random subsampling
# 3. k fold cross validation (and variants)
# 4. bootstrap method

# TODO: begin notes on bagging here
# bagging: bootstrap aggregating
# an ensemble approach to generating N trees
# and choosing the best M from the N trees
# (for the ensemble)
# basic approach
# 1. split your dataset into a test set and a "remainder set"
# 2. using the remainder set, sample N (diff N 
# than number of instances) bootstrap samples
# and use each sample to build a tree
# for each tree's sample:
#   ~63.2% of instances will be sampled into training set
#   ~36.8% of instances will not (form VALIDATION SET)
# 3. measure the performance of the tree on the validation set
# using a performance metric. then choose to retain the 
# M best trees based on their performance scores... that is the ensemble
# 4. using the best M trees, make predictions for each instance in
# the test set (see step 1) using majority voting

# advantages of bagging
# 1. simple idea, simple to implement
# 2. reduces overfitting
# 3. generally increases accuracy
# (reduces the classification variance across classifiers)


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
# returning to approach #1) from above (random subsets)
# let F be the size of random attribute subsets
# F >= 2

# TODO: Ensemble Lab Task 2:
# Define a python function that selects F random attributes
# from an attribute list
# (test your function with att_indexes (or header))
def compute_random_subset(values, num_values):
    # there is a function np.random.choice()
    values_copy = values[:] # shallow copy
    np.random.shuffle(values_copy) # in place shuffle
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