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

# TODO: begin notes on bagging here

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

# TODO: Ensemble Lab Task 2:
# Define a python function that selects F random attributes from an attribute list
# (test your function with att_indexes (or header))

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