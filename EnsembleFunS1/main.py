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

# begin notes on ensemble learning here


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
