from surprise.reader import Reader
from surprise.model_selection import cross_validate, KFold,GridSearchCV
from surprise.trainset import Trainset

from surprise import (
    Dataset,    
    SVD,
    accuracy
)

import copy
from tabulate import tabulate

import numpy as np
import random

import pandas as pd  # noqa

def set_my_folds(dataset, nfolds = 5, shuffle = True):
    
    raw_ratings = dataset.raw_ratings
    if (shuffle): raw_ratings = random.sample(raw_ratings, len(raw_ratings))

    chunk_size = int(1/nfolds * len(raw_ratings))
    thresholds = [chunk_size * x for x in range(0,nfolds)]
    
    print("set_my_folds> len(raw_ratings): %d" % len(raw_ratings))    
    
    folds = []
    
    for th in thresholds:                
        test_raw_ratings = raw_ratings[th: th + chunk_size]
        train_raw_ratings = raw_ratings[:th] + raw_ratings[th + chunk_size:]
    
        print("set_my_folds> threshold: %d, len(train_raw_ratings): %d, len(test_raw_ratings): %d" % (th, len(train_raw_ratings), len(test_raw_ratings)))
        
        folds.append((train_raw_ratings,test_raw_ratings))
       
    return folds

    
# Setting seed to make code reproducible
my_seed = 311 
random.seed(my_seed)
np.random.seed(my_seed)

# Data loading
data = Dataset.load_builtin("ml-100k")                                                   
               
svd_param_grid = {"n_factors": [25, 100], "n_epochs": [10, 20]}

folds = set_my_folds(data)  

for i, (train_ratings, test_ratings) in enumerate(folds):
    
    print('Fold: %d' % i)
    svd_gs = GridSearchCV(SVD, svd_param_grid, measures=["rmse", "mae"], cv=3)
    
    # fit parameter must have a raw_ratings attribute
    train_dataset = copy.deepcopy(data)
    train_dataset.raw_ratings = train_ratings
    svd_gs.fit(train_dataset)
            
    # best MAE score    
    print('Grid search>\nmae=%.3f, cfg=%s' % (svd_gs.best_score["mae"], svd_gs.best_params["mae"]))
    
    # We can now use the algorithm that yields the best MAE
    svd_algo = svd_gs.best_estimator["mae"]
    
    # We train the algorithm with the whole train set
    svd_algo.fit(train_dataset.build_full_trainset())
    
    # test parameter must be a testset
    test_dataset = copy.deepcopy(data) 
    test_dataset.raw_ratings = test_ratings    
    test_set = test_dataset.construct_testset(raw_testset=test_ratings)
    
    svd_predictions = svd_algo.test(test_set)
     
    # Compute and print MAE
    print('Test>')
    accuracy.mae(svd_predictions,verbose=True)   
    