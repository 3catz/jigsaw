#Using random cv-grid search to find optimal hyperparameters 
#for the LGBM model used in the Kaggle/Jigsaw.ai unintentional bias 
#in toxic comments competition of 2019.

import lightgbm as lgb
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

param_test ={
             'num_leaves': sp_randint(30, 100), 
             'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4],
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.5, scale=0.5),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
             'bagging_fraction': sp_uniform(loc=0.5,scale=0.5),
             'feature_fraction':sp_uniform(loc=0.5, scale = 0.5)}



fit_params={"early_stopping_rounds":30, 
            
            "eval_metric" : 'auc', 
            "eval_set" : [(X_test, y_test)],
            'eval_names': ['valid'],
            #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
            'verbose': 100,
            'categorical_feature': 'auto'}
            
from sklearn.metrics import f1_score
n_HP_points_to_test = 3


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
clf = lgb.LGBMClassifier(max_depth=-1, random_state=901, silent=False, 
                         metric="binary", n_jobs=4,
                         n_estimators=5000)
gs = RandomizedSearchCV(
    estimator=clf, param_distributions=param_test, 
    n_iter=n_HP_points_to_test,
    scoring='roc_auc',
    cv=3,
    refit=True,
    random_state=901,
    verbose=True)

gs.fit(X_train, y_train, **fit_params)

print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))
