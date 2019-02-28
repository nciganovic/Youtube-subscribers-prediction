# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 19:58:35 2019

@author: Nikola Ciganović
"""
#LOADING DATA
import pandas as pd
csv_file = "data.csv"
dt = pd.read_csv(csv_file)

#REPLACING -- WITH NULL
import numpy as np
dt = dt.replace('--', np.nan, regex=True)
dt["Rank"] = dt["Rank"].replace('th', "", regex=True)
dt["Rank"] = dt["Rank"].replace('rd', "", regex=True)
dt["Rank"] = dt["Rank"].replace('nd', "", regex=True)
dt["Rank"] = dt["Rank"].replace('st', "", regex=True)
dt["Rank"] = dt["Rank"].replace(',', "", regex=True)

#REPlACING OBJECT TYPE WITH FLOAT64 
dt["Subscribers"] = pd.to_numeric(dt["Subscribers"])
dt["Video Uploads"] = pd.to_numeric(dt["Video Uploads"])
dt["Video views"] = pd.to_numeric(dt["Video views"])
dt["Rank"] = pd.to_numeric(dt["Rank"])

dt["Average views"] = dt["Video views"] / dt["Video Uploads"]

#INSERTING MEDIAN VALUE TO NULL (in dt_full)
from sklearn.impute import SimpleImputer

sample_incomplete_rows = dt[dt.isnull().any(axis=1)].head()

imputer = SimpleImputer(strategy="median")

dt_num = dt.drop(["Channel name","Grade"], axis = 1)

imputer.fit(dt_num)
X = imputer.transform(dt_num)
dt_full = pd.DataFrame(X, columns=dt_num.columns)
dt_full.loc[sample_incomplete_rows.index.values]
dt_full = pd.DataFrame(X, columns=dt_num.columns)
 
#SHOWING DATA ON GRAPH 
import matplotlib.pyplot as plt
dt.hist(bins=10, figsize=(10,8))
plt.show()

#CORRELATION MATRIX
corr_matrix = dt.corr()
print(corr_matrix["Subscribers"].sort_values(ascending=False))

#SCATTER MATRIX FOR COMPARING 
from pandas.plotting import scatter_matrix
attributes = ["Rank", "Video views", "Video Uploads",
"Subscribers"]
scatter_matrix(dt[attributes], figsize=(12, 8))


#GRAPH WITH MOST RELATION TO EACH OTHER
dt.plot(kind="scatter", x="Subscribers", y="Video views",
alpha=0.4)

dt.plot(kind="scatter", x="Subscribers", y="Average views",
alpha=0.4)

#CATEGORICAL DATA
grade_cat = dt[["Grade"]] 

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
dt_cat_1hot = cat_encoder.fit_transform(grade_cat)
dt_one_hot_array = dt_cat_1hot.toarray()
print(dt_one_hot_array)

#TRAIN AND TEST DATA
dt_full_train = dt_full[:4000]
dt_full_test = dt_full[4000:]

subs_train = dt_full_train["Subscribers"].copy()

#PIPELINES
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

dt_num_tr = num_pipeline.fit_transform(dt_full) 

from sklearn.compose import ColumnTransformer

num_attribs = list(dt_full)
cat_attribs = ["Grade"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs)
    ])

dt_prepared = full_pipeline.fit_transform(dt[:4000]) #???
print("Shape of dt_prepared: ", dt_prepared.shape)

#TRAINING MODEL 
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(dt_prepared, subs_train)

some_data = dt.iloc[:10]
some_labels = subs_train.iloc[:10]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

#mean squared error
from sklearn.metrics import mean_squared_error

dt_predictions = lin_reg.predict(dt_prepared)
lin_mse = mean_squared_error(subs_train, dt_predictions)
lin_rmse = np.sqrt(lin_mse)
print("Mean squared error: ",lin_rmse)

#mean absolute error
from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(subs_train, dt_predictions)
print("Mean absolute error: ",lin_mae)

#Decision tree
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(dt_prepared, subs_train)

dt_predictions = tree_reg.predict(dt_prepared)
tree_mse = mean_squared_error(subs_train, dt_predictions)
tree_rmse = np.sqrt(tree_mse)

print("\n")

#FINE TUNE MODEL  
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, dt_prepared, subs_train,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

print("Decision tree cross validation:", display_scores(tree_rmse_scores))

print("\n")

lin_scores = cross_val_score(lin_reg, dt_prepared, subs_train,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print("Linear regression cross validation:", display_scores(lin_rmse_scores))

print("\n")

#Random Forest
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(dt_prepared, subs_train)

dt_predictions = forest_reg.predict(dt_prepared)
forest_mse = mean_squared_error(subs_train, dt_predictions)
forest_rmse = np.sqrt(forest_mse)
print("\n Random forest: ", forest_rmse)

#Grid Search CV
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(dt_prepared, subs_train)

print("Best parametars", grid_search.best_params_)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
table = pd.DataFrame(grid_search.cv_results_)    

print("\n")

#Randomized search CV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(dt_prepared, subs_train)

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
  
feature_importances = grid_search.best_estimator_.feature_importances_
print("\n")


#FINAL PART
final_model = grid_search.best_estimator_

X_test = dt_full_test.drop("Subscribers", axis=1)

all_grades = dt["Grade"]
grade_test = all_grades[4000:]
X_test["Grade"] = grade_test

y_test = dt_full_test["Subscribers"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)

print(final_rmse)
