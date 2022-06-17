import os
from datetime import datetime
from statsmodels import robust
from scipy.stats import kurtosis, skew
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import randint
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# GPS and Distances
import glob
import re
import haversine as hs 
from haversine import Unit
import folium
from folium.plugins import HeatMapWithTime
from geopy import distance


#sklearn 
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
# from xgboost import plot_importance
# from xgboost import XGBClassifier
# import xgboost as xgb 
from sklearn.model_selection import learning_curve, GridSearchCV, train_test_split, StratifiedKFold, validation_curve, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# SHAP and explainability
from pdpbox.pdp import pdp_isolate, pdp_plot
from pdpbox.pdp import pdp_interact, pdp_interact_plot
import shap

# Classification 
# import plotly
# import plotly.express as px 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.inspection import permutation_importance
import eli5
from eli5.sklearn import PermutationImportance
from sklearn import metrics
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split, ShuffleSplit, learning_curve
from sklearn.utils import class_weight
from yellowbrick.classifier import ROCAUC
from yellowbrick.datasets import load_game

#EDA
import sweetviz as sv
import pandas_profiling
from pandas_profiling import ProfileReport


##############################
###### EDA and cleaning ######
##############################
def impute_invalid_vals_col(df, col='gps_height'):
    """Fill the zeros with values from the district and region averages"""
    df[col].replace(0.0, np.nan, inplace=True)
    df[col].fillna(df.groupby(['region', 'district_code'])[col].transform('mean'), inplace=True)
    df[col].fillna(df.groupby(['region'])[col].transform('mean'), inplace=True)
    df[col].fillna(df['gps_height'].mean(), inplace=True)
    return df[col]

def encodeColums(app_train,cardinality_limit=1000):
    """ Encode all the object columns in the dataframe based on a cardinality limit""" 
    le = LabelEncoder()
    le_count = 0

    for col in app_train:
        if(app_train[col].dtype == 'object'):
            # If 2 or fewer unique categories
            if(len(list(app_train[col].unique())) <= cardinality_limit):
                le.fit(app_train[col])
                app_train[col] = le.transform(app_train[col])
                le_count += 1
            else:
                print('too many unique values to encode for:', col, np.unique(col))

    print('%d columns were label encoded.' % le_count)
    return;


def distribution_checker(data, xlabel):
    """Return the distribution of the feature for the target columns"""
    grouped = data.groupby([xlabel, 'Occupancy'])['Taxi'].count().reset_index()
    pivot = grouped.pivot_table(index = xlabel, columns = 'Occupancy', fill_value = 0)
    mi=pivot.columns
    mi.tolist()
    ind = pd.Index([str(e[1])  for e in mi.tolist()])
    pivot.columns = ind
    pivot['nr_classes'] = pivot['0']+pivot['1']
    pivot['all_classes'] = pivot['nr_classes'].sum()
    pivot['perc_0'] = ((pivot['0']/pivot['nr_classes'])*100).round(1)
    pivot['perc_1'] = ((pivot['1']/pivot['nr_classes'])*100).round(1)
    return(pivot)

def show_top_k_pump_cat(df,col='DayOfWeek',k=10):
    agg_df = distribution_checker(df, col)
    agg_df = agg_df.sort_values('nr_classes', ascending= False)
    return(agg_df.head(10))

#############
## Results ##
#############


####################################
###### Modelling functions #########
####################################
def learning_curve_model_cv(X, Y, model, cv, train_sizes):
    """Plot the Sklearn learning curve for the model with the train and test sets over different training samples """
    warnings.filterwarnings('ignore')
    plt.figure()
    plt.title("Learning curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(model, X, Y, cv=cv, n_jobs=4, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")

    plt.legend(loc="best")
    return plt;

def plot_confusion_matrix(y_test,y_pred):
    """Compute and plot confusion matrix from the test and predicted labels"""
    fig, ax = plt.subplots(figsize=(8,5)) 
    data = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(data, columns=np.unique(y_test), index = np.unique(y_test))
    ax = sns.heatmap(df_cm, cmap='Blues', fmt='g' ,annot=True,annot_kws={"size": 14})
    ax.set_xlabel("Predicted")
    ax.set_ylabel ("Actual")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_yticklabels(ax.get_xticklabels(), rotation=0)
    return

def calc_metrics_from_model(model,X_train, X_test,y_test, activate_lc=False):
    """Calc. classification metrics for a model passed like confusion matrix, classification report and AUROC curves"""
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    # print("Accuracy score on train set: {}".format(accuracy_score(y_train, y_pred_train)))
    print("Accuracy score on test set: {:.2f}".format(accuracy_score(y_test, y_pred)))
    print('\n')

    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred)
    plt.show()
    
    if(activate_lc==True):
        kfold = StratifiedKFold(n_splits=5)
        learning_curve_model_cv(X_train, y_train, model, kfold, train_sizes=np.linspace(.1, 1.0, 5))
        
    visualizer = ROCAUC(model, classes=['functional', 'functional needs repair', 'non functional'])

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show()                       # Finalize and render the figure
    return    