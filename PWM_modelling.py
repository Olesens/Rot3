from PWM_Plotting import clean_up_df, cal_prob
from SC_modelling import trial_df, all_trials, cv_models_logit, cv_models_logreg, cv_models_log, plot_all_logs
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
from scipy.optimize import curve_fit
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics
import warnings
from statistics import mean
warnings.filterwarnings('ignore')

# Load in PWM dataframe
pickle_in = open("Rot3_data\\PWM_full_df.pkl","rb")
Animal_df = pickle.load(pickle_in)
animal_list = []  # optional selection of animals to include in dataframe.

pwm = clean_up_df(Animal_df, animallist=animal_list)
pwm = pwm.rename(columns={"history_pair": "history_stim"})  # temporary fix to run some functions
date_list = pwm.index.values
date_list2 = ['2019-07-24',
       '2019-07-25', '2019-07-26', '2019-07-29', '2019-07-30',
       '2019-07-31', '2019-08-01', '2019-08-02', '2019-08-05',
       '2019-08-06', '2019-08-07']  # stage 3 list
date_list3 = ['2019-06-19', '2019-06-20',
       '2019-06-21', '2019-06-24', '2019-06-25', '2019-06-26',
       '2019-06-27', '2019-06-28', '2019-07-01', '2019-07-02',
       '2019-07-03', '2019-07-04', '2019-07-05', '2019-07-08',
       '2019-07-09', '2019-07-10', '2019-07-11', '2019-07-12',
       '2019-07-15', '2019-07-16', '2019-07-17']  # stage 2
animals = ['AA02', 'AA04', 'AA06', 'AA08', 'DO01', 'DO02', 'DO05', 'DO06',
           'SC01', 'SC02', 'SC03', 'SC06', 'VP02', 'VP03', 'VP06']
session_df = all_trials(pwm, 'AA08', date_list3, dummies=False)

# need to give the cal_prob function a stage option
