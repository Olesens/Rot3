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
#session_df = all_trials(pwm, 'AA08', date_list3, dummies=False)

# need to give the cal_prob function a stage option


date_list4 = ['2019-07-24', '2019-07-25',
       '2019-07-26', '2019-07-27', '2019-07-29', '2019-07-30', '2019-07-31',
       '2019-08-01', '2019-08-02', '2019-08-05', '2019-08-06', '2019-08-07',
       '2019-08-08', '2019-08-09', '2019-08-12', '2019-08-13', '2019-08-14',
       '2019-08-15', '2019-08-16', '2019-08-19', '2019-08-20', '2019-08-21',
       '2019-08-22', '2019-08-23', '2019-08-27', '2019-08-28', '2019-08-29',
       '2019-08-30', '2019-09-02', '2019-09-03', '2019-09-04', '2019-09-05',
       '2019-09-06', '2019-09-09', '2019-09-10', '2019-09-11', '2019-09-12',
       '2019-09-13', '2019-09-16']


aa08 = all_trials(pwm, 'AA08', date_list4)
plot_all_logs(aa08)
plt.title(' PsuedoR2 for AA08')

aa02 = all_trials(pwm, 'AA02', date_list4)
plot_all_logs(aa02)
plt.title(' PsuedoR2 for AA02')

do05 = all_trials(pwm, 'DO05', date_list4)
plot_all_logs(do05)
plt.title(' PsuedoR2 for DO05')

sc01 = all_trials(pwm, 'SC01', date_list4)
plot_all_logs(sc01)
plt.title(' PsuedoR2 for SC01')

vp06 = all_trials(pwm, 'VP06', date_list4)
plot_all_logs(vp06)
plt.title(' PsuedoR2 for VP06')
