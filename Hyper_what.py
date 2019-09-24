from Plotting_functions import clean_up_df, cal_prob
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
from sklearn import preprocessing
from sklearn import metrics
import warnings
from statistics import mean
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')
from SC_modelling import cnf_heatmap

# Load in PWM dataframe
pickle_in = open("Rot3_data\\PWM_full_df.pkl","rb")
Animal_df = pickle.load(pickle_in)
animal_list = []  # optional selection of animals to include in dataframe.
pwm = clean_up_df(Animal_df, animallist=animal_list)
pwm = pwm.rename(columns={"history_pair": "history_stim"})  # temporary fix to run some functions
date_list = pwm.index.values
date_list4 = ['2019-07-24', '2019-07-25',
       '2019-07-26', '2019-07-27', '2019-07-29', '2019-07-30', '2019-07-31',
       '2019-08-01', '2019-08-02', '2019-08-05', '2019-08-06', '2019-08-07',
       '2019-08-08', '2019-08-09', '2019-08-12', '2019-08-13', '2019-08-14',
       '2019-08-15', '2019-08-16', '2019-08-19', '2019-08-20', '2019-08-21',
       '2019-08-22', '2019-08-23', '2019-08-27', '2019-08-28', '2019-08-29',
       '2019-08-30', '2019-09-02', '2019-09-03', '2019-09-04', '2019-09-05',
       '2019-09-06', '2019-09-09', '2019-09-10', '2019-09-11', '2019-09-12',
       '2019-09-13', '2019-09-16']

date_list_sc = ['2019-07-26', '2019-07-27', '2019-07-29', '2019-07-30', '2019-07-31',
       '2019-08-01', '2019-08-02', '2019-08-05', '2019-08-06', '2019-08-07',
       '2019-08-08', '2019-08-09', '2019-08-12', '2019-08-13', '2019-08-14',
       '2019-08-15', '2019-08-16', '2019-08-19', '2019-08-20', '2019-08-21',
       '2019-08-22', '2019-08-23', '2019-08-27', '2019-08-28', '2019-08-29',
       '2019-08-30', '2019-09-02', '2019-09-03', '2019-09-04', '2019-09-05',
       '2019-09-06', '2019-09-09', '2019-09-10', '2019-09-11', '2019-09-12',
       '2019-09-13', '2019-09-16']

aa08 = all_trials(pwm, 'AA08', date_list4)
df = aa08.copy()
y = df['Ct_wr']
df['Intercept'] = 1
# I am going to use all the columns for this optimization
x = df[['Ct_Stim', 'Pt_Hit', 'Pt_Stim', 'Pt_csr', 'Pt_wr', 'PPt_stim',
       'Lt_stim', 'Intercept']]

# Load in the SC dataframe
pickle_in = open("Rot3_data\\SC_full_df.pkl","rb")
Animal_SC = pickle.load(pickle_in)
animals = ['AA01', 'AA03', 'AA05', 'AA07', 'DO04', 'DO08', 'SC04', 'SC05',
           'VP01', 'VP07', 'VP08']  # AA03 and SC04 don't do any trials
sc = clean_up_sc(Animal_SC)


# ESTIMATING THE HYPERPARAMETER
# "The hyperparameter value (λ) was selected independently for each rat using evidence optimization,
# on the basis of fivefold cross-validation."

scaler = preprocessing.MinMaxScaler()  # from the scaler transformation the intercept turns to zero
# if does not make a difference whether it is included or not.
x = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

clf = LogisticRegressionCV(cv=5, random_state=0, fit_intercept=True)

# For our example what is baseline accuracy etc.
logreg = LogisticRegression(random_state=0, fit_intercept=True)
logreg.fit(x_train, y_train)  # look up what these values really mean
y_pred = logreg.predict(x_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # Accuracy: 0.6352395672333848
print("Precision:", metrics.precision_score(y_test, y_pred))
    # Precision: 0.628482972136223
print("Recall:", metrics.recall_score(y_test, y_pred))
    # Recall: 0.6363636363636364
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_heatmap(cnf_matrix)

# Normally people then use GridSearch to determine some parameters.
#Grid Search
from sklearn.model_selection import GridSearchCV
grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}
grid_clf_acc = GridSearchCV(logreg, param_grid = grid_values,scoring = 'recall')
grid_clf_acc.fit(x_train, y_train)
y_pred_acc = grid_clf_acc.predict(x_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_acc))
    # Accuracy: 0.633693972179289
print("Precision:", metrics.precision_score(y_test, y_pred_acc))
    # Precision: 0.625
print("Recall:", metrics.recall_score(y_test, y_pred_acc))
    # Recall: 0.6426332288401254
cnf_matrix2 = metrics.confusion_matrix(y_test, y_pred_acc)
cnf_heatmap(cnf_matrix2)
#
hmm = grid_clf_acc.best_params_
{'C': 0.09, 'penalty': 'l1'}

# Try using LOGREG CV
C=[0.001,.009,0.01,.09,1,5,10,25]  # same values as above
clf = LogisticRegressionCV(Cs=C, cv=5, random_state=0, fit_intercept=True)  #think you can add a max_iter of 200
clf.fit(x_train,y_train)
y_pred_cv = clf.predict(x_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_cv))
    # same as gridsearch
print("Precision:", metrics.precision_score(y_test, y_pred_cv))
    # same as grid search
print("Recall:", metrics.recall_score(y_test, y_pred_cv))
    # same as gridsearch
cnf_matrix3 = metrics.confusion_matrix(y_test, y_pred_cv)
cnf_heatmap(cnf_matrix3)
clf.C_  # gets you the C hyperparameter
clf.penalty  # gets you the penalty used.
# It just dosen't look like it makes much difference whether or not you try to optimize the hyperparameter.
# Get the same parameters from Gridsearch and logreg CV so that is good


# The interesting question then is can we use the estimated parameters to feed into logit function and see if it
# improves.
x = df[['Ct_Stim', 'Pt_Hit', 'Pt_Stim', 'Pt_csr', 'Pt_wr', 'PPt_stim',
       'Lt_stim']]
x_scaled = scaler.fit_transform(x)
x_df = pd.DataFrame(x_scaled, columns=x.columns, index=x.index)  # turn array back into df
x_df['intercept'] = 1
x = x_df
# don't think I need to split into test and train.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
logit_model = sm.Logit(y, x)
result = logit_model.fit_regularized(alpha=1, L1_wt=0.0)
#y_pred_logit = result.predict(x_test)
 # L2 regularization doesn't exist yet. Think the above values are as close as you get to l2 regularization.
print(result.summary2())
    # psuedo R: 0.076


alpha = 1/0.09 # inverse of hyperparameter derived above.
logit_model = sm.Logit(y, x)
result = logit_model.fit_regularized(alpha=alpha, L1_wt=0.0)  # this actually create a worse result because you get
# nan for intercept and lt_stim
# changing the l1_wr does not really have an effect
# changing alpha = 0.09 did not change r2, but increased iterations and changed some coefficients and
# p-values


# Try to use the GLM
gamma_model = sm.GLM(y,x, family=sm.families.Binomial())
gamma_results = gamma_model.fit()
print(gamma_results.summary2())
# this gives the same null and log-likelihoods. so can just calculate pseudoR from that
# I can't get the regularized version to work though, it outputs some weird shit

x = df[['Ct_Stim', 'Pt_Hit', 'Pt_Stim', 'Pt_csr', 'Pt_wr', 'PPt_stim',
       'Lt_stim']]
x = scaler.fit_transform(x)
# need to use the sm add constant thing
gamma_model = sm.GLM(y,x, family=sm.families.Binomial())
gamma_results = gamma_model.fit_constrained('x1 + x3 + x6 = 1')
print(gamma_results.summary2())
ll = gamma_results.llf
llnull = gamma_results.llnull
nobs = gamma_results.nobs
((ll-llnull)/nobs)/math.log(2)
pr2_dict = {}
model_dict = {'A: No history':
                      ['Ct_Stim', 'intercept'],
                  'B: Correct-side history':
                      ['Ct_Stim', 'Pt_csr', 'intercept'],
                  'C: Reward history':
                      ['Ct_Stim', 'Pt_Hit', 'intercept'],
                  'D: Correct-side/Action history':
                      ['Ct_Stim', 'Pt_csr', 'Pt_wr', 'intercept'],
                  'E: Correct-side + short-term sens. history':
                      ['Ct_Stim', 'Pt_Stim','Pt_csr', 'PPt_stim', 'intercept'],
                  'F: Correct-side + short and long term sens. history':
                      ['Ct_Stim', 'Pt_Stim', 'Pt_csr', 'PPt_stim', 'Lt_stim', 'intercept'],
                  'G: Reward + short and long term sens history':
                      ['Ct_Stim', 'Pt_Hit', 'Pt_Stim', 'PPt_stim', 'Lt_stim', 'intercept'],
                  'H: Correct-side + short and long term sens. history + Preseverance':
                      ['Ct_Stim', 'Pt_Stim', 'Pt_csr', 'Pt_wr', 'PPt_stim', 'Lt_stim', 'intercept'],
                  'I: Correct-side + long term sens. history':
                      ['Ct_Stim', 'Pt_csr', 'Lt_stim', 'intercept']
                  }
constrain_dict = {'A: No history':
                      [],
                  'B: Correct-side history':
                      [],
                  'C: Reward history':
                      [],
                  'D: Correct-side/Action history':
                      [],
                  'E: Correct-side + short-term sens. history':
                      ['x1 + x2 + x4 = 1'],
                  'F: Correct-side + short and long term sens. history':
                      ['x1 + x2 + x4 = 1'],
                  'G: Reward + short and long term sens history':
                      ['x1 + x3 + x4 = 1'],
                  'H: Correct-side + short and long term sens. history + Preseverance':
                      ['x1 + x2 + x5 = 1'],
                  'I: Correct-side + long term sens. history':
                      []
                  }

pr2_dict = {}
for model in model_dict:
    x = df[model_dict[model]]
    #x_df = x_df.drop(columns=['intercept'])  # for now lets just ignore intercept
    x = scaler.fit_transform(x)
    sm.add_constant(x)

    gamma_model = sm.GLM(y, x, family=sm.families.Binomial())
    try:
        gamma_results = gamma_model.fit_constrained(constrain_dict[model])
    except:
        gamma_results = gamma_model.fit()
    print(gamma_results.summary2())
    ll = gamma_results.llf
    llnull = gamma_results.llnull
    nobs = gamma_results.nobs
    r2 = ((ll - llnull) / nobs) / math.log(2)
    pr2_dict[str(model)] = r2 # at r2 value to dict

pr2_df = pd.DataFrame.from_dict(pr2_dict, orient='index')
pr2_df.plot.bar()
plt.xticks(rotation=10)

pr2_dict = {}
for model in model_dict:
    x = df[model_dict[model]]
    #x_df = x_df.drop(columns=['intercept'])  # for now lets just ignore intercept
    x = scaler.fit_transform(x)

    gamma_model = sm.GLM(y, x, family=sm.families.Binomial())
    gamma_results = gamma_model.fit()
    print(gamma_results.summary2())
    ll = gamma_results.llf
    llnull = gamma_results.llnull
    nobs = gamma_results.nobs
    r2 = ((ll - llnull) / nobs) / math.log(2)
    pr2_dict[str(model)] = r2  # at r2 value to dict

pr2_df = pd.DataFrame.from_dict(pr2_dict, orient='index')
pr2_df.plot.bar()
plt.xticks(rotation=10)


# DUMMY VARIABLES - checking the effect of dummy variables

# there is an option to use a proper label encoder.
# pwm example
aa08 = all_trials(pwm, 'AA08', date_list4, dummies=False)
aa08_dum = all_trials(pwm, 'AA08', date_list4, dummies=True)
notdum = aa08.copy()
dum = aa08_dum.copy()
x_dum = dum[['Pt_Hit', 'Pt_csr', 'Pt_wr', 'Lt_stim', 'Ct_Stim_1.0',
       'Ct_Stim_2.0', 'Ct_Stim_3.0', 'Ct_Stim_4.0', 'Ct_Stim_5.0',
       'Ct_Stim_6.0', 'Ct_Stim_7.0', 'Ct_Stim_8.0', 'Pt_Stim_1.0',
       'Pt_Stim_2.0', 'Pt_Stim_3.0', 'Pt_Stim_4.0', 'Pt_Stim_5.0',
       'Pt_Stim_6.0', 'Pt_Stim_7.0', 'Pt_Stim_8.0', 'PPt_stim_1.0',
       'PPt_stim_2.0', 'PPt_stim_3.0', 'PPt_stim_4.0', 'PPt_stim_5.0',
       'PPt_stim_6.0', 'PPt_stim_7.0', 'PPt_stim_8.0']]


# sc examples
do08 = all_trials(sc, 'DO08', date_list_sc, dummies=False)
do08_dum = all_trials(sc, 'DO08', date_list_sc, dummies=True)
notdum = do08.copy()
dum = do08_dum.copy()

do04 = all_trials(sc, 'DO04', date_list_sc, dummies=False)
do04_dum = all_trials(sc, 'DO04', date_list_sc, dummies=True)
notdum = do04.copy()
dum = do04_dum.copy()

x_dum = dum[['Pt_Hit', 'Pt_csr', 'Pt_wr', 'Lt_stim', 'Ct_Stim_1.0',
       'Ct_Stim_2.0', 'Ct_Stim_3.0', 'Ct_Stim_4.0', 'Ct_Stim_5.0',
       'Ct_Stim_6.0', 'Pt_Stim_1.0', 'Pt_Stim_2.0', 'Pt_Stim_3.0',
       'Pt_Stim_4.0', 'Pt_Stim_5.0', 'Pt_Stim_6.0', 'PPt_stim_1.0',
       'PPt_stim_2.0', 'PPt_stim_3.0', 'PPt_stim_4.0', 'PPt_stim_5.0',
       'PPt_stim_6.0']]  #there is different amounts of stimuli

#


# Rerun from here (same for pwm and sc)
y_notdum = notdum['Ct_wr']
x_notdum = notdum[['Ct_Stim', 'Pt_Hit', 'Pt_Stim', 'Pt_csr', 'Pt_wr', 'PPt_stim',
       'Lt_stim']]
y_dum = dum['Ct_wr']



x_dum = scaler.fit_transform(x_dum)
x_notdum = scaler.fit_transform(x_notdum)

x_train_dum, x_test_dum, y_train_dum, y_test_dum = train_test_split(x_dum, y_dum, test_size=0.20, random_state=0)
x_train_notdum, x_test_notdum, y_train_notdum, y_test_notdum = train_test_split(x_notdum, y_notdum,
                                                                                test_size=0.20, random_state=0)
# not dum
logreg.fit(x_train_notdum, y_train_notdum)  # look up what these values really mean
y_pred_notdum = logreg.predict(x_test_notdum)
print("Accuracy:", metrics.accuracy_score(y_test_notdum, y_pred_notdum))
    # Accuracy: 0.6352395672333848
print("Precision:", metrics.precision_score(y_test_notdum, y_pred_notdum))
    # Precision: 0.628482972136223
print("Recall:", metrics.recall_score(y_test_notdum, y_pred_notdum))
    # Recall: 0.6363636363636364
cnf_matrix_notdum = metrics.confusion_matrix(y_test_notdum, y_pred_notdum)
cnf_heatmap(cnf_matrix_notdum)
plt.title('Cnf_matrix, example: AA08, no dummy variables')


logit_roc_auc_notdum = roc_auc_score(y_test_notdum, y_pred_notdum)
fpr, tpr, thresholds = roc_curve(y_test_notdum, logreg.predict_proba(x_test_notdum)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc_notdum)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic: aa08, no dummies')
plt.legend(loc='lower right')
plt.savefig('Log_ROC')
plt.show()

# dum
logreg.fit(x_train_dum, y_train_dum)  # look up what these values really mean
y_pred_dum = logreg.predict(x_test_dum)
print("Accuracy:", metrics.accuracy_score(y_test_dum, y_pred_dum))
    # Accuracy: 0.6352395672333848
print("Precision:", metrics.precision_score(y_test_dum, y_pred_dum))
    # Precision: 0.628482972136223
print("Recall:", metrics.recall_score(y_test_dum, y_pred_dum))
    # Recall: 0.6363636363636364
cnf_matrix_dum = metrics.confusion_matrix(y_test_dum, y_pred_dum)
cnf_heatmap(cnf_matrix_dum)
plt.title('Cnf_matrix, example: AA08, dummy variables')

logit_roc_auc_dum = roc_auc_score(y_test_dum, y_pred_dum)
fpr, tpr, thresholds = roc_curve(y_test_dum, logreg.predict_proba(x_test_dum)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc_dum)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic: aa08, dummies')
plt.legend(loc='lower right')
plt.savefig('Log_ROC')
plt.show()


# try another example
#sc01 = all_trials(pwm, 'SC01', date_list4, dummies=False)
#sc01_dum = all_trials(pwm, 'SC01', date_list4, dummies=True)
#notdum = sc01.copy()
#dum = sc01_dum.copy()


## DUMMY VARIABLES + CONSTRAINTS
# Try to use the GLM
notdum = aa08.copy()
dum = aa08_dum.copy()


# Rerun from here
y_notdum = notdum['Ct_wr']
x_notdum = notdum[['Ct_Stim', 'Pt_Hit', 'Pt_Stim', 'Pt_csr', 'Pt_wr', 'PPt_stim',
       'Lt_stim']]

y_dum = dum['Ct_wr']
x_dum = dum[['Pt_Hit', 'Pt_csr', 'Pt_wr', 'Lt_stim', 'Ct_Stim_1.0',
       'Ct_Stim_2.0', 'Ct_Stim_3.0', 'Ct_Stim_4.0', 'Ct_Stim_5.0',
       'Ct_Stim_6.0', 'Ct_Stim_7.0', 'Ct_Stim_8.0', 'Pt_Stim_1.0',
       'Pt_Stim_2.0', 'Pt_Stim_3.0', 'Pt_Stim_4.0', 'Pt_Stim_5.0',
       'Pt_Stim_6.0', 'Pt_Stim_7.0', 'Pt_Stim_8.0', 'PPt_stim_1.0',
       'PPt_stim_2.0', 'PPt_stim_3.0', 'PPt_stim_4.0', 'PPt_stim_5.0',
       'PPt_stim_6.0', 'PPt_stim_7.0', 'PPt_stim_8.0']]

x_notdum = scaler.fit_transform(x_notdum)
gamma_model_notdum = sm.GLM(y_notdum,x_notdum, family=sm.families.Binomial())
gamma_results_notdum = gamma_model_notdum.fit()
print(gamma_results_notdum.summary2())
# this gives the same null and log-likelihoods. so can just calculate pseudoR from that
# I can't get the regularized version to work though, it outputs some weird shit



## running constraints with dummies
x_scaled = scaler.fit_transform(x_dum)
x_df = pd.DataFrame(x_scaled, columns=x_dum.columns, index=x_dum.index)  # turn array back into df
#x_df = x_df.drop(columns=['intercept'])  # remove old intercept it gets transformed to 0
x_df['intercept'] = 1  # add the intercept again
x_dum = x_df

gamma_model_dum = sm.GLM(y_dum ,x_dum, family=sm.families.Binomial())
weight_list = 'x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + ' \
              'x23 + x24 + x25 + x26 + x27 + x28 = 1'
gamma_results_dum = gamma_model_dum.fit_constrained(str(weight_list))
print(gamma_results_dum.summary2())
ll = gamma_results.llf
llnull = gamma_results.llnull
nobs = gamma_results.nobs
((ll-llnull)/nobs)/math.log(2)


# iterate throught the models
#df = aa08_dum.copy()
df = sc01_dum.copy()
df['intercept'] = 1
pr2_dict = {}
model_dict = {'A: No history':
                      ['Ct_Stim', 'intercept'],
                  'B: Correct-side history':
                      ['Ct_Stim', 'Pt_csr', 'intercept'],
                  'C: Reward history':
                      ['Ct_Stim', 'Pt_Hit', 'intercept'],
                  'D: Correct-side/Action history':
                      ['Ct_Stim', 'Pt_csr', 'Pt_wr', 'intercept'],
                  'E: Correct-side + short-term sens. history':
                      ['Ct_Stim', 'Pt_Stim','Pt_csr', 'PPt_stim', 'intercept'],
                  'F: Correct-side + short and long term sens. history':
                      ['Ct_Stim', 'Pt_Stim', 'Pt_csr', 'PPt_stim', 'Lt_stim', 'intercept'],
                  'G: Reward + short and long term sens history':
                      ['Ct_Stim', 'Pt_Hit', 'Pt_Stim', 'PPt_stim', 'Lt_stim', 'intercept'],
                  'H: Correct-side + short and long term sens. history + Preseverance':
                      ['Ct_Stim', 'Pt_Stim', 'Pt_csr', 'Pt_wr', 'PPt_stim', 'Lt_stim', 'intercept'],
                  'I: Correct-side + long term sens. history':
                      ['Ct_Stim', 'Pt_csr', 'Lt_stim', 'intercept']
                  }

sen_weights = 'Ct_Stim_1.0 + Ct_Stim_2.0 + Ct_Stim_3.0 + Ct_Stim_4.0 + Ct_Stim_5.0 + Ct_Stim_6.0 + Ct_Stim_7.0 ' \
              '+ Ct_Stim_8.0 + Pt_Stim_1.0+ Pt_Stim_2.0+ Pt_Stim_3.0+ Pt_Stim_4.0+ Pt_Stim_5.0+ Pt_Stim_6.0+ ' \
              'Pt_Stim_7.0+ Pt_Stim_8.0 + PPt_stim_1.0 + PPt_stim_2.0+ PPt_stim_3.0+ PPt_stim_4.0+ PPt_stim_5.0+ ' \
              'PPt_stim_6.0+ PPt_stim_7.0+ PPt_stim_8.0 = 1'
constrain_dict = {'A: No history':
                      ['Ct_Stim_1.0 + Ct_Stim_2.0 + Ct_Stim_3.0 + Ct_Stim_4.0 + Ct_Stim_5.0 + Ct_Stim_6.0 + Ct_Stim_7.0 '
                       '+ Ct_Stim_8.0 = 1'],
                  'B: Correct-side history':
                      ['Ct_Stim_1.0 + Ct_Stim_2.0 + Ct_Stim_3.0 + Ct_Stim_4.0 + Ct_Stim_5.0 + Ct_Stim_6.0 + Ct_Stim_7.0 '
                          '+ Ct_Stim_8.0 = 1'],
                  'C: Reward history':
                      ['Ct_Stim_1.0 + Ct_Stim_2.0 + Ct_Stim_3.0 + Ct_Stim_4.0 + Ct_Stim_5.0 + Ct_Stim_6.0 + Ct_Stim_7.0 '
                          '+ Ct_Stim_8.0 = 1'],
                  'D: Correct-side/Action history':
                      ['Ct_Stim_1.0 + Ct_Stim_2.0 + Ct_Stim_3.0 + Ct_Stim_4.0 + Ct_Stim_5.0 + Ct_Stim_6.0 + Ct_Stim_7.0 '
                          '+ Ct_Stim_8.0 = 1'],
                  'E: Correct-side + short-term sens. history':
                      [sen_weights],
                  'F: Correct-side + short and long term sens. history':
                      [sen_weights],
                  'G: Reward + short and long term sens history':
                      [sen_weights],
                  'H: Correct-side + short and long term sens. history + Preseverance':
                      [sen_weights],
                  'I: Correct-side + long term sens. history':
                      ['Ct_Stim_1.0 + Ct_Stim_2.0 + Ct_Stim_3.0 + Ct_Stim_4.0 + Ct_Stim_5.0 + Ct_Stim_6.0 + Ct_Stim_7.0 '
                          '+ Ct_Stim_8.0 = 1']
                  }





pr2_dict = {}
for model in model_dict:
    # first create the right x
    x_dummy_list = []
    x_list = model_dict[model].copy()
    # print(str(x_list))
    for column_name in model_dict[model]:
        # print(str(column_name))
        if column_name in dummy_list:
            x_list.remove(column_name)
            print('Removed, now' + str(x_list))
    x_without_dummies = df[x_list]
    x_dummy_list.append(x_without_dummies)
    for dummy in dummy_list:
        if dummy in model_dict[model]:
            x_dummy = df.filter(regex='^' + dummy, axis=1)
            x_dummy_list.append(x_dummy)
    x = pd.concat((x_dummy_list), axis=1)

    y_dum = df['Ct_wr']

    # then scale and give back names
    x_scaled = scaler.fit_transform(x)
    x_df = pd.DataFrame(x_scaled, columns=x.columns, index=x.index)  # turn array back into df
    # x_df = x_df.drop(columns=['intercept'])  # remove old intercept it gets transformed to 0
    x_df['intercept'] = 1  # add the intercept again
    x_dum = x_df

    gamma_model_dum = sm.GLM(y_dum, x_dum, family=sm.families.Binomial())
    #gamma_results = gamma_model_dum.fit_constrained(constrain_dict[model])
    gamma_results = gamma_model_dum.fit()
    print(gamma_results.summary2())
    ll = gamma_results.llf
    llnull = gamma_results.llnull
    nobs = gamma_results.nobs
    r2 = ((ll - llnull) / nobs) / math.log(2)
    pr2_dict[str(model)] = r2 # at r2 value to dict

pr2_df = pd.DataFrame.from_dict(pr2_dict, orient='index')
pr2_df.plot.bar()
plt.xticks(rotation=10)

# DUmmy and constraints for SC example
notdum = do04.copy()
dum = do04_dum.copy()

df = do04_dum.copy()
df['intercept'] = 1
pr2_dict = {}
model_dict = {'A: No history':
                      ['Ct_Stim', 'intercept'],
                  'B: Correct-side history':
                      ['Ct_Stim', 'Pt_csr', 'intercept'],
                  'C: Reward history':
                      ['Ct_Stim', 'Pt_Hit', 'intercept'],
                  'D: Correct-side/Action history':
                      ['Ct_Stim', 'Pt_csr', 'Pt_wr', 'intercept'],
                  'E: Correct-side + short-term sens. history':
                      ['Ct_Stim', 'Pt_Stim','Pt_csr', 'PPt_stim', 'intercept'],
                  'F: Correct-side + short and long term sens. history':
                      ['Ct_Stim', 'Pt_Stim', 'Pt_csr', 'PPt_stim', 'Lt_stim', 'intercept'],
                  'G: Reward + short and long term sens history':
                      ['Ct_Stim', 'Pt_Hit', 'Pt_Stim', 'PPt_stim', 'Lt_stim', 'intercept'],
                  'H: Correct-side + short and long term sens. history + Preseverance':
                      ['Ct_Stim', 'Pt_Stim', 'Pt_csr', 'Pt_wr', 'PPt_stim', 'Lt_stim', 'intercept'],
                  'I: Correct-side + long term sens. history':
                      ['Ct_Stim', 'Pt_csr', 'Lt_stim', 'intercept']
                  }

sen_weights = 'Ct_Stim_1.0 + Ct_Stim_2.0 + Ct_Stim_3.0 + Ct_Stim_4.0 + Ct_Stim_5.0 + Ct_Stim_6.0 + ' \
              'Pt_Stim_1.0+ Pt_Stim_2.0+ Pt_Stim_3.0+ Pt_Stim_4.0+ Pt_Stim_5.0+ Pt_Stim_6.0 ' \
              '+ PPt_stim_1.0 + PPt_stim_2.0+ PPt_stim_3.0+ PPt_stim_4.0+ PPt_stim_5.0+ ' \
              'PPt_stim_6.0 = 1'
constrain_dict = {'A: No history':
                      ['Ct_Stim_1.0 + Ct_Stim_2.0 + Ct_Stim_3.0 + Ct_Stim_4.0 + Ct_Stim_5.0 + Ct_Stim_6.0 = 1'],
                  'B: Correct-side history':
                      ['Ct_Stim_1.0 + Ct_Stim_2.0 + Ct_Stim_3.0 + Ct_Stim_4.0 + Ct_Stim_5.0 + Ct_Stim_6.0  = 1'],
                  'C: Reward history':
                      ['Ct_Stim_1.0 + Ct_Stim_2.0 + Ct_Stim_3.0 + Ct_Stim_4.0 + Ct_Stim_5.0 + Ct_Stim_6.0 = 1'],
                  'D: Correct-side/Action history':
                      ['Ct_Stim_1.0 + Ct_Stim_2.0 + Ct_Stim_3.0 + Ct_Stim_4.0 + Ct_Stim_5.0 + Ct_Stim_6.0 = 1'],
                  'E: Correct-side + short-term sens. history':
                      [sen_weights],
                  'F: Correct-side + short and long term sens. history':
                      [sen_weights],
                  'G: Reward + short and long term sens history':
                      [sen_weights],
                  'H: Correct-side + short and long term sens. history + Preseverance':
                      [sen_weights],
                  'I: Correct-side + long term sens. history':
                      ['Ct_Stim_1.0 + Ct_Stim_2.0 + Ct_Stim_3.0 + Ct_Stim_4.0 + Ct_Stim_5.0 + Ct_Stim_6.0  = 1']
                  }





pr2_dict = {}
for model in model_dict:
    # first create the right x
    x_dummy_list = []
    x_list = model_dict[model].copy()
    # print(str(x_list))
    for column_name in model_dict[model]:
        # print(str(column_name))
        if column_name in dummy_list:
            x_list.remove(column_name)
            print('Removed, now' + str(x_list))
    x_without_dummies = df[x_list]
    x_dummy_list.append(x_without_dummies)
    for dummy in dummy_list:
        if dummy in model_dict[model]:
            x_dummy = df.filter(regex='^' + dummy, axis=1)
            x_dummy_list.append(x_dummy)
    x = pd.concat((x_dummy_list), axis=1)

    y_dum = df['Ct_wr']

    # then scale and give back names
    x_scaled = scaler.fit_transform(x)
    x_df = pd.DataFrame(x_scaled, columns=x.columns, index=x.index)  # turn array back into df
    # x_df = x_df.drop(columns=['intercept'])  # remove old intercept it gets transformed to 0
    x_df['intercept'] = 1  # add the intercept again
    x_dum = x_df

    gamma_model_dum = sm.GLM(y_dum, x_dum, family=sm.families.Binomial())
    gamma_results = gamma_model_dum.fit_constrained(constrain_dict[model])
    #gamma_results = gamma_model_dum.fit()
    print(gamma_results.summary2())
    ll = gamma_results.llf
    llnull = gamma_results.llnull
    nobs = gamma_results.nobs
    r2 = ((ll - llnull) / nobs) / math.log(2)
    pr2_dict[str(model)] = r2 # at r2 value to dict

pr2_df = pd.DataFrame.from_dict(pr2_dict, orient='index')
pr2_df.plot.bar()
plt.xticks(rotation=10)

# Test different logit solvers
def logit_solvers(df, animal_id=None, avoid_nan=False, normalise=True, dummies=True):
    dict1 = cv_models_logit(df, plot=False, avoid_nan=avoid_nan, normalise=normalise, dummies=dummies, solver='newton')
    dict2 = cv_models_logit(df, plot=False, avoid_nan=avoid_nan, normalise=normalise, dummies=dummies, solver='bfgs')
    dict3 = cv_models_logit(df, plot=False, avoid_nan=avoid_nan, normalise=normalise, dummies=dummies, solver='lbfgs')
    dict4 = cv_models_logit(df, plot=False, avoid_nan=avoid_nan, normalise=normalise, dummies=dummies, solver='powell')
    dict5 = cv_models_logit(df, plot=False, avoid_nan=avoid_nan, normalise=normalise, dummies=dummies, solver='cg')
    dict6 = cv_models_logit(df, plot=False, avoid_nan=avoid_nan, normalise=normalise, dummies=dummies, solver='ncg')



    dicts = pd.DataFrame.from_dict([dict1, dict2, dict3, dict4, dict5, dict6])
    dicts.set_index([pd.Index(['newton', 'bfgs', 'lbfgs', 'powell', 'cg', 'ncg'])])
    dicts.plot.bar()

    plt.title('PseudoR2 for the different solvers for Logit()')
    if animal_id is not None:
        plt.title('PseudoR2 for the different Logistic functions for animal: ' + str(animal_id))
    plt.xticks(np.arange(3), ('newton', 'bfgs', 'lbfgs', 'powell', 'cg', 'ncg'), rotation=0)
    plt.xlabel('Function used: ', fontsize=14)
    plt.ylabel('PseudoR2', fontsize=14)
    return dicts






# when you do the one with constraints and yoou only have one sensosory variable it gets a little fucky. and they
# become negative.


#Dummy Classifier

#from sklearn.dummy import DummyClassifier
#dum = DummyClassifier(strategy= 'most_frequent').fit(x_train, y_train)
#y_pred = clf.predict(x_test)
#Distribution of y test
#print('y actual : \n' +  str(y_test.value_counts()))
#Distribution of y predicted
#print('y predicted : \n' + str(pd.Series(y_pred).value_counts()))