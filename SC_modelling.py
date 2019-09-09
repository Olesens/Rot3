import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from SC_plotting import clean_up_df, cal_prob

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

pickle_in = open("Rot3_data\\SC_full_df.pkl","rb")
Animal_df = pickle.load(pickle_in)
sc = clean_up_df(Animal_df, ['SC04'])  # create the cleaned up dataframe
date_list = sc.index.values
# do not want to include '2019-08-21' just fyi, should remove that (data is weird).

date = '2019-08-23'
# create trial data set


def trial_df(df, date, ses_no = 1):
    session = df.loc[date]
    # make dataframe of raw trials
    history = cal_prob(df, 'SC04', date, ret_hist=True)

    # if it was not a tm or vio add it to dataset
    trial_index = 1  # we start from the second trial because the first trial dosen't have a previous trial
    trial_dict = {}

    # determine if animal went right or left
    history['went_right'] = 0  # add new column filled with zeros
    mask = (history['hits'] == 1) & (history['side'] == 114)  # correctly went right to stim (1,2 or 3)
    history['went_right'] = history['went_right'].mask(mask, 1)
    mask = (history['hits'] == 0) & (history['side'] == 108)  # incorrectly went right to stim (4,5 or 6)
    history['went_right'] = history['went_right'].mask(mask, 1)

    history['right_side_correct'] = 0
    mask = (history['side'] == 114)  # right side is correct choice
    history['right_side_correct'] = history['right_side_correct'].mask(mask, 1)

    #return history

    # trial[0] is side, trial[1] is stim, trial[2] is hit, trial[3] is went_right
    for tri_number in history.index._values[1:]:
        prev_trial = history.ix[(tri_number - 1)]
        trial = history.ix[tri_number]
        session = 'S' + str(ses_no) + '_ '
        if np.isnan(trial[2]) == False:
            #tri_index_list.append(trial_index)
            if np.isnan(prev_trial[2]) == False:
                key = str(session + str(trial_index))
                trial_dict[key] = [trial[3], trial[1], prev_trial[2], prev_trial[1], prev_trial[4]]
        trial_index += 1

    #probably have to figure out how to deal with if the previous trial was a violation or timeout
        # make the nan values
        # have removed them for now

    # create dict
    data_df = pd.DataFrame.from_dict(trial_dict,
                                     orient='index',
                                     columns=['Ct_wr', 'Ct_Stim', 'Pt_Hit', 'Pt_Stim', 'Pt_csr'])  #csr = correct side right
    return data_df



def all_trials(df, date_list):
    session_list = []
    session_index = 1
    for session in date_list:
        print('date is: ' +  str(session))
        try:
            trials = trial_df(df, session, ses_no = session_index)
            session_list.append(trials)
            session_index += 1
            print('success')
        except:
            print('could not extract trials for date: ' + str(session) + ' continuing to next date')
    session_df = pd.concat(session_list)
    return session_df


def drop_x(df, ct_stim=False, pt_hit=False, pt_stim = False, pt_csr=False):
    x = df[['Ct_Stim', 'Pt_Hit', 'Pt_Stim', 'Pt_csr']]
    if ct_stim is True:
        df = df.drop(columns=['Ct_Stim'])
        x = df[['Pt_Hit', 'Pt_Stim']]
    if pt_stim is True:
        df = df.drop(columns=['Pt_Stim'])
        x = df[['Ct_Stim', 'Pt_Hit']]
    if pt_hit is True:
        df = df.drop(columns=['Pt_Hit'])
        x = df[['Ct_Stim', 'Pt_Stim']]
    if pt_csr is True:
        df = df.drop(columns=['Pt_csr'])
        x = df[['Ct_Stim', 'Pt_Stim', 'Pt_Hit']]
    #else:
    #    x = df[['Ct_Stim', 'Pt_Hit', 'Pt_Stim']]

    return x


def train_data(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)  # look up what these values really mean
    y_pred = logreg.predict(x_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    return cnf_matrix


def cnf_heatmap(cnf_matrix):
    fig, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="viridis", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    return None


def test_model(df, ct_stim = False, pt_hit = False, pt_stim = False, pt_csr=False):
    params = [ct_stim, pt_hit, pt_stim, pt_csr]
    y = df['Ct_wr']
    x = drop_x(df, ct_stim=params[0], pt_hit=params[1], pt_stim= params[2], pt_csr=params[3])
    cnf = train_data(x,y)
    cnf_heatmap((cnf))
    return cnf


# df['Ct_wr'].value_counts()
# session_df['Ct_wr'].value_counts()
# sns.countplot(x='Ct_wr', data=session_df, palette='hls')
# df.isnull().sum()
date = '2019-08-06'
df = trial_df(sc, date)
y = df['Ct_wr']
session_df = all_trials(sc, date_list)