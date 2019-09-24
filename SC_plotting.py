import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy as sy


# DEFINED ERROR CLASSES
class Error(Exception):
    # Error is derived class for Exception, but
    # Base class for exceptions in this module
    pass


class StageError(Error):
    """raised selecting for specific stage which does not correspond to given session"""
    pass


class PCurveError(Error):
    pass


class SessionCheckError(Error):
    pass


# CLEAN UP AND CREATE NEW DF FROM RAW
def clean_up_df(df, animallist=[], index=True, multiindex=True, fixstages=True, duplicates=True, print_dup=False):
    """
    Takes in dataframe and makes alterations, fixing stage issues optimizing index for plotting

    :param df: dataframe to take in and clean up
    :param animallist: optional list of string names for animals to select for in cleaned up dataframe
    :param index: (bool), if True sorts index (date)
    :param multiindex: (bool), if True names index multi-index
    :param fixstages: (bool), if True applies mask to change stage number to correct.
    :param duplicates: (bool), if True removes duplicates
    :param print_dup: (bool), if True print duplicates being removed
    :return:
    """
    if index is True:
        df = df.sort_index()  # sort according to index, this should sort according to date
    if multiindex is True:
        df = df.rename_axis(['animal', 'date2'])  # add labels for Multi-index
    if fixstages is True:
        # add mask to label what task the rats did based on their settings file.
        df['task'] = None  # make the baseline Nan values
        mask = df.file.str.contains('SoundCat')
        df['task'] = df['task'].mask(mask, 'SC')
        mask = df.file.str.contains('DelayComp')
        df['task'] = df['task'].mask(mask, 'PWM')

        # For SC task, turn stage 1 to 0 if reward type is always
        try:
            mask = (df['stage'] == 1) & (df['reward_type'] == 'Always') & (df['task'] == 'SC')
            df['stage'] = df['stage'].mask(mask, 0)  # this is for SC task
        except:
            print('Could not apply SC stage mask to dataframe')

        # For PWM task, turn stage 1 to 2 if A2_time is above 0 and stage 2 to 3 if reward_type is NoReward
        try:
            mask = (df['stage'] == 1) & (df['A2_time'] > 0) & (df['task'] == 'PWM')
            df['stage'] = df['stage'].mask(mask, 2)
            mask = (df['stage'] == 2) & (df['reward_type'] == 'NoReward') & (df['task'] == 'PWM')
            df['stage'] = df['stage'].mask(mask, 3)
        except:
            print('Could not apply PWM stage mask to dataframe')

    if animallist:
        df = df.loc[animallist]  # include only specifically given animals

    if duplicates is True: # remove duplicates
        df = df.swaplevel('animal', 'date2')  # can't remember is this is necessary for duplicate removal
        df = df.reset_index()  # In order to delete duplicates
        dup_r = df[df.duplicated(['date2', 'animal'])]  # this is actually unnecessary
        if print_dup is True:
            print('duplicates are:', dup_r)
        df = df.drop_duplicates(['date2', 'animal'])
        dup_r = df[df.duplicated(['date2', 'animal'])]  # Run a little double check
        if print_dup is True:
            print('After removal duplicates are:', dup_r)
        # Put the dataframe nicely together again
        df = df.set_index(['date2', 'animal'])
        df = df.sort_index()
        df = df.unstack()

    return df


# PLOTTING TRIALS, AND WITHIN SESSION TRIAL TYPES
def plot_cp(df, stage='All'):
    if stage != 'All':  # if stage no provided
        mask = df['stage'] == stage
        df = df[mask]
        df_cp = df['total_CP']
        col_list = df_cp.columns.values  # get list to use for labelling
        cp_plot = plt.plot(df_cp,
                           marker='o',
                           linewidth=1.0,
                           markersize=2.5)
        plt.xticks(rotation=75)
        plt.legend(col_list)

    else:
        print('all stages included')
        df_cp = df['total_CP']
        cp_plot = df_cp.plot(marker='o',
                             linewidth=1.0,
                             markersize=2.5,
                             cmap=plt.cm.RdPu)
        plt.xticks(rotation=0,
                   fontsize='medium')
    plt.ylabel('Total CP')
    plt.xlabel('Date')
    plt.title('Total CP over time for each PWM animal')
    return cp_plot

def plot_trials(df, stage='All', p_type='all_trials'):

    p_types = ['all_trials', 'vio_only', 'tm_only', 'left', 'right']
    if p_type not in p_types:
        raise ValueError("Invalid p_type. Expected one of: %s" % p_type)

    # Filter according to stages
    if stage != 'All':
        mask = df['stage'] == stage
        df = df[mask]
    else:
        print('all stages included')

    # Remove violations and trials from done trials
    df['done_trials'] = df['done_trials'] - ((df['violations'] + df['timeouts']) * df['done_trials'])
    if p_type is 'left' or 'right':  # might not need to be if statement, just do no matter what
        # this should only occur inside the function so change be global just for the division.
        # how would it effect the other calculations
        mask = df['done_trials'] == 0
        df['done_trials'] = df['done_trials'].mask(mask, 1)  # replace all 0s with 1 for division further down

    p_types_dict = {'all_trials': df['done_trials'],
                    'vio_only': df['violations'] * 100,
                    'tm_only': df['timeouts']*100,
                    'left': df['left_trials'] / df['done_trials']*100,
                    'right': df['right_trials'] / df['done_trials']*100}

    p_types_title = {'all_trials': ' done trials for PWM animals minus violations and timeouts ',
                     'vio_only': ': % Violation trials for PWM animals',
                     'tm_only': ': % Timeout trials for PWM animals',
                     'left': ': % Left trials for PWM animals',
                     'right': ': % Right trials for PWM animals'}

    df_trials = p_types_dict[p_type]
    col_list = df_trials.columns.values  # get list to use for labelling
    tri_plot = plt.plot(df_trials,
                        marker='o',
                        linewidth=1.0,
                        markersize=2.5)

    if p_type is not 'all_trials':
        plt.ylim([-5, 105])
        plt.ylabel('Percentage of trials')
    else:
        plt.ylabel('Trials')
    # Settings for all plots
    plt.xticks(rotation=75)
    plt.legend(col_list)
    plt.xlabel('Date')
    plt.title(stage, p_types_title[p_type])

    # how to change line colors by making a loop
    return tri_plot


def boxplot_animal(df, animal_id, stage='All', percentage = False):
    # maybe only for stage 2 and 3 really?
    # select for the single animal, thus animal = level, axis=1 because it is on the column level
    # not on the index level, .xs allows selection at specific often lower level


    single_animal = df.xs(animal_id, level='animal', axis=1).copy()  # single animal dataframe
    vio = vio = single_animal['violations']*100  # this is in percentage
    tm = single_animal['timeouts']*100  # this is in percentage

    mask = single_animal['done_trials'] == 0
    single_animal['done_trials'] = single_animal['done_trials'].mask(mask, 1)
    right = (single_animal['right_trials'] / single_animal['done_trials']) * 100
    left = (single_animal['left_trials'] / single_animal['done_trials']) * 100

    height = np.add(left, right).tolist()
    height2 = np.add(height, vio).tolist()
    barWidth = 1

    # need to have some exception cause I am plotting these on top of each other but some of them have no vio and left right trials
    # have changed the order from the other plotting
    # still gives errors...
    boxplot = plt.bar(single_animal.index, right, color='#045a8d', edgecolor='black', width=barWidth)
    plt.bar(single_animal.index, left, bottom=right, color='#016c59', edgecolor='black', width=barWidth)
    plt.bar(single_animal.index, vio, bottom=height, color='#810f7c', edgecolor='black', width=barWidth)
    plt.bar(single_animal.index, tm, bottom=height2, color='#636363', edgecolor='black', width=barWidth)



    plt.xticks(rotation=75)
    plt.xlabel('Date')
    plt.ylabel('Percentage')
    plt.ylim([0, 125])
    legend = ['Right trials', 'Left trials', 'Violations', 'Timeouts']
    plt.legend(legend, fontsize=14)
    plt.title('Categorization of trials for: ' + animal_id, fontsize=18)

    # Figure out how to add legend

    return boxplot


# RUN ALL PLOTS AND UPDATE THEM IN FOLDERS
def run_all_plots():

    # Done trials for stage 0
    st0_trials = plot_trials(pwm, stage=0, p_type='all_trials')
    plt.savefig('Rot3_data\\st0_trials.png', bbox_inches='tight')
    plt.close()

    # Done trials for stage 1
    st1_trials = plot_trials(pwm, stage=1, p_type='all_trials')
    plt.savefig('Rot3_data\\st1_trials.png', bbox_inches='tight')
    plt.close()

    # Done trials for stage 2
    st2_trials = plot_trials(pwm, stage=2, p_type='all_trials')
    plt.savefig('Rot3_data\\st2_trials.png', bbox_inches='tight')
    plt.close()

    # Violation trials for stage 1
    st1_vio_trials = plot_trials(pwm, stage=1, p_type='vio_only')
    plt.savefig('Rot3_data\\st1_vio_trials.png', bbox_inches='tight')
    plt.close()

    # Violation trials for stage 2
    st2_vio_trials = plot_trials(pwm, stage=2, p_type='vio_only')
    plt.savefig('Rot3_data\\st2_vio_trials.png', bbox_inches='tight')
    plt.close()

    # Timeout trials for stage 1
    st1_tm_trials = plot_trials(pwm, stage=1, p_type='tm_only')
    plt.savefig('Rot3_data\\st1_tm_trials.png', bbox_inches='tight')
    plt.close()

    # Timeout trials for stage 2
    st2_tm_trials = plot_trials(pwm, stage=2, p_type='tm_only')
    plt.savefig('Rot3_data\\st2_tm_trials.png', bbox_inches='tight')
    plt.close()

    # Left trials for stage 1
    st1_left_trials = plot_trials(pwm, stage=1, p_type='left')
    plt.savefig('Rot3_data\\st1_left_trials.png', bbox_inches='tight')
    plt.close()

    # Left trials for stage 2
    st2_left_trials = plot_trials(pwm, stage=2, p_type='left')
    plt.savefig('Rot3_data\\st2_left_trials.png', bbox_inches='tight')
    plt.close()

    # Right trials for stage 1
    st1_right_trials = plot_trials(pwm, stage=1, p_type='right')
    plt.savefig('Rot3_data\\st1_right_trials.png', bbox_inches='tight')
    plt.close()

    # Right trials for stage 2
    st2_right_trials = plot_trials(pwm, stage=2, p_type='right')
    plt.savefig('Rot3_data\\st2_right_trials.png', bbox_inches='tight')
    plt.close()


def run_box_plots():
    for animal in animals:
        fig_name = 'sc_' + animal + '_boxp'
        plot = boxplot_animal(sc, animal)
        plt.show()
        plt.savefig('Rot3_data\\SoundCat\\Boxplots\\' + fig_name + '.png', bbox_inches='tight')
        plt.close()


def run_pcurves():
    for animal in animals:
        fig_name = 'sc_' + animal + '_pcurve'
        plot = animal_pcurve(sc, animal, date_list, stage=2)
        plt.show()
        plt.savefig('Rot3_data\\SoundCat\\' + fig_name + '.png', bbox_inches='tight')
        plt.close()


def run_param_plots():
    param_list = ['A', 'B', 'C', 'D']
    for parameter in param_list:
        for animal in animals:
            fig_name = 'sc_' + animal + '_param_' + parameter
            plot = param_days(sc, animal, date_list, stage=2, param=parameter)
            plt.show()
            plt.savefig('Rot3_data\\SoundCat\\4PL param plots\\' + fig_name + '.png', bbox_inches='tight')
            plt.close()


def run_slope_plots():
    for animal in animals:
        fig_name = 'sc_' + animal + '_slopes'
        plot = param_days(sc, animal, date_list, stage=2, param='B')
        plt.show()
        plt.savefig('Rot3_data\\SoundCat\\' + fig_name + '.png', bbox_inches='tight')
        plt.close()


# FUNCTION TO CHECK SESSION
def check(df, animal_id, date):
    # could maybe modify this a little to be shorter, but it works atm
    single_animal = df.xs(animal_id, level='animal', axis=1).copy()
    session = single_animal.loc[date]
    stim = session['history_stim']
    try:
        stim = pd.DataFrame(stim, columns=['stim']).T

    except ValueError:
        #print('Could not extract stimulus history, check that specific animal and session exists')
        return False
    else:
        return True


# PLOTTING PROBABILITIES AND FITTING P-CURVES
def cal_prob(df, animal_id, date, ret_hist=False):
    # can definitely bring this baby down in size
    single_animal = df.xs(animal_id, level='animal', axis=1).copy()
    session = single_animal.loc[date]
    stim = session['history_stim']
    side = session['history_side']
    hits = session['history_hits']

    if check(df, animal_id, date) is False:
        return None

    stim = pd.DataFrame(stim, columns=['stim']).T
    side = pd.DataFrame(side, columns=['side']).T
    hits = pd.DataFrame(hits, columns=['hits']).T
    dfs = [side, stim, hits]
    history = pd.concat(dfs)
    history = history.T

    if ret_hist is True:
        return history
    # stage condition
    if 6 in history['stim'].values:
        # print('detected 6 stimuli, calculating probabilities..')
        # amount of times the rat went right when stim 1 was on (which is the correct choice)
        history['stim1_wr'] = (history['side'] == 114) & (history['stim'] == 1) & (history['hits'] == 1)
        history['stim2_wr'] = (history['side'] == 114) & (history['stim'] == 2) & (history['hits'] == 1)
        history['stim3_wr'] = (history['side'] == 114) & (history['stim'] == 3) & (history['hits'] == 1)
        # for stim 3 and 4 the rat went right, which would be incorrect for those stimuli
        history['stim4_wr'] = (history['side'] == 108) & (history['stim'] == 4) & (history['hits'] == 0)
        history['stim5_wr'] = (history['side'] == 108) & (history['stim'] == 5) & (history['hits'] == 0)
        history['stim6_wr'] = (history['side'] == 108) & (history['stim'] == 6) & (history['hits'] == 0)
        # sum the above into one variable for each stimuli
        sum_stim1R = history['stim1_wr'].sum()
        sum_stim2R = history['stim2_wr'].sum()
        sum_stim3R = history['stim3_wr'].sum()
        sum_stim4R = history['stim4_wr'].sum()
        sum_stim5R = history['stim5_wr'].sum()
        sum_stim6R = history['stim6_wr'].sum()

        # extract all done trials minus violations and timeouts and calculate percentage
        done_trials = session.loc['done_trials'] - (
                    (session.loc['violations'] + session.loc['timeouts']) * session.loc['done_trials'])
        # could do the above outside of the condition
        stim1_percR = sum_stim1R / done_trials * 100
        stim2_percR = sum_stim2R / done_trials * 100
        stim3_percR = sum_stim3R / done_trials * 100
        stim4_percR = sum_stim4R / done_trials * 100
        stim5_percR = sum_stim5R / done_trials * 100
        stim6_percR = sum_stim6R / done_trials * 100

        # just doing it for left as well to check if it adds up to 100% which it should when timeouts and violations
        # are not included

        # amount of times the rat went left when stim 1,2,3
        history['stim1_wl'] = (history['side'] == 114) & (history['stim'] == 1) & (history['hits'] == 0)
        history['stim2_wl'] = (history['side'] == 114) & (history['stim'] == 2) & (history['hits'] == 0)
        history['stim3_wl'] = (history['side'] == 114) & (history['stim'] == 3) & (history['hits'] == 0)
        # for stim 4,5,6 the rat went left
        history['stim4_wl'] = (history['side'] == 108) & (history['stim'] == 4) & (history['hits'] == 1)
        history['stim5_wl'] = (history['side'] == 108) & (history['stim'] == 5) & (history['hits'] == 1)
        history['stim6_wl'] = (history['side'] == 108) & (history['stim'] == 6) & (history['hits'] == 1)

        # sum the above into one variable for each stimuli
        sum_stim1L = history['stim1_wl'].sum()
        sum_stim2L = history['stim2_wl'].sum()
        sum_stim3L = history['stim3_wl'].sum()
        sum_stim4L = history['stim4_wl'].sum()
        sum_stim5L = history['stim5_wl'].sum()
        sum_stim6L = history['stim6_wl'].sum()
        stim1_percL = sum_stim1L / done_trials * 100
        stim2_percL = sum_stim2L / done_trials * 100
        stim3_percL = sum_stim3L / done_trials * 100
        stim4_percL = sum_stim4L / done_trials * 100
        stim5_percL = sum_stim5L / done_trials * 100
        stim6_percL = sum_stim6L / done_trials * 100

        # calculate probability
        # sum the left and right choices for each given stimulus
        stim1_sum = sum_stim1R + sum_stim1L
        stim2_sum = sum_stim2R + sum_stim2L
        stim3_sum = sum_stim3R + sum_stim3L
        stim4_sum = sum_stim4R + sum_stim4L
        stim5_sum = sum_stim5R + sum_stim5L
        stim6_sum = sum_stim6R + sum_stim6L

        # calculate the percentage of right choices for each given stimulus
        s1_right_prob = sum_stim1R / stim1_sum * 100
        s2_right_prob = sum_stim2R / stim2_sum * 100
        s3_right_prob = sum_stim3R / stim3_sum * 100
        s4_right_prob = sum_stim4R / stim4_sum * 100
        s5_right_prob = sum_stim5R / stim5_sum * 100
        s6_right_prob = sum_stim6R / stim6_sum * 100

        # create a dict with all variables to include in dataframe
        dict = {'Sum right choices': [sum_stim1R, sum_stim2R, sum_stim3R, sum_stim4R, sum_stim5R, sum_stim6R],
                'Perc_R of all done': [stim1_percR, stim2_percR, stim3_percR, stim4_percR, stim5_percR, stim6_percR],
                'Sum left choices': [sum_stim1L, sum_stim2L, sum_stim3L, sum_stim4L, sum_stim5L, sum_stim6L],
                'Perc_L of all done': [stim1_percL, stim2_percL, stim3_percL, stim4_percL, stim5_percL, stim6_percL],
                'Stimulus trials sum': [stim1_sum, stim2_sum, stim3_sum, stim4_sum, stim5_sum, stim6_sum],
                'Right_prob': [s1_right_prob, s2_right_prob, s3_right_prob, s4_right_prob, s5_right_prob, s6_right_prob]
                }
        stim_df = pd.DataFrame(dict, index=['stim1', 'stim2', 'stim3', 'stim4', 'stim5', 'stim6'])
        return stim_df

    if 6 not in history['stim'].values:
        # print('detected 4 stimuli, calculating probabilities...')
        history['stim1_wr'] = (history['side'] == 114) & (history['stim'] == 1) & (history['hits'] == 1)
        history['stim2_wr'] = (history['side'] == 114) & (history['stim'] == 2) & (history['hits'] == 1)
        # for stim 3 and 4 the rat went right, which would be incorrect for those stimuli
        history['stim3_wr'] = (history['side'] == 108) & (history['stim'] == 3) & (history['hits'] == 0)
        history['stim4_wr'] = (history['side'] == 108) & (history['stim'] == 4) & (history['hits'] == 0)
        # sum the above into one variable for each stimuli
        sum_stim1R = history['stim1_wr'].sum()
        sum_stim2R = history['stim2_wr'].sum()
        sum_stim3R = history['stim3_wr'].sum()
        sum_stim4R = history['stim4_wr'].sum()

        # extract all done trials minus violations and timeouts and calculate percentage
        done_trials = session.loc['done_trials'] - ((session.loc['violations'] + session.loc['timeouts']) * session.loc['done_trials'])
        stim1_percR = sum_stim1R/done_trials*100
        stim2_percR = sum_stim2R/done_trials*100
        stim3_percR = sum_stim3R/done_trials*100
        stim4_percR = sum_stim4R/done_trials*100

        # just doing it for left as well to check if it adds up to 100% which it should when timeouts and violations
        # are not included

        # amount of times the rat went left when stim 1
        history['stim1_wrL'] = (history['side'] == 114) & (history['stim'] == 1) & (history['hits'] == 0)
        history['stim2_wrL'] = (history['side'] == 114) & (history['stim'] == 2) & (history['hits'] == 0)
        # for stim 3 and 4 the rat went left
        history['stim3_wrL'] = (history['side'] == 108) & (history['stim'] == 3) & (history['hits'] == 1)
        history['stim4_wrL'] = (history['side'] == 108) & (history['stim'] == 4) & (history['hits'] == 1)
        # sum the above into one variable for each stimuli
        sum_stim1L = history['stim1_wrL'].sum()
        sum_stim2L = history['stim2_wrL'].sum()
        sum_stim3L = history['stim3_wrL'].sum()
        sum_stim4L = history['stim4_wrL'].sum()
        stim1_percL = sum_stim1L / done_trials * 100
        stim2_percL = sum_stim2L / done_trials * 100
        stim3_percL = sum_stim3L / done_trials * 100
        stim4_percL = sum_stim4L / done_trials * 100

        # calculate probability
        # sum the left and right choices for each given stimulus
        stim1_sum = sum_stim1R + sum_stim1L
        stim2_sum = sum_stim2R + sum_stim2L
        stim3_sum = sum_stim3R + sum_stim3L
        stim4_sum = sum_stim4R + sum_stim4L
        # calculate the percentage of right choices for each given stimulus
        s1_right_prob = sum_stim1R / stim1_sum *100
        s2_right_prob = sum_stim2R / stim2_sum * 100
        s3_right_prob = sum_stim3R / stim3_sum * 100
        s4_right_prob = sum_stim4R / stim4_sum * 100

        # create a dict with all variables to include in dataframe
        dict = {'Sum right choices': [sum_stim1R, sum_stim2R, sum_stim3R, sum_stim4R],
                'Perc_R of all done': [stim1_percR, stim2_percR, stim3_percR, stim4_percR],
                'Sum left choices': [sum_stim1L, sum_stim2L, sum_stim3L, sum_stim4L],
                'Perc_L of all done': [stim1_percL, stim2_percL, stim3_percL, stim4_percL],
                'Stimulus trials sum': [stim1_sum, stim2_sum, stim3_sum, stim4_sum],
                'Right_prob': [s1_right_prob, s2_right_prob, s3_right_prob, s4_right_prob]
                }
        stim_df = pd.DataFrame(dict, index=['stim1', 'stim2', 'stim3', 'stim4'])
        return stim_df


def pf(x, A, B, C, D):  # psychometric function
    return D + A / (1 + np.exp((-(x-C))/B)) # athena function
    #return D + (A - D) / (1.0 + ((x / C) ** (B)))  # matlab function
    #return 1. / (1 + np.exp(- (x-A)/B))  #original function used, not enough parameters


def plot_pcurve(big_df, animal_id, date, invert=False, col1='#810f7c', col2='#045a8d',
                stage='ALL', label_type='date', ret_param=False):

    if check(big_df, animal_id, date) is False:  # run check to see if there is data for the session
        raise SessionCheckError

    df = cal_prob(big_df, animal_id, date)
    right_prob = df['Right_prob'].sort_index(ascending=False)
    right_prob = (right_prob/100).values

    if len(right_prob) == 6:  # make length of stim equal to length of right_prob
        stim = np.array([0, 1, 2, 3, 4, 5])  # arbitrary x-axis needs to be same for plotting and p-fit
        stim_list = ('stim 6', 'stim 5', 'stim 4', 'stim 3', 'stim 2', 'stim 1')
        stim_no = 6
        stim2 = np.arange(0, 5, 0.01)
        if stage == 1:  # if only want stage 1 then abort
            print('Stage 2 detected, aborted plotting')
            raise StageError

    elif len(right_prob) == 4:
        stim = np.array([0, 1, 2, 3])
        stim_list = ('stim 4', 'stim 3', 'stim 2', 'stim 1')
        stim_no = 4
        stim2 = np.arange(0, 3, 0.01)
        if stage == 2:  # if only want stage 2 then abort
            print('Stage 1 detected, aborted plotting')
            raise StageError

    try:  # check that you can fit data with p-curve
        par, mcov = curve_fit(pf, stim, right_prob)  # fit p curve to data
        if ret_param is True:  # if condition true just return the estimated parameters
            return par
        plt.plot(stim2, pf(stim2, par[0], par[1], par[2], par[3]), color=col2)  # plot the p-curve
    except:
        raise PCurveError

    if label_type== 'date':
           label = date
    elif label_type== 'animal':
           label = animal_id

    pcurve = plt.plot(stim, right_prob, marker='o', markersize=8, color=col1, linestyle='', label=label)
    if invert is True: # this might be unneccesary now
        ax = plt.gca()  # invert x axis to classic direction of curve
        ax.invert_xaxis()  # invert x axis to classic direction of curve
    plt.ylabel('Probability of Right choice', fontsize=12)
    plt.xlabel('Stimulus', fontsize=12)
    plt.xticks(np.arange(stim_no), stim_list, fontsize=9, rotation=0)
    plt.title('pCurve for '+animal_id + ' on ' + date, fontsize=14)

    return pcurve


def day_pcurve(big_df, animal_list, date, stage='ALL'):
    # could include a condition to exclude animals if they do trials below a certain number
    colorlist = ['#82dae0', '#fff0a5', '#b0e5ca', '#e5b0b1', '#b7adc7']
    col_index = 1
    pic = plot_pcurve(big_df, animal_list[0], date, col1=colorlist[0], col2=colorlist[0], stage=stage,
                      label_type='animal')
    for animal in animal_list[1:]:
        if col_index == len(colorlist):
            col_index = 0
        col1 = colorlist[col_index]
        try:
            plot_pcurve(big_df, animal, date, invert=False, col1=col1, col2=col1, label_type='animal')
        except SessionCheckError:
            print('SessionCheckError for:  ' + animal + ' ...continuing to next animal...')
        except StageError:
            print('StageError for:  ' + animal + ' ...continuing to next animal...')
        except PCurveError:
            print('Failed to fit p-curve for:  ' + animal + ' ...continuing to next animal...')
        col_index += 1
    plt.title('Pcurves for animals on: ' + date)
    plt.legend()
    return pic


def animal_pcurve(big_df, animal_id, date_list, stage='ALL'):
    # need to fix this so different amounts of stimuli are allowed
    colorlist = ['#BCBD52', '#94B85A', '#6FAF67', '#4FA575', '#349981', '#2A8C88', '#337D89', '#436D84', '#515E79',
                 '#5A4E6A', '#5D4057', '#5A3343', '#522930', '#462120']
    colorlist2= ['#42201F', '#4D282F', '#533242', '#553E54', '#504C65', '#455B72', '#376A79', '#2D787A', '#308575'
                , '#44916B', '#619B5F', '#83A353', '#A8A84B', '#CEAB4D', '#F4AB5A']
    col_index = 1
    date_index = 0
    date = date_list[date_index]

    while check(sc, animal_id, date) is False:  # continue to check for data until first successful day
        print('no session date for: ' + date + '   ...proceeding to next date in list')
        date_index += 1
        date = date_list[date_index]
    print('first successful date is: ' + date)


    try:  # problem with this is if its stage 1 the the pic is not created..
        pic = plot_pcurve(big_df, animal_id, date, col1=colorlist[0], col2=colorlist[0], stage=stage) # add label for date
    except StageError:
        col_index=0
    # will get error now, I don't but am unsure if the first gets created properly
    date_index += 1
    for date in date_list[date_index:]:
        if col_index == len(colorlist):
            col_index = 0
        col1 = colorlist[col_index]
        print('next date is: ' + date + 'col = ' + str(col_index))

        try:
            plot_pcurve(big_df, animal_id, date, invert=False, col1=col1, col2=col1, stage=stage)
            col_index += 1
            # might not need this to be a try statement anymore because the plot_pcurve function contains a lot of exceptions
        except SessionCheckError:
            print('SessionCheckError for:  ' + date + ' ...continuing to next date...')
        except StageError:
            print('StageError for:  ' + date + ' ...continuing to next date...')
        except PCurveError:
            print('Failed to fit p-curve for:  ' + date + ' ...continuing to next date...')

    plt.title('Pcurves for: ' + animal_id + ', stages included: ' + str(stage))
    plt.legend()


def param_days(big_df, animal_id, date_list, stage='ALL', param='B'):
    param_list = []
    successful_date_list = []
    title_dict = {'A': 'Minimum asymptote (A)',
                  'B': 'Slope (B)',
                  'C': 'Inflection point (C)',
                  'D': 'Maximum asymptote (D)'}

    param_dict = {'A': 0,
                  'B': 1,
                  'C': 2,
                  'D': 3}

    try:
        param_no = param_dict[param]
    except:
        print('invalid param name entered, try: A, B, C or D')

    for date in date_list:
        print('next date is: ' + date)
        try:
            par = plot_pcurve(big_df, animal_id, date, stage=stage, ret_param=True)
            parameter = par[param_no]
            slope = par[0]/(4*par[1])
            if param == 'B':
                param_list.append(slope)
                print(' its B mate')
            else:
                param_list.append(parameter)
            successful_date_list.append(date)
        except SessionCheckError:
            print('SessionCheckError for:  ' + date + ' ...continuing to next date...')
        except StageError:
            print('StageError for:  ' + date + ' ...continuing to next date...')
        except PCurveError:
            print('Failed to plot curve for:  ' + date + ' ...continuing to next date...')
    alp_plot = plt.plot(param_list, 'go')
    plt.xticks(np.arange(len(successful_date_list)), successful_date_list, fontsize=9, rotation=0)
    plt.title(title_dict[param] + ' over days for: ' + animal_id + ', stages included: ' + str(stage))
    plt.ylabel(title_dict[param])
    return alp_plot






# Plotting params (globally set), these are quite large images
plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams['axes.facecolor'] = 'FFFFFF'
plt.rcParams['text.color'] = 'black'

# Load in SC file
pickle_in = open("Rot3_data\\SC_full_df.pkl", "rb")
SC_df = pickle.load(pickle_in)

# Load in PWM file
pickle_in = open("Rot3_data\\PWM_full_df.pkl","rb")
PWM_df = pickle.load(pickle_in)

# Clean them up
sc = clean_up_df(SC_df)
pwm = clean_up_df(PWM_df)
pwm = pwm.rename(columns={"history_pair": "history_stim"})  # rename this column to same column name as sc


# Create the cleaned up SC dataframe, shouldn't need to select animals
animals = ['AA01', 'AA03', 'AA05', 'AA07', 'DO04', 'DO08', 'SC04', 'SC05',
           'VP01', 'VP07', 'VP08']  # AA03 and SC04 don't do any trials

date_list = sc.index.values

# good example animal and day
#plot_pcurve(sc, 'VP08', '2019-08-06')

cal_prob(sc, 'VP08', '2019-08-06')
