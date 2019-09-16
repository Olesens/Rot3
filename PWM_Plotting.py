import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from SC_plotting import check
import ipython_genutils
# Plotting params (globally set)
plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams['axes.facecolor'] = 'FFFFFF'
plt.rcParams['text.color'] = 'black'

# Load in file
pickle_in = open("Rot3_data\\PWM_full_df.pkl","rb")
Animal_df = pickle.load(pickle_in)


def clean_up_df(df, animallist=[], index=True, multiindex=True, fixstages=True, duplicates=True):
    """

    :param df:
    :param animallist:
    :param index:
    :param multiindex:
    :param fixstages:
    :param duplicates:
    :return:
    """
    if index is True:
        df = df.sort_index()  # sort according to index, this should sort according to date
    if multiindex is True:
        df = df.rename_axis(['animal', 'date2'])  # add labels for Multi-index
    if fixstages is True:
        mask = (df['stage'] == 1) & (df['A2_time'] > 0)  # Fix stages problem!!
        df['stage'] = df['stage'].mask(mask, 2)  # have not yet created mask for stage 3
        # stage 3 equals: stage = 1(after mask 2) and reward type = NoReward
    if animallist:
        df = df.loc[animallist]  # include only specifically given animals
        # should make this dependent on their settings file in the future
    if duplicates is True: # remove duplicates
        df = df.swaplevel('animal', 'date2')  # can't remember is this is necessary for duplicate removal
        df = df.reset_index()  # In order to delete duplicates
        dup_r = df[df.duplicated(['date2', 'animal'])]  # this is actually unnecessary
        # print('dub is:',dup_r)
        df = df.drop_duplicates(['date2', 'animal'])
        dup_r = df[df.duplicated(['date2', 'animal'])]  # Run a little double check
        # print('dub is:',dup_r)
        # Put the dataframe nicely together again
        df = df.set_index(['date2', 'animal'])
        df = df.sort_index()
        df = df.unstack()

        #could include a show removed duplicates function
    return df


def plot_cp(df, stage='All'):
    if stage != 'All':
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
        cp_plot = df_cp.plot(marker='o',  # I might be able to set some of these parameters globally and not need function
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

# Create the cleaned up PWM dataframe, with only the below selected animals
animals = ['AA02', 'AA04', 'AA06', 'AA08', 'DO01', 'DO02', 'DO05', 'DO06',
           'SC01', 'SC02', 'SC03', 'SC06', 'VP02', 'VP03', 'VP06']
pwm = clean_up_df(Animal_df, animallist=animals)


def run_all_plots():
    # plot CP duration for all animals
    CP_fig = plot_cp(pwm)
    plt.savefig('Rot3_data\\CP_fig.png', bbox_inches='tight')
    plt.close()

    # plot CP duration for stage 1
    st1_CP = plot_cp(pwm, stage=1)
    plt.savefig('Rot3_data\\st1_CP.png', bbox_inches='tight')
    plt.close()

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
        fig_name = animal + '_boxp'
        plot = boxplot_animal(pwm, animal)
        plt.savefig('Rot3_data\\' + fig_name + '.png', bbox_inches='tight')
        plt.close()
# Was trying to check if timeouts, violations and left right trials make up all done trials. Some kinks currently
#pwm_trials = Animal_df['done_trials'] - (Animal_df['violations']*Animal_df['done_trials']) - (Animal_df['timeouts']*Animal_df['done_trials']) - Animal_df['left_trials'] - Animal_df['right_trials']
#Animal_df2 = Animal_df.copy()
#Animal_df2['trial_diff'] = pwm_trials


def history(single_session):
    df = single_session
    session_list = []
    hh = (df['history_hits'])[0]
    hv = (df['history_vio'])[0]
    htm = (df['history_tm'])[0]
    for position in range(len(hv)):
        if hh[position] == 1:
            session_list.append('hits')
        elif hv[position] == 1:
            session_list.append('vio')
        elif htm[position] == 1:
            session_list.append('tm')
        else:
            session_list.append('dunno')
    return session_list  # this works!





# PLOTTING PROBABILITIES AND FITTING P-CURVES
def cal_prob(df, animal_id, date, ret_hist=False):
    # still need to add for 10 stimuli, is this dependent on the stage?
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
    if 8 in history['stim'].values:
        # print('detected 8 stimuli, calculating probabilities..')
        # amount of times the rat went right when stim 1 was on (which is the correct choice)
        history['stim1_wr'] = (history['side'] == 114) & (history['stim'] == 1) & (history['hits'] == 1)
        history['stim2_wr'] = (history['side'] == 114) & (history['stim'] == 2) & (history['hits'] == 1)
        history['stim3_wr'] = (history['side'] == 114) & (history['stim'] == 3) & (history['hits'] == 1)
        history['stim4_wr'] = (history['side'] == 114) & (history['stim'] == 4) & (history['hits'] == 1)
        # for stim 5-8 if the rat went right, which would be incorrect for those stimuli
        history['stim5_wr'] = (history['side'] == 108) & (history['stim'] == 5) & (history['hits'] == 0)
        history['stim6_wr'] = (history['side'] == 108) & (history['stim'] == 6) & (history['hits'] == 0)
        history['stim7_wr'] = (history['side'] == 108) & (history['stim'] == 7) & (history['hits'] == 0)
        history['stim8_wr'] = (history['side'] == 108) & (history['stim'] == 8) & (history['hits'] == 0)
        # sum the above into one variable for each stimuli
        sum_stim1R = history['stim1_wr'].sum()
        sum_stim2R = history['stim2_wr'].sum()
        sum_stim3R = history['stim3_wr'].sum()
        sum_stim4R = history['stim4_wr'].sum()
        sum_stim5R = history['stim5_wr'].sum()
        sum_stim6R = history['stim6_wr'].sum()
        sum_stim7R = history['stim7_wr'].sum()
        sum_stim8R = history['stim8_wr'].sum()

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
        stim7_percR = sum_stim7R / done_trials * 100
        stim8_percR = sum_stim8R / done_trials * 100

        # just doing it for left as well to check if it adds up to 100% which it should when timeouts and violations
        # are not included

        # amount of times the rat went left when stim 1,2,3
        history['stim1_wl'] = (history['side'] == 114) & (history['stim'] == 1) & (history['hits'] == 0)
        history['stim2_wl'] = (history['side'] == 114) & (history['stim'] == 2) & (history['hits'] == 0)
        history['stim3_wl'] = (history['side'] == 114) & (history['stim'] == 3) & (history['hits'] == 0)
        history['stim4_wl'] = (history['side'] == 114) & (history['stim'] == 4) & (history['hits'] == 0)
        # for stim 4,5,6 the rat went left
        history['stim5_wl'] = (history['side'] == 108) & (history['stim'] == 5) & (history['hits'] == 1)
        history['stim6_wl'] = (history['side'] == 108) & (history['stim'] == 6) & (history['hits'] == 1)
        history['stim7_wl'] = (history['side'] == 108) & (history['stim'] == 7) & (history['hits'] == 1)
        history['stim8_wl'] = (history['side'] == 108) & (history['stim'] == 8) & (history['hits'] == 1)

        # sum the above into one variable for each stimuli
        sum_stim1L = history['stim1_wl'].sum()
        sum_stim2L = history['stim2_wl'].sum()
        sum_stim3L = history['stim3_wl'].sum()
        sum_stim4L = history['stim4_wl'].sum()
        sum_stim5L = history['stim5_wl'].sum()
        sum_stim6L = history['stim6_wl'].sum()
        sum_stim7L = history['stim7_wl'].sum()
        sum_stim8L = history['stim8_wl'].sum()
        stim1_percL = sum_stim1L / done_trials * 100
        stim2_percL = sum_stim2L / done_trials * 100
        stim3_percL = sum_stim3L / done_trials * 100
        stim4_percL = sum_stim4L / done_trials * 100
        stim5_percL = sum_stim5L / done_trials * 100
        stim6_percL = sum_stim6L / done_trials * 100
        stim7_percL = sum_stim7L / done_trials * 100
        stim8_percL = sum_stim8L / done_trials * 100

        percentage_sum = stim1_percR + stim2_percR + stim3_percR + stim4_percR + stim5_percR + stim6_percR \
                         + stim7_percR + stim8_percR + stim1_percL + stim2_percL + stim3_percL + stim4_percL + \
                         stim5_percL + stim6_percL + stim7_percL + stim8_percL
        print('Sum of percentages for left and right choices are: ' +str(percentage_sum))

        # calculate probability
        # sum the left and right choices for each given stimulus
        stim1_sum = sum_stim1R + sum_stim1L
        stim2_sum = sum_stim2R + sum_stim2L
        stim3_sum = sum_stim3R + sum_stim3L
        stim4_sum = sum_stim4R + sum_stim4L
        stim5_sum = sum_stim5R + sum_stim5L
        stim6_sum = sum_stim6R + sum_stim6L
        stim7_sum = sum_stim7R + sum_stim7L
        stim8_sum = sum_stim8R + sum_stim8L

        # calculate the percentage of right choices for each given stimulus
        s1_right_prob = sum_stim1R / stim1_sum * 100
        s2_right_prob = sum_stim2R / stim2_sum * 100
        s3_right_prob = sum_stim3R / stim3_sum * 100
        s4_right_prob = sum_stim4R / stim4_sum * 100
        s5_right_prob = sum_stim5R / stim5_sum * 100
        s6_right_prob = sum_stim6R / stim6_sum * 100
        s7_right_prob = sum_stim7R / stim7_sum * 100
        s8_right_prob = sum_stim8R / stim8_sum * 100

        # create a dict with all variables to include in dataframe
        dict = {'Sum right choices': [sum_stim1R, sum_stim2R, sum_stim3R, sum_stim4R, sum_stim5R, sum_stim6R, sum_stim7R, sum_stim8R],
                'Perc_R of all done': [stim1_percR, stim2_percR, stim3_percR, stim4_percR, stim5_percR, stim6_percR, stim7_percR, stim8_percR],
                'Sum left choices': [sum_stim1L, sum_stim2L, sum_stim3L, sum_stim4L, sum_stim5L, sum_stim6L, sum_stim7L, sum_stim8L],
                'Perc_L of all done': [stim1_percL, stim2_percL, stim3_percL, stim4_percL, stim5_percL, stim6_percL, stim7_percL, stim8_percL],
                'Stimulus trials sum': [stim1_sum, stim2_sum, stim3_sum, stim4_sum, stim5_sum, stim6_sum, stim7_sum, stim8_sum],
                'Right_prob': [s1_right_prob, s2_right_prob, s3_right_prob, s4_right_prob, s5_right_prob, s6_right_prob, s7_right_prob, s8_right_prob]
                }
        stim_df = pd.DataFrame(dict, index=['stim1', 'stim2', 'stim3', 'stim4', 'stim5', 'stim6', 'stim7', 'stim8'])
        return stim_df