import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy as sy
import ipython_genutils

# Plotting params (globally set)
plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams['axes.facecolor'] = 'FFFFFF'
plt.rcParams['text.color'] = 'black'

# Load in file
pickle_in = open("Rot3_data\\SC_full_df.pkl","rb")
Animal_df = pickle.load(pickle_in)


def clean_up_df(df, animallist=[], index=True, multiindex=True, fixstages=True, duplicates=True):
    """
    :param df: dataframe to take in a clean up
    :param animallist:
    :param index:
    :param multiindex:
    :param fixstages:
    :param duplicates
    :return:
    """
    if index is True:
        df = df.sort_index()  # sort according to index, this should sort according to date, think it sort animals
    if multiindex is True:
        df = df.rename_axis(['animal', 'date2'])  # add labels for Multi-index
    if fixstages is True:
        mask = (df['stage'] == 1) & (df['reward_type'] == 'Always')  # turn stage 1 to 0 if reward type is always
        df['stage'] = df['stage'].mask(mask, 0)  # this is for SC task

    if animallist:
        df = df.loc[animallist]  # include only specifically given animals

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


def plot_cp(df, stage='All'):  # will I even need this for SC?
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
        fig_name = 'sc_' + animal + '_boxp'
        plot = boxplot_animal(sc, animal)
        plt.show()
        plt.savefig('Rot3_data\\' + fig_name + '.png', bbox_inches='tight')
        plt.close()


def check(df, animal_id, date):
    # could maybe modify this a little to be shorter, but it works atm
    single_animal = df.xs(animal_id, level='animal', axis=1).copy()
    session = single_animal.loc[date]
    stim = session['history_stim']
    try:
        stim = pd.DataFrame(stim, columns=['stim']).T

    except ValueError:
        print('Could not extract stimulus history, check that specific animal and session exists')
        return False
    else:
        return True


def cal_prob(df, animal_id, date):
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


    # stage condition
    if 6 in history['stim'].values:
        print('detected 6 stimuli, calculating probabilities..')
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
        print('detected 4 stimuli, calculating probabilities...')
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


def pf(x, alpha, beta):  # pschymetric function
    return 1. / (1 + np.exp( - (x-alpha)/beta))


def plot_pcurve(big_df, animal_id, date, invert=False, col1='#810f7c', col2='#045a8d'):
    df = cal_prob(big_df, animal_id, date)
    right_prob = df['Right_prob'].sort_index(ascending=False)
    right_prob = (right_prob/100).values
    stim = np.array([0, 1, 2, 3, 4, 5])  #arbitrary x-axis needs to be same for plotting and p-fit
    pcurve = plt.plot(stim, right_prob, marker='o', markersize=8, color=col1, linestyle='', label=animal_id)
    if invert is True: # this might be unneccesary now
        ax = plt.gca()  # invert x axis to classic direction of curve
        ax.invert_xaxis()  # invert x axis to classic direction of curve
    plt.ylabel('Probability of Right choice', fontsize=12)
    plt.xlabel('Stimulus', fontsize=12)
    plt.xticks(np.arange(6), ('stim 6', 'stim 5', 'stim 4', 'stim 3', 'stim 2', 'stim 1'), fontsize=9, rotation=0)
    plt.title('pCurve for '+animal_id + ' on ' + date, fontsize=14)

    # par0 = sy.array([100., 1.]) or sy.array([0., 1.])
    par, mcov = curve_fit(pf, stim, right_prob)  # fit p curve to data
    plt.plot(stim, pf(stim, par[0], par[1]), color=col2)  # plot on top of data

    return pcurve


def day_pcurve(big_df, animal_list, date):
    # could include a condition to exclude animals if they do trials below a certain number
    # need to fix this so different amounts of stimuli are allowed
    colorlist = ['#82dae0', '#fff0a5', '#b0e5ca', '#e5b0b1', '#b7adc7']
    col_index = 1
    pic = plot_pcurve(big_df, animal_list[0], date, col1=colorlist[0], col2=colorlist[0])
    #animal_names = []
    #animal_names.append(animal_list[0])

    for animal in animal_list[1:]:
        print(col_index)
        if col_index == 5:
            col_index = 0
        col1 = colorlist[col_index]
        try:
            plot_pcurve(big_df, animal, date, invert=False, col1=col1, col2=col1)
            #animal_names.append(animal)
        except:
            continue
        col_index += 1
    plt.title('Pcurves for animals on: ' + date)
    plt.legend()
    return pic


def animal_pcurve(big_df, animal_id, date_list):
    # need to fix this so different amounts of stimuli are allowed
    colorlist = ['#82dae0', '#fff0a5', '#b0e5ca', '#e5b0b1', '#b7adc7']
    col_index = 1
    date_index = 0
    date = date_list[date_index]
    # include a check for if there is data in the df for the animal
    while check(sc, animal_id, date) is False:
        print('no session date for: ' + date)
        print('proceeding to next date in list')
        date_index += 1
        date = date_list[date_index]

    print('first succesful date is: ' + date)
    pic = plot_pcurve(big_df, animal_id, date, col1=colorlist[0], col2=colorlist[0]) # add a label for date
    date_index += 1
    for date in date_list[date_index:]:
        print('date is: '+ date )
        print(col_index)
        if col_index == 5:
            col_index = 0
        col1 = colorlist[col_index]
        try:
            plot_pcurve(big_df, animal_id, date, invert=False, col1=col1, col2=col1)
            #animal_names.append(animal)
        except:
            continue
        col_index += 1
    plt.title('Pcurves for animals on: ' + date)
    #plt.legend()
    return pic

# Create the cleaned up SC dataframe, shouldn't need to select animals
animals = ['AA01', 'AA03', 'AA05', 'AA07', 'DO04', 'DO08', 'SC04', 'SC05',
           'VP01', 'VP07', 'VP08']

sc = clean_up_df(Animal_df)

date = '2019-08-06'
day_pcurve(sc, animals, date)

# good example animal and day
plot_pcurve(sc, 'VP08', '2019-08-06')
df = cal_prob(sc, 'VP08', '2019-08-06')
plt.close()
date_list = sc.index.values
plot_pcurve(sc, 'VP08', date_list[0])
