import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
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


# Create the cleaned up SC dataframe, shouldn't need to select animals
animals = ['AA01', 'AA03', 'AA05', 'AA07', 'DO04', 'DO08', 'SC04', 'SC05',
           'VP01', 'VP07', 'VP08']

sc = clean_up_df(Animal_df)

