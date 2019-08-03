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
pickle_in = open("Rot3_data\\Rat_full_df.pkl","rb")
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
        df = df.sort_index()  # sort according to index, this should sort according to date
    if multiindex is True:
        df = df.rename_axis(['animal', 'date2'])  # add labels for Multi-index
    if fixstages is True:
        mask = (df['stage'] == 1) & (df['A2_time'] > 0)  # Fix stages problem!!
        df['stage'] = df['stage'].mask(mask, 2)  # have not yet created mask for stage 3
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


def plot_trials(df, stage='All', p_type = 'all_trials', inc_vio=True, perc = True, ):

    p_types = ['all_trials', 'vio_only', 'tm_only', 'left', 'right']
    if p_type not in p_types:
        raise ValueError("Invalid p_type. Expected one of: %s" % p_type)

    # Filter according to stages
    if stage != 'All':
        mask = df['stage'] == stage
        df = df[mask]
    else:
        print('all stages included')

    if p_type is 'left' or 'right':  # might not need to be if statement, just do no matter what
        # this should only occur inside the function so change be global just for the division.
        # how would it effect the other calculations

         # replace
        df['done_trials'] = df['done_trials'] - ((df['violations'] + df['timeouts'])*df['done_trials'])
        mask = df['done_trials'] == 0
        df['done_trials'] = df['done_trials'].mask(mask, 1)

        #df_correct = df['done_trials'] - (df['done_trials'] * df['violations']) - (df['done_trials'] * df['timeouts'])
    p_types_dict = {'all_trials': df['done_trials'] - (df['done_trials'] * df['violations']),  #not %
                    'vio_only': df['violations'] * 100,
                    'tm_only': df['timeouts']*100,
                    'left': df['left_trials'] / df['done_trials']*100,  #might have to make this minus vio and timeouts
                    'right': df['left_trials'] / df['done_trials']*100}

    p_types_title = {'all_trials': ': Done trials for PWM animals minus violations ',
                    'vio_only': ': % Violation trials for PWM animals',
                    'tm_only': ': % Timeout trials for PWM animals',
                    'left': '% left trials for PWM animals',
                    'right': '% right trials for PWM animals'}

     #if trials = true:
     #   take the thing and multiply by done_trials
    df_trials = p_types_dict[p_type]
    col_list = df_trials.columns.values  # get list to use for labelling
    tri_plot = plt.plot(df_trials,
                        marker='o',
                        linewidth=1.0,
                        markersize=2.5)
    if p_type is not 'all_trials':
        plt.ylim([-5, 105])

    plt.xticks(rotation=75)
    plt.legend(col_list)
    plt.xlabel('Date')
    plt.ylabel('Trials')
    plt.title(p_types_title[p_type])

    # Labels for plot title
    violations_label = ''
    #title = ': Done trials for PWM animals '

    # if vio_only is True:

    #   df_trials = (df['done_trials'] * df['violations'])
    #title = ': Violation trials for PWM animals'
    #if tm_only is True:
       # df_trials = (df['done_trials'] * df['timeouts'])
        #title = ': Timeout trials for PWM animals'
    #elif inc_vio is True:
        #df_trials = df['done_trials'] - (df['done_trials'] * df['violations'])
        #violations_label = 'minus violations'
    #else:
        #df_trials = df['done_trials']

    # plt.plot instead of df.plot fixed my problem with the x-axis, but the colors are worse, and legends are gone\n",
    # ideally figure out how to fix things in either\n",
    # how to change line colors by making a loop
    # colormap = cmap=plt.cm.RdPu
    return tri_plot


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
    st0_trials = plot_trials(pwm, stage=0)
    plt.savefig('Rot3_data\\st0_trials.png', bbox_inches='tight')
    plt.close()

    # Done trials for stage 1
    st1_trials = plot_trials(pwm, stage=1)
    plt.savefig('Rot3_data\\st1_trials.png', bbox_inches='tight')
    plt.close()

    # Done trials for stage 2
    st2_trials = plot_trials(pwm, stage=2)
    plt.savefig('Rot3_data\\st2_trials.png', bbox_inches='tight')
    plt.close()

    # Violation trials for stage 1
    st1_vio_trials = plot_trials(pwm, stage=1, vio_only=True)
    plt.savefig('Rot3_data\\st1_vio_trials.png', bbox_inches='tight')
    plt.close()

    # Violation trials for stage 2
    st2_vio_trials = plot_trials(pwm, stage=2, vio_only=True)
    plt.savefig('Rot3_data\\st2_vio_trials.png', bbox_inches='tight')
    plt.close()

    # Timeout trials for stage 1
    st1_tm_trials = plot_trials(pwm, stage=1, tm_only=True)
    plt.savefig('Rot3_data\\st1_tm_trials.png', bbox_inches='tight')
    plt.close()

    # Timeout trials for stage 2
    st2_tm_trials = plot_trials(pwm, stage=2, tm_only=True)
    plt.savefig('Rot3_data\\st2_tm_trials.png', bbox_inches='tight')
    plt.close()




#pwm_trials = Animal_df['done_trials'] - (Animal_df['violations']*Animal_df['done_trials']) - (Animal_df['timeouts']*Animal_df['done_trials']) - Animal_df['left_trials'] - Animal_df['right_trials']
#Animal_df2 = Animal_df.copy()
#Animal_df2['trial_diff'] = pwm_trials

