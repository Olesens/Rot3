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
        cp_plot = df_cp.plot(marker='o',  # I might be able to set some of these parameters globally and not need functionlinewidth=1.0,
                             markersize=2.5,
                             cmap=plt.cm.RdPu)
        plt.xticks(rotation=0,
                   fontsize='medium')
    plt.ylabel('Total CP')
    plt.xlabel('Date')
    plt.title('Total CP over time for each PWM animal')
    return cp_plot


def plot_trials(df, stage='All', inc_vio=True):
    violations_label = ''
    if stage != 'All':
        mask = df['stage']== stage
        df = df[mask]
    else:
        print('all stages included')
    if inc_vio is True:
        df_trials = df['done_trials'] - (df['done_trials'] * df['violations'])
        violations_label = 'minus violations'

    else:
        df_trials = df['done_trials']
    col_list = df_trials.columns.values  # get list to use for labelling
    tri_plot = plt.plot(df_trials,
                        marker='o',
                        linewidth=1.0,
                        markersize=2.5)
    plt.xticks(rotation=75)
    plt.legend(col_list)
    plt.ylabel('Done Trials')
    plt.xlabel('Date')
    plt.title('Stage ' + str(stage) + ': Done trials for PWM animals ' + str(violations_label))

    # plt.plot instead of df.plot fixed my problem with the x-axis, but the colors are worse, and legends are gone\n",
    # ideally figure out how to fix things in either\n",
    # how to change line colors by making a loop
    # colormap = cmap=plt.cm.RdPu
    return tri_plot


# Create the cleaned up PWM dataframe, with only the below selected animals
animals = ['AA02', 'AA04', 'AA06', 'AA08', 'DO01', 'DO02', 'DO05', 'DO06',
           'SC01', 'SC02', 'SC03', 'SC06', 'VP02', 'VP03', 'VP06']
pwm = clean_up_df(Animal_df, animallist=animals)

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




