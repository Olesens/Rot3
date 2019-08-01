import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import ipython_genutils
# Plotting params (globally set)
plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams['axes.facecolor'] = '555555'
plt.rcParams['text.color'] = 'white'

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


def plot_cp(df):
    df_cp = df['total_CP']
    cp_plot = df_cp.plot(marker='o',  # I might be able to set some of these parameters globally and not need functionlinewidth=1.0,
                    markersize=2.5,
                    cmap=plt.cm.RdPu)
    plt.xticks(rotation=75,
               fontsize='medium')
    plt.ylabel('Total CP')
    plt.xlabel('Date')
    plt.title('Total_CP over time for each animal')
    return cp_plot


# Create the cleaned up PWM dataframe, with only the below selected animals
animals = ['AA02', 'AA04', 'AA06', 'AA08', 'DO01', 'DO02', 'DO05', 'DO06',
           'SC01', 'SC02', 'SC03', 'SC06', 'VP02', 'VP03', 'VP06']
pwm = clean_up_df(Animal_df, animallist=animals)
# Create CP plot
CP_fig = plot_cp(pwm)
# plot CP duration for all animals
