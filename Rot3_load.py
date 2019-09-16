import scipy.io as sio
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import pickle

# notes:...
# multi index accessing (non-transposed)
# rat_frame['AA01','02-May-2019']
# rat_frame['AA01']
# rat_frame.ix['stage',['AA01']] # index comes first cause .ix
# rat_co = pd.merge(rat_frame,rat2_frame,left_index=True,right_index=True,how='outer')

# list containing the relevant parameter features from the experiment as written in excel file
param_list = ['ProtocolsSection_n_done_trials',
              'SavingSection.settings.file',
              'SavingSection.settings.file.load.time',
              'SavingSection.experimenter',
              'SavingSection.ratname',
              'SavingSection.SaveTime',
              'SideSection.A2.time',
              'SideSection.init.CP.duration',
              'SideSection.Total.CP.duration',
              'SideSection.reward.type',
              'SideSection.training.stage',
              'StimulusSection.nTrialsClass1',
              'StimulusSection.nTrialsClass2',
              'StimulusSection.nTrialsClass3',
              'StimulusSection.nTrialsClass4',
              'StimulusSection.nTrialsClass5',
              'StimulusSection.nTrialsClass6',
              'StimulusSection.nTrialsClass7',
              'StimulusSection.nTrialsClass8',
              'OverallPerformanceSection.violation.rate',
              'OverallPerformanceSection.timeout.rate',
              'OverallPerformanceSection.Left.hit.frac',
              'OverallPerformanceSection.Right.hit.frac',
              'OverallPerformanceSection.hit.frac',
              'AthenaDelayComp.violation.history',
              'AthenaDelayComp.hit.history',
              'AthenaDelayComp.pair.history',
              'AthenaDelayComp.timeout.history',
              'SideSection.previous.sides'
              ]

# shorter param names, same as Viktor
params_as_headers = ['file',
                     'settings_file',
                     'experimenter',
                     'animal_id',
                     'date',
                     'start_time',
                     'save_time',
                     'right_trials',
                     'left_trials',
                     'stage',
                     'init_CP',
                     'total_CP',
                     'done_trials',
                     'A2_time',
                     'reward_type',
                     'violations',
                     'timeouts',
                     'hits_left',
                     'hits_right',
                     'hits_total',
                     'history_vio',
                     'history_hits',
                     'history_pair',
                     'history_tm',
                     'history_side']  # not actually side history but info about which side would be correct

# dictionary mapping param names to their headers
# this might not be useful actually, it is currently not being used mainly for own ease
params_to_headers = {'ProtocolsSection_n_done_trials': 'done_trials',
                     'SavingSection_settings_file': 'settings_file',
                     'SavingSection_settings_file_load_time': 'start_time',
                     'SavingSection_experimenter': 'experimenter',
                     'SavingSection_ratname': 'animal_id',
                     'SavingSection_SaveTime': 'save_time',
                     'SideSection_A2_time': 'A2_time',
                     'SideSection_init_CP_duration': 'init_CP',
                     'SideSection_Total_CP_duration': 'total_CP',
                     'SideSection_reward_type': 'reward_type',
                     'SideSection_training_stage': 'stage',
                     'StimulusSection_nTrialsClass1': 'right_trials',
                     'StimulusSection_nTrialsClass2': 'right_trials',
                     'StimulusSection_nTrialsClass3': 'right_trials',
                     'StimulusSection_nTrialsClass4': 'right_trials',
                     'StimulusSection_nTrialsClass5': 'left_trials',
                     'StimulusSection_nTrialsClass6': 'left_trials',
                     'StimulusSection_nTrialsClass7': 'left_trials',
                     'StimulusSection_nTrialsClass8': 'left_trials',
                     'OverallPerformanceSection_violation_rate': 'violations',
                     'OverallPerformanceSection_timeout_rate': 'timeouts',
                     'OverallPerformanceSection_Left_hit_frac': 'hits_left',
                     'OverallPerformanceSection_Right_hit_frac': 'hits_right',
                     'OverallPerformanceSection_hit_frac': 'hits_total',
                     'AthenaDelayComp_violation_history': 'history_vio',
                     'AthenaDelayComp_hit_history': 'history_hits',
                     'AthenaDelayComp_pair_history': 'history_pair',
                     'AthenaDelayComp_timeout_history': 'history_tm'
                     # add side history
                     }

# create new list with parameters, replace . with _ to correctly extract from matlab file
param_list_refined = []
for param in param_list:
    param_list_refined.append(param.replace('.', '_'))

# define data folder and file name
data_folder = r'H:\ratter\SoloData\Data\athena\AA01'
file_name = 'data_@AthenaDelayComp_athena_AA01_190502a.mat'


def create_rat_dict(file_name=file_name, data_folder=data_folder):
    """
    :param file_name: full matlab file name, .mat
    :param data_folder: full path to folder containing above file
    :return: dictionary containing values for all headers
    NB! does not check for new headers from the param_list_refined, must be done manually atm.
    """

    # load in .mat file contained data, in a format where we can easily extract each parameter value
    full_path = data_folder + '\\' + file_name # combine filename and folder to create full path
    mat = sio.loadmat(full_path, struct_as_record=False, squeeze_me=True)
    mat_saved = mat['saved']  # save the 'saved' part of the mat files as its own entity

    # extract values from mat file and create dictionary containing values for relevant parameters
    rat_values = {'file_name': file_name}  # add in the name of file loaded in script to avoid confusion
    for param in param_list_refined:
        var = 'mat_saved.' + param
        try:
            rat_values[param] = eval(var)  # eval() runs the var expression
        except:
            print("failed to extract param, most likely soundcategorization file. Nothing returned")
            return None

    # calculate sum of left and right trials
    right_trials = rat_values['StimulusSection_nTrialsClass1'] + rat_values['StimulusSection_nTrialsClass2'] \
                   + rat_values['StimulusSection_nTrialsClass3'] + rat_values['StimulusSection_nTrialsClass4']
    left_trials = rat_values['StimulusSection_nTrialsClass5'] + rat_values['StimulusSection_nTrialsClass6'] \
                  + rat_values['StimulusSection_nTrialsClass7'] + rat_values['StimulusSection_nTrialsClass8']
    # create separate values for date and save time
    split = rat_values['SavingSection_SaveTime'].split()  # split into date and time
    date = split[0]
    save_time = split[1]

    # create a version of the date in correct pd format, this is important for sorting later
    date_pd = str(pd.to_datetime(date))
    split_date = date_pd.split(' ')  # split into date and time
    date_pd = split_date[0]  # keep date

    # convert Matlab time format into timestamp, remove date and ms from timestamp
    try:
        start = str(pd.to_datetime(rat_values['SavingSection_settings_file_load_time'] - 719529, unit='D'))
        split_start = start.split(' ')  # split into date and time
        start_time1 = split_start[1]  # keep time
        start_time = start_time1[:8]  # keep only hr, min, s in timestamp
    except:
        print("time formatting failed")
        start_time = rat_values['SavingSection_settings_file_load_time']

    try:  # Refactor and remove these later
        a2_time = rat_values['SideSection_A2_time']

    except:
        a2_time = None

    # create new dictionary for rat with headers as keys and add appropriate values
    rat_val_headers = {'file': rat_values['file_name'],
                       'settings_file': rat_values['SavingSection_settings_file'],
                       'experimenter': rat_values['SavingSection_experimenter'],
                       'animal_id': rat_values['SavingSection_ratname'],
                       'date': date,
                       'start_time': start_time,
                       'save_time': save_time,
                       'right_trials': right_trials,
                       'left_trials': left_trials,
                       'stage': rat_values['SideSection_training_stage'],
                       'init_CP': rat_values['SideSection_init_CP_duration'],
                       'total_CP': rat_values['SideSection_Total_CP_duration'],
                       'done_trials': rat_values['ProtocolsSection_n_done_trials'],
                       'A2_time': a2_time,
                       'reward_type': rat_values['SideSection_reward_type'],
                       'violations': rat_values['OverallPerformanceSection_violation_rate'],
                       'timeouts': rat_values['OverallPerformanceSection_timeout_rate'],
                       'hits_left': rat_values['OverallPerformanceSection_Left_hit_frac'],
                       'hits_right': rat_values['OverallPerformanceSection_Right_hit_frac'],
                       'hits_total': rat_values['OverallPerformanceSection_hit_frac'],
                       'history_vio': rat_values['AthenaDelayComp_violation_history'],
                       'history_hits': rat_values['AthenaDelayComp_hit_history'],
                       'history_pair': rat_values['AthenaDelayComp_pair_history'],
                       'history_tm': rat_values['AthenaDelayComp_timeout_history'],
                       'history_side': rat_values['SideSection_previous_sides']
                       }

    # create nested dict with date as key and rat+date as name
    # use tuple as dict key to create multi-indexing when creating dataframe
    rat = {(rat_val_headers['animal_id'], date_pd): rat_val_headers}
    # considering making the dict into a transposed dataframe so just need to merge later on
    return rat


def create_df_from_dict(rat_dict):
    """
    :param rat_dict: dictionary
    :return: dataframe in preferred format to easily merge
    """
    # create dataframe
    rat_frame = DataFrame(rat_dict, index=params_as_headers)  # make the indexing in the same order as Viktor
    rat_frame = rat_frame.T
    # append to desired dataframe
    return rat_frame


def whole_animal_df(animal_folder):
    df_list = []  # list to contain all dfs
    for filename in os.listdir(animal_folder):
        if filename.endswith('.mat'):
            print("extracting data from..", filename)
            rat_session = filename
            session = create_rat_dict(rat_session, animal_folder)
            session_df = create_df_from_dict(session)
            df_list.append(session_df)
            continue
        else:
            continue
    animal_df = pd.concat(df_list)
    return animal_df


def create_all_dfs(data_folder):
    # can make a list of experimenters to include if only some are required
    rat_df_list = []
    for experimenter in os.listdir(data_folder):
        if experimenter != 'experimenter':
            print('Iterating through:', experimenter, 'folder')
            exp_folder = data_folder + '\\' + experimenter
            for animal in os.listdir(exp_folder):
                animal_folder = exp_folder + '\\' + animal
                print('animal folder is: ', animal_folder)
                animal_full_df = whole_animal_df(animal_folder)
                rat_df_list.append(animal_full_df)
        else:
            print('skipping experimenter folder: ', experimenter)  # skip the experimenter folder it is weird

    return rat_df_list



def save_dataframe(dataframe, name = 'Rat_full_df'):
    # Save large dataframe in project
    with open("Rot3_data\\" + name + ".pkl", "wb") as f:
        pickle.dump(dataframe, f)

#data_folder = r'H:\ratter\SoloData\Data'
#rat_df_list = create_all_dfs(data_folder)
#Rat_full = pd.concat(rat_df_list)
#save_dataframe(Rat_full, name='Rat_full_df')

create_rat_dict('data_@AthenaDelayComp_athena_AA02_190807a.mat', r'H:\ratter\SoloData\Data\athena\AA02' )








