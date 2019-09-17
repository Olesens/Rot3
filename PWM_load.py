import scipy.io as sio
import pandas as pd
import os
import pickle
from pandas import DataFrame


# This below dictionary is not used, but a nice reference for what matlab features are extracted
# and what column name they are under in the dataframes.
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
                     'AthenaDelayComp_timeout_history': 'history_tm',
                     'SideSection_previous_sides': 'history_side'
                     }


# INCLUDE IN PARAM_LIST FEATURES TO BE EXTRACTED FROM MATLAB FILE
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


# Create new list with parameters, replace . with _ to correctly extract from matlab file
param_list_refined = []
for parameter in param_list:
    param_list_refined.append(parameter.replace('.', '_'))


# Functions for generating dictionaries and subsequently dataframes from matlab files
def create_rat_dict(file_name='', data_folder='', return_keys=False):
    """
    Extract features specified in param_list from matlab file in data_folder and creates a dictionary of the features
    with a tuple (animal_id, date) as key

    NB: Make sure any features are both in the param_list and also in the rat_val_headers dictionary in this
    function to make sure it is included in the dictonary (and subsequent dataframes).

    :param file_name: full matlab file name, .mat.
            Example: file_name = 'data_@AthenaDelayComp_athena_AA01_190502a.mat'
    :param data_folder: full path to folder containing above file.
            Example: data_folder = r'H:\ratter\SoloData\Data\athena\AA01'
    :param return_keys: only return the keys(headers of column names for the matlab features)
    :type return_keys: bool

    :return: dictionary containing values for all headers
    """

    # load in .mat file contained data, in a format where we can easily extract each parameter value
    full_path = data_folder + '\\' + file_name  # combine filename and folder to create full path
    mat = sio.loadmat(full_path, struct_as_record=False, squeeze_me=True)
    mat_saved = mat['saved']  # save the 'saved' part of the mat files as its own entity

    # extract values from mat file and create dictionary containing values for relevant parameters
    rat_values = {'file_name': file_name}  # add in the name of file loaded in script to avoid confusion
    for param in param_list_refined:
        var = 'mat_saved.' + param
        try:
            # this expression usually fails if the file is from soundcategorization because some of the matlab
            # feature names will not be the same for the two types of files. This way we select from pwm.
            rat_values[param] = eval(var)  # eval() runs the var expression
        except:
            print("failed to extract param, most likely soundcategorization file. Nothing returned")
            return None

    # calculate sum of left and right trials
    right_trials = rat_values['StimulusSection_nTrialsClass1'] + rat_values['StimulusSection_nTrialsClass2'] \
                   + rat_values['StimulusSection_nTrialsClass3'] + rat_values['StimulusSection_nTrialsClass4']
    left_trials = rat_values['StimulusSection_nTrialsClass5'] + rat_values['StimulusSection_nTrialsClass6'] \
                  + rat_values['StimulusSection_nTrialsClass7'] + rat_values['StimulusSection_nTrialsClass8']
    # create separate values for date and savetime
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

    # create new dictionary for rat with headers as keys and add appropriate values
    # ADD FEATURES HERE IF YOU WANT THEM INCLUDED
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
                       'A2_time': rat_values['SideSection_A2_time'],
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

    # use tuple as dict key to create multi-indexing when creating dataframe
    rat_dict = {(rat_val_headers['animal_id'], date_pd): rat_val_headers}
    if return_keys is True:
        return rat_val_headers.keys()
    else:
        return rat_dict


def create_df_from_dict(rat_dict, rat_dict_values_keys):
    """
    Creates a dataframe from the given rat_dicts, with rat_dict_values_keys as column names.
    :param rat_dict:
    :param rat_dict_values_keys:
    :return: dataframe for given rat
    """
    # create dataframe
    rat_frame = DataFrame(rat_dict, index=rat_dict_values_keys)
    rat_frame = rat_frame.T  # transpose so index becomes column names instead.
    return rat_frame


def whole_animal_df(animal_folder):
    """
    Iterates through all the files in the animal_folder and extract features from the matlab files and create a
    dataframe to contain it all.
    :param animal_folder: str
    :return: Dataframe containing data from all relevant sessions in animal folder
    """
    df_list = []  # list to contain all dfs
    for filename in os.listdir(animal_folder):
        if filename.endswith('.mat'):
            print("extracting data from..", filename)
            rat_session = filename
            session = create_rat_dict(rat_session, animal_folder)
            session_dict_values_keys = create_rat_dict(rat_session, animal_folder, return_keys=True)
            session_df = create_df_from_dict(session, session_dict_values_keys)
            df_list.append(session_df)
            print('extraction completed for', filename)
            continue
        else:
            continue
    try:
        animal_df = pd.concat(df_list)
        return animal_df
    except:
        print('No dataframes to concatenate, check that there are files in this folder...')
        return None


def create_list_of_dfs(data_folder):
    """
    Iterates through animal folders in each experimenter folder to extract relevant data from matlab files
    :param data_folder: datafolder to iterate through. Usually the ratter/SoloData/Data
    :return: list of dataframes for all relevant animal sessions in datafodler.
    """
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


# GENERATE AND SAVE THE FULL DATAFRAME
def save_df(dataframe, name='PWM_full_df', folder="Rot3_data\\"):
    """
    Save dataframe under specified name in folder using pickle

    Current default parameters is a folder within the python project.
    NB: function overwrites, without warning, any files given by the same name
    :param dataframe: Dataframe
    :param name: Name to save Dataframe under
    :type name: str
    :param folder: the folder to save the dataframe in
    :type folder: str

    :return: None
    """
    with open(folder + name + ".pkl", "wb") as f:
        pickle.dump(dataframe, f)
    print('Dataframe has been saved in folder: ' + folder + ' as: ' + name)


def create_pwm_df(datafolder=r'H:\ratter\SoloData\Data', save=True, name='PWM_full_df', folder="Rot3_data\\" ):
    """
    Function creates a complete dataframe for all animals in datafolder which has a pwm setting file.

    :param datafolder: Location of ratter SoloData folder
    :param save: Whether or not to save the newly created dataframe
    :type save: bool
    :param name: if save = True, what name to save the dataframe under
    :param folder: of save = True, what folder should the dataframe be pickle dumped in.

    :return: Complete PWM dataframe
    """
    pwm_df_list = create_list_of_dfs(data_folder)  # creates all dfs for all relevant files in given datafolder
    pwm_full_df = pd.concat(pwm_df_list)  # concatenates all dfs into one dataframe
    if save is True:
        save_df(pwm_full_df, name=name, folder=folder)
    return pwm_full_df











