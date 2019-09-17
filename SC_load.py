import scipy.io as sio
import pandas as pd
import os
import pickle
from pandas import DataFrame


# list containing the relevant parameter features from the experiment as written in excel file
param_list = ['ProtocolsSection_n_done_trials',
              'SavingSection.settings.file',
              'SavingSection.settings.file.load.time',
              'SavingSection.experimenter',
              'SavingSection.ratname',
              'SavingSection.SaveTime',
              'SideSection.A1.time',
              # 'SideSection.A2.time',
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
              # 'StimulusSection.nTrialsClass7',
              # 'StimulusSection.nTrialsClass8',
              'OverallPerformanceSection.violation.rate',
              'OverallPerformanceSection.timeout.rate',
              'OverallPerformanceSection.Left.hit.frac',
              'OverallPerformanceSection.Right.hit.frac',
              'OverallPerformanceSection.hit.frac',
              # 'SideSection.deltaf.history',
              'SoundCategorization.violation.history',
              'SoundCategorization.hit.history',
              # 'SoundCategorization.pair.history',
              'SoundCategorization.timeout.history',
              'StimulusSection.thisclass',
              'SideSection.previous.sides',
              'SideSection.ThisTrial'
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
                     'A1_time',
                     # 'A2_time',
                     'reward_type',
                     'violations',
                     'timeouts',
                     'hits_left',
                     'hits_right',
                     'hits_total',
                     'history_vio',
                     'history_hits',
                     # 'history_pair',
                     'history_tm',
                     'history_stim',
                     'history_side',
                     'last_choice'
                     ]

# dictionary mapping param names to their headers
# this might not be useful actually, it is currently not being used mainly for own ease
params_to_headers = {'ProtocolsSection_n_done_trials': 'done_trials',
                     'SavingSection_settings_file': 'settings_file',
                     'SavingSection_settings_file_load_time': 'start_time',
                     'SavingSection_experimenter': 'experimenter',
                     'SavingSection_ratname': 'animal_id',
                     'SavingSection_SaveTime': 'save_time',
                     'SideSection_A1_time': 'A1_time',
                     # 'SideSection_A2_time': 'A2_time',
                     'SideSection_init_CP_duration': 'init_CP',
                     'SideSection_Total_CP_duration': 'total_CP',
                     'SideSection_reward_type': 'reward_type',
                     'SideSection_training_stage': 'stage',
                     'StimulusSection_nTrialsClass1': 'right_trials',
                     'StimulusSection_nTrialsClass2': 'right_trials',
                     'StimulusSection_nTrialsClass3': 'right_trials',
                     'StimulusSection_nTrialsClass4': 'right_trials',
                     # 'StimulusSection_nTrialsClass5': 'left_trials',
                     # 'StimulusSection_nTrialsClass6': 'left_trials',
                     # 'StimulusSection_nTrialsClass7': 'left_trials',
                     # 'StimulusSection_nTrialsClass8': 'left_trials',
                     'OverallPerformanceSection_violation_rate': 'violations',
                     'OverallPerformanceSection_timeout_rate': 'timeouts',
                     'OverallPerformanceSection_Left_hit_frac': 'hits_left',
                     'OverallPerformanceSection_Right_hit_frac': 'hits_right',
                     'OverallPerformanceSection_hit_frac': 'hits_total',
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
    full_path = data_folder + '\\' + file_name  # combine filename and folder to create full path
    mat = sio.loadmat(full_path, struct_as_record=False, squeeze_me=True)
    mat_saved = mat['saved']  # save the 'saved' part of the mat files as its own entity

    # extract values from mat file and create dictionary containing values for relevant parameters
    rat_values = {'file_name': file_name}  # add in the name of file loaded in script to avoid confusion
    for param in param_list_refined:
        var = 'mat_saved.' + param
        try:
            rat_values[param] = eval(var)  # eval() runs the var expression
        except:
            print("failed to extract", param)

    # calculate sum of left and right trials, try to include 5 and 6 if not only use 1-4
    try:
        right_trials = rat_values['StimulusSection_nTrialsClass1'] + rat_values['StimulusSection_nTrialsClass2'] \
                       + rat_values['StimulusSection_nTrialsClass3']
        left_trials = rat_values['StimulusSection_nTrialsClass4'] + rat_values['StimulusSection_nTrialsClass5']\
                       + rat_values['StimulusSection_nTrialsClass6']
    except:
        right_trials = rat_values['StimulusSection_nTrialsClass1'] + rat_values['StimulusSection_nTrialsClass2']
        left_trials = rat_values['StimulusSection_nTrialsClass3'] + rat_values['StimulusSection_nTrialsClass4']

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
                       'A1_time': rat_values['SideSection_A1_time'],
                       'reward_type': rat_values['SideSection_reward_type'],
                       'violations': rat_values['OverallPerformanceSection_violation_rate'],
                       'timeouts': rat_values['OverallPerformanceSection_timeout_rate'],
                       'hits_left': rat_values['OverallPerformanceSection_Left_hit_frac'],
                       'hits_right': rat_values['OverallPerformanceSection_Right_hit_frac'],
                       'hits_total': rat_values['OverallPerformanceSection_hit_frac'],
                       'history_vio': rat_values['SoundCategorization_violation_history'],
                       'history_hits': rat_values['SoundCategorization_hit_history'],
                       # 'history_pair': rat_values['SoundCategorization_pair_history'],
                       'history_tm': rat_values['SoundCategorization_timeout_history'],
                       'history_stim': rat_values['StimulusSection_thisclass'],
                       'history_side': rat_values['SideSection_previous_sides'],
                       'last_choice': rat_values['SideSection_ThisTrial']}

    # create nested dict with date as key and rat+date as name
    # use tuple as dict key to create multi-indexing when creating dataframe
    rat = {(rat_val_headers['animal_id'], date_pd): rat_val_headers}
    # considering making the dict into a transposed dataframe so just need to merge later on
    return rat


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
            if 'DelayComp' in filename:
                print('extraction aborted since settings file is for PWM')
            else:
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


def save_dataframe(dataframe, name='Rat_full_df'):
    # Save large dataframe in project
    with open("Rot3_data\\" + name + ".pkl", "wb") as f:
        pickle.dump(dataframe, f)


def create_sc_df():  # Create the dataframe
    data_folder = r'H:\ratter\SoloData\Data'
    rat_df_list = create_all_dfs(data_folder)
    SC_full_df = pd.concat(rat_df_list)
    save_dataframe(SC_full_df, name='SC_full_df')


#AA01 = whole_animal_df(data_folder)



def mini_hist():
    hits = (AA01['history_hits'])[0]
    tm = (AA01['history_tm'])[0]
    stim = (AA01['history_stim'])[0]
    side = (AA01['history_side'])[0]
    vio = (AA01['history_vio'])[0]
    vio = pd.DataFrame(vio, columns=['vio'])
    vio = vio.T
    tm = (AA01['history_tm'])[0]
    tm = pd.DataFrame(tm, columns=['tm'])
    tm = tm.T
    stim = (AA01['history_stim'])[0]
    stim = pd.DataFrame(stim, columns=['stim'])
    stim = pd.DataFrame(stim, columns=['stim'])
    stim = stim.T
    side = (AA01['history_side'])[0]
    side = pd.DataFrame(side, columns=['side'])
    side = side.T
    hits = (AA01['history_hits'])[0]
    hits = pd.DataFrame(hits, columns=['hits'])
    hits = hits.T
    list = [vio, tm, stim, side, hits]
    history = pd.concat(list)

    history['righty'] = (history['side'] == 114) & (history['hits'] == 1)
    history['righty'].sum()


    return history

create_sc_df()
#file_name = 'data_@SoundCategorization_athena_AA01_190807a.mat'
#create_rat_dict(file_name=file_name, data_folder=data_folder)

