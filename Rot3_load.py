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
                     'hits_total']

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
                       'A2_time': rat_values['SideSection_A2_time'],
                       'reward_type': rat_values['SideSection_reward_type'],
                       'violations': rat_values['OverallPerformanceSection_violation_rate'],
                       'timeouts': rat_values['OverallPerformanceSection_timeout_rate'],
                       'hits_left': rat_values['OverallPerformanceSection_Left_hit_frac'],
                       'hits_right': rat_values['OverallPerformanceSection_Right_hit_frac'],
                       'hits_total': rat_values['OverallPerformanceSection_hit_frac']}

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


def create_all_dfs():
    # this is the temporary solution
    animal_folder1 = r'H:\ratter\SoloData\Data\athena\AA01'
    animal_folder2 = r'H:\ratter\SoloData\Data\athena\AA02'
    animal_folder3 = r'H:\ratter\SoloData\Data\athena\AA03'
    animal_folder4 = r'H:\ratter\SoloData\Data\athena\AA04'
    animal_folder5 = r'H:\ratter\SoloData\Data\athena\AA05'
    animal_folder6 = r'H:\ratter\SoloData\Data\athena\AA06'
    animal_folder7 = r'H:\ratter\SoloData\Data\athena\AA07'
    animal_folder8 = r'H:\ratter\SoloData\Data\athena\AA08'

    rat_df_list = []

    AA01_full_df = whole_animal_df(animal_folder1)
    rat_df_list.append(AA01_full_df)
    AA02_full_df = whole_animal_df(animal_folder2)
    rat_df_list.append(AA02_full_df)
    AA03_full_df = whole_animal_df(animal_folder3)
    rat_df_list.append(AA03_full_df)
    AA04_full_df = whole_animal_df(animal_folder4)
    rat_df_list.append(AA04_full_df)
    AA05_full_df = whole_animal_df(animal_folder5)
    rat_df_list.append(AA05_full_df)
    AA06_full_df = whole_animal_df(animal_folder6)
    rat_df_list.append(AA06_full_df)
    AA07_full_df = whole_animal_df(animal_folder7)
    rat_df_list.append(AA07_full_df)
    AA08_full_df = whole_animal_df(animal_folder8)
    rat_df_list.append(AA08_full_df)

    animal_folderD1 = r'H:\ratter\SoloData\Data\dammy\DO01'
    animal_folderD2 = r'H:\ratter\SoloData\Data\dammy\DO02'
    animal_folderD3 = r'H:\ratter\SoloData\Data\dammy\DO03'
    animal_folderD4 = r'H:\ratter\SoloData\Data\dammy\DO04'
    animal_folderD5 = r'H:\ratter\SoloData\Data\dammy\DO05'
    animal_folderD6 = r'H:\ratter\SoloData\Data\dammy\DO06'
    animal_folderD7 = r'H:\ratter\SoloData\Data\dammy\DO07'
    animal_folderD8 = r'H:\ratter\SoloData\Data\dammy\DO08'

    DO01_full_df = whole_animal_df(animal_folderD1)
    rat_df_list.append(DO01_full_df)
    DO02_full_df = whole_animal_df(animal_folderD2)
    rat_df_list.append(DO02_full_df)
    DO03_full_df = whole_animal_df(animal_folderD3)
    rat_df_list.append(DO03_full_df)
    DO04_full_df = whole_animal_df(animal_folderD4)
    rat_df_list.append(DO04_full_df)
    DO05_full_df = whole_animal_df(animal_folderD5)
    rat_df_list.append(DO05_full_df)
    DO06_full_df = whole_animal_df(animal_folderD6)
    rat_df_list.append(DO06_full_df)
    DO07_full_df = whole_animal_df(animal_folderD7)
    rat_df_list.append(DO07_full_df)
    DO08_full_df = whole_animal_df(animal_folderD8)
    rat_df_list.append(DO08_full_df)

    animal_folderS1 = r'H:\ratter\SoloData\Data\sharbat\SC01'
    animal_folderS2 = r'H:\ratter\SoloData\Data\sharbat\SC02'
    animal_folderS3 = r'H:\ratter\SoloData\Data\sharbat\SC03'
    animal_folderS4 = r'H:\ratter\SoloData\Data\sharbat\SC04'
    animal_folderS5 = r'H:\ratter\SoloData\Data\sharbat\SC05'
    animal_folderS6 = r'H:\ratter\SoloData\Data\sharbat\SC06'

    SC01_full_df = whole_animal_df(animal_folderS1)
    rat_df_list.append(SC01_full_df)
    SC02_full_df = whole_animal_df(animal_folderS2)
    rat_df_list.append(SC02_full_df)
    SC03_full_df = whole_animal_df(animal_folderS3)
    rat_df_list.append(SC03_full_df)
    SC04_full_df = whole_animal_df(animal_folderS4)
    rat_df_list.append(SC04_full_df)
    SC05_full_df = whole_animal_df(animal_folderS5)
    rat_df_list.append(SC05_full_df)
    SC06_full_df = whole_animal_df(animal_folderS6)
    rat_df_list.append(SC06_full_df)

    animal_folderV1 = r'H:\ratter\SoloData\Data\viktor\VP01'
    animal_folderV2 = r'H:\ratter\SoloData\Data\viktor\VP02'
    animal_folderV3 = r'H:\ratter\SoloData\Data\viktor\VP03'
    animal_folderV4 = r'H:\ratter\SoloData\Data\viktor\VP04'
    animal_folderV5 = r'H:\ratter\SoloData\Data\viktor\VP05'
    animal_folderV6 = r'H:\ratter\SoloData\Data\viktor\VP06'
    animal_folderV7 = r'H:\ratter\SoloData\Data\viktor\VP07'
    animal_folderV8 = r'H:\ratter\SoloData\Data\viktor\VP08'

    VP01_full_df = whole_animal_df(animal_folderV1)
    rat_df_list.append(VP01_full_df)
    VP02_full_df = whole_animal_df(animal_folderV2)
    rat_df_list.append(VP02_full_df)
    VP03_full_df = whole_animal_df(animal_folderV3)
    rat_df_list.append(VP03_full_df)
    VP04_full_df = whole_animal_df(animal_folderV4)
    rat_df_list.append(VP04_full_df)
    VP05_full_df = whole_animal_df(animal_folderV5)
    rat_df_list.append(VP05_full_df)
    VP06_full_df = whole_animal_df(animal_folderV6)
    rat_df_list.append(VP06_full_df)
    VP07_full_df = whole_animal_df(animal_folderV7)
    rat_df_list.append(VP07_full_df)
    VP08_full_df = whole_animal_df(animal_folderV8)
    rat_df_list.append(VP08_full_df)

    return rat_df_list


def save_dataframe(dataframe, name = 'Rat_full_df'):
    # Save large dataframe in project
    with open("Rot3_data\\" + name + ".pkl", "wb") as f:
        pickle.dump(dataframe, f)


#rat_df_list = create_all_dfs()
#Rat_full = pd.concat(rat_df_list)
#save_dataframe(Rat_full, name='Rat_full_df')

rat_df_list2 = create_all_dfs()
Rat_full2 = pd.concat(rat_df_list2)
save_dataframe(Rat_full2, name='Rat_full2_df')

# Create full df containing all animals from all experimenter folder
# core working directory
#work_dir = r'H:\ratter\SoloData\Data'
# might want to skip the experimenter folder dosen't look like content is super important

#pickle_out = open("Animal.pickle","wb")
#pickle.dump(AA01_full_df, pickle_out)

# create first rat df
#rat = create_rat_dict()
#rat_frame = create_df_from_dict(rat)

# create second rat df
#file_name2 = 'data_@AthenaDelayComp_athena_AA01_190510a.mat'
#rat2 = create_rat_dict(file_name2)
#rat2_frame = create_df_from_dict(rat2)

# create third rat df
#file_name3 = 'data_@AthenaDelayComp_athena_AA01_190514a.mat'
#rat3 = create_rat_dict(file_name3)
#rat3_frame = create_df_from_dict(rat3)

# concatenate
# this works for now but I don't know how it deals with missing data, being different lengths.
#rat_co = pd.concat([rat_frame,rat2_frame])




