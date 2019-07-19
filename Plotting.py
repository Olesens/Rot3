import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import ipython_genutils


# load in file from Rot3_data folder using pickle
pickle_in = open("Rot3_data\\Animal_df.pkl","rb")
Animal_df = pickle.load(pickle_in)
Animal_df = Animal_df.sort_index()  # sort according to index, this should sort according to date


