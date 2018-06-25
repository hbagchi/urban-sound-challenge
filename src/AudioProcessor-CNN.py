"""
Urban Sound Challenge - Sound Classification using Convolutional Neural Network (CNN)
@author: - Hitesh Bagchi
Version: 1.0
Date: 25th June 2018
"""
#--------------------------------------------------------------------------------------------------
# Python libraries import
#--------------------------------------------------------------------------------------------------
from pathlib import Path

# %%

data_dir = "./data/trainShort"

def extract_features(data_dir, bands = 60, frames = 41, file_ext="*.wav"):
    pathlist = Path(data_dir).glob(file_ext)
    for file_path in pathlist:
        print(file_path)

# %%
extract_features(data_dir)