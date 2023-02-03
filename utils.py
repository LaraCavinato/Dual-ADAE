import pathlib
import pandas as pd
import numpy as np
import os


# Setting up directories
base = pathlib.Path(__file__).parent

csv_path = base + 'DATA/'
if not os.path.isdir(csv_path):
    os.makedirs(csv_path)

param_path = base + 'PARAMS/'
if not os.path.isdir(param_path):
    os.makedirs(param_path)

adv_path_dual = base + 'ADV_FILES_DUAL/'
if not os.path.isdir(adv_path_dual):
    os.makedirs(adv_path_dual)




