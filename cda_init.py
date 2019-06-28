# ==============================================================================
# Autor: Joseph Jiang
#
# cda_input.py:  read training data from bmp file.
#
# ==============================================================================

import numpy as NP
import os as OS
import tensorflow as tf

import cda_data as CDA_DATA

print("---------------cda----init---------begin---------------")
print("it will take some times, please wait ...")
CDA_DATA.gen_train_data_batch_npy_file_all()
print("---------------cda----init---------end-----------------")
