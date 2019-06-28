# ==============================================================================
# Autor: Joseph Jiang
#
# cda_train.py:  training model from bmp file.
#
# ==============================================================================

import numpy as NP
import os as OS
import tensorflow as tf

import cda_model as CDA_MODEL
import cda_data as CDA_DATA

# train selection define in the model.py
#---------------------------------------------
# CDA_TRAIN_S1_ONLY = 1
# CDA_TRAIN_S2_ONLY = 2
# CDA_TRAIN_S3_ONLY = 3
# CDA_TRAIN_S4_ONLY = 4
# CDA_TRAIN_S3_S4   = 5
# CDA_TRAIN_ALL = 6
#---------------------------------------------

#---------------------------------------------
# modi TRAIN_NUM in cda_model.py  set the train times.
#---------------------------------------------
def cda_model_train():
  flag1 = not OS.path.exists(CDA_DATA.train_data_dir)
  flag2 = not OS.path.exists(CDA_DATA.preprocess_data_dir)
  flag3 = not OS.path.exists(CDA_DATA.preprocess_data_dir_lr)
  flag4 = not OS.path.exists(CDA_DATA.preprocess_data_dir_hr)
  if flag1 or flag2  or flag3  or flag4 : 
    print("Error: No train data require...")
    print("please run cda.init.py first....")
    return

  #train_step_select = CDA_MODEL.CDA_TRAIN_ALL
  #train_step_select = CDA_MODEL.CDA_TRAIN_S1_ONLY
  #train_step_select = CDA_MODEL.CDA_TRAIN_S2_ONLY
  #train_step_select = CDA_MODEL.CDA_TRAIN_S3_ONLY
  #train_step_select = CDA_MODEL.CDA_TRAIN_S4_ONLY
  train_step_select = CDA_MODEL.CDA_TRAIN_S3_S4

  print("---------------cda----train---------begin---------------")
  print("it will take some times, please wait ...")
  CDA_MODEL.cda_train(train_step_select)
  print("---------------cda----train---------end-----------------")


#run from here.
cda_model_train()

