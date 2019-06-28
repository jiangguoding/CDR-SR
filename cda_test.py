# ==============================================================================
# Autor: Joseph Jiang
#
# cda_test.py:  test model from set5/set14 file.
#
# ==============================================================================

import numpy as NP
import os as OS
import tensorflow as tf

import cda_model as CDA_MODEL
import cda_data as CDA_DATA

def cda_model_test():
  model_file1 = CDA_MODEL.model_test_file + ".index"
  model_file2 = CDA_MODEL.model_test_file + ".meta"
  if not OS.path.exists(model_file1) or not OS.path.exists(model_file2):
    model_dir  = OS.getcwd() + "/model/test"
    print("---------------cda----test---------Fail---------------")
    print(CDA_MODEL.model_test_file, " do not exist!  please run  cda_train.py first")
    print("After training,  please copy the train file to dir : ",  model_dir)
    print("------------------------------------------------------")
    return 

  print("---------------cda----test---------begin---------------")
  CDA_MODEL.cda_test()
  print("---------------cda----test---------end-----------------")

#run from here.
cda_model_test()

