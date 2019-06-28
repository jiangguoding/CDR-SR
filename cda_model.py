# ==============================================================================
# Autor: Joseph Jiang
#
# cda_model.py:  cda train model definition 
#                cda test model definition
#
# ==============================================================================

import argparse
import sys
import math
import tensorflow as tf
import cda_data as CDA_DATA
import os as OS



#------------------------------------------------------------------------
#  Training model Parameter define
#------------------------------------------------------------------------
FLAGS = None
UP_SCALE = 3
YL_UNITS = 243  # 81 * 3
HL_UNITS =  (int)(YL_UNITS * 2.5)
YH_UNITS = 243
HH_UNITS =  (int)(YH_UNITS * 2.5)
TRAIN_GRADIENT = 0.01


#27000 <--->  1 Hour
#650000 ------> 24 Hour
TRAIN_NUM = 400000
TRAIN_SAVE_DATA_NUM = 10000

#DEBUG
TRAIN_PRINT_NUM = 10

weights = {
  #step1 weights 
  'w1' : tf.Variable(tf.truncated_normal([YL_UNITS, HL_UNITS], stddev=0.1)),
  'wt1' : tf.Variable(tf.truncated_normal([HL_UNITS, YL_UNITS], stddev=0.1)),

  #step2 weights  
  'wt3' : tf.Variable(tf.truncated_normal([YH_UNITS,HH_UNITS], stddev=0.1)),
  'w3' : tf.Variable(tf.truncated_normal([HH_UNITS, YH_UNITS],  stddev=0.1)),

  #step3 weights  
  'w2' : tf.Variable(tf.truncated_normal([HL_UNITS, HH_UNITS],  stddev=0.1))
}

biases = {
  #step1 biases 
  'b1' : tf.Variable(tf.zeros([HL_UNITS])),
  'bt1' : tf.Variable(tf.zeros([YL_UNITS])),

  #step2 biases 
  'bt3' : tf.Variable(tf.zeros([HH_UNITS])),
  'b3' : tf.Variable(tf.zeros([YH_UNITS])),

   #step3 biases 
   'b2' : tf.Variable(tf.zeros([HH_UNITS]))
}

#if training model,  Model save to model_train_file
model_train_file = OS.getcwd() + "/model/train/cda_model.ckpt" 

#after training model,  copy the model data to model_test_file for testing
model_test_file  = OS.getcwd() + "/model/test/cda_model.ckpt" 

#------------------------------------------------------------------------
#  training model Parameter save and restore 
#------------------------------------------------------------------------


#  training model step select define. 
#------------------------------------------------------------------------
CDA_TRAIN_S1_ONLY = 1
CDA_TRAIN_S2_ONLY = 2
CDA_TRAIN_S3_ONLY = 3
CDA_TRAIN_S4_ONLY = 4
CDA_TRAIN_S3_S4   = 5
CDA_TRAIN_ALL = 6


#step1 model_parameter
def get_cda_train_model_saver_and_filename(step):
  train_saver = ""
  file_name = ""
  if step == CDA_TRAIN_S1_ONLY: 
    file_name = model_train_file + ".onlystep1"
    train_saver = tf.train.Saver([weights['w1'], biases['b1'], weights['wt1'], biases['bt1']])  
  if step == CDA_TRAIN_S2_ONLY:
    file_name = model_train_file + ".onlystep2"
    train_saver = tf.train.Saver([weights['wt3'], biases['bt3'], weights['w3'], biases['b3']])  
  if step == CDA_TRAIN_S3_ONLY or step == CDA_TRAIN_S3_S4:
    file_name = model_train_file + ".onlystep3"
    train_saver = tf.train.Saver() 
  if step == CDA_TRAIN_ALL:
    file_name = model_train_file
    train_saver = tf.train.Saver()

  #print("get_cda_train_model_saver_and_filename---", file_name)
  return train_saver, file_name

def cda_train_save_model_parameter_step_i(sess, step):
  train_saver, file_name = get_cda_train_model_saver_and_filename(step) 
  if train_saver != "" :
    train_saver.save(sess, file_name)

def cda_train_restore_model_parameter_step(sess, step):  
  train_saver, file_name = get_cda_train_model_saver_and_filename(step)
  file_name1 = file_name + ".index"
  if OS.path.exists(file_name1):
    print("cda_train_restore_model_parameter---", file_name)
    train_saver.restore(sess,  file_name)

def cda_train_save_model_parameter_step(sess, train_select_flag, step):  
    #print("--------", train_select_flag, step)
   
    #step4: save all model paramters.
    if step == 4:  
      cda_train_save_model_parameter_step_i(sess, CDA_TRAIN_ALL)
      return

    #For other step:
    # CDA_TRAIN_ALL: save checkpoint for all check...
    # ONLY:  save to only.
    if train_select_flag == CDA_TRAIN_ALL:  
      savefilepath = model_train_file + ".s" + str(step)
      cda_saver = tf.train.Saver()
      savefilepath = cda_saver.save(sess, savefilepath)
      print("savefilepath = ",  savefilepath) 
    else : 
      cda_train_save_model_parameter_step_i(sess, train_select_flag)




##############################################################################################
#     cda model training begin
##############################################################################################


#  training model define. 
#------------------------------------------------------------------------
def cda_train(train_select_flag):

  print("--------------------------cda--train--begin---------------------------------------------")  

  train_run_num = TRAIN_NUM
  train_print_num = TRAIN_PRINT_NUM
  #train_run_num = 1


  #------------------------------------------------------------------------
  #  LR training  step1 model define
  #------------------------------------------------------------------------
  print("-------------------------step1--------------------------------------------------")
  # Create lr  encode model 
  yl = tf.placeholder(tf.float32, [None, YL_UNITS])
  hl = tf.nn.sigmoid(tf.matmul(yl, weights['w1']) + biases['b1'])

   # Create lr  decode model
  yl_cal = tf.nn.sigmoid(tf.matmul(hl, weights['wt1']) + biases['bt1'])
  
  # Define loss and optimizer
  mse_yl = tf.reduce_sum(tf.square(yl_cal -  yl))
  train_step1 = tf.train.GradientDescentOptimizer(TRAIN_GRADIENT).minimize(mse_yl)


  #------------------------------------------------------------------------
  #  HR training  step2 model define
  #------------------------------------------------------------------------
  print("--------------------------step2-------------------------------------------------")
  # Create hr  encode model 
  yh = tf.placeholder(tf.float32, [None, YH_UNITS])
  hh = tf.nn.sigmoid(tf.matmul(yh, weights['wt3']) + biases['bt3'])

  # Create hr  decode model
  yh_cal = tf.nn.sigmoid(tf.matmul(hh, weights['w3']) + biases['b3'])
  
  # Define loss and optimizer
  mse_yh = tf.reduce_sum(tf.square(yh_cal - yh))
  train_step2 = tf.train.GradientDescentOptimizer(TRAIN_GRADIENT).minimize(mse_yh)

  #------------------------------------------------------------------------
  #  maping training step3  HL--->HH model define
  #------------------------------------------------------------------------
  print("--------------------------step3-------------------------------------------------")
  # create mapping model
  hl_in = tf.placeholder(tf.float32, [None, HL_UNITS])
  hh_out = tf.placeholder(tf.float32, [None, HH_UNITS])

  hh_cal = tf.nn.sigmoid(tf.matmul(hl_in,  weights['w2']) + biases['b2'])
 
  # Define loss and optimizer
  mse_maping = tf.reduce_sum(tf.square(hh_out - hh_cal))
  train_step3 = tf.train.GradientDescentOptimizer(TRAIN_GRADIENT).minimize(mse_maping)

  #------------------------------------------------------------------------
  #  fine-tuning step4 model define
  #------------------------------------------------------------------------
  print("-----------------------step4----------------------------------------------------")
  # create mapping model
  hh_cal_m = tf.nn.sigmoid(tf.matmul(hl, weights['w2']) + biases['b2'])
  x_cal_m = tf.nn.sigmoid(tf.matmul(hh_cal_m, weights['w3']) + biases['b3'])
 
  # Define loss and optimizer
  mse_x = tf.reduce_sum(tf.square(yh - x_cal_m))
  train_step4 = tf.train.GradientDescentOptimizer(TRAIN_GRADIENT).minimize(mse_x)

  #------------------------------------------------------------------------
  #  define sess,  init variable, prepare to run.
  #------------------------------------------------------------------------
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  #obeserve: 
  #merged_summary_op = tf.merge_all_summaries()
  #summary_writer = tf.train.SummaryWriter('/home/tcl/tensor/src/test/bmp/board', sess.graph)


  #------------------------------------------------------------------------
  #  Saver or restore the previous model/variable 
  #------------------------------------------------------------------------
  cda_saver = tf.train.Saver(max_to_keep=10)
  model_dir = OS.getcwd() + "/model"
  if not OS.path.exists(model_dir):
    OS.mkdir(model_dir)  

  model_dir = model_dir + "/train"
  if not OS.path.exists(model_dir):
    OS.mkdir(model_dir)  

  print("model_train_file = ", model_train_file)
  
  # previous training file exist, then restore, and continue to train.
  cda_train_restore_model_parameter_step(sess, CDA_TRAIN_ALL)

  #if seperate training model exist,  load step-only model parameter
  #step only parameter loading will not affect the others.
  cda_train_restore_model_parameter_step(sess, CDA_TRAIN_S1_ONLY)
  cda_train_restore_model_parameter_step(sess, CDA_TRAIN_S2_ONLY)
  cda_train_restore_model_parameter_step(sess, CDA_TRAIN_S3_ONLY)

  #------------------------------------------------------------------------
  #  LR training  step1 RUN
  #------------------------------------------------------------------------
  if train_select_flag == CDA_TRAIN_S1_ONLY or train_select_flag == CDA_TRAIN_ALL:
    print("--------------------------step1--train-----------------------------------------------")
    # Train
    for i in range(train_run_num):
      batch_yl = CDA_DATA.load_train_batch_random_lr()
      sess.run(train_step1, feed_dict={yl:batch_yl})  
      mse_yl_res, _ = sess.run([mse_yl, train_step1], feed_dict={yl:batch_yl})  
      if i % train_print_num == 0:
        print("mse_yl_res = ", mse_yl_res, "--num---",  i)  
      if i % TRAIN_SAVE_DATA_NUM == 0:
        cda_train_save_model_parameter_step(sess, train_select_flag, 1) 
   
  
    # Test trained model
    #mse_yl_res = sess.run(mse_yl,  feed_dict={yl:batch_yl})
    #print("mse_yl_res = ", mse_yl_res)
    
    # save checkpoint s1
    cda_train_save_model_parameter_step(sess, train_select_flag, 1)  

  #------------------------------------------------------------------------
  #  HR training  step2 RUN
  #------------------------------------------------------------------------
  if train_select_flag == CDA_TRAIN_S2_ONLY or train_select_flag == CDA_TRAIN_ALL:
    print("--------------------------step2--train-----------------------------------------------")
    # Train
    for i in range(train_run_num):
      batch_yh = CDA_DATA.load_train_batch_random_lr()
      #sess.run(train_step2, feed_dict={yh:batch_yh})
      mse_yh_res, _ = sess.run([mse_yh, train_step2], feed_dict={yh:batch_yh})
      if i % train_print_num == 0:
        print("mse_yh_res = ", mse_yh_res, "--num---",  i)  
      if i % TRAIN_SAVE_DATA_NUM == 0:
         cda_train_save_model_parameter_step(sess, train_select_flag, 2) 

    # Test trained model
    #mse_yh_res = sess.run(mse_yh,  feed_dict={yh:batch_yh})
    #print("mse_yh_res = ", mse_yh_res) 

    # save checkpoint s1
    cda_train_save_model_parameter_step(sess, train_select_flag, 2) 
  

  #------------------------------------------------------------------------
  #  maping training step3  HL--->HH run
  #------------------------------------------------------------------------
  if train_select_flag == CDA_TRAIN_S3_ONLY or train_select_flag == CDA_TRAIN_ALL or train_select_flag == CDA_TRAIN_S3_S4:
    print("--------------------------step3--train-----------------------------------------------")
    # Train
    for i in range(train_run_num):
      batch_yl, batch_yh = CDA_DATA.load_train_batch_random_lr_and_hr()
      hl_cal3 = sess.run(hl, feed_dict={yl:batch_yl})
      hh_cal3 = sess.run(hh, feed_dict={yh:batch_yh})
      #print("--hl_cal3--", hl_cal3.shape)
      #print("--hh_cal3--", hh_cal3.shape)
      #sess.run(train_step3, feed_dict={hl_in:hl_cal3, hh_out:hh_cal3})
      mse_maping_res, _ = sess.run([mse_maping, train_step3], feed_dict={hl_in:hl_cal3, hh_out:hh_cal3}) 
      if i % train_print_num == 0:
        print("mse_maping_res = ", mse_maping_res, "--num---",  i)
      if i % TRAIN_SAVE_DATA_NUM == 0:
         cda_train_save_model_parameter_step(sess, train_select_flag, 3) 

    # Test trained model
    mse_maping_res = sess.run(mse_maping, feed_dict={hl_in:hl_cal3, hh_out:hh_cal3})
    print("mse_maping_res = ", mse_maping_res)

    # save checkpoint s3
    cda_train_save_model_parameter_step(sess, train_select_flag, 3) 

  #------------------------------------------------------------------------
  #  fine-tuning step4 run
  #------------------------------------------------------------------------
  if train_select_flag == CDA_TRAIN_S4_ONLY or train_select_flag == CDA_TRAIN_ALL or train_select_flag == CDA_TRAIN_S3_S4:
    print("--------------------------step4--train-----------------------------------------------")
    # Train
    for i in range(train_run_num):
      batch_yl, batch_yh = CDA_DATA.load_train_batch_random_lr_and_hr()
      #sess.run(train_step4, feed_dict={yl:batch_yl, yh:batch_yh})
      mse_x_res, _ = sess.run([mse_x, train_step4], feed_dict={yl:batch_yl, yh:batch_yh})
      if i % train_print_num == 0:
        print("mse_x_res = ", mse_x_res, "--num---",  i)
      if i % TRAIN_SAVE_DATA_NUM == 0:
        cda_train_save_model_parameter_step(sess, train_select_flag, 4) 

    # save checkpoint s4
    cda_train_save_model_parameter_step(sess, train_select_flag, 4)   
  

  print("--------------------------cda--train--end---------------------------------------------")


##############################################################################################
#    cda model training end
##############################################################################################


##############################################################################################
#    cda model test  begin
##############################################################################################

def cda_test():  
  model_dir = OS.getcwd() + "/model/test"
  model_file1 = model_test_file + ".index"
  if not OS.path.exists(model_file1):
    print(model_test_file, " do not exist!  please run  cda_train.py first")
    print("After training,  please copy the train file to dir: ",  model_dir)
    return 

  print("-------------------------cda_test---init-----------------------------------------------")

  #------------------------------------------------------------------------
  #  define cda model
  #------------------------------------------------------------------------ 
  yl_t = tf.placeholder(tf.float32, [None, YH_UNITS])
  hl_t = tf.nn.sigmoid(tf.matmul(yl_t, weights['w1']) + biases['b1'])
  hh_t = tf.nn.sigmoid(tf.matmul(hl_t,  weights['w2']) + biases['b2'])
  x_t = tf.nn.sigmoid(tf.matmul(hh_t, weights['w3']) + biases['b3'])

   
  #------------------------------------------------------------------------
  #  define sess,  init variable, prepare to run.
  #------------------------------------------------------------------------
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  #------------------------------------------------------------------------
  #  Saver: define saver for model/variable restore.
  #------------------------------------------------------------------------
  cda_saver = tf.train.Saver()  
  #print("-cda_test--w1--", sess.run(weights['w1']))
  cda_saver.restore(sess, model_test_file)
  #print("-cda_test--w1--", sess.run(weights['w1']))


  #------------------------------------------------------------------------
  #  cda generate set5 test file
  #------------------------------------------------------------------------
  print("-------------Process---test---set5---files-----------------")
  CDA_DATA.test_data_set5_init()
  file_num = CDA_DATA.get_test_file_num()
  #file_num = 0
  for i in range(file_num):
    #No overlap output
    batch_yl = CDA_DATA.get_test_file_set(i)  
    result = sess.run(x_t, feed_dict={yl_t:batch_yl})
    CDA_DATA.save_cda_gen_data_to_bmp_file(result)
    #CDA_DATA.save_cda_gen_data_to_bmp_file(batch_yl)

    #overlapped output for comparation
    batch_yl = CDA_DATA.get_test_file_set_overlap(i) 
    result = sess.run(x_t, feed_dict={yl_t:batch_yl}) 
    CDA_DATA.save_cda_gen_data_to_bmp_file_overlap(result)
    #CDA_DATA.save_cda_gen_data_to_bmp_file_overlap(batch_yl)

  #------------------------------------------------------------------------
  #  cda generate set14 test file
  #------------------------------------------------------------------------
  print("-------------Process---test---set14---files-----------------")
  CDA_DATA.test_data_set14_init()
  file_num = CDA_DATA.get_test_file_num()
  #file_num = 0
  for i in range(file_num):
    #No overlap output
    batch_yl = CDA_DATA.get_test_file_set(i)  
    result = sess.run(x_t, feed_dict={yl_t:batch_yl})
    CDA_DATA.save_cda_gen_data_to_bmp_file(result)
    #CDA_DATA.save_cda_gen_data_to_bmp_file(batch_yl)

    #overlapped output for comparation
    batch_yl = CDA_DATA.get_test_file_set_overlap(i)  
    result = sess.run(x_t, feed_dict={yl_t:batch_yl}) 
    CDA_DATA.save_cda_gen_data_to_bmp_file_overlap(result)
    #CDA_DATA.save_cda_gen_data_to_bmp_file_overlap(batch_yl)


  print("-------------cda_test-----------------") 



##############################################################################################
#    cda model test  end
##############################################################################################


