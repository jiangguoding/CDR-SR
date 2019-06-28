# ==============================================================================
# Autor: Joseph Jiang
#
# cda_data.py:  data management
#               1.  train data:  preprocess ,  batch API
#               2.  test data:   set5 and set14,  test data API
#               3.  cda  generate data:   gen data according to 2
# ==============================================================================
from PIL import Image
import numpy as NP
import os as OS
import tensorflow as tf
import random as Rand

# define color format to be used during init,  training,  test.
# 1: RGB  2: YCbCr  3:  YCbCr,  but Y for training only.
CDA_COLOR_FORMAT = 1


#----------------------------------------------------------------------------------------
#  BMP file operation begin.
#----------------------------------------------------------------------------------------

#Get 9*9 block data
#---------------------------------------------------------
def getdatafromarray(img_arr, row_b, col_b, row_num, col_num):
    # row * col * 1,  ele contains RGB
    red_arr = NP.zeros(81, 'uint8');
    green_arr = NP.zeros(81, 'uint8');
    blue_arr = NP.zeros(81, 'uint8');
    
    row_start = row_b * 9
    row_end = row_start + row_num
    col_start = col_b * 9
    col_end = col_start + col_num
    index = 0;
    for i in range(row_start, row_end):
        for j in range(col_start,  col_end):
            color = img_arr[i][j]  # (R, G, B)
            #print("------------color----", color)
            red_arr[index] = color[0]
            green_arr[index] = color[1]
            blue_arr[index] = color[2]
             
            index += 1

    #print("red_arr==", red_arr.shape,  red_arr)
    #print("green_arr==", red_arr.shape, green_arr)
    #print("blue_arr==", red_arr.shape, blue_arr)
    return red_arr,  green_arr,  blue_arr


#Get  9*9 block data list from one bmp file
#flag:  1: LR  2: HR
#CDA_COLOR_FORMAT:  1 RGB  2: YCbCr
#---------------------------------------------------------
def getbmparray_orig(filename, flag):
    im_rgb = Image.open(filename) # 读取图片
    #print("getbmparray_orig---",filename, im_rgb.width, im_rgb.height)
   
    #print("getbmparray_orig---mode------", im_rgb.mode,  flag)
    #print("getbmparray_orig--CDA_COLOR_FORMAT=", CDA_COLOR_FORMAT)    

    im = im_rgb
    if CDA_COLOR_FORMAT == 2 or CDA_COLOR_FORMAT == 3: 
      im = im_rgb.convert("YCbCr")
      #print("-getbmparray_orig-converted mode------", im.mode,  flag)

    #init variable 
    img_arr = ""

    #print("mode------", im.mode,  flag)

    #Get HR image to array
    if flag == 2 :
       img_arr = NP.array(im)
       #print("hr_img_arr---", img_arr.shape, img_arr.dtype)
    else:
        orgsize = im.width, im.height
        scalesize = im.width * 3, im.height * 3
        im.resize(scalesize)
        im.resize(orgsize, Image.BICUBIC)

        #Get lR image to array
        img_arr = NP.array(im)
        #print("lr_img_arr---", img_arr.shape, img_arr.dtype)
   
    im.close()
 
    return img_arr


def readfromfile_rgb(filename, flag, callback = ''):
 
    #init variable 
    img_arr = ""
    rb_num = 0
    cb_num = 0
    block_num = 0

    img_arr = getbmparray_orig(filename, flag)

    #print("lr_img_arr---", img_arr.shape, img_arr.dtype)

    #cal block to read.
    rb_num = int(img_arr.shape[0]/9)
    cb_num = int(img_arr.shape[1]/9)
    block_num = int(rb_num * cb_num)

    #print("rb_num=", rb_num,  "---cb_num=", cb_num)  
    #print("img_arr---read-", img_arr)  

    shape_result = [block_num, 243]  # 81 * 3
    result_data = NP.zeros(shape_result, 'uint8');
    block_count = 0;
    for rr in range(0, rb_num):
        for cc in range(0, cb_num):           
           red, green, blue = getdatafromarray(img_arr, rr, cc, 9, 9)   
           #color preprocesing
           for kk in range(81): 
              ll = kk * 3
              result_data[block_count][ll] = red[kk]
              result_data[block_count][ll+1] = green[kk]
              result_data[block_count][ll+2] = blue[kk]  

           block_count += 1

       
    if callback != '' :
        callback(rb_num, cb_num)

    #print("result_data-----", result_data)

    return result_data


# 1:  yes  0 : no
def is_bmp_file(filename):

    fnamelen = len(filename)
    if fnamelen < 4 :
      return 0

    file_name_ext = filename[fnamelen-4:fnamelen]
    file_name_ext.lower()

    #print("fnamelen = ", file_name_ext)
    if file_name_ext != ".bmp" :
      #print("file ext ...")
      return 0
    
    im_rgb = Image.open(filename) # 读取图片
    #print("getbmparray_orig---mode------", im_rgb.mode)

    if im_rgb.mode != "RGB" :
       #print("EEEE")
       return 0

    im_rgb.close()

    return 1 
    


#--------------------------------------------------------------------
# new add for overlaping process...
#--------------------------------------------------------------------
BMP_OVERLAPPED_PIX_NUM = 3
BMP_OVERLAPPED_POS_COEF = 6

def getdatafromarray_overlap(img_arr, row_b, col_b, row_num, col_num):
    # row * col * 1,  ele contains RGB
    red_arr = NP.zeros(81, 'uint8');
    green_arr = NP.zeros(81, 'uint8');
    blue_arr = NP.zeros(81, 'uint8');
    
    row_start = row_b * BMP_OVERLAPPED_POS_COEF
    row_end = row_start + row_num
    col_start = col_b * BMP_OVERLAPPED_POS_COEF
    col_end = col_start + col_num
    index = 0;
    for i in range(row_start, row_end):
        for j in range(col_start,  col_end):
            color = img_arr[i][j]  # (R, G, B)
            #print("------------color----", color)
            red_arr[index] = color[0]
            green_arr[index] = color[1]
            blue_arr[index] = color[2]
            index += 1

    #print("red_arr==", red_arr.shape,  red_arr)
    #print("green_arr==", red_arr.shape, green_arr)
    #print("blue_arr==", red_arr.shape, blue_arr)
    return red_arr,  green_arr,  blue_arr


#Get  9*9 block data list from one bmp file
#flag:  1: LR  2: HR
#CDA_COLOR_FORMAT:  1 RGB  2: YCbCr
#---------------------------------------------------------
def readfromfile_rgb_overlap(filename, flag, callback = ''):

    #init variable 
    img_arr = ""
    rb_num = 0
    cb_num = 0
    block_num = 0

    img_arr = getbmparray_orig(filename, flag)
    #print("lr_img_arr---", img_arr.shape, img_arr.dtype)

    #cal block to read.
    rb_num = int((img_arr.shape[0] - 3)/BMP_OVERLAPPED_POS_COEF)
    cb_num = int((img_arr.shape[1] - 3)/BMP_OVERLAPPED_POS_COEF)
    block_num = int(rb_num * cb_num)

    #print("rb_num=", rb_num,  "---cb_num=", cb_num)  
    #print("img_arr---read-", img_arr)  

    shape_result = [block_num, 243]  # 81 * 3
    result_data = NP.zeros(shape_result, 'uint8');
    block_count = 0;
    for rr in range(0, rb_num):
        for cc in range(0, cb_num):           
           red, green, blue = getdatafromarray_overlap(img_arr, rr, cc, 9, 9)   
           #color preprocesing
           for kk in range(81): 
              ll = kk * 3
              result_data[block_count][ll] = red[kk]
              result_data[block_count][ll+1] = green[kk]
              result_data[block_count][ll+2] = blue[kk]  

           block_count += 1

    #im.close()
    
    if callback != '' :
        callback(rb_num, cb_num)

    #print("result_data-----", result_data)

    return result_data




#----------------------------------------------------------------------------------------
#  BMP file operation end.
#----------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------
#  Training batch data Generated begin
#----------------------------------------------------------------------------------------

#Global variable define
#---------------------------------------------------------
train_data_dir = OS.getcwd() + "/data/training/"
train_filelist = OS.listdir(train_data_dir)
train_file_num = len(train_filelist)
train_file_index = 0
preprocess_data_dir = OS.getcwd() + "/data/preprocess"
preprocess_data_dir_lr = preprocess_data_dir + "/lr"
preprocess_data_dir_hr = preprocess_data_dir + "/hr"
#for Generate batch npy data
g_batch_train_data_size = 100
g_batch_shape = [g_batch_train_data_size, 243]  # 81 * 3
g_batch_data = NP.zeros(g_batch_shape, 'uint8');
g_batch_file_count = 0
g_batch_cur_index = 0
# for get data
g_batch_shape_file_num = 0  


def check_train_dir_files():
   global train_filelist
   global train_file_num
   
   #print("check_train_dir_files-----")
   for i in range(train_file_num):
     filename = train_data_dir + "/" + train_filelist[i]
     flag = is_bmp_file(filename)
     if flag == 0: 
       OS.remove(filename)
   train_filelist = OS.listdir(train_data_dir)
   train_file_num = len(train_filelist)

def get_batch_data_file_dir(flag) :
    tmp_dir = preprocess_data_dir_lr
    if flag == 2 :
       tmp_dir = preprocess_data_dir_hr
    return tmp_dir

def get_batch_data_file_name(index, flag) :  
    batch_dir =  get_batch_data_file_dir(flag)
    batch_file_name1 = batch_dir + "/batch" + str(index) + ".npy"
    return batch_file_name1

# flag:  1: LR  2: HR
def save_batch_data_to_file(flag):
    global g_batch_file_count
    #if g_batch_file_count < 3 :
    #  print("batch----", g_batch_file_count)
    #  print(g_batch_data)
    batch_file_name = get_batch_data_file_name(g_batch_file_count, flag)
    NP.save(batch_file_name, g_batch_data)
    g_batch_file_count += 1

    

# flag:  1: LR  2: HR
def save_bmp_array_by_batch(bmp_data_arr, flag):
   global g_batch_cur_index  

   rnum = bmp_data_arr.shape[0]   
   for i in range(rnum) :
     for j in range(243) :  # pixel copy
       g_batch_data[g_batch_cur_index][j] =  bmp_data_arr[i][j]
       if CDA_COLOR_FORMAT == 3:  # Only input Y
         if j % 3 != 0 :  # CbCr set to 0
           g_batch_data[g_batch_cur_index][j] = 0
       

     g_batch_cur_index += 1       
     if g_batch_cur_index == g_batch_train_data_size :           
        save_batch_data_to_file(flag) 
        g_batch_cur_index = 0  

#create the preprocess dir
def batch_dir_init() :
  if not OS.path.exists(preprocess_data_dir):
    OS.mkdir(preprocess_data_dir) 

  if OS.path.exists(preprocess_data_dir_lr):
    #OS.rmdir(preprocess_data_dir_lr)
    cmd = "rm -rf " + preprocess_data_dir_lr
    OS.system(cmd)

  if OS.path.exists(preprocess_data_dir_hr):
    #OS.rmdir(preprocess_data_dir_hr)
    cmd = "rm -rf " + preprocess_data_dir_hr
    OS.system(cmd)

  OS.mkdir(preprocess_data_dir_lr) 
  OS.mkdir(preprocess_data_dir_hr) 
         

def get_train_data_from_specific_file_rgb(index, img_flag):
    result = ""
    if train_file_num > 0 :       
        if index < train_file_num :
            filename = train_data_dir + train_filelist[index]   
            #print("-----get_train_data_from_specific_file_rgb--------",  filename)
            result = readfromfile_rgb(filename, img_flag)     
    return result

#generate training preprocess data
def gen_train_data_batch_npy_file(flag) :
  global g_batch_file_count
  global g_batch_cur_index
  g_batch_file_count = 0
  g_batch_cur_index = 0

  #bmp data preprocess
  bmp_data_arr = ""  
  #for i in range(1):
  for i in range(train_file_num):
    bmp_data_arr = get_train_data_from_specific_file_rgb(i, flag)
    #print("bmp_data_arr.shape = ", bmp_data_arr.shape)
    #print("bmp_data_arr : ", bmp_data_arr)
    #print("------------------------------------------------")
    save_bmp_array_by_batch(bmp_data_arr, flag)


#preprcess:  convert bmpfile to  batch data used for training.
#---------------------------------------------------------
def gen_train_data_batch_npy_file_all():

   print("-----gen_train_data_batch_npy_file_all-----")
   check_train_dir_files()
   batch_dir_init()
   #LR batch init.
   gen_train_data_batch_npy_file(1)
   #HR batch init.
   gen_train_data_batch_npy_file(2)


def load_train_batch_from_npy_file(index, flag) :
   filename = get_batch_data_file_name(index, flag)
   batach_data = NP.load(filename)
   #print("load_train_batch_npy_file--------", index, filename)
   #print(batach_data)

   return batach_data

#
def generate_train_batch_random_index():
   global g_batch_shape_file_num
   if g_batch_shape_file_num == 0 : 
       batch_dir = get_batch_data_file_dir(1)
       batch_filelist = OS.listdir(batch_dir)
       g_batch_shape_file_num = len(batch_filelist)

   if g_batch_shape_file_num <= 0 : 
       return -1

   #print("g_batch_shape_file_num----", g_batch_shape_file_num)
   index = Rand.randint(0, g_batch_shape_file_num - 1)
   #print("index ----",  index)
 
   return index




# uint8 to float32
# NP.zeros(81, 'uint8');
def format_batch_data_to_float32(batch_data):
  
   #print("-----------format_batch_data_to_float32---", batch_data)
   #print("------------", batch_data.shape)

   rnum = batch_data.shape[0]
   cnum = batch_data.shape[1]

   batch_float_arr = ""
   if rnum <= 0 and cnum <= 0:
      return batch_float_arr

   #-------------------------------------------------------------------   
   # must smaller for training.
   # color RGB  [0, 255]  change to [-0.5, 0.5]
   # color/256 - 0.5
   #-------------------------------------------------------------------
   batch_float_arr = NP.zeros([rnum, cnum], 'float32')
   for i in range(rnum):
     for j in range(cnum) :
        batch_float_arr[i][j] =  batch_data[i][j]
        # RGB : training data using 256 , to be compitible with training data.
        #YCbCr: training data using 255,  255 is better!!!
        if CDA_COLOR_FORMAT == 1 :
          batch_float_arr[i][j] = batch_float_arr[i][j]/255
        
        if CDA_COLOR_FORMAT == 2: 
          batch_float_arr[i][j] = batch_float_arr[i][j]/255

        #Y only.
        if CDA_COLOR_FORMAT == 3: 
          batch_float_arr[i][j] = batch_float_arr[i][j]/255
          if j % 3 != 0 :
            batch_float_arr[i][j] = 0


   #print("------batch_data----RGB--------")
   #print(batch_data)

   #print("------batch_data----float32--------")
   #print(batch_float_arr)

   return batch_float_arr

   
def load_train_batch_random(flag) :
   batch_data = ""
   index = generate_train_batch_random_index()
   if index < 0 :
       return

   batch_data = load_train_batch_from_npy_file(index, flag)

   return format_batch_data_to_float32(batch_data)


#
# define API function for train use 
#---------------------------------------------------------
def load_train_batch_random_lr_and_hr_old() :
   batch_data = ""
   index = generate_train_batch_random_index()
   if index < 0 :
       return

   batch_data1 = load_train_batch_from_npy_file(index, 1)
   batch_data2 = load_train_batch_from_npy_file(index, 2)
   #print("--------batch_data1-------", batch_data1)
   #print("--------batch_data2-------", batch_data2)
   lr_data = format_batch_data_to_float32(batch_data1)
   hr_data = format_batch_data_to_float32(batch_data2)

   return lr_data, hr_data

def load_train_batch_random_lr_old(): 
   return load_train_batch_random(1)
        
def load_train_batch_random_hr_old(): 
   return load_train_batch_random(2)

#-----------------------------------------------------------------------------
# second times training data.  based on previous
#-----------------------------------------------------------------------------
def generate_train_batch_random_pos(random_num):
   pos = Rand.randint(0, random_num)
   #print("position ----",  pos,  random_num)  
   return pos

def load_train_batch_random_s2(flag) :
   global g_batch_train_data_size
   global g_batch_shape
   
   index1 = generate_train_batch_random_index() 
   index2 = generate_train_batch_random_index()
   if index1 < 0 or index2 < 0:
       return

   #debug   
   #print("-load_train_batch_random_s2--", index1, index2)

   #pos 
   random_num = 2 * g_batch_train_data_size - 1   
   batch_data1 = load_train_batch_from_npy_file(index1, flag)
   batch_data2 = load_train_batch_from_npy_file(index2, flag)

   #print("batch_data1--", batch_data1)
   #print("batch_data2--", batch_data2)

   bdata = ""
   batch_data = NP.zeros(g_batch_shape, 'uint8');
   for i in range(g_batch_train_data_size) :
       pos = generate_train_batch_random_pos(random_num)
       #print("pos --",  pos)
       if pos < g_batch_train_data_size :
          bdata = batch_data1
       else :
          bdata = batch_data2
          pos = pos - g_batch_train_data_size
       for j in range(243) :
          batch_data[i][j] = bdata[pos][j]
    
   return format_batch_data_to_float32(batch_data)

def load_train_batch_random_s2_lr_hr() :
   global g_batch_train_data_size
   
   index1 = generate_train_batch_random_index() 
   index2 = generate_train_batch_random_index()
   if index1 < 0 or index2 < 0:
       return
       
   #print("-load_train_batch_random_s2_lr_hr--", index1, index2)

   #pos 
   random_num = 2 * g_batch_train_data_size - 1   
   batch_data1 = load_train_batch_from_npy_file(index1, 1) #LR
   batch_data2 = load_train_batch_from_npy_file(index2, 1)

   batch_data3 = load_train_batch_from_npy_file(index1, 2) #HR
   batch_data4 = load_train_batch_from_npy_file(index2, 2)
   
   bdatalr = ""  #tmp var
   bdatahr = ""  #tmp var

   batch_data_lr = NP.zeros(g_batch_shape, 'uint8');
   batch_data_hr = NP.zeros(g_batch_shape, 'uint8');
   
   for i in range(g_batch_train_data_size) :
       pos = generate_train_batch_random_pos(random_num)
       if pos < g_batch_train_data_size :
          bdatalr = batch_data1
          bdatahr = batch_data3
       else :
          bdatalr = batch_data2
          bdatahr = batch_data4 
          pos = pos - g_batch_train_data_size
          
       for j in range(243) :
          batch_data_lr[i][j] = bdatalr[pos][j]
          batch_data_hr[i][j] = bdatahr[pos][j]

   batch_data_lr_float32 = format_batch_data_to_float32(batch_data_lr)
   batch_data_hr_float32 = format_batch_data_to_float32(batch_data_hr)

   #print("batch_data_lr--", batch_data_lr)
   #print("batch_data_lr_float32--", batch_data_lr_float32)

   #print("batch_data_hr--", batch_data_hr)
   #print("batch_data_hr--", batch_data_hr_float32)

   return  batch_data_lr_float32,  batch_data_hr_float32

def load_train_batch_random_lr_and_hr() :
   return load_train_batch_random_s2_lr_hr()

def load_train_batch_random_lr(): 
   return load_train_batch_random_s2(1)
        
def load_train_batch_random_hr(): 
   return load_train_batch_random_s2(2)

############test...
#print(load_train_batch_random_lr())
#print(load_train_batch_random_hr())
#load_train_batch_random_lr_and_hr()


#----------------------------------------------------------------------------------------
#  Training batch data Generated end
#----------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------
#  Test data function define begin
#----------------------------------------------------------------------------------------
test_data_dir = OS.getcwd() + "/data/test"
test_data_dir_set5 = test_data_dir + "/set5"
test_data_dir_set14 = test_data_dir + "/set14"

test_cur_file_dir = ""
test_cur_file_list = "" 
test_cur_file_num = ""
test_cur_file_name = ""
test_cur_file_row_num = 0
test_cur_file_col_num = 0


test_cda_data_file_dir = OS.getcwd() + "/result_cda"
test_cda_data_file_dir_set5 = test_cda_data_file_dir + "/set5"
test_cda_data_file_dir_set14 = test_cda_data_file_dir + "/set14"
test_cda_data_file_cur_dir = ""


#print("train_imgfile_dir ====",  test_data_dir)
#print("file ====",  test_filelist)
#print("filenum ====",  test_file_num)


#
# define function for get test file data
#---------------------------------------------------------
def test_data_directory_init(test_dir):
   global test_cur_file_dir
   global test_cur_file_list
   global test_cur_file_num 
   global test_cur_file_name
   global test_cur_file_row_num
   global test_cur_file_col_num

   print("test_data_directory_init-----", test_dir)

   test_cur_file_dir = test_dir
   test_cur_file_list = OS.listdir(test_dir) 
   test_cur_file_num = len(test_cur_file_list)
   print("test_cur_filelist-----", test_cur_file_list)
   print("test_cur_file_num-----", test_cur_file_num)

   test_cur_file_name = ""
   test_cur_file_row_num = 0
   test_cur_file_col_num = 0

def test_cda_gen_data_directory_init(test_cda_dir):
   if not OS.path.exists(test_cda_data_file_dir):
      OS.mkdir(test_cda_data_file_dir)

   if not OS.path.exists(test_cda_dir):
      OS.mkdir(test_cda_dir)
 
   global test_cda_data_file_cur_dir
   test_cda_data_file_cur_dir = test_cda_dir




def set_bmp_info_callback(row_n, col_n):
    global test_cur_file_row_num
    global test_cur_file_col_num
    test_cur_file_row_num = row_n
    test_cur_file_col_num = col_n
    #print("set_bmp_info_callback----",  row_n, col_n, test_cur_file_row_num, test_cur_file_col_num)


def get_test_data_from_specific_file(index):
    global test_cur_file_name
    global test_cur_filelist

    test_cur_file_name = ""
    #print("get_test_data_from_specific_file-----", index)
    #print("get_test_data_from_specific_file-----", test_cur_file_list)

    callback = set_bmp_info_callback
    result = ""
    if test_cur_file_num > 0 :       
        if index < test_cur_file_num :
            test_cur_file_name =  test_cur_file_list[index]
            filename = test_cur_file_dir + "/" + test_cur_file_name    
            print("read: ", filename)                  
            result = readfromfile_rgb(filename, 1, callback)   # 1:  lr img
            
    return result


def get_test_data_from_specific_file_overlap(index):
    global test_cur_file_name
    global test_cur_filelist

    test_cur_file_name = ""
    #print("get_test_data_from_specific_file_overlap-----", index)
    #print("get_test_data_from_specific_file_overlap-----", test_cur_file_list)

    callback = set_bmp_info_callback
    result = ""
    if test_cur_file_num > 0 :       
        if index < test_cur_file_num :
            test_cur_file_name =  test_cur_file_list[index]
            filename = test_cur_file_dir + "/" + test_cur_file_name    
            print("overlap read: ", filename)                  
            result = readfromfile_rgb_overlap(filename, 1, callback)   # 1: lr img   
            
    return result


def check_test_dir_files(file_dir):  
   print("check_test_dir_files-----", file_dir) 
   tmp_file_list = OS.listdir(file_dir) 
   #print("tmp_file_list-----", tmp_file_list)
   tmp_file_num = len(tmp_file_list)  
   for i in range(tmp_file_num):
     filename = file_dir + "/" + tmp_file_list[i]
     flag = is_bmp_file(filename)
     #print("check_test_dir_files--flag---", flag)
     if flag == 0: 
       #print("remove....",  filename)
       OS.remove(filename)

def test_data_set5_init():
    #print("test_data_set5_init----")
    check_test_dir_files(test_data_dir_set5)
    test_data_directory_init(test_data_dir_set5)
    test_cda_gen_data_directory_init(test_cda_data_file_dir_set5)


def test_data_set14_init():
    #print("test_data_set5_init----")
    check_test_dir_files(test_data_dir_set14)
    test_data_directory_init(test_data_dir_set14)
    test_cda_gen_data_directory_init(test_cda_data_file_dir_set14)

def get_test_file_num(): 
    return test_cur_file_num


#def


def get_test_file_set(index):
    batch_data = get_test_data_from_specific_file(index)
    batch_data_float = format_batch_data_to_float32(batch_data)
    return batch_data_float


def get_test_file_set_overlap(index):
    batch_data = get_test_data_from_specific_file_overlap(index)
    batch_data_float = format_batch_data_to_float32(batch_data)
    return batch_data_float

#
# define function for save cda generate data to file.
#---------------------------------------------------------
def get_file_name_without_suffix(file_name):
   #new_img.show()
   #find position of .bmp
   tmp_file_name = file_name
   tmp_file_name.lower()
   pos_bmp_suffix =  tmp_file_name.index(".bmp")
   #print("-----------", tmp_file_name,  pos_bmp_suffix)

   #get substring
   tmp_file_name = file_name[0:pos_bmp_suffix]
   #print("-------get_file_name_without_suffix---------", tmp_file_name)
   return tmp_file_name

# cal R G B value
# RGB : training data using 256 , to be compitible with training data.
#YCbCr: training data using 255,  255 is better!!!
def convert_cda_pixel_data_to_color(c1, c2, c3):
    #RGB
    if CDA_COLOR_FORMAT == 1 :
      red = int((c1)*255)
      green = int((c2)*255)
      blue = int((c3)*255)
  
    #YCbCr
    if CDA_COLOR_FORMAT == 2:
      red = int((c1)*255)
      green = int((c2)*255)
      blue = int((c3)*255)

    #3: YCbCr, only Y used, other not used.
    if CDA_COLOR_FORMAT == 3:
      red = int((c1)*255)
      green = 0
      blue = 0

    return red, green, blue


#data block   81 * float
def fill_data_block_to_bmp_array(data_block, row_b, col_b,  pixels_per_row, img_arr):
  #cal position in img_arr
  img_row_i = row_b * 9
  img_col_j = col_b * 9
  
  #fill 9*9 block by 81 float variable
  #data_block:  243 * float: 
  #formate:  RGB -- RGB---RGB 
  
  aa = NP.zeros(1, 'uint8')
  b = -120

  for i in range(81):

    #print("----",  data_block[3*i],  data_block[3*i + 1],  data_block[3*i + 2])
    #print("RGB--",  red, green, blue)
    red, green, blue = convert_cda_pixel_data_to_color(data_block[3*i], data_block[3*i + 1],  data_block[3*i + 2])
    # save back to img_arr.
    img_arr[img_row_i][img_col_j][0] = red
    img_arr[img_row_i][img_col_j][1] = green
    img_arr[img_row_i][img_col_j][2] = blue

    img_col_j += 1
    if (i + 1) % 9 == 0 :
       img_row_i += 1
       img_col_j = col_b * 9

#for 3 only:
def set_data_array_CbCr(img_arr) :
  if CDA_COLOR_FORMAT != 3 :
     return

  filename = test_cur_file_dir + "/" + test_cur_file_name
  bmp_arr = getbmparray_orig(filename, 2)

  rnum = img_arr.shape[0]
  cnum = img_arr.shape[1]
  print("set_data_array_CbCr---", filename)
  print("set_data_array_CbCr-bmp_arr.shape--", bmp_arr.shape)
  print("set_data_array_CbCr---", rnum, cnum)

  #copy CbCr back.
  for i in range(rnum): 
    for j in range(cnum):
      img_arr[i][j][1] = bmp_arr[i][j][1] #Cb
      img_arr[i][j][2] = bmp_arr[i][j][2] #Cr


#data_arr:  color data.  
#CDA_COLOR_FORMAT:  1 RGB  2: YCbCr
#overlap: 0 -- no overlap  1 --- overlap
def save_data_array_to_bmp_file(img_arr, overlap = 0):
  #if img_arr == "" :
  #  return
  
  #print("save_data_array_to_bmp_file--CDA_COLOR_FORMAT=", CDA_COLOR_FORMAT)
  # data is YCbCr, only Y is used, other should copy back.
  if CDA_COLOR_FORMAT == 3:  
    set_data_array_CbCr(img_arr)

  new_img_rgb = ""    
  #print("save_data_array_to_bmp_file----")
  if CDA_COLOR_FORMAT == 1 :  # data is RGB
     new_img_rgb = Image.fromarray(img_arr, "RGB")

  if CDA_COLOR_FORMAT == 2  or CDA_COLOR_FORMAT == 3:  # data is YCbCr
    new_img = Image.fromarray(img_arr, "YCbCr")
    new_img_rgb = new_img.convert("RGB") 
    #print("-------convert--YCbCr--to-----RGB-----") 

  #print("-------test_cur_file_name-----------", test_cur_file_name) 
  tmp_file_name = get_file_name_without_suffix(test_cur_file_name)
  tmp_file_name1 =  test_cda_data_file_cur_dir + "/" +  tmp_file_name + "_cda.bmp"
  if overlap == 1:
    tmp_file_name1 =  test_cda_data_file_cur_dir + "/" +  tmp_file_name + "_cda_overlap.bmp"

  print("save_data_array_to_bmp_file-----------", tmp_file_name1)

  new_img_rgb.save(tmp_file_name1)
 

#resul_data format:  [none, 243] * float32
#block: 81
def save_cda_gen_data_to_bmp_file(result_data):
  #print("--save_cda_gen_data_to_bmp_file---",  test_cur_file_row_num,  test_cur_file_col_num)
  rb_num = test_cur_file_row_num
  cb_num = test_cur_file_col_num
  row_pixel_num = rb_num * 9
  col_pixel_num = cb_num * 9  
  pixel_shape = [row_pixel_num, col_pixel_num, 3]
  pixels_per_row = col_pixel_num

  #print("pixel_shape---", pixel_shape)
  img_arr = NP.zeros(pixel_shape, 'uint8')
  
  #print("result_data---------", result_data)

  index = 0  
  for row_b in range(rb_num):    
    for col_b in range(cb_num):
      fill_data_block_to_bmp_array(result_data[index], row_b, col_b, pixels_per_row, img_arr)
      index += 1

  save_data_array_to_bmp_file(img_arr)

  
# overlapped process.
#--------------------------------------------------------------------------
#data block   81 * float

def fill_data_block_to_bmp_array_overlap_averaged(i, j, img_arr, avg_img_arr):
    rflag = (int) (i / BMP_OVERLAPPED_POS_COEF)
    rflag1 = (int) (i % BMP_OVERLAPPED_POS_COEF)
    colflag = (int) (j / BMP_OVERLAPPED_POS_COEF)
    colflag1 = (int) (j % BMP_OVERLAPPED_POS_COEF)

    if rflag == 0 :  # row [0--5]
      if colflag == 0 :  # col [0--5]
        avg_img_arr[i][j][0] = (int)(img_arr[i][j][0])
        avg_img_arr[i][j][1] = (int)(img_arr[i][j][1])
        avg_img_arr[i][j][2] = (int)(img_arr[i][j][2])
      else:  # other cols  in the [0--5]
        if colflag1 < 3 :  # the first 3 overlapped.  x/2
          avg_img_arr[i][j][0] = (int)(img_arr[i][j][0]/2)
          avg_img_arr[i][j][1] = (int)(img_arr[i][j][1]/2)
          avg_img_arr[i][j][2] = (int)(img_arr[i][j][2]/2)
        else :
          avg_img_arr[i][j][0] = (int)(img_arr[i][j][0])
          avg_img_arr[i][j][1] = (int)(img_arr[i][j][1])
          avg_img_arr[i][j][2] = (int)(img_arr[i][j][2])

    # [6, xx] row.           
    if rflag != 0 :
      if colflag == 0 :  # col [0--5]
        if rflag1 < 3 :  # the first 3 overlapped.  x/2
          avg_img_arr[i][j][0] = (int)(img_arr[i][j][0]/2)
          avg_img_arr[i][j][1] = (int)(img_arr[i][j][1]/2)
          avg_img_arr[i][j][2] = (int)(img_arr[i][j][2]/2)
        else :
          avg_img_arr[i][j][0] = (int)(img_arr[i][j][0])
          avg_img_arr[i][j][1] = (int)(img_arr[i][j][1])
          avg_img_arr[i][j][2] = (int)(img_arr[i][j][2])
      else : # col[6, xx]
        if colflag1 < 3 :   # /4
          if rflag1 < 3 : 
            avg_img_arr[i][j][0] = (int)(img_arr[i][j][0]/4)
            avg_img_arr[i][j][1] = (int)(img_arr[i][j][1]/4)
            avg_img_arr[i][j][2] = (int)(img_arr[i][j][2]/4) 
          else :
            avg_img_arr[i][j][0] = (int)(img_arr[i][j][0]/2)
            avg_img_arr[i][j][1] = (int)(img_arr[i][j][1]/2)
            avg_img_arr[i][j][2] = (int)(img_arr[i][j][2]/2)
        else: # col 3--6              # /2
          if rflag1 < 3 : 
            avg_img_arr[i][j][0] = (int)(img_arr[i][j][0]/2)
            avg_img_arr[i][j][1] = (int)(img_arr[i][j][1]/2)
            avg_img_arr[i][j][2] = (int)(img_arr[i][j][2]/2)
          else :
            avg_img_arr[i][j][0] = (int)(img_arr[i][j][0])
            avg_img_arr[i][j][1] = (int)(img_arr[i][j][1])
            avg_img_arr[i][j][2] = (int)(img_arr[i][j][2])

          

def fill_data_block_to_bmp_array_overlap(data_block, row_b, col_b,  pixels_per_row, img_arr):
  #cal position in img_arr
  img_row_i = row_b * BMP_OVERLAPPED_POS_COEF
  img_col_j = col_b * BMP_OVERLAPPED_POS_COEF

  img_row_start = img_row_i
  img_col_start = img_col_j
  
  #print("row_b =",  row_b)
  #print("col_b =",  col_b)

  #fill 9*9 block by 81 float variable
  #data_block:  243 * float: 
  #formate:  RGB -- RGB---RGB 
  for i in range(81):
    # cal R G B value
    red, green, blue = convert_cda_pixel_data_to_color(data_block[3*i], data_block[3*i + 1], data_block[3*i + 2])

    #print("img_row_i---", img_row_i)
    #print("img_col_j---", img_col_j)

    # save back to img_arr.
    img_arr[img_row_i][img_col_j][0] += red
    img_arr[img_row_i][img_col_j][1] += green
    img_arr[img_row_i][img_col_j][2] += blue
   
    img_col_j += 1
    if (i + 1) % 9 == 0 :  # next line.
       img_row_i += 1
       img_col_j = img_col_start


#resul_data format:  [none, 243] * float32
#block: 81
#overlap: 0 --- no overlap;  1 overlap.
def save_cda_gen_data_to_bmp_file_overlap(result_data):
  #print("--save_cda_gen_data_to_bmp_file_overlap---",  test_cur_file_row_num,  test_cur_file_col_num)
  rb_num = test_cur_file_row_num
  cb_num = test_cur_file_col_num
  row_pixel_num = rb_num * BMP_OVERLAPPED_POS_COEF + 3
  col_pixel_num = cb_num * BMP_OVERLAPPED_POS_COEF + 3 
  pixel_shape = [row_pixel_num, col_pixel_num, 3]
  pixels_per_row = col_pixel_num

  #print("pixel_shape---", pixel_shape)
  img_arr = NP.zeros(pixel_shape, 'uint8')
  img_arr1 = NP.zeros(pixel_shape, 'float32')
  
  #print("pixel_shape-111--", img_arr1)

  index = 0  
  for row_b in range(rb_num):    
    for col_b in range(cb_num):
      fill_data_block_to_bmp_array_overlap(result_data[index], row_b, col_b, pixels_per_row, img_arr1)
      index += 1


  #print("img_arr1--222-", img_arr1)
  #unoverlaped
  for i in range(row_pixel_num):    
    for j in range(col_pixel_num):
      fill_data_block_to_bmp_array_overlap_averaged(i, j, img_arr1, img_arr)

  
  save_data_array_to_bmp_file(img_arr, 1)
 


#######################test
#print("test-------------------------------")
#test_data_set5_init()
#print("test--------------------111-----------")
#data = get_test_file_set(0)
#save_cda_gen_data_to_bmp_file(data)

#----------------------------------------------------------------------------------------
#  Test data function define end
#----------------------------------------------------------------------------------------


#test...
#----------------------------------------------------------
#print("-----gen_train_data_batch_npy_file_all-----")
#gen_train_data_batch_npy_file_all()
#load_train_batch_from_npy_file(0, 1)
#load_train_batch_from_npy_file(1)
#load_train_batch_from_npy_file(2)

#load_train_batch_random_lr()
#load_train_batch_random_hr()

#load_train_batch_random_lr_and_hr()

#------------------------------------------------------------------------
#  Training batch data Generated end
#------------------------------------------------------------------------


#lr hr: shape like: 
#shape_result = [block_num, 81]
#lr, hr = readfromfile('t1.bmp')
#hr = get_train_data_from_file(2, 1)
#print("---hr-------", hr.shape)
#print("----train_file_index-----", train_file_index)
#hr = get_train_data_from_next_file(1)
#print("----train_file_index-----", train_file_index)

#tttresult = [11, 81, 22]
#for i in range(len(tttresult)):
#    tttresult[i] *= 10


#tttresult1 = [[10, 20], [40,60]]
#tttresult2 = [[110, 120], [140, 160]]

#tttresult3 = tttresult2 - tttresult1
#print("-----------tttresult3----------", tttresult3)

##print("-----------tttresult1----------", tttresult1, 0x1000000)
#tttttt = NP.reshape(tttresult1, 4)
#print("-----------tttresult1----------", tttttt)
#print("-----------tttresult1----------", tttttt[3],  len(tttttt))
#get_hr_train_data()


#save test
#weghts = {
#    'b1' : tf.Variable(tf.zeros([3])),
#    'w2' :tf.Variable(tf.truncated_normal([3, 3],  stddev=0.1))
#}


#tttresult1 = NP.array([(10, 20), (40,60)] )
#tttresult2 = NP.array([(110, 27), (140, 160)])
#t1 = tf.convert_to_tensor(tttresult1)
#t2 = tf.convert_to_tensor(tttresult2)
#t3 = t1 - t2
#t4 = tf.square(t3)
#t5 = tf.reduce_sum(t4)
#sess = tf.InteractiveSession()
#tf.global_variables_initializer().run()

#print("t1----", sess.run(t1))
#print("t2----", sess.run(t2))
#print("t3----", sess.run(t3))
#print("t4----", sess.run(t4))
#print("t5----", sess.run(t5))

#print(sess.run(weghts['b1']))
#print(sess.run(weghts['w2']))

#------------------------------------------------------------------------
#  data output function define
#------------------------------------------------------------------------


#res = get_hr_train_data_specific(1)
#print("res---", res)


#save_test_data_to_bmp_file(res, 12, 13)

#tttresult1 = NP.array([(10, 20), (40,60)] )
#tttresult2 = NP.array([(110, 27), (140, 160)])
#filename11 = "/home/tcl/tensor/src/test/bmp/temp.npy"
#NP.save(filename11, tttresult1)
#tttresult3 = NP.load(filename11)
#print("tttresult3---", tttresult3)


#filehandle = open("/home/tcl/tensor/src/test/bmp/temp.bin", 'rb+')
#filehandle.write(tttresult1)
#filehandle.write(tttresult2)
#tttresult3 = filehandle.read(len(tttresult1))
#filehandle.seek(0)
#tttresult3 = filehandle.read()
#tttresult4 = filehandle.read(len(tttresult2))
#print("tttresult3---", tttresult3)
#print("tttresult4---", tttresult4)
#filehandle.close()








