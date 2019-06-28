# CDR-SR

This is implementation of Paper:
    Coupled Deep Autoencoder for Single Image Super-Resolution 
    by Kun Zeng, Jun Yu,  RuXin Wang, Cuihua Li, and Dacheng Tao.

It is my first case of my AI study.  Now it is free to share with loved friends. 


Run Enviroment:
   Linux + python + tensorflow + Numpy + PIL

Note:   
 For the directory structure,  please refer to the doc file.
 all the following exec in "src" directory

1.  test
-----------------------------------------------------------------
   run command:
     python cda_test.py

   function: 
     this step is to train the model.  
     You can set training parameters:
     1. train step:
         You can select which step to train by modifying the cda_train.py (unmask one sentence...)
     2. train times:
         change TRAIN_NUM in cda_model.py, default 500000     
    
    test files is in following directories:
         data/set5
         data/set14
    output result is in the direcotry:
         result_cda/set5
         result_cda/set14
    
    2 kinds of files to be created:
      1.   directory mode created: 
          no overlap,  just input whole file data to cda,  then get the output.
      2.  overlapped mode created:
           overlap reading from bmp file,  then feed to cda,  get the output ,  then unoverlaped it.

    if you want to have others bmp file to test,  
       copy to data/set5  or data/set14 directory,  and run cda_test.py


2.  train the model  if you want to: 
-----------------------------------------------------------------

2.1  train init
--------------------------------------------
   run command:
      python cda_init.py
 
   function: 
     this step is to create the preprocessing files for traing.
     Preprocess files are saved in dir:  data/preprocess
     If you have the preprocessing files,  just skip this step.


2.2  training.
--------------------------------------------
   run command:
      python cda_train.py
 
   function: 
     this step is to train the model.  
     You can set training parameters:
     1. train step:
         You can select which step to train by modifying the cda_train.py (unmask one sentence...)
     2. train times:
         change TRAIN_NUM in cda_model.py, default 500000     
     
     After training,  model/paramters will be saved to files in the direcory:
        "model/train" directory.
     every 10000 times,  model/paramters will be saved by overwriting the previouse one.
     
     For single step training:  you will get step-only files like: 
        cda_model.ckpt.onlystepX.meta
        cda_model.ckpt.onlystepX.index
        cda_model.ckpt.onlystepX.data-00000-of-00001
      X = [1, 2, 3]

     You ran step1 and step2 training in paralell training, and get the step-only files.
     Before you run step3 and step4,  
        you can copy the step-only files to  model/train in your running directory.
        modify cda_train.py and run:  python cda_train.py
        Then the training will first load model/paramter from step-only files,  then start training.
      
     ############!!!!!!!!!!!!!!!!!!!!###################
     If you finish all your train,  
         you should copy  the following 3 files to  directory : {{ model/test }}
            cda_model.ckpt.index
            cda_model.ckpt.meta
            cda_model.ckpt.data-00000-of-00001
         Then you can test you model.
                   


