# YOLO_v3_tf
reconstruct yolo_v3 model with tf.estimator for distributed training         

1.data prepare         
image_file_path: /data/data/xxx_train_data/images/xxx.jpg           
label_file_path: /data/data/xxx_train_data/labels/xxx.txt            
            
2.label content        
x0,y0,x1,y1,x2,y2,x3,y3,class_name1,class_idx1                    
x0,y0,x1,y1,x2,y2,x3,y3,class_name2,class_idx2                 
x0,y0,x1,y1,x2,y2,x3,y3,class_name3,class_idx3            
......            
......             
       
3.generate tfrecord             
cd batch_convert              
python ./gen_tfrecord_multi.py             
(you can set pid_num to use multiprocessing in generating)                
            
4.start training               
python yolo_train_two.py               
(using estimator in tensorflow==1.14 & python3.7)                 
test on a distributed training system with a configuration of 4*(4*tesla_P40)         
              
