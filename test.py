# # # # -*- coding: utf-8 -*-
import torch
flag = torch.cuda.is_available()
print(flag)

ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3,3).cuda())


#ipython
import tensorflow as tf
print(tf.test.is_gpu_available())
#如果显示True，说明gpu版本已经安装成功
print(tf.test.gpu_device_name())