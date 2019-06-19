# -*- coding: utf-8 -*-

# a config file for quantize

scale=True
offset=True 
#base 对称的定点表示
#scale 有缩放的定点表示
#offset 有缩放非对称的定点表示
shift_interval_update=False
scale_interval_update=False
offset_interval_update=False


nbit_update='none' #不动态更新bit数
update_th=0.1
update_num=8

alpha=0.01
beta=0.025
gama=2
sigma=25
step1=200
step2=454
 
