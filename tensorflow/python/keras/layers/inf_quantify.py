import numpy as np
import tensorflow as tf 
def fixed_inf(data, flag, bitnum=8, mode=0, isfp16=False):
    if mode == 0: 
        shift = tf.Variable(0, trainable=False, name="lc_shift", dtype=tf.float32)
        new_shift = get_shift(data, shift, bitnum, flag)
        assign_shift = tf.assign(shift, tf.cast(new_shift, tf.float32))
        output = float2fix(data, assign_shift, bitnum)
        return output, shift
    if mode == 1:
        shift = tf.Variable(0, trainable=False, name="lc_shift", dtype=tf.float32)
        f = tf.Variable(0, trainable=False, name="lc_f", dtype=tf.float32)
        new_shift, new_f = get_shift_f(data, shift, f, bitnum, flag)
        assign_shift = tf.assign(shift, new_shift)
        assign_f = tf.assign(f, new_f)
        output = float2fix(data,assign_shift, bitnum,f=assign_f,isfp16=isfp16)
        return output, shift, assign_f
    if mode == 2:
        shift = tf.Variable(0, trainable=False, name="lc_shift", dtype=tf.float32)
        f = tf.Variable(0, trainable=False, name="lc_f", dtype=tf.float32)
        offset = tf.Variable(0, trainable=False, name="lc_offset", dtype=tf.float32)
        new_shift, new_f, new_offset = get_shift_f_offset(data, shift, f, offset, bitnum, flag)
        assign_shift = tf.assign(shift, new_shift)
        assign_f = tf.assign(f, new_f)
        assign_offset = tf.assign(offset, new_offset)
        output = float2fix(data,assign_shift, bitnum,f=assign_f,offset=assign_offset, isfp16=isfp16)
        return output, shift, assign_f,assign_offset

def get_shift(data, shift, bitnum, flag):
    new_shift = tf.cond(tf.equal(flag,0), lambda:get_new_shift(data,bitnum), lambda:shift)
    return new_shift

def get_new_shift(data, bitnum):
    data_abs = tf.abs(data)
    max_abs = tf.reduce_max(data_abs)
    shift = tf.cast( - tf.ceil(tf.log(max_abs)/tf.log(2.0) - bitnum + 1 + 0.0001), tf.int32)
    return shift

def get_shift_f(data, shift, f, bitnum, flag):
    new_shift, new_f = tf.cond(tf.equal(flag, 0), lambda:get_new_shift_f(data,bitnum), lambda:get_old_s_f(shift,f))
    return new_shift, new_f
def get_old_s_f(shift,f):
    return shift,f
def get_new_shift_f(data, bitnum):
    data_abs = tf.abs(data)
    max_abs = tf.reduce_max(data_abs)
    shift =  - tf.ceil(tf.log(max_abs)/tf.log(2.0) - bitnum + 1 + 0.0001)
    f = max_abs/tf.pow(2.0, -shift)/(tf.pow(2.0, bitnum - 1) - 1)
    return shift, f
def get_shift_f_offset(data, shift, f, offset, bitnum, flag):
    new_shift, new_f, new_offset = tf.cond(tf.equal(flag, 0), lambda:get_new_shift_f_offset(data,bitnum), lambda:get_old_s_f_o(shift, f, offset))
    return new_shift, new_f, new_offset

def get_old_s_f_o(shift, f, offset):
    return shift,f, offset
def get_new_shift_f_offset(data, bitnum):
    mmin = tf.reduce_min(data)
    mmax = tf.reduce_max(data)
    shift = - tf.ceil(tf.log(mmax - mmin)/tf.log(2.0) - bitnum + 0.0001)
    f = (mmax-mmin)/2/tf.pow(2.0, -shift)/(tf.pow(2.0, bitnum - 1) - 1)
    offset = (mmax + mmin)/2.0
    return shift, f,offset

def float2fix(data, shift, bitnum, f=1, offset=0, isfp16=False):
    t = data
    neg_b = -(tf.pow(2.0, bitnum - 1.0)) * tf.pow(2.0, - shift)*f+offset
    pos_b = (tf.pow(2.0, bitnum - 1.0) - 1.0) * tf.pow(2.0, - shift)*f+offset
    step = tf.pow(2.0, - shift)*f
    data = tf.maximum(data, neg_b)
    data = tf.minimum(data, pos_b)
    data = data - offset
    temp0 = tf.divide(data, step)
    with tf.get_default_graph().gradient_override_map({"Round":"Identity"}):
        temp1 = tf.round(temp0)
    output = temp1*step+offset
    if isfp16 is True:
        return temp1
    return  output

