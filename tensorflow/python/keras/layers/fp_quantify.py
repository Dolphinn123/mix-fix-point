import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops 
def fixed(data, bitnum = 8, interval=1, steps_per_epoch=None):
    global_step = tf.train.get_or_create_global_step()
    shift = tf.Variable(0, trainable=False, name="lc_shift", dtype=tf.float32)
    new_shift = get_shift2(data, shift, bitnum, global_step,interval,steps_per_epoch)
    assign_shift = tf.assign(shift, new_shift)
    output = float2fix(data, assign_shift, bitnum)
    return output, shift

def get_shift2(data, shift, bitnum, global_step, interval=1, steps_per_epoch=None):
    if steps_per_epoch is None:
        new_shift = tf.cond(tf.equal(global_step%interval,0), 
            lambda:get_new_shift(data, bitnum),
            lambda:shift)
    else:
        new_shift = tf.cond(tf.equal(global_step%interval,0) & tf.less(global_step,steps_per_epoch) 
            | tf.equal(global_step%steps_per_epoch,0) | tf.equal(global_step,1), 
            lambda:get_new_shift(data, bitnum),
            lambda:shift)
    return new_shift


def get_new_shift(data, bitnum):
    data_abs = tf.abs(data)
    max_abs = tf.reduce_max(data_abs)
    shift =  - tf.ceil(tf.log(max_abs)/tf.log(2.0) - bitnum + 1 + 0.0001)
    return shift


def float2fix(data, shift, bitnum, scale=1, offset=0):
#    with tf.control_dependencies([assign_shift]):
#    shift = tf.cast(shift, tf.float32)
#    bitnum = tf.cast(bitnum, tf.float32)
    neg_b = -(tf.pow(2.0, bitnum - 1.0)) * tf.pow(2.0, - shift)*scale+offset
    pos_b = (tf.pow(2.0, bitnum - 1.0) - 1.0) * tf.pow(2.0, - shift)*scale+offset
    step = tf.pow(2.0, - shift)*scale
    with tf.get_default_graph().gradient_override_map({"Maximum":"MyMaxMinGrad"}):
        data = tf.maximum(data, neg_b)
    with tf.get_default_graph().gradient_override_map({"Minimum":"MyMaxMinGrad"}):
        data = tf.minimum(data, pos_b)
    data = data - offset
    temp0 = tf.divide(data, step)
    with tf.get_default_graph().gradient_override_map({"Round":"Identity"}):
        temp1 = tf.round(temp0)
    output = temp1*step+offset
    return  output

@ops.RegisterGradient("MyMaxMinGrad")
def _MyMaxMinGrad(op, grad): 
    return (grad,tf.reduce_mean(grad))
