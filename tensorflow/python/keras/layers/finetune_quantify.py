import tensorflow as tf
from tensorflow.python.framework import ops
_START = 0.0 #525420
def quantify(data, name, steps_per_epoch=None, th=0.5): 
    global_step = tf.train.get_or_create_global_step()
    global_step = tf.cast(global_step, tf.float32)
    shift_op = tf.Variable(0, name=name+'_shift', trainable=False, dtype=tf.float32)
#    scale_op = tf.Variable(0, name=name+'_scale', trainable=False, dtype=tf.float32)
#    offset_op = tf.Variable(0, name=name+'_offset', trainable=False, dtype=tf.float32)
    bitnum_op = tf.Variable(4, name=name+'_bitnum', trainable=False, dtype=tf.float32) 
    update_step_op = tf.Variable(_START+1, name=name+'_update_step', trainable=False, dtype=tf.float32) 
    m_op = tf.Variable(100, name=name+'_m', trainable=False, dtype=tf.float32) #就是浮点数
    new_shift_op, new_bitnum_op, new_update_step_op, new_m_op = get_dynamic_all(data, global_step, shift_op, bitnum_op, m_op, update_step_op, steps_per_epoch,th)
    shift_op_assign = tf.assign(shift_op, new_shift_op)
    bitnum_op_assign = tf.assign(bitnum_op, new_bitnum_op)
    update_step_op_assign = tf.assign(update_step_op, new_update_step_op)
    m_op_assign = tf.assign(m_op, new_m_op)
  
    with tf.control_dependencies([shift_op_assign, bitnum_op_assign, update_step_op_assign, m_op_assign]):
        data = tf.identity(data)
        outdata = float2fix(data, shift_op_assign, bitnum_op_assign)
    return outdata, shift_op_assign, bitnum_op_assign,update_step_op_assign,m_op_assign
       
_alpha = 0.04
_beta = 0.1
_gamma = 2
_delta = 100

 #---------get update step ----------

def get_update_step(data, bitnum, shift, m, update_step, global_step, steps_per_epoch):
    return tf.cond(tf.less(global_step, _START+steps_per_epoch),
            lambda:get_new_update_step1(data, bitnum, shift, m, update_step, global_step, steps_per_epoch),
            lambda:get_new_update_step1(data, bitnum, shift, m, update_step, global_step, steps_per_epoch))

def get_new_update_step1(data, bitnum, shift, m, update_step, global_step, steps_per_epoch):
    return tf.cond(tf.less(global_step, _START+steps_per_epoch/10), 
            lambda:get_next_step(global_step,shift,m),
            lambda:get_next_update_step(data, bitnum, shift, m, update_step, global_step))
#    return tf.cond(tf.less(new_update_step,_START+steps_per_epoch), lambda:new_update_step, lambda:_START+steps_per_epoch), new_m

def get_new_update_step2(data, bitnum, shift, m, update_step, global_step, steps_per_epoch):
    return get_next_epoch(global_step, shift, m, update_step, steps_per_epoch)

def get_next_step(global_step, shift, m):
    new_m = get_new_m(shift, m)
    return tf.cast(global_step + 1, tf.float32), new_m

def get_next_epoch(global_step, shift, m, update_step, steps_per_epoch):
    new_m = get_new_m(shift, m)
    new_update_step = tf.cond(tf.equal(global_step, update_step), lambda: global_step+steps_per_epoch/2.0, lambda: update_step)
    return new_update_step, new_m

def get_next_update_step(data, bitnum, shift, m, update_step, global_step):
    next_update_step, next_m = tf.cond(tf.equal(global_step, update_step), 
            lambda:get_new_update_step(data, bitnum, shift, m, global_step),
            lambda:get_old_update_step(update_step, shift, m))
    return next_update_step, next_m
def get_old_update_step(update_step, shift, m):
    return update_step, get_new_m(shift,m)

def get_new_update_step(data, bitnum, shift, m, global_step):

    diff1 = _alpha * tf.abs(shift - m)
    diff_bit = get_diff(data, shift)
    diff2 = _delta * diff_bit * diff_bit
    diff = tf.maximum(diff1, diff2)

    new_m = get_new_m(shift, m)
    with tf.get_default_graph().gradient_override_map({"Round":"Identity"}):
        interval = tf.round(_beta / diff - _gamma)
    interval = tf.maximum(interval, 1.0) #tf.cond(tf.less(interval, 1.0), lambda:1.0, lambda:interval)

    update_step = global_step + interval
    return update_step, new_m

def get_new_m(shift, m):
    shift=tf.cast(shift,tf.float32)
    new_m = _alpha * shift + (1 - _alpha) * m
    return new_m

def get_dynamic_all(data, global_step, shift, bitnum, m, update_step, steps_per_epoch, th):
    new_shift, new_bitnum, new_m = get_dynamic_n_s_interval(data,
        shift, bitnum, m, global_step, update_step, th, steps_per_epoch)
                        
    new_update_step, new_m = get_update_step(data,
        new_bitnum, new_shift, new_m, update_step, global_step, steps_per_epoch)

    return new_shift, new_bitnum, new_update_step, new_m

def get_dynamic_n_s_interval(data, shift, bitnum, m, global_step, update_step, th, steps_per_epoch):
    new_shift, new_bitnum = tf.cond(tf.equal(global_step, update_step) | tf.equal(global_step%steps_per_epoch,0), 
                                lambda:dynamic_n(data, bitnum, th),lambda:get_last(shift, bitnum))
  
    new_m = m + tf.cast(new_bitnum - bitnum, tf.float32)
    return new_shift, new_bitnum, new_m
def get_last(shift,bitnum):
    return shift, bitnum

#-----------dynamic n-------------

_TH = 0.5
_ADDBIT = 4
_MAXBIT = 8

def dynamic_n(data, bitnum, th):
    shift = get_new_shift(data, bitnum)
    diff = get_diff(data, shift)
    
    loop = [diff, bitnum, shift, data]
    cond = lambda diff, bitnum, shift, data: tf.greater(diff, th) & tf.less(bitnum, _MAXBIT)
    body = lambda diff, bitnum, shift, data: loop_body(diff, bitnum, shift, data)
    diff, bitnum, shift, data = tf.while_loop(cond, body, loop)
    return shift, bitnum

def loop_body(diff, bitnum, shift, data):
    bitnum = bitnum + _ADDBIT
    new_shift = get_new_shift(data, bitnum) 
    outdata = float2fix(data, new_shift, bitnum) #使用新diff注释掉
    diff = get_diff(data, outdata) #使用新diff get_diff(data, new_shift)
    return diff, bitnum, new_shift, data


def get_diff(data, outdata):
    #计算量化误差
    data_mean = tf.reduce_mean(data)
    outdata_mean = tf.reduce_mean(outdata)
    diff = tf.log1p(tf.divide(tf.reduce_mean(tf.abs(data - outdata)), tf.reduce_mean(tf.abs(data)))) / tf.log(2.0)
    return diff

#def get_diff(data, shift):
#   shift=tf.cast(shift, tf.float32)
#    r = tf.pow(2.0, -shift)
#    diff = tf.log1p(r/16/tf.reduce_mean(tf.abs(data)))/tf.log(2.0)
#    return diff


def get_new_shift_scale_offset(data, bitnum):
    mmin = tf.reduce_min(data)
    mmax = tf.reduce_max(data)
    shift = - tf.ceil(tf.log(mmax - mmin)/tf.log(2.0) - bitnum + 0.0001)
    scale = (mmax-mmin)/2/tf.pow(2.0, -shift)/(tf.pow(2.0, bitnum - 1) - 1)
    offset = (mmax + mmin)/2.0
    shift = tf.cast(shift,tf.int32)
    return shift, scale,offset

def get_new_shift(data, bitnum):
    data_abs = tf.abs(data)
    max_abs = tf.reduce_max(data_abs)
    shift =  tf.cond(tf.equal(max_abs,0),lambda:30.0, lambda:- tf.ceil(tf.log(max_abs)/tf.log(2.0) - bitnum + 1 + 0.0001))
    return shift


def float2fix(data, shift, bitnum,scale=1,offset=0):
   # with tf.control_dependencies([assign]):
    shift = tf.cast(shift, tf.float32)
    bitnum = tf.cast(bitnum, tf.float32)
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

#@ops.RegisterGradient("MyMaxMinGrad")
#def _MyMaxMinGrad(op, grad): 
#    return (grad,tf.reduce_mean(grad))
