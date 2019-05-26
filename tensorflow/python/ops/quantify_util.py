import tensorflow as tf

_alpha = 0.04
_beta = 0.1
_gamma = 2
_delta = 100

 #---------get update step ----------

def get_update_step(data, bitnum, shift, m, update_step, global_step, steps_per_epoch):
    return tf.cond(tf.less(global_step, steps_per_epoch),
            lambda:get_new_update_step1(data, bitnum, shift, m, update_step, global_step, steps_per_epoch),
            lambda:get_new_update_step2(data, bitnum, shift, m, update_step, global_step, steps_per_epoch))

def get_new_update_step1(data, bitnum, shift, m, update_step, global_step, steps_per_epoch):
    return tf.cond(tf.less(global_step, steps_per_epoch/10), 
            lambda:get_next_step(global_step,shift,m),
            lambda:get_next_update_step(data, bitnum, shift, m, update_step, global_step))

def get_new_update_step2(data, bitnum, shift, m, update_step, global_step, steps_per_epoch):
    return get_next_epoch(global_step, shift, m, update_step, steps_per_epoch)

def get_next_step(global_step, shift, m):
    new_m = get_new_m(shift, m)
    return global_step, new_m

def get_next_epoch(global_step, shift, m, update_step, steps_per_epoch):
    new_m = get_new_m(shift, m)
    new_update_step = tf.cond(tf.equal(global_step, update_step), lambda: global_step+steps_per_epoch, lambda: update_step)
    return new_update_step, new_m

def get_next_update_step(data, bitnum, shift, m, update_step, global_step):
    next_update_step, next_m = tf.cond(tf.equal(global_step, update_step), 
            lambda:get_new_update_step(data, bitnum, shift, m, global_step),
            lambda:get_old_update_step(update_step, shift, m))
    return next_update_step, next_m
def get_old_update_step(update_step, shift, m):
    return update_step, get_new_m(shift,m)

def get_new_update_step(data, bitnum, shift, m, global_step):

    shift=tf.cast(shift,tf.float32)
    diff1 = _alpha * tf.abs(shift - m)
    diff_bit = get_diff(data, shift)
    diff2 = _delta * diff_bit * diff_bit
    diff = tf.maximum(diff1, diff2)

    new_m = get_new_m(shift, m)
    interval = _beta / diff - _gamma
    interval = tf.maximum(interval, 1.0) #tf.cond(tf.less(interval, 1.0), lambda:1.0, lambda:interval)

    update_step = global_step + interval
 #   shift=tf.cast(shift, tf.float32)
 #   update_step=tf.cast(update_step, tf.int64)
    return update_step, new_m

def get_new_m(shift, m):
    shift=tf.cast(shift,tf.float32)
    new_m = _alpha * shift + (1 - _alpha) * m
    shift=tf.cast(shift, tf.int32)
    return new_m

#------------get quantify params ------------
def get_fix_update_step(global_step, update_step, interval, steps_per_epoch=None):
    if steps_per_epoch is None:        
        return global_step+interval
    else:
        return tf.cond(tf.less(global_step, steps_per_epoch),
            lambda: global_step+interval, 
            lambda:global_step+steps_per_epoch)

def get_fixn_shift(data, global_step, shift, bitnum, update_step, interval, steps_per_epoch=None):
    return tf.cond(tf.equal(tf.cast(global_step,tf.float32),update_step) | tf.equal(global_step,0), 
        lambda:get_fixn_shift_update(data, global_step, bitnum, update_step, interval, steps_per_epoch),
        lambda:get_last(shift, update_step=update_step))

def get_fixn_shift_update(data, global_step, bitnum, update_step, interval, steps_per_epoch=None):
    new_shift = get_new_shift(data, bitnum)
    new_update_step = get_fix_update_step(global_step, update_step, interval, steps_per_epoch)
    new_update_step = tf.cast(new_update_step, tf.float32)
    return new_shift, new_update_step

def get_last(shift, bitnum=None, update_step=None):
    #返回上一代参数
    if bitnum is None:
        if update_step is None:
            return shift
        else:
            return shift, update_step
    elif update_step is None:
        return shift, bitnum
    else:
        return shift, bitnum, update_step


def get_dynamic_bitnum_shift(data, global_step, shift, bitnum, update_step, interval, steps_per_epoch=None):
    #返回本代量化的shift，bitnum和下一次更新的代数:
    #若global_step等于update_step，参数更新，否则使用上一代参数
    return tf.cond(tf.equal(tf.cast(global_step,tf.float32),update_step), 
        lambda:get_dynamic_bitnum_shift_update(data, global_step, bitnum, update_step, interval, steps_per_epoch),
        lambda:get_last(shift, bitnum=bitnum, update_step=update_step))

def get_dynamic_bitnum_shift_update(data, global_step, bitnum, update_step, interval, steps_per_epoch=None):
    #返回更新的shift和更新的下一次更新代数
    new_shift, new_bitnum = dynamic_n(data, bitnum)
    new_update_step = get_fix_update_step(global_step, update_step, interval, steps_per_epoch)
    return new_shift, new_bitnum, new_update_step
def get_dynamic_all(data, global_step, shift, bitnum, m, update_step, steps_per_epoch):
    global_step=tf.cast(global_step, tf.float32)
    new_shift, new_bitnum, new_m = get_dynamic_n_s_interval(data,
        shift, bitnum, m, global_step, update_step)
                        
    new_update_step, new_m = get_update_step(data,
        new_bitnum, new_shift, new_m, update_step, global_step, steps_per_epoch)

    return new_shift, new_bitnum, new_update_step, new_m

def get_dynamic_n_s_interval(data, shift, bitnum, m, global_step, update_step):
    new_shift, new_bitnum = tf.cond(tf.equal(global_step, update_step), lambda:dynamic_n(data, bitnum),lambda:get_last(shift, bitnum=bitnum))
    
    new_m = m + tf.cast(new_bitnum - bitnum, tf.float32)
    return new_shift, new_bitnum, new_m


#-----------dynamic n-------------

_TH = 0.03
_ADDBIT = 8
_MAXBIT = 48
_MINBIT = 8

def dynamic_n(data, bitnum):
    shift = get_new_shift(data, bitnum)
    diff = get_diff(data, shift)
    
    loop = [diff, bitnum, shift, data]
    cond = lambda diff, bitnum, shift, data: tf.greater(diff, _TH) & tf.less(bitnum, _MAXBIT)
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
#计算量化误差，使用r(更新中)
#   shift=tf.cast(shift, tf.float32)
#    r = tf.pow(2.0, -shift)
#    diff = tf.log1p(r/16/tf.reduce_mean(tf.abs(data)))/tf.log(2.0)
#    return diff



def get_new_shift(data, bitnum):
    bitnum=tf.cast(bitnum, tf.float32)
    data_abs = tf.abs(data)
    max_abs = tf.reduce_max(data_abs)
    shift = tf.cond(tf.less(max_abs,1e-10),lambda:32.0,lambda:- tf.ceil(tf.log(max_abs)/tf.log(2.0) - bitnum +1 + 0.0001))
    return shift

def float2fix(data, shift, bitnum):
  #  with tf.control_dependencies([assign_shift,update_step]):
   # if m is None:
    #    with tf.control_dependencies([update_step]):
    neg_b = - tf.pow(2.0, bitnum - 1.0) * tf.pow(2.0, - shift)
    pos_b = (tf.pow(2.0, bitnum - 1.0) - 1.0) * tf.pow(2.0, - shift)

    step = tf.pow(2.0, - shift) 

    data = tf.maximum(data, neg_b)
    data = tf.minimum(data, pos_b)
    output =tf.multiply(tf.round(tf.divide(data, step)), step)
 #   else:
 #       with tf.control_dependencies([update_step, m]):
 #           neg_b = - tf.pow(2.0, bitnum - 1.0) * tf.pow(2.0, - shift)
 #           pos_b = (tf.pow(2.0, bitnum - 1.0) - 1.0) * tf.pow(2.0, - shift)

 #           step = tf.pow(2.0, - shift) 

 #           data = tf.maximum(data, neg_b)
 #           data = tf.minimum(data, pos_b)
 #           output =tf.multiply(tf.round(tf.divide(data, step)), step)

    return output
