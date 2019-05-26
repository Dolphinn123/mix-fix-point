from .quantify_util import *

def quantify(data, global_step, mode=0, interval=None, steps_per_epoch=None): 
    if mode == 0:
        return data,None,None,None,None
    if mode == 1: 
        bitnum = 16 #8
        shift_op = tf.Variable(0, name='lc_grad_shift', trainable=False, dtype=tf.float32) #tf.float32 for multi-gpu, e.g Tensorpack
    #    update_step_op = tf.Variable(0, name='lc_grad_update_step', trainable=False, dtype=tf.float32) 
   #     new_shift_op,new_update_step_op = get_fixn_shift(data, global_step, shift_op, bitnum, update_step_op, interval, steps_per_epoch=steps_per_epoch) # 
#        new_shift_op = get_new_shift(data, bitnum)
#        new_update_step_op = update_step_op + 1 
        global_step_fp32 = tf.cast(global_step, tf.float32)
        new_shift_op = tf.cond(tf.equal(global_step_fp32 // interval, global_step_fp32 / interval) & tf.less(global_step_fp32, steps_per_epoch)
                               | tf.equal(global_step_fp32 // steps_per_epoch, global_step_fp32 / steps_per_epoch)
                               | tf.equal(global_step_fp32, 1),# // steps_per_epoch, global_step_fp32 / steps_per_epoch),
                               lambda: get_new_shift(data, bitnum),
                               lambda: shift_op)     
        shift_op_assign = tf.assign(shift_op, new_shift_op)
     #   update_step_op_assign = tf.assign(update_step_op, new_update_step_op)
        outdata = float2fix(data, shift_op_assign, bitnum)#,update_step_op_assign,None)
        return outdata, shift_op_assign, None, None, None
    if mode == 2:
        shift_op = tf.Variable(0, name='lc_grad_bitnum', trainable=False, dtype=tf.float32)
        bitnum_op = tf.Variable(0, name='lc_grad_bitnum', trainable=False, dtype=tf.float32) 
        update_step_op = tf.Variable(0, name='lc_grad_update_step', trainable=False, dtype=tf.float32) 
        new_shift_op, new_bitnum_op, new_update_step_op = get_dynamic_bitnum_shift(data, global_step, shift_op, bitnum_op, update_step_op, interval, steps_per_epoch)
        shift_op_assign = tf.assign(shift_op, new_shift_op)
        bitnum_op_assign = tf.assign(bitnum_op, new_bitnum_op)
        update_step_op_assign = tf.assign(update_step_op, new_update_step_op)
        outdata = float2fix(data, shift_op_assign, bitnum_op_assign,update_step_op_assign,None)
        return outdata, shift_op, bitnum_op,None, None
    if mode ==3:
        shift_op = tf.Variable(0, name='lc_grad_bitnum', trainable=False, dtype=tf.float32)
        bitnum_op = tf.Variable(8, name='lc_grad_bitnum', trainable=False, dtype=tf.float32) 
        update_step_op = tf.Variable(0, name='lc_grad_update_step', trainable=False, dtype=tf.float32) 
        m_op = tf.Variable(100, name='lc_grad_m', trainable=False, dtype=tf.float32) #就是浮点数
        new_shift_op, new_bitnum_op, new_update_step_op, new_m_op = get_dynamic_all(data, global_step, shift_op, bitnum_op, m_op, update_step_op, steps_per_epoch)
        shift_op_assign = tf.assign(shift_op, new_shift_op)
        bitnum_op_assign = tf.assign(bitnum_op, new_bitnum_op)
        update_step_op_assign = tf.assign(update_step_op, new_update_step_op)
        m_op_assign = tf.assign(m_op, new_m_op)
        with tf.control_dependencies(update_step_op_assign, m_op_assign):
            outdata = float2fix(data, shift_op_assign, bitnum_op_assign)#, update_step_op_assign, m_op_assign)
        return outdata, shift_op_assign, bitnum_op_assign,update_step_op_assign,m_op_assign
       
