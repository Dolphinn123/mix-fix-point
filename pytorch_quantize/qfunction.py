import torch
from torch.autograd.function import Function
import config

class base_quantize(Function):

    @classmethod
    def forward(cls,ctx, input, num_bits=8,inplace=False,quantize_para=torch.tensor([1,0,0],dtype=torch.float),global_step=None,update=None):
        Z=input.abs().max()
        shift=quantize_para[0]
        scale=quantize_para[1]
        offset=quantize_para[2]
        if config.offset:
            if  config.offset_interval_update and not global_step>=update.shift_updatestep:
                Z=(input-offset).abs().max()
                offset_update_flag=0
            else:
                Z_max=input.max()
                Z_min=input.min()
                offset= (Z_max+Z_min)/2
                Z=Z_max-offset
                offset_update_flag=1
        else:
            offset=torch.tensor(0,dtype=torch.float).cuda()
            offset_update_flag=0
        #shift 
        if config.shift_interval_update and not global_step>=update.shift_updatestep:
            step=2**shift
            A=(2**(num_bits-1)-1)*step
            A_=(2**(num_bits-1))*step
            shift_update_flag=0
        else:
            if Z>0:
                shift=torch.ceil(torch.log2(Z/(2**(num_bits-1)-1)))
                step=2**shift
                A=(2**(num_bits-1)-1)*step
                A_=(2**(num_bits-1))*step
            else:
                shift=torch.tensor(-12.0).cuda()
                step=2**shift
                A=(2**(num_bits-1)-1)*step
                A_=(2**(num_bits-1))*step
            shift_update_flag=1
        #scale
        if config.scale:
             if config.scale_interval_update and not global_step>=update.shift_updatestep:
                 A=A*scale
                 A_=A_*scale
                 scale_update_flag=0
             else:
                 scale_update_flag=1
                 if Z>0:
                     scale=Z.div(A)
                 else:
                     scale=torch.tensor(1,dtype=torch.float)
                 A=Z
                 A_=A_*scale
                 
        else:
             scale_update_flag=0
             scale=torch.tensor(1,dtype=torch.float).cuda()
        
        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if  not scale_update_flag or not shift_update_flag:
            #output=torch.max(output,-A_+offset)
            #output=torch.min(output,A+offset)
            output.sub_(offset).div_(step*scale).round_().mul_(step).mul_(scale).add_(offset)
        else:
            step=2**shift
            output.sub_(offset).div_(step*scale).round_().mul_(step).mul_(scale).add_(offset)
        quantize_para[0]=shift
        quantize_para[1]=scale
        quantize_para[2]=offset
        if not update==None:
            update.getdiffnum(input,output)
        if shift_update_flag and config.shift_interval_update:
            update.update_shift(shift,global_step)
        '''
        if scale_update_flag and config.scale_interval_update:
            update.update_scale(scale,global_step)
        if offset_update_flag and config.offset_interval_update:
            update.update_offset(offset,global_step)
        '''
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None, None,None,None
import numpy as np
class QuantizeGrad(Function):

    @classmethod
    def forward(cls, ctx, input, num_bits=[8], inplace=False,quantize_para=torch.tensor([0,1,0],dtype=torch.float),global_step=None,update=None):
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.quantize_para=quantize_para
        ctx.global_step=global_step
        ctx.update=update
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = base_quantize().apply(grad_output, ctx.num_bits[0], ctx.inplace,ctx.quantize_para,ctx.global_step,ctx.update)
        #grad_input=grad_output
        if config.nbit_update=='mean_diff':
            if ctx.update.diffnum>config.update_th:
                ctx.num_bits[0]=torch.min(torch.tensor(32,dtype=torch.float).cuda(),ctx.num_bits[0]+8)
        
        return grad_input, None, None,None,None,None
class QuantizeGrad1(Function):

    @classmethod
    def forward(cls, ctx, input, num_bits=[8], inplace=False,quantize_para=torch.tensor([0,1,0],dtype=torch.float),global_step=None,update=None):
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.quantize_para=quantize_para
        ctx.global_step=global_step
        ctx.update=update
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = base_quantize().apply(grad_output, ctx.num_bits[0], ctx.inplace,ctx.quantize_para,ctx.global_step,ctx.update)
        #grad_input=grad_output
        if config.nbit_update=='mean_diff':
            if ctx.update.diffnum>config.update_th:
                ctx.num_bits[0]=torch.min(torch.tensor(32,dtype=torch.float).cuda(),ctx.num_bits[0]+8)
        
        return grad_input, None, None,None,None,None


def quantize(x, num_bits=8,inplace=False,quantize_para=torch.tensor([1,0,0],dtype=torch.float),global_step=None,update=None):
    return base_quantize().apply(x, num_bits,inplace,quantize_para,global_step,update)
    
def quantize_grad1(x, num_bits=8,inplace=False,quantize_para=torch.tensor([1,0,0],dtype=torch.float),global_step=None,update=None):
    return QuantizeGrad1().apply(x, num_bits,inplace,quantize_para,global_step,update)



def quantize_grad(x, num_bits=8,inplace=False,quantize_para=torch.tensor([1,0,0],dtype=torch.float),global_step=None,update=None):
    return QuantizeGrad().apply(x, num_bits,inplace,quantize_para,global_step,update)

class update(object):
    def __init__(self,mode='forward',shift_updatestep=0,offset_updatestep=0,scale_updatestep=0):
        self.mode=mode
        self.shift_updatestep=torch.tensor(shift_updatestep,dtype=torch.float).cuda()
        self.scale_updatestep=torch.tensor(scale_updatestep,dtype=torch.float).cuda()
        self.offset_updatestep=torch.tensor(offset_updatestep,dtype=torch.float).cuda()
        
        self.shift_mv=torch.tensor(0,dtype=torch.float).cuda()
        self.offset_mv=torch.tensor(0,dtype=torch.float).cuda()
        self.scale_mv=torch.tensor(0,dtype=torch.float).cuda()
        self.diffnum=torch.tensor(0,dtype=torch.float).cuda()
    def getdiffnum(self,origin,quantize):
        if self.mode=='backward' or 1:
            self.diffnum=torch.log2((torch.sum(torch.abs(quantize-origin))/torch.sum(torch.abs(origin)))+1)

    def updatefunction(self,mv,newnum,step):
        if step<config.step1:
            return config.alpha*newnum+(1-config.alpha)*mv,1+step
        elif step<config.step2:

            diffupdate1=config.alpha*(torch.abs(newnum-mv))
            #print('diffupdate1',diffupdate1)
            diffupdate2=config.sigma*self.diffnum**2
            #print('diffupdate2',diffupdate2)
            diffupdate=torch.max(diffupdate1,diffupdate2)
            return config.alpha*newnum+(1-config.alpha)*mv,torch.min(torch.max(torch.round(config.beta/diffupdate-config.gama),torch.tensor(1,dtype=torch.float).cuda()),torch.tensor(45,dtype=torch.float).cuda())+step
        else:
            diffupdate2=config.sigma*self.diffnum**2
            if diffupdate2>config.update_th/10:
                
                diffupdate1=config.alpha*(torch.abs(newnum-mv))
                diffupdate=torch.max(diffupdate1,diffupdate2)
                return config.alpha*newnum+(1-config.alpha)*mv,torch.min(torch.max(torch.round(config.beta/diffupdate-config.gama),torch.tensor(1,dtype=torch.float).cuda()),torch.tensor(45,dtype=torch.float).cuda())+step
            else: 
                return config.alpha*newnum+(1-config.alpha)*mv,config.step2+step

    def update_shift(self,newnum,step):
        if step<config.step1:
            self.shift_mv=newnum
        else:
            pass
            #print('shift_mv',self.shift_mv)
            #print('newnum',newnum)
        self.shift_mv,self.shift_updatestep=self.updatefunction(self.shift_mv,newnum,step)
        #print 'shift',self.shift_updatestep
'''
    def update_scale(self,newnum,step):
        self.scale_mv,self.scale_updatestep=self.updatefunction(self.scale_mv,newnum,step)
        #print 'scale',self.scale_updatestep
    def update_offset(self,newnum,step):
        self.offset_mv,self.offset_updatestep=self.updatefunction(self.offset_mv,newnum,step)
        #print('offset',self.offset_updatestep)
'''            
