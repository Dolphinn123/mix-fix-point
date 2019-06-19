import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import fp16qfunction as QF
import math
from torch.autograd import Variable as Vb
def relu(x):
    return F.relu(x.half()).float()
def sigmoid(x):
    return torch.sigmoid(x.half()).float()
def softmax(x,dim):
    return F.softmax(x.half(),dim).float()
class Linear(nn.Linear):
    """quantized Linear."""

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_grad=8,split=1):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.updateweight=QF.update()
        self.updateinput=QF.update()
        self.updategrad=QF.update()

        self.register_buffer('num_bits',torch.tensor([num_bits],dtype=torch.float))
        #self.num_bits_weight = num_bits_weight or num_bits 
        self.register_buffer('num_bits_grad',torch.tensor([num_bits_grad],dtype=torch.float))

        
        self.register_buffer('step',torch.tensor(0,dtype=torch.float))
        self.register_buffer('weight_quantize_para',torch.tensor([0,1,0],dtype=torch.float))
        self.register_buffer('input_quantize_para',torch.tensor([0,1,0],dtype=torch.float))
        self.register_buffer('grad_quantize_para',torch.tensor([0,1,0],dtype=torch.float))
        self.split=split
        self.split_size=math.ceil(out_features/split)


    def forward(self, input):
        if self.split>1:
            qweightlist=[QF.quantize(x,num_bits=self.num_bits,quantize_para=self.weight_quantize_para,global_step=self.step,update=self.updateweight) for x in torch.split(self.weight,self.split_size)]
        else:
            qweight,weight_scale=QF.quantize(self.weight, num_bits=self.num_bits,quantize_para=self.weight_quantize_para,global_step=self.step,update=self.updateweight)
        qinput,input_scale=QF.quantize(input,num_bits=self.num_bits,quantize_para=self.input_quantize_para,global_step=self.step,update=self.updateinput)
        if self.split>1:
             
            output=torch.cat([F.linear(qinput,qweight[0]).half().float()*input_scale*qweight[1] for qweight in qweightlist],1)
        else:
            output = F.linear(qinput, qweight).half().float()*weight_scale*input_scale
        if self.bias is not None:
            output=(output+self.bias)

        if not self.num_bits_grad==0:
            #self.numbit_grad_list=[self.num_bits_grad]
            output = QF.quantize_grad(output, num_bits=self.num_bits_grad,quantize_para=self.grad_quantize_para,global_step=self.step,update=self.updategrad)
            #self.num_bits_grad=self.numbit_grad_list[0]
        if self.training:
            self.step=self.step+1
        return output

class Conv1d(nn.Conv1d):
    """quantized Conv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, num_bits_grad=8):
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        
        self.updateweight=QF.update()
        self.updateinput=QF.update()
        self.updategrad=QF.update()

        self.register_buffer('num_bits',torch.tensor([num_bits],dtype=torch.float))
        #self.num_bits_weight = num_bits_weight or num_bits
        self.register_buffer('num_bits_grad',torch.tensor([num_bits_grad],dtype=torch.float))

        self.register_buffer('step',torch.tensor(0,dtype=torch.float))
        self.register_buffer('weight_quantize_para',torch.tensor([0,1,0],dtype=torch.float))
        self.register_buffer('input_quantize_para',torch.tensor([0,1,0],dtype=torch.float))
        self.register_buffer('grad_quantize_para',torch.tensor([0,1,0],dtype=torch.float))

    def forward(self, input):
        qweight,scale_weight=QF.quantize(self.weight, num_bits=self.num_bits,quantize_para=self.weight_quantize_para,global_step=self.step,update=self.updateweight)
        qinput,scale_input=QF.quantize(input,num_bits=self.num_bits,quantize_para=self.input_quantize_para,global_step=self.step,update=self.updateinput)

        output = F.conv1d(qinput, qweight,None,self.stride,
                              self.padding, self.dilation, self.groups).half().float()*scale_weight*scale_input
        if self.bias is not None:
            output=(output.transpose(1,2)+self.bias).transpose(1,2)

        if not self.num_bits_grad==0:
            output = QF.quantize_grad(output, num_bits=self.num_bits_grad,quantize_para=self.grad_quantize_para,global_step=self.step,update=self.updategrad)
        if self.training:
            self.step=self.step+1
        return output

class Conv2d(nn.Conv2d):
    """quantized Conv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=16, num_bits_grad=20):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)

        self.updateweight=QF.update()
        self.updateinput=QF.update()
        self.updategrad=QF.update()

        self.register_buffer('num_bits',torch.tensor([num_bits],dtype=torch.float))
        #self.num_bits_weight = num_bits_weight or num_bits
        self.register_buffer('num_bits_grad',torch.tensor([num_bits_grad],dtype=torch.float))

        self.register_buffer('step',torch.tensor(0,dtype=torch.float))
        self.register_buffer('weight_quantize_para',torch.tensor([1,0,0],dtype=torch.float))
        self.register_buffer('input_quantize_para',torch.tensor([1,0,0],dtype=torch.float))
        self.register_buffer('grad_quantize_para',torch.tensor([1,0,0],dtype=torch.float))

    def forward(self, input):
        qweight,scale_weight=QF.quantize(self.weight, num_bits=self.num_bits,quantize_para=self.weight_quantize_para,global_step=self.step,update=self.updateweight)
        qinput,scale_input=QF.quantize(input,num_bits=self.num_bits,quantize_para=self.input_quantize_para,global_step=self.step,update=self.updateinput)
        loss=torch.abs((qweight-self.weight)/self.weight)
        #print torch.mean(torch.abs((qweight-self.weight)/self.weight))
        #print torch.mean(torch.abs((qinput-input)/input))
        output = F.conv2d(qinput, qweight,None, self.stride,
                              self.padding, self.dilation, self.groups).half().float()*scale_weight*scale_input
        if self.bias is not None:
            output=(output.transpose(1,3)+self.bias).transpose(1,3)
       
        if not self.num_bits_grad==0:
            output = QF.quantize_grad(output, num_bits=self.num_bits_grad,quantize_para=self.grad_quantize_para,global_step=self.step,update=self.updategrad)

        if self.training:
            self.step=self.step+1
        return output

# test
'''
input=Vb(torch.from_numpy(np.random.uniform(low=0.1,high=0.5,size=[32,32,128]))).cuda().float()
a=Conv1d(32,32,3).cuda()

b=nn.Conv1d(32,32,3).cuda()
b.bias.data=a.bias.data
b.weight.data=a.weight.data
aout=a(input)
bout=b(input)
print(torch.mean(aout-bout))
'''
'''
loss=torch.abs((aout-bout)/bout)
id=torch.argmax(loss).cpu().item()
print 'max',id,loss.max()
print aout.view(-1)[id]
print bout.view(-1)[id]
print 'mean',torch.mean(loss)
'''

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.fc=Linear(3,3,num_bits_grad=4)
    def forward(self,x):
        return self.fc(x)
'''
#a=Linear(3,3,num_bits_grad=4).cuda()
a=Test().cuda()
#b=QLinear(3,3,num_bits_grad=8)
import numpy as np
k=Vb(torch.from_numpy(np.array([[10.0,20.0,30.0]]))).float().cuda()
x=Vb(torch.from_numpy(np.array([[1.0,4.0,128.0]]))).float().cuda()
loss=(a(x)-k).mean()
loss.backward()
'''
#print a(x)
#loss.backward()
#a.num_bits=a.num_bits-14
#print a(x)
#loss1=(a(x)-k).mean()
#loss1.backward()
'''
#print a.numbit_grad_list
#print a.weight_quantize_para
#print a.step
'''
#print a.state_dict()
#torch.save(a.state_dict(),'a.pth')
'''
b=Linear(3,3,num_bits_grad=8)
b.load_state_dict(torch.load('a.pth'),False)
print b.weight_quantize_para
'''
'''
a=nn.Linear(3,3)
a.load_state_dict(torch.load('a.pth'),False)
print a.weight
'''

