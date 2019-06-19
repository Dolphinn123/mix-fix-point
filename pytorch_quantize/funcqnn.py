import torch
import torch.nn as nn
import torch.nn.functional as F
import qfunction as QF
from torch.autograd import Variable as Vb
def LinearFunction(input, weight, bias=None,
                    num_bits=torch.tensor([8],dtype=torch.float),
                    num_bits_grad=torch.tensor([16],dtype=torch.float),
                    input_quantize_para=torch.tensor([1,0,0],dtype=torch.float),
                    weight_quantize_para=torch.tensor([1,0,0],dtype=torch.float),
                    grad_quantize_para=torch.tensor([1,0,0],dtype=torch.float),
                    step=[0]):
        
        qweight=QF.quantize(weight, num_bits=num_bits,quantize_para=weight_quantize_para,global_step=step)
        qinput=QF.quantize(input,num_bits=num_bits,quantize_para=input_quantize_para,global_step=step)
        output = F.linear(input, qweight, self.bias)
        if not self.num_bits_grad.numpy()==0:
            #self.numbit_grad_list=[self.num_bits_grad]
            output = QF.quantize_grad(output, num_bits=num_bits_grad,quantize_para=grad_quantize_para)
            #self.num_bits_grad=self.numbit_grad_list[0]
        step[0]=step[0]+1
        return output
linear=LinearFunction
