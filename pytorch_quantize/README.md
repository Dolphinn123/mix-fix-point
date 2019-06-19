# mix-fix-point

This is a fix-point version of pytorch.

## train

1. install pytorch
```
pip install torch
```
2. use the qnn.module in your project.
```
cp pytorch_quantize/ your_path/ -r
edit __init__.py
edit parameters in qnn.py and config.py

from .pytorch_quantize import qnn
use qnn.Linear or qnn.Conv2d

cp pytorch_quantize/ torch/nn/modules/
cp rnn.py torch/nn/modules/
```


