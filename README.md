# mix-fix-point

This is a fix-point version of tensorpack (backend is tensorflow).

## train

1. install tensorflow-gpu v1.9.0 and tensorpack.
```
pip install tensorflow-gpu==1.9.0
pip install tensorpack
```

2. use the folds and files in this repository to replace the origin ones.
```
cp tensorflow/python/keras/layers/* YOUR_PATH_TO_TENSORFLOW/python/keras/layers/
cp tensorflow/python/ops/* YOUR_PATH_TO_TENSORFLOW/python/ops/
cp tensorflow/python/training/optimizer.py YOUR_PATH_TO_TENSORFLOW/python/training/
cp tensorpack/train/tower.py YOUR_PATH_TO_TENSORPACK/train/
```

3. change the parameters.


