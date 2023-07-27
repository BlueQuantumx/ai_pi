# 7 基于NumPy的MLP实现

直接参考 Pytorch 架构实现了基于计算图自动求导，理论上可支持任意网络结构，但是受限于测试时间仅实现了一些基础的网络模型和优化器。

```
|── mlp
|   ├── autograd.py 计算图相关
|   ├── tensor.py ndarray 的带有 grad 的包装，类似 Pytorch 的 Tensor
|   ├── nn 一些基础的网络模型
|   └── optim 优化器
```