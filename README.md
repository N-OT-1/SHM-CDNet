# SHM-CDNet core

最小变化检测代码：共享 **MambaVision-Base** 编码器、差分条件空间/通道门控、乘积融合、FPN解码、深监督及边缘损失。模型从已有冻结实现提取；未启用的新方向、搜索队列及实验记录不属于此仓库。

## 环境

使用Linux、NVIDIA CUDA和匹配的PyTorch/CUDA工具链。先安装适合本机CUDA的PyTorch与MMCV，再安装其余依赖；`mamba-ssm`可能需要本地CUDA编译。核心来源环境使用PyTorch 2.1/CUDA 11.8、MMCV 2.1、MMSegmentation 1.2.2；清理后的CPU兼容性检查另见提交说明，不代表完成新训练。

```bash
pip install -r requirements.txt
```

## 数据与预训练

自行准备数据，训练/验证/测试划分分别使用如下目录，三类文件同名：

```text
LEVIR-CD/
  train/{t1,t2,label}/*.png
  val/{t1,t2,label}/*.png
  test/{t1,t2,label}/*.png
```

输入为RGB图像；掩码支持0/1或0/255，灰度值用固定128阈值二值化。数据、预训练、训练权重均不附带。

从[NVIDIA MambaVision](https://github.com/NVlabs/MambaVision)获取Base分类预训练，注意使用对应许可证。路径通过环境变量指定：

```bash
export CD_DATA_ROOT=/path/to/LEVIR-CD
export MAMBAVISION_PRETRAINED=/path/to/mambavision_base_1k.pth.tar
python train.py configs/shm_base.py
```

示例为20k、batch2、lr5e-5、seed3407、验证F1选模。它是可移植运行入口；简化数据加载及显式掩码规则与旧实验管线不同，**不声称复现论文历史表格分数**。

## 评估

```bash
python evaluate.py configs/shm_base.py /path/to/best_checkpoint.pth --split val
```

默认阈值0.5，累计全划分TP/FP/FN/TN后计算前景F1、IoU、Precision、Recall、OA和背景FPR。确定方案与阈值后才使用独立test划分，示例不做阈值搜索。空分母指标记0。

## 文件与许可

- `shm/model.py`：变化检测网络及损失。
- `shm/mamba_vision.py`：保留原版权声明的MambaVision实现。
- `shm/data.py`、`shm/metrics.py`：成对数据加载与指标。
- `configs/shm_base.py`、`train.py`、`evaluate.py`：配置及入口。

保留[NVIDIA MambaVision](https://github.com/NVlabs/MambaVision)来源与仓库中的NVIDIA Source Code License-NC。代码及其衍生部分受非商业研究/评估限制；预训练权重许可见上游。本仓库不另行将上游代码改授MIT等许可证。
