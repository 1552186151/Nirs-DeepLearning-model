# Nirs-DeepLearning-model

本项目旨在利用深度学习技术对近红外光谱（NIR）数据进行回归分析，特别是预测梨果的糖含量。该项目实现了多种深度学习模型，包括卷积神经网络（ConvNet）、深度视觉Transformer（DeepVit）、视觉Transformer（VitNet）、以及专门为光谱数据设计的SpectFormer。项目同时提供了这些模型的迁移学习（Transfer Learning, TL）版本，以提升模型的泛化能力和小样本性能。模型精度经过测试可达到日常使用。

## 项目结构

项目主要包含以下文件：

- **ConvNet.py**: 实现了卷积神经网络（ConvNet）模型，用于从近红外光谱数据中提取特征并进行糖含量回归预测。
- **ConvNetTL.py**: ConvNet的迁移学习版本，适用于在已有预训练模型基础上对小数据集进行微调。
- **DeepVit.py**: 实现了深度视觉Transformer模型（DeepVit），通过自注意力机制提取全局特征，适合光谱数据的回归任务。
- **DeepVitTL.py**: DeepVit的迁移学习版本，通过微调最后几层使模型适应特定任务。
- **SpectFormer.py**: 实现了SpectFormer模型，一种专为光谱数据设计的Transformer变种，优化了光谱特征提取和回归能力。
- **SpectFormerTL.py**: SpectFormer的迁移学习版本，微调后适应梨果糖含量预测。
- **VitNet.py**: 实现了视觉Transformer（ViT）模型，通过对光谱数据应用自注意力机制进行糖含量回归。
- **VitNetTL.py**: VitNet的迁移学习版本，通过加载预训练模型并微调提高预测准确性。

## 环境依赖

请确保您已安装以下依赖项：

- Python 3.x
- PyTorch >= 1.8
- TorchVision
- 其他深度学习和科学计算库（如Numpy、Matplotlib等）

可以通过以下命令安装依赖：

```bash
pip install -r requirements.txt
