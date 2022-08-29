# Paddle-Cutpaste

## 目录

- [1. 简介]()
- [2. 数据集]()
- [3. 复现精度]()
- [4. 模型数据与环境]()
    - [4.1 目录介绍]()
    - [4.2 准备环境]()
    - [4.3 准备数据]()
- [5. 开始使用]()
    - [5.1 模型训练]()
    - [5.2 模型评估]()
    - [5.3 模型预测]()
- [6. 自动化测试脚本]()
- [7. LICENSE]()
- [8. 模型信息]()

## 1. 简介
论文中提出了一种图像缺陷异常检测模型，可以不依赖于异常数据来检测未知的异常缺陷。框架整体属于 two-stage：首先通过自监督学习方法来学习正常图像的表示，然后基于学习到的图像表示来构建单分类器。CutPaste 技术主要是通过图片剪切然后再粘贴至其它位置来构造负样本。实验部分在 MVTec 数据集中验证了模型对图片缺陷检测的有效性，如果不使用预训练那么可以比当前 baselines 的 AUC 提升 3.1，如果基于 ImageNet 进行迁移学习那么 AUC 可以达到 96.6。

[aistudio在线运行](https://aistudio.baidu.com/aistudio/projectdetail/4460691)

**论文:** [CutPaste: Self-Supervised Learning for Anomaly Detection and Localization](https://arxiv.org/pdf/2104.04015v1.pdf)

**参考repo:** [pytorch-cutpaste](https://github.com/Runinho/pytorch-cutpaste)


## 2. 数据集

MVTec AD是MVtec公司提出的一个用于异常检测的数据集。与之前的异常检测数据集不同，该数据集模仿了工业实际生产场景，并且主要用于unsupervised anomaly detection。数据集为异常区域都提供了像素级标注，是一个全面的、包含多种物体、多种异常的数据集。数据集包含不同领域中的五种纹理以及十种物体，且训练集中只包含正常样本，测试集中包含正常样本与缺陷样本，因此需要使用无监督方法学习正常样本的特征表示，并用其检测缺陷样本。

数据集下载链接：[AiStudio数据集](https://aistudio.baidu.com/aistudio/datasetdetail/116034) 解压到data文件夹下


## 3. 复现精度

| defect_type   |   CutPaste(3-way)(复现) |  CutPaste (3-way) |
|:--------------|-----------------------:|------------------:|
| bottle        |                  100.0 |              98.3 |
| cable         |                   94.7 |              80.6 |
| capsule       |                   89.9 |              96.2 |
| carpet        |                   94.3 |              93.1 |
| grid          |                   98.4 |              99.9 |
| hazelnut      |                   98.8 |              97.3 |
| leather       |                  100.0 |             100.0 |
| metal_nut     |                   95.9 |              99.3 |
| pill          |                   91.7 |              92.4 |
| screw         |                   82.3 |              86.3 |
| tile          |                   99.5 |              93.4 |
| toothbrush    |                   97.7 |              98.3 |
| transistor    |                   93.6 |              95.5 |
| wood          |                   99.1 |              98.6 |
| zipper        |                   99.9 |              99.4 |
| average       |                   95.7 |              95.2 |


## 4. 模型数据与环境

### 4.1 目录介绍

```
    |--images                         # 测试使用的样例图片，两张
    |--deploy                         # 预测部署相关
        |--export_model.py            # 导出模型
        |--infer.py                   # 部署预测
    |--data                           # 训练和测试数据集
    |--lite_data                      # 自建立的小数据集，含有bottle
    |--logdirs                        # 训练train和测试eval打印的日志信息  
    |--eval                           # eval输出文件
    |--models                         # 训练的模型权值
    |--test_tipc                      # tipc代码
    |--cutpaste.py                    # cutpaste代码
    |--dataset.py                     # 数据加载
    |--density.py                     # 高斯聚类代码
    |--model.py                       # resnet模型
    |--predict.py                     # 预测代码
    |--eval.py                        # 评估代码
    |--train.py                       # 训练代码
    |----README.md                    # 用户手册
```

### 4.2 准备环境

- 框架：
  - PaddlePaddle >= 2.3.1
- 环境配置：使用`pip install -r requirement.txt`安装依赖。


### 4.3 准备数据

- 全量数据训练：
  - 数据集下载链接：[AiStudio数据集](https://aistudio.baidu.com/aistudio/datasetdetail/116034) 解压到data文件夹下
- 少量数据训练：
  - 无需下载数据集，直接使用lite_data里的数据
  
## 5. 开始使用
### 5.1 模型训练

- 全量数据训练：
  - `python train.py --type all --batch_size 64 --test_epochs 10 --head_layer 1 --seed 102`
- 少量数据训练：
  - `python train.py --data_dir lite_data --type bottle --epochs 10 --test_epochs 5 --batch_size 5`
  
模型训练权重保存在models文件下，日志保存在logdirs文件下

可以将训练好的模型权重[下载](https://aistudio.baidu.com/aistudio/datasetdetail/162384) 解压为models文件放在本repo/下，直接对模型评估和预测

### 5.2 模型评估(通过5.1完成训练后)

- 全量数据模型评估：`python eval.py --type all --data_dir data --head_layer 8 --density paddle`
- 少量数据模型评估：`python eval.py --data_dir lite_data --type bottle`

评估会生成验证结果保存在项目evel文件下

### 5.3 模型预测（需要预先完成5.1训练以及5.2的评估）

- 模型预测：`python predict.py --data_type bottle --img_file images/good.png`

结果如下：
```
预测结果为：正常 预测分数为：21.0923
```

- 基于推理引擎的模型预测：
```
python deploy/export_model.py
python deploy/infer.py --data_type bottle --img_path images/good.png
```
结果如下：
```
> python deploy/export_model.py
inference model has been saved into deploy

> python deploy/infer.py --data_type bottle --img_path images/good.png
image_name: images/good.png, data is normal, score is 21.092235565185547, threshold is 57.449745178222656
```


## 6. 自动化测试脚本
- tipc 所有代码一键测试命令（少量数集）
```
bash test_tipc/test_train_inference_python.sh test_tipc/configs/resnet18/train_infer_python.txt lite_train_lite_infer 
```

结果日志如下
```
[Run successfully with command - python3.7 train.py --type bottle --test_epochs 3 --model_dir=./log/resnet18/lite_train_lite_infer/norm_train_gpus_0 --epochs=2   --batch_size=1!]
[Run successfully with command - python3.7 eval.py --type bottle --pretrained=./log/resnet18/lite_train_lite_infer/norm_train_gpus_0/model-bottle.pdparams! ]
[Run successfully with command - python3.7 deploy/export_model.py  --pretrained=./log/resnet18/lite_train_lite_infer/norm_train_gpus_0/model-bottle.pdparams --save-inference-dir=./log/resnet18/lite_train_lite_infer/norm_train_gpus_0!  ]
[Run successfully with command - python3.7 deploy/infer.py --use-gpu=True --model-dir=./log/resnet18/lite_train_lite_infer/norm_train_gpus_0 --batch-size=1   --benchmark=False > ./log/resnet18/lite_train_lite_infer/python_infer_gpu_batchsize_1.log 2>&1 !  ]
```

## 7. LICENSE

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。

## 8. 模型信息

| 信息 | 描述 |
| --- | --- |
| 作者 | Lieber|
| 日期 | 2022年8月 |
| 框架版本 | PaddlePaddle==2.3.1 |
| 应用场景 | 异常检测 |
| 硬件支持 | GPU、CPU |
| 在线体验 | [notebook](https://aistudio.baidu.com/aistudio/projectdetail/4460691)
