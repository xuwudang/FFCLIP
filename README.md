## 多模态对比学习在弱监督语义分割的方法研究

## 数据集

### VOC dataset

#### 1. 下载数据集

``` bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar –xvf VOCtrainval_11-May-2012.tar
```
#### 2. 下载augmented annotations
augmented annotations来自 [SBD dataset](http://home.bharathh.info/pubs/codes/SBD/download.html) 。增强注释的下载链接在 [DropBox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip？dl=0)。下载 `SegmentationClassAug.zip` 后，您应将其解压并移动到 `VOCdevkit/VOC2012`。因此，目录结构应为

``` bash
VOCdevkit/
└── VOC2012
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages
    ├── SegmentationClass
    ├── SegmentationClassAug
    └── SegmentationObject
```

### COCO dataset

#### 1. 下载数据集
``` bash
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
```
解压下载的文件后，为了方便起见，我建议按照 VOC 格式对其进行整理。

``` bash
MSCOCO/
├── JPEGImages
│    ├── train
│    └── val
└── SegmentationClass
     ├── train
     └── val
```

#### 2. 为 COCO 生成 VOC 风格的分割标签
要为 COCO 数据集生成 VOC 风格的分割标签，您可以使用此 [repo](https://github.com/alicranck/coco2voc)中提供的脚本。或者，直接从 [Google Drive](https://drive.google.com/file/d/1pRE9SEYkZKVg0Rgz2pi9tg48j7GlinPV/view) 下载已生成的掩码。

### 创建并激活 conda 环境

```bash
conda create --name py38 python=3.8
conda activate py38
pip install -r requirments.txt
```


### 下载预训练的 CLIP-VIT/16 权重

从官方  [link](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)下载预训练的 CLIP-VIT/16 权重。

然后，将此模型移动到 `pretrained/` 目录下。


### 修改配置
根据您的路径，需要修改三个参数：

(1) root_dir: `your/path/VOCdevkit/VOC2012` or `your/path/MSCOCO`

(2) name_list_dir: `your/path/FFCLIP/datasets/voc` or `your/path/FFCLIP/datasets/coco`

(3) clip_pretrain_path: `your/path/FFCLIP/pretrained/ViT-B-16.pt`

VOC在此修改 `configs/voc_attn_reg.yaml`.

COCO在此修改 `configs/coco_attn_reg.yaml`. 

### Train
要开始训练，只需运行以下代码。
```bash
# train on voc
python scripts/dist_clip_voc.py --config your/path/FFCLIP/configs/voc_attn_reg.yaml
# train on coco
python scripts/dist_clip_coco.py --config your/path/FFCLIP/configs/coco_attn_reg.yaml
```

### Inference
要进行inference，首先在 `test_msc_flip_voc` 或 `test_msc_flip_voc` 中修改推理模型路径 `--model_path` 。

```bash
# inference on voc
python test_msc_flip_voc.py --model_path your/inference/model/path/FFCLIP_model_iter_30000.pth
# inference on coco
python test_msc_flip_coco.py --model_path your/inference/model/path/FFCLIP_model_iter_80000.pth
```
