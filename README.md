# ViX     
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vision-xformers-efficient-attention-for-image/image-classification-on-cifar-10)](https://paperswithcode.com/sota/image-classification-on-cifar-10?p=vision-xformers-efficient-attention-for-image)
## Vision Xformers: Efficient Attention for Image Classification

![image](https://user-images.githubusercontent.com/15833382/172207987-e07bb02b-4a1e-430c-a1bf-bc78af87976b.png)


We use Linear Attention mechanisms to replace quadratic attention in ViT for image classification. We show that models using linear attention and CNN embedding layers need less parameters and low GPU requirements for achieving good accuracy. These improvements can be used to democratize the use of transformers by practitioners who are limited by data and GPU.

Hybrid ViX uses convolutional layers instead of linear layer for generating embeddings

Rotary Postion Embedding (RoPE) is also used in our models instead of 1D learnable position embeddings

Nomenclature:
We replace the X in ViX with the starting alphabet of the attention mechanism used
Eg. When we use Performer in ViX, we replace the X with P, calling it ViP (Vision Performer)

'Hybrid' prefix is used in models which uses convolutional layers instead of linear embeddding layer. 

We have added RoPE in the title of models which used Rotary Postion Embedding

The code for using all for these models for classification of CIFAR 10/Tiny ImageNet dataset is provided

### Models

- Vision Transformer (ViT)
- Vision Linformer (ViL)
- Vision Performer (ViP)
- Vision Nyströmformer (ViN)
- FNet
- Hybrid Vision Transformer (HybridViT)
- Hybrid Vision Linformer (HybridViL)
- Hybrid Vision Performer (HybridViP)
- Hybrid Vision Nyströmformer (HybridViN)
- Hybrid FNet
- LeViN (Replacing Transformer in LeViT with Nyströmformer)
- LeViP (Replacing Transformer in LeViT with Performer)
- CvN (Replacing Transformer in CvT with Nyströmformer)
- CvP (Replacing Transformer in CvT with Performer)
- CCN (Replacing Transformer in CCT with Nyströmformer)
- CCP(Replacing Transformer in CCT with Performer)

We have adapted the codes for ViT and linear transformers from @lucidrains 

## Install
```bash
$ pip install vision-xformer
```
## Usage
### Image Classification
#### Vision Nyströmformer (ViN)

```python
import torch, vision_xformer
from vision_xformer import ViN

model = ViN(
    image_size = 32,
    patch_size = 1,
    num_classes = 10,             
    dim = 128,  
    depth = 4,             
    heads = 4,      
    mlp_dim = 256,
    num_landmarks = 256,
    pool = 'cls',
    channels = 3,
    dropout = 0.,
    emb_dropout = 0.
    dim_head = 32
)

img = torch.randn(1, 3, 32, 32)

preds = model(img) # (1, 10)
```

#### Vision Performer (ViP)

```python
import torch, vision_xformer
from vision_xformer import ViP

model = ViP(
    image_size = 32,
    patch_size = 1,
    num_classes = 10,             
    dim = 128,  
    depth = 4,             
    heads = 4,      
    mlp_dim = 256,
    dropout = 0.25,
    dim_head = 32
)

img = torch.randn(1, 3, 32, 32)

preds = model(img) # (1, 10)
```

#### Vision Linformer (ViL)

```python
import torch, vision_xformer
from vision_xformer import ViL

model = ViL(
    image_size = 32,
    patch_size = 1,
    num_classes = 10,             
    dim = 128,  
    depth = 4,             
    heads = 4,      
    mlp_dim = 256,
    dropout = 0.25,
    dim_head = 32
)

img = torch.randn(1, 3, 32, 32)

preds = model(img) # (1, 10)
```
## Parameters

- `image_size`: int.  
Size of input image. If you have rectangular images, make sure your image size is the maximum of the width and height
- `patch_size`: int.  
Number of patches. `image_size` must be divisible by `patch_size`.
- `num_classes`: int.  
Number of classes to classify.
- `dim`: int.  
Final dimension of token emeddings after linear layer. 
- `depth`: int.  
Number of layers.
- `heads`: int.  
Number of heads in multi-head attention
- `mlp_dim`: int.  
Embedding dimension in the MLP (FeedForward) layer. 
- `num_landmarks`: int.
Number of landmark points. Use one-fourth the number of patches.
- `pool`: str.
Pool type must be either `'cls'` (cls token) or `'mean'` (mean pooling)
- `dropout`: float between `[0, 1]`, default `0.`.  
Dropout rate. 
- `dim_head`: int.  
Embedding dimension of token in each head of mulit-head attention.


More information about these models can be obtained from our paper : [ArXiv Paper](https://arxiv.org/abs/2107.02239), [WACV 2022 Paper](https://openaccess.thecvf.com/content/WACV2022/html/Jeevan_Resource-Efficient_Hybrid_X-Formers_for_Vision_WACV_2022_paper.html)

If you wish to cite this, please use:
```
@misc{jeevan2021vision,
      title={Vision Xformers: Efficient Attention for Image Classification}, 
      author={Pranav Jeevan and Amit Sethi},
      year={2021},
      eprint={2107.02239},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
@InProceedings{Jeevan_2022_WACV,
    author    = {Jeevan, Pranav and Sethi, Amit},
    title     = {Resource-Efficient Hybrid X-Formers for Vision},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {2982-2990}
}
```
