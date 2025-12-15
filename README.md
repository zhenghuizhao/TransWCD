# TransWCD: Transformer-based Weakly-Supervised Change Detection
## :notebook_with_decorative_cover: Code for Paper: Exploring Effective Priors and Efficient Models for Weakly-Supervised Change Detection [[arXiv]](https://arxiv.org/abs/2307.10853) 
Accepted to IEEE TGRS as: TransWCD: A Scene-Adaptive Joint-Constrained Framework for Weakly-Supervised Change Detection

## Update
| :zap:        | Higher-performing TransWCD baselines have been released, with F1 score of +2.47 on LEVIR-CD and +5.72 on DSIFN-CD compared to those mentioned in our paper. |
|--------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
##

## Abastract <p align="justify">
<blockquote align="justify">Weakly-supervised change detection (WSCD) aims to detect pixel-level changes with only image-level (i.e., scene-level) annotations. We develop TransWCD, a simple yet powerful transformer-based model, showcasing the potential of weakly-supervised learning in change detection. 
</p>


## :speech_balloon: TransWCD Architectures (Encoder-Only):
<p align="center">
    <img src="./tutorials/TransWCD.png" width="75%" height="75%">
</p>



##
## A. Preparations
### 1. Download Dataset
You can download [WHU-CD](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html), [DSIFN-CD](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset), [LEVIR-CD](http://chenhao.in/LEVIR/), and other CD datasets, then use our `data_and_label_processing` to convert these raw change detection datasets into cropped weakly-supervised change detection datasets.

Or use the processed weakly-supervised datasets from [`here`](https://drive.google.com/drive/folders/1Ee4T4-pOhZSe9NJ4av4cPBkXh6PX8w71?usp=sharing). Please cite their papers and ours.
``` bash
WSCD dataset with image-level labels:
├─A
├─B
├─label
├─imagelevel_labels.npy
└─list
```

### 2. Download Pre-trained Weights

Download the pre-trained weights from [SegFormer](https://github.com/NVlabs/SegFormer) and move them to `transwcd/pretrained/`.

### 3.Create and activate conda environment

```bash
conda create --name transwcd python=3.6
conda activate transwcd
pip install -r requirments.txt
```

##
## B. Train and Test
```bash
# train 
python train_transwcd_sa.py

```
You can modify the corresponding implementation settings `WHU.yaml`, `LEVIR.yaml`, and `DSIFN.yaml` in `train_transwcd.py` for different datasets.

###
```bash
# test
python test_sa.py
```
Please remember to modify the corresponding configurations in `test.py`, and the visual results can be found at `transwcd/results/`

##
## C. Performance and Best Models
| TransWCD      |    WHU-CD |             LEVIR-CD |             DSIFN-CD |
|:--------------:|:----------:|:---------------------:|:---------------------:|  
| Single-Stream | 67.81/[Best model](https://drive.google.com/file/d/1ZK9aNG-RG26ybLGAu9NqaG3-yPChL4Kx/view?usp=drive_link) | 51.06/[Best model](https://drive.google.com/file/d/1z_7e057spJPP4BW_6Ujz8ws-PZ-GWufA/view?usp=drive_link) | 57.28/[Best model](https://drive.google.com/file/d/1i9farsfLxDQBUxxfhjjOJz1zNgjavjoI/view?usp=drive_link) | 
| Dual-Stream   | 68.73/[Best model](https://drive.google.com/file/d/1us1TCqkSfjNjuubasXmRN0vBJ2OAI2vl/view?usp=drive_link) | 62.55/[Best model](https://drive.google.com/file/d/1cpr3ICsR4Ro5XtWA3I8RzDNzxkp1A_Fz/view?usp=drive_link) | 59.13/[Best model](https://drive.google.com/file/d/1oGSlT64WRuzyY1vqmXb1Y55T0VrBJnhn/view?usp=drive_link) | 

*Average F1 score / Best model

On both WHU-CD and LEVIR-CD datasets, the test performance closely matches that of the validation, with differences < 3% F1 score. 


## Citation
If it's helpful to your research, please kindly cite. Here is an example BibTeX entry:

``` bibtex
@ARTICLE{zhao2025transwcd,
  author={Zhao, Zhenghui and Ru, Lixiang and Wu, Chen and Wang, Di},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={TransWCD: Scene-Adaptive Joint Constrained Framework for Weakly Supervised Change Detection}, 
  year={2025},
  volume={63},
  pages={1-12},
  doi={10.1109/TGRS.2025.3545051}
}

@article{zhao2023exploring,
  title={Exploring Effective Priors and Efficient Models for Weakly-Supervised Change Detection},
  author={Zhao, Zhenghui and Ru, Lixiang and Wu, Chen},
  journal={arXiv preprint arXiv:2307.10853},
  year={2023}
}
```

## Acknowledgement
Thanks to these brilliant works [BGMix](https://github.com/tsingqguo/bgmix), [ChangeFormer](https://github.com/wgcban/ChangeFormer), and [SegFormer](https://github.com/NVlabs/SegFormer)!


