# VSDF: A variation-based spatiotemporal data fusion method
Python code for VSDF
Available time: 23/10/2022

## Requirement for Python:
* Python 3.6
* gdal 3.1.4
* torch 1.10.2
* numpy 1.19.0
* skimage 0.17.2
* sklearn 0.24.2
* guided-filter-pytorch 3.7.5 

## Requirement for input images:
* Format: recognized by GDAL, GeoTif is recomended
* Size: fine image and coarse images should be in the same size (e.g., 800*800)
* Band number: only 6 bands is tested

## Input:
1. L1: Fine image at T1
2. M1: Coarse image at T1
3. M2: Coarse image at T2

## Output:
* Fusion_L2 Tif


## Cite
If you find VSDF is helpful, please cite the following work:
VSDF [[Paper]](https://www.sciencedirect.com/science/article/pii/S0034425722004151) [[Code]](https://github.com/ChenXuAxel/VSDF)
```
@article{XU2022113309,
title = {VSDF: A variation-based spatiotemporal data fusion method},
journal = {Remote Sensing of Environment},
volume = {283},
pages = {113309},
year = {2022},
issn = {0034-4257},
doi = {https://doi.org/10.1016/j.rse.2022.113309},
}
```

## NEW! FastVSDF 
Speed up VSDF with 40+ times! [[Paper]](https://ieeexplore.ieee.org/document/10399795) [[Code]](https://github.com/ChenXuAxel/FastVSDF)
```
@ARTICLE{10399795,
  author={Xu, Chen and Du, Xiaoping and Fan, Xiangtao and Jian, Hongdeng and Yan, Zhenzhen and Zhu, Junjie and Wang, Robert},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={FastVSDF: An Efficient Spatiotemporal Data Fusion Method for Seamless Data Cube}, 
  year={2024},
  doi={10.1109/TGRS.2024.3353758}}
```
