# VSDF: A variation-based spatiotemporal data fusion method
Python code for VSDF (v0.1).

Requirement for Python:
* Python 3.6
* gdal 3.1.4
* torch 1.10.2
* numpy 1.19.0
* skimage 0.17.2
* sklearn 0.24.2
* guided-filter-pytorch 3.7.5 

Requirement for input images:
* Format: recognized by GDAL, GeoTif is recomended
* Size: fine image and coarse images should be in the same size (e.g., 800*800)
* Band number: only 6 bands is tested

Input:
1. L1: Fine image at T1
2. M1: Coarse image at T1
3. M2: Coarse image at T2

Output:
* Fusion_L2 Tif
