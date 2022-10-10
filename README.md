# VSDF
Requirement for input images:
  Format: recognized by GDAL, GeoTif is recomended
  Size: fine image and coarse images should be in the same size (e.g., 800*800)
  Band number: only 6 bands is tested

Input:
  L1: Fine image at T1
  M1: Coarse image at T1
  M2: Coarse image at T2

Output:
  Fusion_L2 Tif
