# VSDF: A variation-based spatiotemporal data fusion method
# Author: Chen Xu, Xiaoping Du, Zhenzhen Yan, Junjie Zhu, Shu Xu and Xiangtao Fan.
# Version: 0.11
# Date:24/11/2022

from guided_filter_pytorch.guided_filter import FastGuidedFilter,GuidedFilter
import collections
import os
from sklearn.cluster import KMeans
from skimage import feature
import gdal
import numpy as np
import torch


def rmse(a, b):
    a = a.astype("float")
    b = b.astype("float")

    band_num = a.shape[-1]
    r = []
    for band in range(band_num):
        r.append(np.sqrt(np.sum(((a[:, :, band] - b[:, :, band])) ** 2) / np.prod(a[:, :, band].shape)))
    # return np.sqrt(np.sum(((a - b) / np.max(a)) ** 2) / np.prod(a.shape))
    return r


# read from Tif file
def read_img(path):
    try:
        naip_ds = gdal.Open(path)
        nbands = naip_ds.RasterCount
        band_data = []
        for b in range(nbands):
            band_data.append(naip_ds.GetRasterBand(b + 1).ReadAsArray())
        img = np.dstack(band_data)
        return img.astype("float")
    except Exception:
        return None


# Save result into Tif
def save_img(array, path):
    driver = gdal.GetDriverByName("GTiff")
    if len(array.shape) == 2:
        dst = driver.Create(path, array.shape[1], array.shape[0], 1, 2)
        dst.GetRasterBand(1).WriteArray(array)
    else:
        # save all bands
        n_band = array.shape[-1]
        dst = driver.Create(path, array.shape[1], array.shape[0], n_band, 6)
        for b in range(n_band):
            dst.GetRasterBand(b + 1).WriteArray(array[:, :, b])
    del dst


# Downsample with origin size
def downsample_ori_size(array, factor):
    out = np.zeros(shape=array.shape).astype(array.dtype)
    for x in range(int(array.shape[1] / factor)):
        for y in range(int(array.shape[0] / factor)):
            out[y * factor:(y + 1) * factor, x * factor:(x + 1) * factor, :] = np.average(
                array[y * factor:(y + 1) * factor, x * factor:(x + 1) * factor, :])
    return out


# Downsample to small size
def downsample_small_size(array, factor):
    x_size = int(array.shape[1] / factor)
    y_size = int(array.shape[0] / factor)
    band_num = array.shape[2]
    out = np.zeros(shape=(y_size, x_size, band_num)).astype(array.dtype)
    for x in range(x_size):
        for y in range(y_size):
            for b in range(band_num):
                out[y, x, b] = np.average(array[y * factor:(y + 1) * factor, x * factor:(x + 1) * factor, b])
    return out


# enlarge downscaled image
def enlarge_size(array, factor):
    if len(array.shape) == 3:
        out = np.zeros(shape=(array.shape[0] * factor, array.shape[1] * factor, array.shape[-1])).astype(array.dtype)
    else:
        out = np.zeros(shape=(array.shape[0] * factor, array.shape[1] * factor)).astype(array.dtype)

    for x in range(array.shape[1]):
        for y in range(array.shape[0]):
            out[y * factor:(y + 1) * factor, x * factor:(x + 1) * factor] = array[y, x]

    return out


# get number of each type
def get_para_from_count(count, obj_list):
    para_list = []
    for obj in obj_list:
        para_list.append(count[obj])
    return para_list


# transform the array into tensor
def make_tensor(in_array):
    out_array = np.zeros(shape=[1, in_array.shape[-1], in_array.shape[0], in_array.shape[1]]).astype("float32")
    for band in range(in_array.shape[-1]):
        out_array[0][band] = in_array[:, :, band]
    return torch.from_numpy(out_array)


# transform the tensor into array
def make_array(tensor):
    tensor = tensor.detach().numpy()
    out_array = np.zeros(shape=[tensor.shape[2], tensor.shape[3], tensor.shape[1]]).astype(tensor.dtype)
    for band in range(tensor.shape[1]):
        out_array[:, :, band] = tensor[0][band]
    return out_array


def get_distance(x, y, x_window_size, y_window_size):
    x_dis = np.tile(range(x_window_size), (y_window_size, 1)) - x
    y_dis = np.tile(np.array(range(y_window_size)).reshape(y_window_size, 1), (1, x_window_size)) - y
    distance = x_dis ** 2 + y_dis ** 2
    return 1 + distance ** 0.5 / (x_window_size / 2)


def get_band_distance(x_in_window, y_in_window, array):
    distance = np.zeros(shape=array.shape[:2]).astype("float64")
    for band in range(array.shape[-1]):
        distance += abs(array[:, :, band] - array[y_in_window, x_in_window, band]) / (
                array[y_in_window, x_in_window, band] + 0.000001)
    return distance


def VSDF_clean(L1_path, M1_path, M2_path, out_folder, P, max_value, n_f=5, skip_edge=False):
    # create output folder if not exist
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # read image into array
    L1_img = read_img(L1_path)[:, :, :]
    M1_img = read_img(M1_path)[:, :, :]
    M2_img = read_img(M2_path)[:, :, :]

    L1_img = L1_img / max_value
    M1_img = M1_img / max_value
    M2_img = M2_img / max_value

    # size of low-resolution image
    x_n_sum = int(L1_img.shape[1] / P)
    y_n_sum = int(L1_img.shape[0] / P)

    # difference of low-resolution images
    delta_M_img_small = downsample_small_size(M2_img - M1_img, P)

    # number of bands (only images with 6 bands are tested)
    band_num = L1_img.shape[-1]

    """""""""""""""""""""""""""
    Step1: Evaluating input data
    """""""""""""""""""""""""""
    # RRI
    # Eq. (1)
    rmse_L1_M1 = np.average(rmse(downsample_small_size(L1_img, P), downsample_small_size(M1_img, P)))
    # Eq. (2)
    rmse_M2_M1 = np.average(rmse(downsample_small_size(M1_img, P), downsample_small_size(M2_img, P)))
    # Eq. (3)
    RRI = rmse_M2_M1 / rmse_L1_M1

    # pixels with significant error will be abandoned
    if RRI < 10:
        # difference between M1 and M2
        dif_M2_M1 = np.sqrt(np.sum(delta_M_img_small ** 2, axis=-1) / band_num) / rmse_M2_M1
        # difference between M1 and L2
        dif_M1_L1 = np.sqrt(np.sum((downsample_small_size(L1_img, P) - downsample_small_size(M1_img, P)) ** 2,
                                   axis=-1) / band_num) / rmse_L1_M1

        # error
        error_metrics = dif_M1_L1 / dif_M2_M1
        error_metrics = error_metrics / np.max(error_metrics)

        # threshold: 95%
        error_metrics_sorted = error_metrics.ravel().copy()
        error_metrics_sorted.sort()
        threshold = error_metrics_sorted[int(error_metrics_sorted.shape[0] * 0.95)]

        # find pixels with large error
        M_error_position = np.where(error_metrics >= threshold, 1, 0)

    # If RRI is larger than 10, we suppose the input low-resolution images are resampled from the high-resolution images.
    else:
        M_error_position = np.zeros(shape=(y_n_sum, x_n_sum))

    """""""""""""""""""""""""""
    Step2: Unmixing
    """""""""""""""""""""""""""
    # give the cluster number
    # If RRI is larger than 10, we suppose the input low-resolution images are resampled from the high-resolution images.
    if RRI > 10:
        n_clusters = 45
    # Eq. (9)
    else:
        n_clusters = int((3 - 1/RRI) * 2 * n_f)

    # guided filter
    # Parameter for guided filter can be changed according to the input images.
    M1_sr = make_tensor(M1_img)
    L1_sr = make_tensor(L1_img)
    delta_M_img_FGF_sr = FastGuidedFilter(P, 1e-7)(M1_sr, make_tensor(M2_img - M1_img), L1_sr)
    delta_M_img_GF_sr = GuidedFilter(P, 1e-7)(L1_sr, delta_M_img_FGF_sr)
    delta_M_img_GF = make_array(delta_M_img_GF_sr)
    u = np.zeros([L1_img.shape[0], L1_img.shape[1], band_num * 2])
    u[:, :, :band_num] = delta_M_img_GF
    u[:, :, band_num:] = L1_img.copy()

    # K-Means clustering
    classifer = KMeans(n_clusters=n_clusters)
    classes_img = classifer.fit_predict(u.reshape(L1_img.shape[0] * L1_img.shape[1], band_num * 2)).reshape(
        [L1_img.shape[0], L1_img.shape[1]])
    label_list = np.unique(classes_img.flatten())

    # unmixing
    # pre_delta_img: △F_AVC in Eq.(11)
    pre_delta_img = np.zeros(shape=L1_img.shape)
    para_list = []
    y_list = []
    for i in range(x_n_sum):
        for j in range(y_n_sum):
            para = get_para_from_count(collections.Counter(classes_img[j * P:(j + 1) * P, i * P:(i + 1) * P].flatten()),
                                       label_list)
            if M_error_position[j, i] == 0:
                para_list.append(para)
                y_list.append(delta_M_img_small[j, i, :])

    para_mat = np.mat(para_list)
    y_mat = np.mat(y_list)

    for b in range(band_num):
        b_para = np.linalg.lstsq(para_mat, y_mat[:, b] * P * P, rcond=None)[0]
        for i in range(label_list.shape[0]):
            pre_delta_img[:, :, b] += np.where(classes_img == label_list[i], b_para[i], 0)

    """""""""""""""""""""""""""
    Distributing the reisiduals
    """""""""""""""""""""""""""
    # determine the loop time
    # Eq.(14)
    if RRI > 1:
        loop_n = int((1 - 1/RRI) ** 2 * 5)
        for i in range(loop_n):
            # Eq. (12)
            difference_small = downsample_small_size(M2_img - M1_img + L1_img, P) - downsample_small_size(
                pre_delta_img + L1_img, P)

            for band in range(band_num):
                difference_small[:, :, band] = np.where(M_error_position == 1, 0, difference_small[:, :, band])

            # Eq. (13)
            # Parameter for guided filter can be changed according to the input images.
            difference = enlarge_size(difference_small, P)
            difference_FGF_sr = FastGuidedFilter(int(P / 2), 1e-3)(M1_sr, make_tensor(difference), L1_sr)
            difference_GF_sr = GuidedFilter(1, 1e-3)(L1_sr, difference_FGF_sr)
            difference_new = make_array(difference_GF_sr)

            pre_delta_img = difference_new + pre_delta_img


    """""""""""""""""""""""""""
    Introducing neighbouring information
    """""""""""""""""""""""""""
    refine_window = 31
    similar_pixels = 30
    margin = int((refine_window - 1) / 2)

    pre_delta_img_new = pre_delta_img.copy()

    for y in range(pre_delta_img.shape[0]):
        for x in range(pre_delta_img.shape[1]):
            window_x_1 = max(0, x - margin)
            window_x_2 = min(L1_img.shape[1], x + margin + 1)
            window_y_1 = max(0, y - margin)
            window_y_2 = min(L1_img.shape[0], y + margin + 1)

            x_in_window = x - window_x_1
            y_in_window = y - window_y_1

            pre_delta_img_window = pre_delta_img[window_y_1:window_y_2, window_x_1:window_x_2, :]
            L1_img_window = L1_img[window_y_1:window_y_2, window_x_1:window_x_2, :]

            band_distance = get_band_distance(x_in_window, y_in_window, L1_img_window)
            k = band_distance.copy()
            k_sorted = k.ravel()
            k_sorted.sort()
            threshold = k_sorted[similar_pixels]

            distance = get_distance(x_in_window, y_in_window, window_x_2 - window_x_1, window_y_2 - window_y_1)
            assert (distance[y_in_window, x_in_window] == 1)

            distance_result = np.where(band_distance <= threshold, 1 / distance, 0)
            distance_result = distance_result / np.sum(distance_result)

            for band in range(band_num):
                # Eq. (15)
                pre_delta_img_new[y, x, band] = np.sum(pre_delta_img_window[:, :, band] * distance_result)

    pre_delta_img = pre_delta_img_new
    
    """""""""""""""""""""""""""
    Edge fusion
    """""""""""""""""""""""""""
    if not skip_edge:
        edges = feature.canny(L1_img[:, :, 0], sigma=0)
        for b in range(1, band_num):
            edges += feature.canny(L1_img[:, :, b], sigma=0)

        # Eq. (16)
        pre_delta_img_sr = FastGuidedFilter(2, 0.1)(M1_sr, make_tensor(pre_delta_img), L1_sr)
        pre_delta_img_sr = GuidedFilter(2, 0.1)(L1_sr, pre_delta_img_sr)

        # Eq. (17)
        for band in range(band_num):
            pre_delta_img[:, :, band] = np.where(edges, make_array(pre_delta_img_sr)[:, :, band],
                                                 pre_delta_img[:, :, band])

    pre_L2_img = pre_delta_img + L1_img

    for b in range(band_num):
        pre_L2_img[:, :, b] = np.where(pre_L2_img[:, :, b] > 1, 1, pre_L2_img[:, :, b])
        pre_L2_img[:, :, b] = np.where(pre_L2_img[:, :, b] < 0, 0, pre_L2_img[:, :, b])

    save_img(pre_L2_img * max_value, os.path.join(out_folder, "VSDF_result.tif"))

    return os.path.join(out_folder, "VSDF_result.tif")


# ratio between the fine and coarse images
P = 20

# max value for the image, minimum is 0 as default
max_value = 10000

L1_path =
L2_path =
M1_path =
M2_path =
out_folder =

# if the input datasets are not reliable, a smaller n_f is recommended
n_f = 5 # in Eq.(9)

# if the input datasets are not reliable, skip_edge is recommended to be set as True
skip_edge = False

import time
a = time.time()
VSDF_path = VSDF_clean(L1_path, M1_path, M2_path, out_folder, P, max_value, n_f=n_f, skip_edge=skip_edge)
print(time.time()-a)


