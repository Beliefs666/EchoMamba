import os
import configparser
import SimpleITK as sitk

import matplotlib.pylab as plt
import matplotlib.pyplot as pltt
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
import cv2 as cv


# %%
def smooth_binary_image_and_keep_binary(im_array):
    # 进行中值滤波平滑处理
    smoothed_image = cv2.medianBlur(im_array, 9)  # 5 是滤波器的大小，可以根据需要调整

    # 对平滑后的图像进行二值化处理
    _, smoothed_binary_image = cv2.threshold(smoothed_image, 127, 255, cv2.THRESH_BINARY)
    return smoothed_binary_image


# %%
def split_gt(filepath, file_name, view, save_path=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image = sitk.ReadImage(str(filepath))
    file_name = str(file_name)
    frames = sitk.GetArrayFromImage(image).shape[0]
    for frame in range(frames):
        im_array = np.squeeze(sitk.GetArrayFromImage(image)[frame, ...])
        im_array[im_array == 1] = 0
        im_array[im_array == 2] = 0
        im_array[im_array > 0] = 255
        im_array = im_array.astype(np.uint8)
        # im_array = smooth_binary_image_and_keep_binary(im_array)
        im_array = cv2.resize(im_array, (512, 512), interpolation=cv2.INTER_NEAREST)

        _, im_array = cv2.threshold(im_array, 127, 255, cv2.THRESH_BINARY)
        # 平滑处理是否有效
        im_array = smooth_binary_image_and_keep_binary(im_array)

        # contours, thinned = thin_demo(im_array)
        #
        # thinned = thinned.astype(np.uint8)
        im_array = im_array.astype(np.uint8)

        # concatenated_array = np.stack((im_array, thinned, thinned), axis=2)

        cv2.imwrite(os.path.join(save_path, f'{file_name}' + f'_{view}_' + f'{frame + 1}' + '.png'), im_array)


# %%
def split(filepath, file_name, view, save_path=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image = sitk.ReadImage(str(filepath))
    file_name = str(file_name)
    frames = sitk.GetArrayFromImage(image).shape[0]
    for frame in range(frames):
        im_array = sitk.GetArrayFromImage(image)[frame, ...]
        im_array = cv2.resize(im_array, (512, 512))
        im_array = np.expand_dims(im_array, axis=0)
        im_array = np.transpose(im_array, (1, 2, 0))
        im_array = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(save_path, f'{file_name}' + f'_{view}_' + f'{frame + 1}' + '.png'), im_array)


# %%
def split_camus_squencese(path_list):
    patient_list = os.listdir(path_list)

    for patient in tqdm(patient_list):
        _2CH_dir_sequences = os.path.join(path_list, patient, f"{patient}" + "_2CH_half_sequence.nii.gz")
        _4CH_dir_sequences = os.path.join(path_list, patient, f"{patient}" + "_4CH_half_sequence.nii.gz")
        _2CH_dir_sequences_gt = os.path.join(path_list, patient, f"{patient}" + "_2CH_half_sequence_gt.nii.gz")
        _4CH_dir_sequences_gt = os.path.join(path_list, patient, f"{patient}" + "_4CH_half_sequence_gt.nii.gz")

        split(_2CH_dir_sequences, patient, '2CH', save_path='./camus/image/')
        split(_4CH_dir_sequences, patient, '4CH', save_path='./camus/image/')
        split_gt(_2CH_dir_sequences_gt, patient, '2CH', save_path='./camus/mask/')
        split_gt(_4CH_dir_sequences_gt, patient, '4CH', save_path='./camus/mask/')


# %%

# %%
if __name__ == '__main__':
    split_camus_squencese('E:\CAMUS\database_nifti')
# %%
