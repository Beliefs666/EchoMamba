import os

import h5py
import numpy
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader




class ECHODataset(Dataset):
    def __init__(self, meta: pd.DataFrame, base_path: str, file_name: str,  frac: float = 1):
        super(ECHODataset, self).__init__()
        if frac < 1:
            meta = meta.sample(frac=frac)
            meta.reset_index(inplace=True)
        self.meta = meta
        self.data = []
        self.label = []

        # 加载单一 HDF5 文件
        self.data_dict = h5py.File(os.path.join(base_path, file_name), "r")

        for idx in tqdm.tqdm(range(len(self.meta))):
            echo_id = self.meta.loc[idx, "ECHO_ID"]
            data = np.array(self.data_dict[echo_id]["video"], dtype=np.uint8)

            label = np.array(self.data_dict[echo_id]["mask"], dtype=np.uint8)
            # print(f"Shape of data: {data.shape}")
            # print(f"Shape of label: {label.shape}")
            h, w = data.shape[-2:]

            if label.shape[0] != data.shape[1]:
                self.data.append(torch.tensor(data[0, 0, :, :], dtype=torch.float32).reshape(3, h, w))
                self.data.append(torch.tensor(data[0, -1, :, :], dtype=torch.float32).reshape(3, h, w))
                self.label.append(torch.tensor(label[0, :, :], dtype=torch.long))
                self.label.append(torch.tensor(label[-1, :, :], dtype=torch.long))
            else:
                for i in range(label.shape[0]):
                    self.data.append(torch.tensor(data[:, i, :, :], dtype=torch.float32))
                    self.label.append(torch.tensor(label[i, :, :], dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data_item = self.data[item]
        label_item = self.label[item]

        # 打印数据和标签的形状
        # print(f"Shape of data: {data_item.shape}")
        # print(f"Shape of label: {label_item.shape}")

        return data_item, label_item,

    def _close_hdf5(self):
        self.data_dict.close()

    def __del__(self):
        if hasattr(self, 'data_dict'):
            self._close_hdf5()



def get_hmcqu_dataset(
        data_list: list,
        base_path: str,
        file_name: str,
):
    # 读取所有元数据文件
    meta = pd.concat(
        [pd.read_csv(data, dtype={"ECHO_ID": str}) for data in data_list]
    )
    meta.reset_index(inplace=True)

    # 创建数据集
    dataset = ECHODataset(
        meta, base_path, file_name
    )
    return dataset

