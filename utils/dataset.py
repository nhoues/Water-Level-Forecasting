import pandas as pd
import numpy as np
import torch


class hydro_dataset:
    def __init__(
        self, data, window_size, num_feat=None, cat_feat=None, is_auto_encoder=False
    ):

        self.window_size = window_size
        self.num_feat = num_feat
        self.cat_feat = cat_feat
        self.target = data.Value.values[window_size:]

        if is_auto_encoder:
            final_step = -1
        else:
            final_step = 0

        num_data = []
        data_dict = dict()

        if num_feat is not None:
            for feat in num_feat:
                temp = []
                for step in range(window_size, final_step, -1):
                    temp.append(np.expand_dims(data[feat].shift(step), 1))
                temp = np.concatenate(temp, axis=1)
                num_data.append(np.expand_dims(temp, axis=2))
            num_data = np.concatenate(num_data, axis=2)
            data_dict["num_data"] = num_data[window_size:]

        if cat_feat is not None:
            for cat in cat_feat:
                temp = []
                for step in range(window_size, final_step, -1):
                    temp.append(np.expand_dims(data[cat].shift(step), 1))
                temp = np.concatenate(temp, axis=1)
                data_dict[cat] = temp[window_size:]

        self.data_dict = data_dict

    def __len__(self):
        return len(self.target)

    def __getitem__(self, item):
        out = dict()
        out["target"] = torch.tensor(self.target[item], dtype=torch.float)
        if self.num_feat is not None:
            out["num_feat"] = torch.tensor(
                self.data_dict["num_data"][item], dtype=torch.float
            )
        if self.cat_feat is not None:
            for cat in self.cat_feat:
                out[cat] = torch.tensor(self.data_dict[cat][item], dtype=torch.long)
        return out
