import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

labels = ["whatdoing", "whereare", "withwhom", "mood", "stress", "depression"]
# one_hot_size = {'whatdoing': 23, 'withwhom': 8, 'whereare': 16, 'mood': 4, 'depression': 4, 'stress': 4}
use_label = labels[0]

class gen_iLog_train_test(data.Dataset):
    def __init__(self, ds_path, mode):
        self.features = []
        self.one_hot_label = []
        self.labels_t = []
        files = os.listdir(ds_path)

        for file in files:
            if mode in file:
                ds_file = os.path.join(ds_path, file)
                data = pd.read_pickle(ds_file)
                data = data.to_dict('list')
               
                self.features.extend(data['features'])
                self.one_hot_label.extend(data[use_label])
                self.labels_t.extend(data[use_label+"_t"])

    def __getitem__(self, index):
        feature = np.array(self.features[index])
        one_hot_label = np.array(self.one_hot_label[index])
        label_t = np.array(self.labels_t[index])
        return feature, one_hot_label, label_t

    def __len__(self):
        return len(self.features)




def get_dataloader(ds_pth, mode, batch):
    shuffle = True if mode == "train" else False
    print("mode:", mode, "shuffle:", shuffle)
    ds =gen_iLog_train_test(ds_path=ds_pth, mode=mode)

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch,
        num_workers=8,
        drop_last=True,
        shuffle=shuffle
        )
    return dataloader

if __name__ == "__main__":
    ds_pth = "/myData/iLog/iLog数据处理代码/dataset_new/byuser"
    
    for user in ["a1be2dfcd1a9133fe88a1349039616dce39e3326"]:
        cnt_batch = 0
        train_test_file_pth = os.path.join(ds_pth, user)
        print("\n", train_test_file_pth)
        for idx, (features, labels, labels_t) in enumerate(get_dataloader(ds_pth=train_test_file_pth, mode="test", batch=64)):
            print(features.shape)  #   torch.Size([BN, 6, 21, 26])
            print(labels.shape)  # torch.Size([BN, 23])
            print(labels_t.shape)  # torch.Size([BN])

            new_l = torch.argmax(labels, dim=1)  # 等价于 labels_t
            print(new_l.shape)  # # torch.Size([BN])
            break
        