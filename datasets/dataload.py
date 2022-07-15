from torch.utils.data import Dataset, DataLoader, random_split
from pandas import read_csv
from numpy import load


class SoundDataSet(Dataset):
    """
    初始化函数:
    输入:info_data文件的数据框格式读取信息+数据集路径
    并且写入相关属性。
    """

    def __init__(self, csv_path, data_path):
        self.df = read_csv(csv_path, encoding='utf-8')
        self.data_path = data_path

    # 一般重写这三个方法就够了
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.df['label'][index]
        data = load(self.df['filepath'][index])
        return data, label


def get_dataloader(data_dir, csv_path, batch_size, train_percent=0.9):
    dataset = SoundDataSet(data_dir, csv_path)
    num_sample = len(dataset)
    num_train = int(train_percent * num_sample)
    num_valid = num_sample - num_train
    train_ds, valid_ds = random_split(dataset, [num_train, num_valid])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                          persistent_workers=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                          persistent_workers=True)
    return train_dl, valid_dl, len(dataset), len(train_ds), len(valid_ds)
