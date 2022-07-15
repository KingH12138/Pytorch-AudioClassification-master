import os

import tqdm
from pandas import DataFrame
from data.utils.datautils import choice_and_process
import numpy as np
"""
A demo for generator DIF(datasets' information file) according to "Scatter Classification" format.
"""


class ScatterData(object):
    """
    -dataset
        -1.wav
        -2.wav
        -.......
    """
    def __init__(self, csv_path, scatter_dir, classes_path, npy_dir):
        self.csv_path = csv_path
        self.scatter_dir = scatter_dir
        self.classes_path = classes_path
        self.npy_dir = npy_dir
        self.classes_list = []

    def classtxt2list(self):
        with open(self.classes_path, 'r') as f:
            classes_list = f.read().split('\n')[:-1]
            try:
                classes_list = [int(i) for i in classes_list] # 尝试化为整型标签
                self.classes_list = classes_list
                return classes_list
            except:
                self.classes_list = classes_list
                return classes_list

    def getclasstxt(self):
        """
        每种scatter数据集命名格式都不太一样，其类别信息存储在名字中，建议自定义
        """
        classes_list = []
        for filename in os.listdir(self.scatter_dir):
            label = int(filename[0])
            if label in classes_list:
                continue
            else:
                classes_list.append(label)
        content = ""
        for class_ in classes_list:
            content = content + str(class_) + "\n"
        with open(self.classes_path,'w') as f:
            f.write(content)
        return classes_list

    def process(self):
        """
        最后导出的数据格式可能不符合需求，可能需要改一改
        """
        print("Processing original sound data to npy data......")
        os.makedirs(self.npy_dir,exist_ok=True)
        for filename in tqdm.tqdm(os.listdir(self.scatter_dir)):
            filepath = self.scatter_dir + '/{}'.format(filename)
            sound_data = choice_and_process(filepath)
            npy_data = sound_data.numpy()
            np.save(self.npy_dir + '/{}'.format(filename.split('.')[0]+'.npy'), npy_data)
        print("Done.")

    def generator(self):
        """
        data items:filename,filepath,label
        每种scatter数据集命名格式都不太一样，其类别信息存储在名字中，建议自定义
        """
        if not os.path.exists(self.classes_path):
            self.getclasstxt()
        classes_list = self.classtxt2list()
        data = {'filename': [], 'filepath': [], 'label': []}
        for filename in tqdm.tqdm(os.listdir(self.npy_dir)):
            filepath = self.npy_dir + '/{}'.format(filename)
            #######
            label_name = filename[0]
            # 这里因为读取label方式比较独特所以采用这种写法，其他情况下大概率得改
            #######
            label = classes_list.index(int(label_name))
            data['filename'].append(filename.split('.')[0])
            data['filepath'].append(filepath)
            data['label'].append(label)
        dataframe = DataFrame(data=data)
        dataframe.to_csv(self.csv_path, encoding='utf-8')


scatterdata = ScatterData(
    csv_path=r"D:\PythonCode\Pytorch-AudioClassification-master\data\scatter\refer.csv",
    scatter_dir=r"D:\machine_learning\音频识别\datasets\data",
    classes_path=r"D:\PythonCode\Pytorch-AudioClassification-master\data\scatter\classes.txt",
    npy_dir=r"D:\PythonCode\Pytorch-AudioClassification-master\data\scatter\npy_data",
)
# scatterdata.process()
scatterdata.generator()