import os
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch


def dataset_distribution(refer_path, cls, save_dir=None):
    """
    tips:make sure your refer.csv have "label" key.
    """
    sns.set_style("whitegrid")
    df = pd.read_csv(refer_path,encoding='utf-8')
    cls_list = cls
    plt.figure(dpi=300,figsize=(4,3))
    plt.rc('font', family='Times New Roman', size=14)
    count_df = pd.DataFrame(df['label'].value_counts()).T
    cols = list(count_df.columns)
    cols = [cls_list[i] for i in cols]
    count_df.columns = cols
    sns.barplot(data=count_df)
    if save_dir:
        plt.savefig(save_dir + '/dataset_distribution.svg')


def torch2onnx(model_path,outputs_path, inputs_shape,device='cuda'):
    """
    tips:你保存的模型一定是直接通过torch.save(model)保存的——也就是保存整个模型
    :param model_path:.pth文件路径
    :param outputs_path:输出的.onnx文件路径
    :param inputs_shape:(bs,x,x,h,w)
    :param device:cuda or cpu
    :return:
    """
    dummy_input = torch.randn(*inputs_shape, device=device)
    input_names = ["input"]
    output_names = ["output"]
    model = torch.load(model_path)
    torch.onnx.export(model, dummy_input, outputs_path, verbose=True, input_names=input_names, output_names=output_names)


def df_generator(epoches, tags, save_path=None):
    """
    2022/5/24:update Auc and mAP
    """
    keys = ['Epoch', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1']
    values = [range(1, epoches + 1)] + tags
    data = dict(zip(keys, values))
    df = pd.DataFrame(data=data)
    if save_path:
        df.to_csv(save_path, encoding='utf-8')
    return df


def log_plot(epoches, tags, save_fig_dir, csv_save_path=None):
    """
    :param epoches:迭代次数
    :param tags:[loss,acc,precision,recall,f1]
    :param save_fig_path:保存路径
    """
    df = df_generator(epoches, tags, save_path=csv_save_path)
    sns.set_style("whitegrid")
    indicators = list(df.columns[2:])
    data = df.to_numpy()[:, 2:]
    index = list(df.index)
    df_ = pd.DataFrame(data=data,index=index,columns=indicators)
    plt.rc('font', family='Times New Roman', size=14)
    plt.figure(dpi=300,figsize=(4,3))
    plt.xlabel("Epoch")
    plt.ylabel("Indicator")
    sns.lineplot(data=df_, markers=True)
    plt.savefig(save_fig_dir + '/indicators.svg')

    plt.figure(dpi=300,figsize=(4,3))
    sns.lineplot(data=df,y='Loss',x="Epoch")
    plt.savefig(save_fig_dir + '/loss_epoch.svg')
    #------------------------------------
    """
    子图代码
    """
    # rows = int(sqrt(len(indicators)))
    # cols = int(len(indicators)/rows)+1
    # plt.subplots(nrows=rows,ncols=cols)
    # for i in range(len(indicators)):
    #     plt.subplot(rows,cols,i+1)
    #     sns.lineplot(data=df,x='Epoch',y=indicators[i])
    # plt.subplots_adjust(wspace=0.3, hspace=0.6)
    # ------------------------------------


def log_generator(train_theme_name, csv_path, cls, inputs_shape, duration,
                  dataset_info_table, classes_info_table,
                  training_device_table, training_info_table,
                  optimizer, model, epoches,
                  tags, log_save_dir, best_cp):
    nowtime = datetime.now()
    year = str(nowtime.year)
    month = str(nowtime.month)
    day = str(nowtime.day)
    hour = str(nowtime.hour)
    minute = str(nowtime.minute)
    second = str(nowtime.second)
    nowtime_strings = year + '/' + month + '/' + day + '/' + hour + ':' + minute + ':' + second
    workplace_path = os.getcwd()
    content = """
Theme:{}\n
Date:{}\n
Time used:{}\n
workplace:{}\n
folder information:\n{}\n
classes:\n{}\n
training device:\n{}\n
training basic configuration:\n{}\n
Optimizer:\n{}\n
Model:\n{}\n,
    """.format(
        train_theme_name,
        nowtime_strings,
        duration,
        workplace_path,
        dataset_info_table,
        classes_info_table,
        training_device_table,
        training_info_table,
        str(optimizer),
        str(model)
    )
    exp_name = 'exp-{}_{}_{}_{}_{}_{}'.format(
        train_theme_name,
        year, month, day,
        hour, minute, second)
    exp_path = log_save_dir + '/' + exp_name
    checkpoints_path = exp_path + '/checkpoints'
    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)
    # logger
    log_name = '{}_{}_{}_{}_{}_{}.log'.format(
        train_theme_name,
        year, month, day,
        hour, minute, second)
    log_file = open(exp_path + '/' + log_name, 'w', encoding='utf-8')
    log_file.write(content)
    log_file.close()
    # checkpoints
    # final_checkpoints
    torch.save(model, checkpoints_path + '/' + 'final_model.pth'.format(
        train_theme_name,
        year, month, day, hour,
        minute, second
    ))
    torch2onnx(
        model_path=checkpoints_path + '/' + 'final_model.pth'.format(
        train_theme_name,
        year, month, day, hour,
        minute, second
    ),
        outputs_path=checkpoints_path + '/' + 'final_model.onnx',
        inputs_shape=inputs_shape
    )
    # best_checkpoints
    torch.save(best_cp, checkpoints_path + '/' + 'best_f1.pth'.format(
        train_theme_name,
        year, month, day, hour,
        minute, second
    ))
    torch2onnx(
        model_path=checkpoints_path + '/' + 'best_f1.pth'.format(
            train_theme_name,
            year, month, day, hour,
            minute, second
        ),
        outputs_path=checkpoints_path + '/' + 'best_f1.onnx',
        inputs_shape=inputs_shape
    )
    # indicator.csv and indicator.jpg
    log_plot(epoches, tags, save_fig_dir=exp_path, csv_save_path=exp_path + '/indicators.csv')
    print("Training log has been saved to path:{}".format(exp_path))
    # datasets' distribution
    dataset_distribution(refer_path=csv_path, cls=cls, save_dir=exp_path)


