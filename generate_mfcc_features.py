"""
读取wav文件夹，生成其mfcc特征，并保存在mfcc文件夹下
"""
import os

import librosa.feature
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from python_speech_features import delta
from my_mfcc_features import get_feature
import librosa


def generate_mfcc_features():
    # 读取wav目录下的所有文件
    indir = 'wav'  # 输入的目录-wav
    outdir = 'mfcc'  # 输出的目录-mfcc
    dir_list = os.listdir(indir)  # wav文件夹下的所有目录list
    dir_len = len(dir_list)  # 目录list的长度
    file_list = os.listdir(indir + '/' + dir_list[0])  # 每个目录下的文件名是相同的，获得文件list
    file_len = len(file_list)  # 获得每个目录的文件个数
    if not os.path.exists(outdir):  # 如果未创建输出目录，则创建输出目录
        os.mkdir(outdir)
    for i in range(dir_len):
        # 如果未创建输出目录，则创建输出目录
        dir_path = outdir + '/' + dir_list[i]  # 输出目录路径
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        for j in range(file_len):
            # 读取文件
            # print(indir + '/' + dir_list[i] + '/' + file_list[j])
            file_path = indir + '/' + dir_list[i] + '/' + file_list[j]  # 文件路径
            if not os.path.exists(file_path):  # 文件路径不存在
                break
            fs, signal = wav.read(file_path)  # fs-采样率，signal-wav文件的内容
            # python_speech_features的mfcc生成13维特征，分别做两次差分，然后合并，得到mfcc特征
            # 默认使用hamming窗
            # mfcc函数解析：https://blog.csdn.net/u011898542/article/details/84255420
            mfcc_feature = mfcc(signal, fs)
            d_mfcc = delta(mfcc_feature, 1)  # 一阶差分
            dd_mfcc = delta(mfcc_feature, 2)  # 二阶差分
            feature = np.hstack((mfcc_feature, d_mfcc, dd_mfcc))
            out = (outdir + '/' + dir_list[i] + '/' + file_list[j]).split('.')[0]  # 输出路径
            np.savetxt(out + '.csv', feature, delimiter=',')


# 使用自己写的mfcc
def generate_mfcc_by_myself():
    # 读取wav目录下的所有文件
    indir = 'wav'  # 输入的目录-wav
    outdir = 'my_mfcc'  # 输出的目录-my_mfcc
    dir_list = os.listdir(indir)  # wav文件夹下的所有目录list
    dir_len = len(dir_list)  # 目录list的长度
    file_list = os.listdir(indir + '/' + dir_list[0])  # 每个目录下的文件名是相同的，获得文件list
    file_len = len(file_list)  # 获得每个目录的文件个数
    if not os.path.exists(outdir):  # 如果未创建输出目录，则创建输出目录
        os.mkdir(outdir)
    for i in range(dir_len):
        # 如果未创建输出目录，则创建输出目录
        dir_path = outdir + '/' + dir_list[i]  # 输出目录路径
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        for j in range(file_len):
            # 读取文件
            # print(indir + '/' + dir_list[i] + '/' + file_list[j])
            file_path = indir + '/' + dir_list[i] + '/' + file_list[j]  # 文件路径
            if not os.path.exists(file_path):  # 文件路径不存在
                break
            feature = get_feature(file_path)
            out = (outdir + '/' + dir_list[i] + '/' + file_list[j]).split('.')[0]  # 输出路径
            np.savetxt(out + '.csv', feature, delimiter=',')

# 使用Librosa的mfcc接口
# filepath = "/Users/birenjianmo/Desktop/learn/librosa/mp3/in.wav"
# y,sr = librosa.load(filepath)
# mfcc = librosa.feature.mfcc( y,sr,n_mfcc=13 )
def generate_mfcc_features_by_librosa():
    # 读取wav目录下的所有文件
    indir = 'wav'  # 输入的目录-wav
    outdir = 'l_mfcc'  # 输出的目录-mfcc
    dir_list = os.listdir(indir)  # wav文件夹下的所有目录list
    dir_len = len(dir_list)  # 目录list的长度
    file_list = os.listdir(indir + '/' + dir_list[0])  # 每个目录下的文件名是相同的，获得文件list
    file_len = len(file_list)  # 获得每个目录的文件个数
    if not os.path.exists(outdir):  # 如果未创建输出目录，则创建输出目录
        os.mkdir(outdir)
    for i in range(dir_len):
        # 如果未创建输出目录，则创建输出目录
        dir_path = outdir + '/' + dir_list[i]  # 输出目录路径
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        for j in range(file_len):
            # 读取文件
            # print(indir + '/' + dir_list[i] + '/' + file_list[j])
            file_path = indir + '/' + dir_list[i] + '/' + file_list[j]  # 文件路径
            if not os.path.exists(file_path):  # 文件路径不存在
                break
            # Librosa得到的是13*x的特征
            y, sr = librosa.load(file_path)
            mfcc_feature = librosa.feature.mfcc(y,sr,n_mfcc=13)
            d_mfcc = delta(mfcc_feature, 1)  # 一阶差分
            dd_mfcc = delta(mfcc_feature, 2)  # 二阶差分
            feature = np.vstack((mfcc_feature, d_mfcc, dd_mfcc))
            feature=feature.T
            out = (outdir + '/' + dir_list[i] + '/' + file_list[j]).split('.')[0]  # 输出路径
            np.savetxt(out + '.csv', feature, delimiter=',')