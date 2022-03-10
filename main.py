import numpy as np
from generate_mfcc_features import generate_mfcc_features, generate_mfcc_by_myself,generate_mfcc_features_by_librosa
from generate_list import generate_training_list, generate_testing_list
from EM_HMM_training import EM_HMM_training
from HMM_testing import HMM_testing
import time


def main():
    # print('Start!')
    # 读取wav文件，生成mfcc特征
    # 生成好了先注释掉,为了更快地测试
    # print('生成mfcc特征')
    # generate_mfcc_features()
    # generate_mfcc_by_myself()
    # generate_mfcc_features_by_librosa()
    # print('特征生成完毕')
    # 划分训练数据和测试数据
    # print('划分训练数据和测试数据')
    generate_training_list()  # 训练数据
    generate_testing_list()  # 测试数据
    # print('划分完毕')
    training_list_name = 'training_list.csv'
    testing_list_name = 'testing_list.csv'

    DIM = 39  # 特征的维数：39维
    num_of_model = 11  # 11个模型：1~9，Zero，O

    # 训练状态从12~15
    num_of_state_start = 12
    num_of_state_end = 15

    accuracy_rate = np.zeros(num_of_state_end+1)  # 准确率

    for i in range(num_of_state_start, num_of_state_end + 1):  # 12~15
        time_start = time.time()
        # print('HMM训练')
        # 使用EM算法进行训练
        # （训练样本，训练维数：39维，训练模型个数，状态个数）
        # print('使用EM算法进行训练')
        mean, var, Aij = EM_HMM_training(training_list_name, DIM, num_of_model, i)
        time_end = time.time()
        # print('训练时间：', time_end - time_start)

        # 测试训练的模型
        # 测试使用Viterbi算法
        time_start = time.time()
        # print('测试模型')
        accuracy_rate[i] = HMM_testing(mean, var, Aij, testing_list_name)
        time_end = time.time()
        # print('测试时间：', time_end - time_start)
        print('状态数为%d的准确率: %.2f %%' % (i, accuracy_rate[i]))
    # 将准确率保存
    np.savetxt('accuracy_rate.csv', accuracy_rate, delimiter=',')


if __name__ == '__main__':
    main()
