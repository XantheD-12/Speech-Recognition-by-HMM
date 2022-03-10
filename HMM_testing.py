import numpy as np
from EM_HMM_training import logGaussian


# 测试模型
def HMM_testing(mean, var, Aij, testing_list_name):
    """
    对模型进行测试
    :param mean: 均值矩阵
    :param var: 方差矩阵
    :param Aij: 状态转移矩阵
    :param testing_list_name:测试文件
    :return: 训练出的模型的准确率（accuracy_rate）
    """
    num_of_model = 11  # 模型个数：11
    num_of_error = 0  # 错误的数量
    num_of_testing = 0  # 测试的数量

    testing_file_list = np.loadtxt(testing_list_name, dtype=str, delimiter=',')  # 载入测试样本
    num_of_uter = testing_file_list.shape[0]  # 测试样本的数量
    # 使用Viterbi算法
    for u in range(num_of_uter):
        k = int(testing_file_list[u][0])  # 得到是哪个模型（1~9，Zero，O）
        file_name = testing_file_list[u][1]  # 得到文件名
        mfcc_file = np.loadtxt(file_name, delimiter=',')
        features = mfcc_file.T

        num_of_testing = num_of_testing + 1
        # 预测发出的是哪个音
        fopt_max = -np.inf
        digit = -1
        for p in range(num_of_model):
            # 计算哪个模型的概率最大，选择该模型
            fopt = viterbi_dist_FR(mean[p, :, :], var[p, :, :], Aij[p, :, :], features)  # 第k个模型
            if fopt > fopt_max:
                digit = p
                fopt_max = fopt
        if digit != k:  # 如果和测试样本的声音不同，则说明识别错误
            num_of_error = num_of_error + 1
    accuracy_rate = float((num_of_testing - num_of_error) * 100) / float(num_of_testing)  # 计算准确率
    return accuracy_rate


def viterbi_dist_FR(mean, var, aij, obs):
    """
    Viterbi算法
    :param mean:均值矩阵
    :param var: 方差矩阵
    :param aij: 状态转移矩阵
    :param obs: 观察矩阵（mfcc特征）
    :return: 该模型的最大概率
    """
    dim, t_len = obs.shape  # dim-特征维数：39；t_len： 一个训练样本的特征的帧数
    mean = np.hstack((np.full((dim, 1), np.nan), mean, np.full((dim, 1), np.nan)))  # dim X (state+2)
    var = np.hstack((np.full((dim, 1), np.nan), var, np.full((dim, 1), np.nan)))  # dim X (state+2)
    aij[-1, -1] = 1
    timing = np.arange(0, t_len)  # t=0~t_len-1时刻
    m_len = mean.shape[1]  # state+2
    f_jt = np.ones((m_len, t_len)) * -np.inf  # viterbi矩阵
    s_chain = np.empty((m_len, t_len), dtype=object)  # backtrace pointer，用于存储状态链
    dt = timing[0]
    for j in range(1, m_len - 1):  # 从状态2到状态13（num_of_state+2=14），第1个状态和第14个状态分别是开始和结束的状态
        # 初始化viterbi矩阵
        f_jt[j, 0] = np.log(aij[0, j]) + logGaussian(mean[:, j], var[:, j], obs[:, 0])
        if f_jt[j, 0] > -np.inf:
            s_chain[j, 0] = np.array([0, j])
    for t in range(1, t_len):
        dt = timing[t] - timing[t - 1]
        for j in range(1, m_len - 1):
            f_max = -np.inf
            i_max = -1
            f = -np.inf
            for i in range(1, j + 1):
                if f_jt[i, t - 1] > -np.inf:
                    f = f_jt[i, t - 1] + np.log(aij[i, j]) + logGaussian(mean[:, j], var[:, j], obs[:, t])
                if f > f_max:  # 找到最大的f，为了计算Viterbi矩阵的下一列
                    f_max = f
                    i_max = i  # 索引
            if i_max != -1:
                s_chain[j, t] = np.append(s_chain[i_max, t - 1], j)
                f_jt[j, t] = f_max

    dt = timing[-1] - timing[-1 - 1]  # t=end
    fopt = -np.inf
    iopt = -1
    for i in range(1, m_len - 1):
        f = f_jt[i, t_len - 1] + np.log(aij[i, m_len - 1])
        if f > fopt:
            fopt = f
            iopt = i
    if iopt != -1:
        chain_op = np.append(s_chain[iopt, t_len - 1], m_len - 1)  # 最优状态链
    return fopt
