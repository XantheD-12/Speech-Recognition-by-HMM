import time
import numpy as np
import matplotlib.pyplot as plt


def EM_HMM_training(training_file_list_name, DIM, num_of_model, num_of_state):
    """
    根据模型个数、状态个数、训练样本进行训练
    :param training_file_list_name: 训练文件
    :param DIM: 维数
    :param num_of_model: 模型个数
    :param num_of_state: 模型状态个数
    :return: 训练出的模型的几个要素：均值矩阵、方差矩阵、状态转移矩阵
    """
    mean = np.zeros((num_of_model, DIM, num_of_state))  # 均值矩阵
    var = np.zeros((num_of_model, DIM, num_of_state))  # 方差矩阵
    # 状态转移矩阵（从状态i转移到状态j的概率），加上了start和end（+2）
    Aij = np.zeros((num_of_model, num_of_state + 2, num_of_state + 2))

    # 初始化HMM
    EMM_initialization(mean, var, Aij, training_file_list_name, DIM, num_of_state, num_of_model)

    # 开始训练
    # 首先使用前向-后向算法
    num_of_iteration = 5  # 迭代5次，使得到的模型是收敛的
    log_likelihood_iter = np.zeros(num_of_iteration)  # 极大似然估计取对数
    likelihood_iter = np.zeros(num_of_iteration)  # 极大似然估计

    training_file_list = np.loadtxt(training_file_list_name, dtype=str, delimiter=',')  # 载入训练样本
    num_of_uter = training_file_list.shape[0]  # 训练样本的总数:440个样本

    # 开始迭代训练
    sum_mean_numerator = np.zeros((num_of_model, DIM, num_of_state))  # 计算均值的分子
    sum_var_numerator = np.zeros((num_of_model, DIM, num_of_state))  # 计算方差的分子
    sum_aij_numerator = np.zeros((num_of_model, num_of_state, num_of_state))  # 计算状态转移矩阵的分子
    sum_denominator = np.zeros((num_of_state, num_of_model))  # 分母

    # 实际上，迭代次数在5左右就已经可以看作收敛了
    for iteration in range(num_of_iteration):
        time_start = time.time()
        # 计算前，置为0
        sum_mean_numerator = sum_mean_numerator * 0  # 计算均值的分子
        sum_var_numerator = sum_var_numerator * 0  # 计算方差的分子
        sum_aij_numerator = sum_aij_numerator * 0  # 计算状态转移矩阵的分子
        sum_denominator = sum_denominator * 0  # 分母
        log_likelihood = 0
        likelihood = 0
        print('对一个完整的帧使用EM算法，第%d次迭代' % (iteration + 1))
        for u in range(num_of_uter):
            k = int(training_file_list[u][0])  # 得到是哪个模型（1~9，Zero，O）
            file_name = training_file_list[u][1]  # 得到文件名
            mfcc_file = np.loadtxt(file_name, delimiter=',')
            features = mfcc_file.T
            # 训练第k个model
            mean_numerator, var_numerator, aij_numerator, denominator, log_likelihood_i, likelihood_i = \
                EM_HMM_FR(mean[k, :, :], var[k, :, :], Aij[k, :, :], features)

            sum_mean_numerator[k, :, :] = sum_mean_numerator[k, :, :] + mean_numerator[:, 1: -1]
            sum_var_numerator[k, :, :] = sum_var_numerator[k, :, :] + var_numerator[:, 1: -1]
            sum_aij_numerator[k, :, :] = sum_aij_numerator[k, :, :] + aij_numerator[1: -1, 1: -1]
            sum_denominator[:, k] = sum_denominator[:, k] + denominator[1: -1]

            log_likelihood = log_likelihood + log_likelihood_i
            likelihood = likelihood + likelihood_i

        # 计算平均值、方差，状态转移矩阵
        for k in range(num_of_model):
            for n in range(num_of_state):
                mean[k, :, n] = sum_mean_numerator[k, :, n] / sum_denominator[n, k]
                var[k, :, n] = sum_var_numerator[k, :, n] / sum_denominator[n, k] - np.square(mean[k, :, n])
        for k in range(num_of_model):
            for i in range(1, num_of_state + 1):
                for j in range(1, num_of_state + 1):
                    Aij[k, i, j] = sum_aij_numerator[k, i - 1, j - 1] / sum_denominator[i - 1, k]
            Aij[k, num_of_state, num_of_state + 1] = 1 - Aij[k, num_of_state, num_of_state]
            Aij[k, num_of_state + 1, num_of_state + 1] = 1

        log_likelihood_iter[iteration] = log_likelihood
        likelihood_iter[iteration] = likelihood
        time_end = time.time()
        print(time_end - time_start)  # 每次迭代的用时
    print('迭代完成')

    # 迭代完成
    # 绘图（极大似然估计）
    plt.figure()
    plt.plot(log_likelihood_iter, '-*')
    plt.xlabel('iterations')
    plt.ylabel('log likelihood')
    title = 'number of state:'
    plt.title('number of states: %d' % num_of_state)
    plt.savefig('状态数为 %d 的log_likelihood' % num_of_state)
    # 可以通过图像看到模型的收敛情况
    # plt.show()
    return mean, var, Aij


def EM_HMM_FR(mean, var, Aij, obs):
    """
    使用EM算法训练一个模型的几个要素：均值、方差、状态转移矩阵的分子；它们的分母；极大似然估计
    :param mean: 平均值矩阵(dim X state)
    :param var: 方差矩阵(dim X state)
    :param Aij: 状态转移矩阵((state+2) X (state+2))
    :param obs: 观察序列（获得的mfcc特征）
    :return: 均值、方差、状态转移矩阵的分子；它们的分母；极大似然估计
    """
    dim, T = obs.shape  # dim-特征维度：39；T-观察的长度或观察帧的数量
    # a = np.full((5), np.nan)
    mean = np.hstack((np.full((dim, 1), np.nan), mean, np.full((dim, 1), np.nan)))
    var = np.hstack((np.full((dim, 1), np.nan), var, np.full((dim, 1), np.nan)))
    Aij[-1][-1] = 1
    N = mean.shape[1]  # 增加了Start和End状态
    log_alpha = np.array(np.ones((N, T + 1)) * -np.inf)  # 前向算法的alpha初始化（取对数为了计算更准确）
    log_beta = np.array(np.ones((N, T + 1)) * -np.inf)  # 后向算法的beta初始化（取对数为了计算更准确）

    # E-step
    # 开始计算alpha
    for i in range(N):  # 初始化第一列
        log_alpha[i, 0] = np.log(Aij[0, i]) + logGaussian(mean[:, i], var[:, i], obs[:, 0])
        # mean[:][i]，var[:][i]：状态i下的平均值和方差

    for t in range(1, T):  # t=1~T-1
        for j in range(1, N - 1):  # j=1~N-2
            # log_sum_alpha(log_alpha(2:N-1,t-1),aij(2:N-1,j))
            log_alpha[j, t] = log_sum_alpha(log_alpha[1:N - 1, t - 1], Aij[1:N - 1, j]) + logGaussian(mean[:, j],
                                                                                                      var[:, j],
                                                                                                      obs[:, t])
    log_alpha[N - 1, T] = log_sum_alpha(log_alpha[1:N - 1, T - 1], Aij[1:N - 1, N - 1])

    # 开始计算beta
    log_beta[:, T - 1] = np.log(Aij[:, - 1])  # 初始化T时刻的β
    for t in range(T - 2, -1, -1):  # t=T-2~0
        for i in range(1, N - 1):  # i=1~N-2
            log_beta[i, t] = log_sum_beta(Aij[i, 1:N - 1], mean[:, 1:N - 1], var[:, 1:N - 1], obs[:, t + 1],
                                          log_beta[1:N - 1, t + 1])
    log_beta[N - 1, 0] = log_sum_beta(Aij[0, 1:N - 1], mean[:, 1:N - 1], var[:, 1:N - 1], obs[:, 0],
                                      log_beta[1:N - 1, 0])

    # 计算γ
    log_gamma = np.array(np.ones((N, T)) * -np.inf)
    for t in range(T):
        for i in range(1, N - 1):
            log_gamma[i, t] = log_alpha[i, t] + log_beta[i, t] - log_alpha[N - 1, T]
    gamma = np.exp(log_gamma)

    # 计算ξ
    log_Xi = np.array(np.ones((T, N, N)) * -np.inf)
    for t in range(T - 1):
        for j in range(1, N - 1):
            for i in range(1, N - 1):
                log_Xi[t, i, j] = log_alpha[i, t] + np.log(Aij[i, j]) + logGaussian(mean[:, j], var[:, j],
                                                                                    obs[:, t + 1]) + log_beta[
                                      j, t + 1] - log_alpha[N - 1, T]
    # 计算T时刻的ξ，即t=T-1
    for i in range(N):
        log_Xi[T - 1, i, N - 1] = log_alpha[i, T - 1] + np.log(Aij[i, N - 1]) - log_alpha[N - 1, T]

    # 计算均值、方差、状态转移矩阵的分子以及分母（M-step）
    mean_numerator = np.zeros((dim, N))
    var_numerator = np.zeros((dim, N))
    aij_numerator = np.zeros((N, N))
    denominator = np.zeros(N)
    for j in range(1, N - 1):
        for t in range(T):
            mean_numerator[:, j] = mean_numerator[:, j] + gamma[j, t] * obs[:, t]
            var_numerator[:, j] = var_numerator[:, j] + gamma[j, t] * np.square(obs[:, t])
            denominator[j] = denominator[j] + gamma[j, t]

    for i in range(1, N - 1):
        for j in range(1, N - 1):
            for t in range(T):
                aij_numerator[i, j] = aij_numerator[i, j] + np.exp(log_Xi[t, i, j])

    log_likelihood = log_alpha[N - 1, T]
    likelihood = np.exp(log_alpha[N - 1, T])

    return mean_numerator, var_numerator, aij_numerator, denominator, log_likelihood, likelihood


# 计算Σα(t-1,j)a(i,j)
# log_sum_alpha(log_alpha_t, aij_j)=Σ log(α(t,i)*a(i,j))
def log_sum_alpha(log_alpha_t, aij_j):
    """
    计算Σ alpha(t-1,j)a(i,j)
    log_sum_alpha(log_alpha_t, aij_j)=Σ log(alpha(t,i)*a(i,j))
    :param log_alpha_t: t时刻下的alpha，即第t列：alpha[:][t]
    :param aij_j: Aij[:][j]，即状态转移矩阵的第j列
    :return: 返回Σα(t-1,j)a(i,j)
    """
    len_x = log_alpha_t.shape[0]
    y = np.array(np.ones(len_x) * -np.inf)
    y_max = -np.inf
    for i in range(len_x):
        y[i] = log_alpha_t[i] + np.log(aij_j[i])
        if y[i] > y_max:
            y_max = y[i]
    if y_max == np.inf:
        logsumalpha = np.inf
    else:
        sum_exp = 0
        for i in range(len_x):
            if y_max == -np.inf and y[i] == -np.inf:
                sum_exp = sum_exp + 1
            else:
                sum_exp = sum_exp + np.exp(y[i] - y_max)
        logsumalpha = y_max + np.log(sum_exp)
    return logsumalpha


# log_sum_beta(aij_i, mean, var, obs, beta_t) = Σ a(i,j)*b(t,j)*β(t,j)
def log_sum_beta(aij_i, mean, var, obs, beta_t):
    """
    计算 Σ a(i,j)*b(t,j)*β(t,j)
    :param aij_i:Aij[i][:]，即第i汉堡
    :param mean:
    :param var:
    :param obs:
    :param beta_t:
    :return:
    """
    len_x = mean.shape[1]  # state的数量
    y = np.array(np.ones(len_x) * -np.inf)
    y_max = -np.inf
    for j in range(len_x):
        y[j] = np.log(aij_i[j]) + logGaussian(mean[:, j], var[:, j], obs) + beta_t[j]
        if y[j] > y_max:
            y_max = y[j]
    if y_max == np.inf:
        logsumbeta = np.inf
    else:
        sum_exp = 0
        for i in range(len_x):
            if y_max == -np.inf and y[i] == -np.inf:
                sum_exp = sum_exp + 1
            else:
                sum_exp = sum_exp + np.exp(y[i] - y_max)
        logsumbeta = y_max + np.log(sum_exp)
    return logsumbeta


# log_b，发射概率取对数
# logGaussian(mean_i,var_i,o_t)=log(bi(xt))
def logGaussian(mean_i, var_i, o_t):
    dim = len(var_i)  # 39维
    log_b = -1 / 2 * ((dim * np.log(2 * np.pi) + np.sum(np.log(var_i))) + np.sum(
        np.divide(np.square(o_t - mean_i), var_i)))
    return log_b


# 初始化HMM
def EMM_initialization(mean, var, Aij, training_file_list_name, DIM, num_of_state, num_of_model):
    """
    初始化EMM
    :param mean:均值矩阵
    :param var:方差矩阵
    :param Aij:状态转移矩阵
    :param training_file_list_name:训练样本文件名
    :param DIM:维数：39
    :param num_of_state:模型的状态数量
    :param num_of_model:模型的数量：11
    :return:初始后的均值、方差、状态转移矩阵
    """
    sum_of_features = np.zeros(DIM)  # 计算平均值
    sum_of_features_square = np.zeros(DIM)  # 计算方差
    num_of_feature = 0  # 模型k中状态m的元素(特征向量)个数
    training_file_list = np.loadtxt(training_file_list_name, dtype=str, delimiter=',')  # 载入训练样本
    num_of_uter = training_file_list.shape[0]  # 样本总数

    # 初始化样本均值、方差、状态转移矩阵
    # print('初始化样本均值、方差、状态转移矩阵')
    for i in range(num_of_uter):
        file_name = training_file_list[i][1]
        mfcc_file = np.loadtxt(file_name, delimiter=',')  # 读取mfcc特征，是帧数X维数（n_samples X dim）
        n_samples = mfcc_file.shape[0]  # 特征的帧数
        dim = mfcc_file.shape[1]  # 维数：39
        features = mfcc_file.T  # 维数X帧数（dim X n_samples）

        # 生成（dim X 1）
        temp = np.sum(features, axis=1)
        sum_of_features = np.add(sum_of_features, temp)
        temp = np.sum(np.square(features), axis=1)
        sum_of_features_square = np.add(sum_of_features_square, temp)
        num_of_feature = num_of_feature + features.shape[1]  # 特征的总的帧数

    # 初始化HMM
    # print('初始化HMM')
    calculate_initial_EM_HMM_items(mean, var, Aij, num_of_state, num_of_model, sum_of_features,
                                   sum_of_features_square, num_of_feature)


def calculate_initial_EM_HMM_items(mean, var, Aij, num_of_state, num_of_model, sum_of_features,
                                   sum_of_features_square, num_of_feature):
    """
    初始化HMM的几个参数
    :param mean:均值矩阵
    :param var:方差矩阵
    :param Aij:状态转移矩阵
    :param num_of_state:状态个数
    :param num_of_model:模型个数
    :param sum_of_features:用于计算均值
    :param sum_of_features_square:用于计算方差
    :param num_of_feature:特征的总的帧数
    :return:初始化后的参数
    """
    # print('初始化HMM')
    for i in range(num_of_model):
        for j in range(num_of_state):
            # 初始化平均值和方差
            mean[i, :, j] = sum_of_features / num_of_feature
            var[i, :, j] = sum_of_features_square / num_of_feature - np.square(mean[i, :, j])
        # -np.square(mean[i,:,j])
        for k in range(1, num_of_state + 1):
            # 初始化状态转移矩阵
            Aij[i, k, k + 1] = 0.4  # 从k到k+1的概率初始化为0.4
            Aij[i, k, k] = 1 - Aij[i][k][k + 1]  # 从k到k的概率初始化为1-0.4=0.6
        Aij[i, 0, 1] = 1  # 初始化时，初始概率为1
