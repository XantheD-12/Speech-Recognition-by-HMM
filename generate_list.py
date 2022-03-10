import numpy as np


# 获取训练数据，并保存在training_list中
def generate_training_list():
    # 共440个训练数据
    # print('获取训练数据')
    dir1 = 'mfcc'
    dir2 = ['AE', 'AJ', 'AL', 'AW', 'BD', 'CB', 'CF', 'CR', 'DL', 'DN', 'EH', 'EL', 'FC', 'FD', 'FF', 'FI', 'FJ', 'FK',
            'FL', 'GG']
    words = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'O', 'Z']
    # 获取测试数据，并保存在training_list.csv中
    training_list = [['' for col in range(2)] for row in range(len(dir2) * len(words) * 2)]
    # training_list = np.array(training_list, dtype=str)
    k = 0
    for i in range(len(dir2)):
        for j in range((len(words))):
            training_list[k][0] = j
            training_list[k][1] = dir1 + '/' + dir2[i] + '/' + words[j] + 'A' + '_endpt.csv'
            training_list[k + 1][0] = j
            training_list[k + 1][1] = dir1 + '/' + dir2[i] + '/' + words[j] + 'B' + '_endpt.csv'
            k += 2
    training_list = np.array(training_list)
    np.savetxt('training_list.csv', training_list, fmt='%s', delimiter=',')
    # print('训练数据生成划分完成')


def generate_testing_list():
    # 共1232个测试数据
    # print('获取测试数据')
    dir1 = 'mfcc'
    dir2 = ['AH', 'AR', 'AT', 'BC', 'BE', 'BM', 'BN', 'CC', 'CE', 'CP', 'DF', 'DJ', 'ED', 'EF', 'ET', 'FA', 'FG', 'FH',
            'FM', 'FP', 'FR', 'FS', 'FT', 'GA', 'GP', 'GS', 'GW', 'HC', 'HJ', 'HM', 'HR', 'IA', 'IB', 'IM', 'IP', 'JA',
            'JH', 'KA', 'KE', 'KG', 'LE', 'LG', 'MI', 'NL', 'NP', 'NT', 'PC', 'PG', 'PH', 'PR', 'RK', 'SA', 'SL', 'SR',
            'SW', 'TC']
    words = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'O', 'Z']
    # 获取测试数据，并保存在testing_list.csv中
    testing_list = [['' for col in range(2)] for row in range(len(dir2) * len(words) * 2)]
    # testing_list = np.array(testing_list)
    k = 0
    for i in range(len(dir2)):
        for j in range((len(words))):
            testing_list[k][0] = j
            testing_list[k][1] = dir1 + '/' + dir2[i] + '/' + words[j] + 'A' + '_endpt.csv'
            testing_list[k + 1][0] = j
            testing_list[k + 1][1] = dir1 + '/' + dir2[i] + '/' + words[j] + 'B' + '_endpt.csv'
            k += 2
    training_list = np.array(testing_list)
    np.savetxt('testing_list.csv', testing_list, fmt='%s', delimiter=',')
    # print('测试数据生成划分完成')
