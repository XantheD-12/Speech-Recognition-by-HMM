import wave
import numpy as np
import math


# 读取wav文件并简单地提取参数
def wav_read(file_name):
    """
    读取wav文件并简单地提取参数
    :param file_name: 打开的wav文件名
    :return: data,time
    """
    wav_file = wave.open(file_name, "rb")
    num_channel = wav_file.getnchannels()  # 声道数
    sample_width = wav_file.getsampwidth()  # 采样字节长度
    framerate = wav_file.getframerate()  # 采样频率
    num_frames = wav_file.getnframes()  # 音频总帧数
    # print("声道数：", num_channel)
    # print("采样字节长度：", sample_width)
    # print("采样频率：", framerate)
    # print("音频总帧数：", num_frames)
    data = wav_file.readframes(num_frames)  # readframes(n):读取并返回以 bytes 对象表示的最多 n 帧音频
    data = np.frombuffer(data, dtype=np.int16)  # 将采样的点变为数组
    data = data.T  # 转置
    # print("采样的n帧音频：", data)
    time = np.arange(0, num_frames) * (1 / framerate)  # 时间
    # print("时间：", time)
    return data, time, framerate


# pre-emphasis 预加重
# x'[t-1]=x[t]-α*x[t-1]
# α=0.97
def pre_emphasis(data):
    """
    pre-emphasis 预加重
    :param data: 未处理的信号
    :return: signal--预加重后的信号
    """
    signal = []
    for i in range(1, len(data)):
        signal.append(data[i] - 0.97 * data[i - 1])
    signal = np.array(signal)  # 将数组转成NumPy数组
    # print("预加重后的采样点：", signal)
    return signal


# Windowing 加窗函数
# w[n]=(1-α)-α*cos(2pai*n/(L-1)) L:window width
# 取α=0.4614，汉明窗
def hamming(frame_width, framerate=44100):  # 默认帧长和帧移的单位为ms
    """
    汉明窗权重
    :param frame_width: 帧长，默认单位为ms
    :param framerate:采样率
    :return: w--相应的汉明窗权重
    """
    l = float(framerate / 1000)
    width = round(frame_width * l)  # 帧长为25ms，采样率为44100Hz。width=round(frame_width*44.1)
    w = []  # window权重
    for i in range(0, width):
        w.append(0.54 - 0.46 * math.cos(2 * np.pi * i / (width - 1)))  # 计算汉明窗的权重
    w = np.array(w)
    # print("汉明窗权重：", w)
    return w


# 分帧并且加窗，在这里，我选择的汉明窗
def window(signal, w, frame_shift, framerate=44100):  # 参数:采样信号，窗函数，帧移
    """
    分帧并加窗
    :param signal:采样信号
    :param w:窗函数
    :param frame_shift:帧移
    :param framerate:采样率
    :return:分帧并加窗后的信号
    """
    n = len(signal)  # 采样点个数
    width = len(w)  # 帧长
    l = float(framerate / 1000)
    shift = round(frame_shift * l)  # 帧移为10ms，采样率为44100Hz。shift=round(frame_shift*44.1)
    nf = (n - width + shift) // shift  # //-整除（向下取整）
    # 计算分帧后的帧数：(n-overlap)/shift
    # 重叠部分：overlap=width-shift
    window_signal = np.zeros((nf, width))  # 初始化
    df = np.multiply(shift, np.array([i for i in range(nf)]))  # 设置每帧在x中的位移量位置
    for i in range(nf):
        window_signal[i, :] = signal[df[i]:df[i] + width]  # 将数据分帧，即nf X width
    window_signal = np.multiply(window_signal, np.array(w))
    # print("加窗后的采样点：", window_signal)
    return window_signal


# 计算振幅频谱
def get_magnitude(dft_signal):
    """
    计算振幅频谱
    :param dft_signal:
    :return:
    """
    m = dft_signal.shape[0]  # 行数
    n = dft_signal.shape[1]  # 列数
    magnitude = []
    for i in range(m):
        for j in range(n):
            temp = np.absolute(dft_signal[i, j])
            magnitude.append(temp)
    # magnitude = np.array(magnitude)
    magnitude = np.reshape(magnitude, (m, n))
    # print("振幅频谱：", magnitude)
    return magnitude


# 计算能量
def get_energy(magnitude):
    """
    计算能量谱，通过振幅频谱计算
    :param magnitude: 做DFT后的振幅频谱
    :return: energy[]
    """
    m = magnitude.shape[0]  # 行数
    n = magnitude.shape[1]  # 列数
    energy = []
    temp = 0
    for i in range(m):
        for j in range(n):
            temp = temp + np.square(magnitude[i, j])
        energy.append(temp)
    energy = np.array(energy)
    # print("能量：", energy)
    return energy


# 计算得到功率谱
def get_power(window_signal, magnitude):
    """
    计算功率谱，通过振幅频谱计算
    :param window_signal: 加窗后的信号
    :param magnitude: 做DFT后的振幅频谱
    :return: power[]
    """
    m = magnitude.shape[0]  # 行数
    n = magnitude.shape[1]  # 列数
    power = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            temp = np.square(magnitude[i, j]) / window_signal.shape[0]
            power[i, j] = temp
    # print("功率谱的形状：", power.shape)
    # print("功率谱：", power)
    return power


# Mel滤波器
def mel_filters(m, n, fl, fs=44100):
    """
    Mel滤波器
    :param m:滤波器个数
    :param n:一帧FFT后保留的点数
    :param fs:采样频率
    :param fl:最低频率
    :return:
    """
    fh = fs / 2  # 最高频率，为采样频率fs的一半
    ml = 1127 * np.log(1 + fl / 700)
    mh = 1127 * np.log(1 + fh / 700)  # 将Hz转换为Mel
    mel = np.linspace(ml, mh, m + 2)  # 将Mel刻度等间隔
    # print("mel", mel)
    h = 700 * (np.exp(mel / 1127) - 1)  # 将Mel转换为Hz
    # print("h:", h)
    f = np.floor((n + 1) * h / fs)
    # print("f:", f)
    w = int(n / 2 + 1)  # 采样频率为fs/2的FFT的点数
    freq = [int(i * fs / n) for i in range(w)]  # 采样频率值，为了画图
    bank = np.zeros((m, w))
    for k in range(1, m + 1):
        f0 = f[k]
        f1 = f[k - 1]
        f2 = f[k + 1]
        for i in range(1, w):
            if i < f1:
                continue
            elif f1 <= i <= f0:
                bank[k - 1, i] = (i - f1) / (f0 - f1)
            elif f0 <= i <= f2:
                bank[k - 1, i] = (f2 - i) / (f2 - f0)
            else:
                break
        # 画图
        # plt.plot(freq, bank[k - 1, :], 'r')
        # plt.grid(True)
        # plt.show()
        # plt.savefig('./' + '%d' % k + '.png')
    return bank


# 得到fbank feature
def get_fbank_feature(power, mel):
    """
    计算得到FBANK feature
    :param power: 功率谱
    :param mel: mel滤波器
    :return: fbank feature
    """
    # 功率谱*Mel滤波器
    # yt[m]=Σ(1,N)mel[m,k]power[t,k]
    # power(T,K) mel(M,K)
    temp = power[:, :mel.shape[1]]  # 切片，为了做矩阵乘法
    y = np.dot(temp, mel.T)

    # 得到第一个特征值
    fbank_features = np.log(y)
    # print(fbank_features.shape)
    return fbank_features


# 对fbank feature做倒谱分析，得到MFCC特征系数
def get_mfcc(n_dct, mel, fbank_features):
    """
    对fbank feature做倒谱分析，得到MFCC特征系数
    :param n_dct: DCT后的谱线，为了做差分，会增加几位
    :param mel: mel滤波器
    :param fbank_features: fbank特征
    :return: MFCC特征系数（未进行选择）
    """
    # 倒谱分析，DCT-离散余弦变换，得到MFCC系数
    # 取前12个系数
    # ∑(0,M−1)FBANK_features*cos(n(m+0.5)𝜋/M)
    # n--DCT后的谱线 m--第m个滤波器 M--滤波器总数
    n_dct = 16  # 取前2~13个，但是为了做差分，所以取2~17
    M = mel.shape[0]  # mel滤波器个数
    m = np.array([i for i in range(M)])  # mel.shape[0]-mel滤波器的个数
    # mfcc = np.zeros((fbank_features.shape[0] - 4, n_dct))  # 因为做差分后会有值的减少，所以会从原来的m行变为m-4行
    temp = np.zeros(fbank_features.shape)
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            temp[i, j] = np.sum(np.multiply(fbank_features[i, :], np.cos((m + 0.5) * j * np.pi / M)))
    mfcc = temp[:, 2:18]  # 从第二个开始取
    return mfcc


# 计算差分
def get_diff(mfcc, energy):
    """
    计算差分，最后得到每行39维的矩阵
    :param mfcc: MFCC特征系数
    :param energy: 能量谱
    :return: 动态特征
    """
    # 做差分，12+一个能量信息，共39维
    # 差分：d(t)=(c(t+1)-c(t-1))/2
    # 因为做了差分，所以值会有缩减，如能量
    # 先分别做差分
    m, n = mfcc.shape
    # 一阶差分
    d_mfcc = np.zeros((m, n - 2))
    for i in range(m):
        for j in range(1, n - 1):
            d_mfcc[i, j - 1] = (mfcc[i, j + 1] - mfcc[i, j - 1]) / 2
    d_energy = np.zeros(len(energy) - 2)
    for i in range(1, len(d_energy) - 1):
        d_energy[i - 1] = (energy[i + 1] - energy[i - 1]) / 2
    # print("d_energy:", d_energy[:5])

    # 二阶差分
    dd_mfcc = np.zeros((m, n - 4))
    for i in range(m):
        for j in range(1, n - 3):
            dd_mfcc[i, j - 1] = (d_mfcc[i, j + 1] - d_mfcc[i, j - 1]) / 2
    # print("dd_mfcc", dd_mfcc[1])
    dd_energy = np.zeros(len(d_energy) - 2)
    for i in range(1, len(dd_energy) - 1):
        dd_energy[i - 1] = (d_energy[i + 1] - d_energy[i - 1]) / 2
    # print("dd_energy:", dd_energy[:3])

    m = len(dd_energy)
    n = 3 * (dd_mfcc.shape[1] + 1)  # 应该是3*（12个MFCC系数+1个能量）
    # print(m, n)
    diff = np.zeros((m, n))
    for i in range(m):
        diff[i, :int(n / 3 - 1)] = mfcc[i, :int(n / 3 - 1)]  # 取mfcc的前12个[0:12]
        diff[i, int(n / 3 - 1)] = energy[i]
        diff[i, int(n / 3):int(2 * n / 3 - 1)] = d_mfcc[i, int(n / 3 - 1)]  # 取d_mfcc的前12个#[13:25]
        diff[i, int(2 * n / 3 - 1)] = d_energy[i]
        diff[i, int(2 * n / 3):n - 1] = dd_mfcc[i]  # [26:38]
        diff[i, n - 1] = dd_energy[i]
    return diff


# 标准化
def normalize(diff):
    """
    标准化
    :param diff:动态特征
    :return: 标准化后的特征
    """
    # 标准化
    # 先计算每一维的均值和标准差
    avr = np.mean(diff, axis=0)
    std = np.std(diff, axis=0)

    # 进行标准化 normalize
    # y'=(y-a(y))/v(y)
    # a(y)--平均值 v(y)--方差
    normalization = np.zeros((diff.shape[0], diff.shape[1]))
    for i in range(normalization.shape[0]):
        normalization[i, :] = np.divide(np.subtract(diff[i, :], avr), std)
    return normalization


def get_feature(file_name):
    # 处理wav文件，得到基本参数
    wav_data, wav_time, framerate = wav_read(file_name)  # 得到采样点和时间

    # pre-emphasis 预加重
    emphasis_signal = pre_emphasis(wav_data)  # 预加重后的采样点

    # 加窗函数
    win = hamming(25, framerate=framerate)  # 帧长为25ms

    # Windowing 加窗操作
    window_signal = window(emphasis_signal, win, 10, framerate=framerate)  # 帧移为10ms

    # 对每一帧做DFT
    # 进行了补零操作，DFT长度为2048
    # dft_signal = np.zeros((window_signal.shape[0], 2048), dtype=complex)
    # # print("DFT的形状：",dft_signal.shape)
    # for i in range(window_signal.shape[0]):
    #     temp = np.fft.fft(window_signal[i], 2048)
    #     dft_signal[i] = temp
    # # print("DFT后：", dft_signal)
    # 进行了补零操作，DFT长度为512
    dft_signal = np.zeros((window_signal.shape[0], 512), dtype=complex)
    # print("DFT的形状：",dft_signal.shape)
    for i in range(window_signal.shape[0]):
        temp = np.fft.fft(window_signal[i], 512)
        dft_signal[i] = temp
    # print("DFT后：", dft_signal)

    # 计算振幅频谱
    magnitude = get_magnitude(dft_signal)

    # 计算得到功率谱
    # Pi(k)=(Di(k)^2)/N  N--窗长  1<=k<=K  K--DFT的长度
    # 在这里，N=window_signal.shape[0]，即1102；K=dft_signal.shape[0]，即2048
    # 资料里写到：通常会执行512点FFT，并仅保留前 257 个系数
    # 所以在这里，我是执行的2048点FFT，保留帧长的采样点个数，即1102
    power = get_power(window_signal, magnitude)
    # print("power:", power.shape)

    # 得到能量频谱（通过振幅频谱得到）
    energy = get_energy(magnitude)
    # print(energy.shape)

    # Mel滤波器
    mel = mel_filters(26, power.shape[1], 300, fs=framerate)
    # print("mel:", mel.shape)

    # 得到第一个特征值--fbank feature
    fbank_features = get_fbank_feature(power, mel)
    # print(fbank_features.shape)

    # 倒谱分析，DCT-离散余弦变换，得到MFCC系数
    mfcc = get_mfcc(16, mel, fbank_features)

    # 计算差分，12+一个能量信息，共39维
    diff = get_diff(mfcc, energy)

    # 标准化
    normalization = normalize(diff)

    # print(normalization.shape)

    # 将标准化后的特征保存为txt文件
    # np.savetxt("feature.txt", normalization)
    # np.savetxt("feature.csv", normalization, delimiter=',')

    return normalization
