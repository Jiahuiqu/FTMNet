
import torch
import torch.utils
import scipy.io as sio
from sklearn import preprocessing


def print_data(gt, class_count):
    gt_reshape = np.reshape(gt, [-1])
    for i in range(class_count):
        idx = np.where(gt_reshape == i + 1)[-1]
        samplesCount = len(idx)
        print('第' + str(i + 1) + '类的个数为' + str(samplesCount))


def get_dataset(dataset):

    data_HSI = []
    data_MSI = []
    data_HR = []
    gt = []
    val_ratio = 0
    class_count = 0
    learning_rate = 0
    max_epoch = 0
    dataset_name = ''
    trainloss_result = []

    if dataset == 'paviau':
        data_HSI_mat = sio.loadmat('/home/amax/ZYS/数据集/paviau/paviau/HSIpaviau4.mat')
        data_HSI = data_HSI_mat['HSIpaviau4']
        data_MSI_mat = sio.loadmat('/home/amax/ZYS/数据集/paviau/paviau/MSIpaviau.mat')
        data_MSI = data_MSI_mat['MSIpaviau']
        data_HR_mat = sio.loadmat('/home/amax/ZYS/数据集/paviau/paviau/paviaU.mat')
        data_HR = data_HR_mat['paviaU']

        gt_mat = sio.loadmat('/home/amax/ZYS/数据集/paviau/paviau/paviaU_gt.mat')
        gt = gt_mat['paviaU_gt']

        # 参数预设
        val_ratio = 0.01  # 验证机比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 9  # 样本类别数
        learning_rate = 2e-4  # 学习率
        max_epoch = 500  # 迭代次数
        dataset_name = "pavia"  # 数据集名称
        trainloss_result = np.zeros([max_epoch + 1, 1])
        pass


    elif dataset == 'houston':
        data_HSI_mat = sio.loadmat('/home/amax/ZYS/数据集/houston/HSIhouston4.mat')
        data_HSI = data_HSI_mat['HSIhouston4']
        data_MSI_mat = sio.loadmat('/home/amax/ZYS/数据集/houston/MSIhouston.mat')
        data_MSI = data_MSI_mat['MSIhouston']
        data_HR_mat = sio.loadmat('/home/amax/ZYS/数据集/houston/HSI_data.mat')
        data_HR = data_HR_mat['HSI_data']

        gt_mat = sio.loadmat('/home/amax/ZYS/数据集/houston/All_Label.mat')
        gt = gt_mat['All_Label']

        # 参数预设
        val_ratio = 0.01  # 验证机比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 15  # 样本类别数
        learning_rate = 2e-4  # 学习率
        max_epoch = 500  # 迭代次数
        dataset_name = "Huston2013"  # 数据集名称
        trainloss_result = np.zeros([max_epoch + 1, 1])
        pass

    elif dataset == 'indian':
        data_HSI_mat = sio.loadmat('/home/amax/ZYS/数据集/indian2/HSIindian.mat')
        data_HSI = data_HSI_mat['HSIindian']
        data_MSI_mat = sio.loadmat('/home/amax/ZYS/数据集/indian2/MSIindian.mat')
        data_MSI = data_MSI_mat['MSIindian']
        data_HR_mat = sio.loadmat('/home/amax/ZYS/数据集/indian2/Indian_pines.mat')
        data_HR = data_HR_mat['indian_pines']

        gt_mat = sio.loadmat('/home/amax/ZYS/数据集/indian2/Indian_pines_gt.mat')
        gt = gt_mat['indian_pines_gt']

        # 参数预设
        val_ratio = 0.01  # 验证机比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 16  # 样本类别数
        learning_rate = 2e-4  # 学习率
        max_epoch = 500  # 迭代次数
        dataset_name = "indian"  # 数据集名称
        trainloss_result = np.zeros([max_epoch + 1, 1])
        pass

    elif dataset == 'salinas':
        data_HSI_mat = sio.loadmat('/home/amax/ZYS/数据集/Salinas/HSIsalinas.mat')
        data_HSI = data_HSI_mat['HSIsalinas']
        data_MSI_mat = sio.loadmat('/home/amax/ZYS/数据集/Salinas/MSIsalinas.mat')
        data_MSI = data_MSI_mat['MSIsalinas']
        data_HR_mat = sio.loadmat('/home/amax/ZYS/数据集/Salinas/Salinas_corrected.mat')
        data_HR = data_HR_mat['salinas_corrected']

        gt_mat = sio.loadmat('/home/amax/ZYS/数据集/Salinas/Salinas_gt.mat')
        gt = gt_mat['salinas_gt']

        # 参数预设
        val_ratio = 0.01  # 验证机比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 16  # 样本类别数
        learning_rate = 2e-4  # 学习率
        max_epoch = 500  # 迭代次数
        dataset_name = "indian"  # 数据集名称
        trainloss_result = np.zeros([max_epoch + 1, 1])
        pass

    elif dataset == 'hanchuan':
        data_HSI_mat = sio.loadmat('/media/xd132/USER_new/ZYS/数据集/WHU-Hi-HanChuan/HSIhanchuan.mat')
        data_HSI = data_HSI_mat['HSIhanchuan']
        data_MSI_mat = sio.loadmat('/media/xd132/USER_new/ZYS/数据集/WHU-Hi-HanChuan/MSIhanchuan.mat')
        data_MSI = data_MSI_mat['MSIhanchuan']
        data_HR_mat = sio.loadmat('/media/xd132/USER_new/ZYS/数据集/WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat')
        data_HR = data_HR_mat['WHU_Hi_HanChuan']

        gt_mat = sio.loadmat('/media/xd132/USER_new/ZYS/数据集/WHU-Hi-HanChuan/WHU_Hi_HanChuan_gt.mat')
        gt = gt_mat['WHU_Hi_HanChuan_gt']

        # 参数预设
        val_ratio = 0.01  # 验证机比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 16  # 样本类别数
        learning_rate = 2e-4  # 学习率
        max_epoch = 500  # 迭代次数
        dataset_name = "hanchuan"  # 数据集名称
        trainloss_result = np.zeros([max_epoch + 1, 1])
        pass

    elif dataset == 'LongKou':
        data_HSI_mat = sio.loadmat('/media/xd132/USER_new/ZYS/数据集/WHU-Hi-LongKou/HSILongKou.mat')
        data_HSI = data_HSI_mat['HSILongKou']
        data_MSI_mat = sio.loadmat('/media/xd132/USER_new/ZYS/数据集/WHU-Hi-LongKou/MSILongKou.mat')
        data_MSI = data_MSI_mat['MSILongKou']
        data_HR_mat = sio.loadmat('/media/xd132/USER_new/ZYS/数据集/WHU-Hi-LongKou/WHU_Hi_LongKou.mat')
        data_HR = data_HR_mat['WHU_Hi_LongKou']

        gt_mat = sio.loadmat('/media/xd132/USER_new/ZYS/数据集/WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat')
        gt = gt_mat['WHU_Hi_LongKou_gt']

        # 参数预设
        val_ratio = 0.01  # 验证机比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 9  # 样本类别数
        learning_rate = 2e-4  # 学习率
        max_epoch = 500  # 迭代次数
        dataset_name = "LongKou"  # 数据集名称
        trainloss_result = np.zeros([max_epoch + 1, 1])
        pass

    return [data_HSI, data_MSI, data_HR, gt, val_ratio, class_count,
            learning_rate, max_epoch, dataset_name, trainloss_result]



def data_standard(data_HSI):

    height, width, bands = data_HSI.shape  # 原始高光谱数据的三个维度

    data_HSI = np.reshape(data_HSI, [height * width, bands])  # 将数据转为HW * B
    minMax = preprocessing.StandardScaler()
    data_HSI = minMax.fit_transform(data_HSI)  # 这两行用来归一化数据，归一化时需要进行数据转换
    data_HSI = np.reshape(data_HSI, [height, width, bands])  # 将数据转回去 H * W * B

    return [data_HSI]

def data_standard2(data_HSI):
    min_val = np.min(data_HSI)
    max_val = np.max(data_HSI)
    normalized_data = (data_HSI - min_val) / (max_val - min_val)

    # normalized_data 的值范围在 [0, 1] 之间

    return [normalized_data]


import numpy as np
import random


def data_partition(samples_type, class_count, gt, train_ratio, val_ratio, height, width):
    train_rand_idx = []
    train_data_index = []
    test_data_index = []
    val_data_index = []
    val_samples = class_count

    # 将 ground truth 展平为一维数组
    gt_reshape = np.reshape(gt, [-1])

    # 总样本数量
    total_samples = len(gt_reshape)

    if samples_type == 'ratio':  # 从所有样本中按比例取样
        # 随机打乱所有样本的索引
        all_data_index = [i for i in range(total_samples)]
        random.shuffle(all_data_index)

        # 计算训练集、验证集和测试集的样本数量
        train_count = int(np.ceil(total_samples * train_ratio))  # 训练集数量
        val_count = int(np.ceil(total_samples * val_ratio))  # 验证集数量
        test_count = total_samples - train_count - val_count  # 测试集数量

        # 根据比例从所有样本中随机选取训练集、验证集和测试集
        train_data_index = all_data_index[:train_count]
        val_data_index = all_data_index[train_count:train_count + val_count]
        test_data_index = all_data_index[train_count + val_count:]

    # 将索引转换为列表并返回
    train_data_index = list(train_data_index)
    val_data_index = list(val_data_index)
    test_data_index = list(test_data_index)

    if samples_type == 'same_num':  # 取固定数量训练
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            real_train_samples_per_class = train_ratio
            rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
            if real_train_samples_per_class > samplesCount:
                real_train_samples_per_class = samplesCount
            rand_idx = random.sample(rand_list,
                                     real_train_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
            rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
            train_rand_idx.append(rand_real_idx_per_class_train)
        train_rand_idx = np.array(train_rand_idx)
        train_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_data_index.append(a[j])
        train_data_index = np.array(train_data_index)

        train_data_index = set(train_data_index)
        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)

        # 背景像元的标签
        background_idx = np.where(gt_reshape == 0)[-1]
        background_idx = set(background_idx)
        # test_data_index = all_data_index - train_data_index - background_idx
        test_data_index = all_data_index - background_idx

        # 从测试集中随机选取部分样本作为验证集
        val_data_count = int(val_samples)  # 验证集数量
        val_data_index = random.sample(test_data_index, val_data_count)
        val_data_index = set(val_data_index)
        test_data_index = test_data_index
        # test_data_index = test_data_index - val_data_index

        # 将训练集 验证集 测试集 整理
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        val_data_index = list(val_data_index)
        pass

    # 获取训练样本的标签图
    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_data_index)):
        train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
        pass
    train_label = np.reshape(train_samples_gt, [height, width])

    # 获取测试样本的标签图
    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_data_index)):
        test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
        pass
    test_label = np.reshape(test_samples_gt, [height, width])  # 测试样本图

    # 获取验证集样本的标签图
    val_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(val_data_index)):
        val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
        pass
    val_label = np.reshape(val_samples_gt, [height, width])

    return [train_label, test_label, val_label]


def target_data_partition(gt, height, width):
    gt_reshape = np.reshape(gt, [-1])
    # 背景像元的标签
    background_idx = np.where(gt_reshape == 0)[-1]
    background_idx = set(background_idx)
    all_data_index = [i for i in range(len(gt_reshape))]
    all_data_index = set(all_data_index)
    test_data_index = all_data_index - background_idx
    test_data_index = list(test_data_index)

    # 获取测试样本的标签图
    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_data_index)):
        test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
        pass
    test_label = np.reshape(test_samples_gt, [height, width])  # 测试样本图
    test_label_tar = test_label
    return test_label

def closest_smaller(number):
    if number % 2 == 0:
        return number - 1  # 如果给定数是偶数，返回比它小的最大奇数
    else:
        return number - 0

def gen_cnn_data1(data_HSI, data_MSI, data_HR, patchsize_HSI, patchsize_MSI, patchsize_HR, train_label, test_label, down_ratio=4):
    height, width, bands = data_HSI.shape
    # ##### 给HSI打padding #####
    # 先给第一个维度打padding，确定打完padding的矩阵的大小后，建立一个[H,W,C]的空矩阵，再用循环给所有维度打padding
    temp = data_HSI[:, :, 0]
    pad_width = np.floor(patchsize_HSI / 2)
    pad_width = int(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [h_pad, w_pad] = temp2.shape
    data_HSI_pad = np.empty((h_pad, w_pad, bands), dtype='float16')

    for i in range(bands):
        temp = data_HSI[:, :, i]
        pad_width = np.floor(patchsize_HSI / 2)
        pad_width = int(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        data_HSI_pad[:, :, i] = temp2

    # ##### 给MSI打padding #####

    height_MSI, width_MSI, bands_MSI = data_MSI.shape
    temp_msi = data_MSI[:, :, 0]
    pad_width_msi = np.floor(patchsize_MSI / 2)
    pad_width_msi = int(pad_width_msi)
    temp2_msi = np.pad(temp_msi, pad_width_msi, 'symmetric')
    [h_pad_msi, w_pad_msi] = temp2_msi.shape
    data_MSI_pad = np.empty((h_pad_msi, w_pad_msi, bands_MSI), dtype='float16')

    for i in range(bands_MSI):
        temp = data_MSI[:, :, i]
        pad_width_msi = np.floor(patchsize_MSI / 2)
        pad_width_msi = int(pad_width_msi)
        temp2_msi = np.pad(temp, pad_width_msi, 'symmetric')
        data_MSI_pad[:, :, i] = temp2_msi

    # ##### 给HR打padding #####

    height_HR, width_HR, bands_HR = data_HR.shape
    temp_hr = data_HR[:, :, 0]
    pad_width_hr = np.floor(patchsize_HR / 2)
    pad_width_hr = int(pad_width_hr)
    temp2_hr = np.pad(temp_hr, pad_width_hr, 'symmetric')
    [h_pad_hr, w_pad_hr] = temp2_hr.shape
    data_HR_pad = np.empty((h_pad_hr, w_pad_hr, bands_HR), dtype='float16')

    for i in range(bands_HR):
        temp = data_HR[:, :, i]
        pad_width_hr = np.floor(patchsize_HR / 2)
        pad_width_hr = int(pad_width_hr)
        temp2_hr = np.pad(temp, pad_width_hr, 'symmetric')
        data_HR_pad[:, :, i] = temp2_hr



    # #### 构建高光谱的训练集和测试集 #####
    [ind1, ind2] = np.where(train_label != 0)  # ind1和ind2是符合where中条件的点的横纵坐标
    TrainNum = len(ind1)
    TrainPatch_HSI = np.empty((TrainNum, bands, patchsize_HSI, patchsize_HSI), dtype='float16')
    TrainLabel_HSI = np.empty(TrainNum)
    ind3 = ind1 + pad_width  # 因为在图像四周打了padding，所以横纵坐标要加上打padding的像素个数
    ind4 = ind2 + pad_width

    # #### 构建MSI的训练集和测试集 #####
    train_label_MSI = np.repeat(train_label, down_ratio, axis=1)
    train_label_MSI = np.repeat(train_label_MSI, down_ratio, axis=0)
    ind1_msi = ind1 * down_ratio
    ind2_msi = ind2 * down_ratio
    TrainNum_msi = len(ind1_msi)
    TrainPatch_MSI = np.empty((TrainNum_msi, bands_MSI, patchsize_MSI, patchsize_MSI), dtype='float16')
    TrainLabel_MSI = np.empty(TrainNum_msi)
    ind3_msi = ind1_msi + pad_width_msi  # 因为在图像四周打了padding，所以横纵坐标要加上打padding的像素个数
    ind4_msi = ind2_msi + pad_width_msi

    # #### 构建HR的训练集和测试集 #####
    train_label_HR = np.repeat(train_label, down_ratio, axis=1)
    train_label_HR = np.repeat(train_label_HR, down_ratio, axis=0)
    ind1_hr = ind1 * down_ratio
    ind2_hr = ind2 * down_ratio
    TrainNum_hr = len(ind1_hr)
    TrainPatch_HR = np.empty((TrainNum_hr, bands_HR, patchsize_HR, patchsize_HR), dtype='float16')
    TrainLabel_HR = np.empty(TrainNum_hr)
    ind3_hr = ind1_hr + pad_width_hr  # 因为在图像四周打了padding，所以横纵坐标要加上打padding的像素个数
    ind4_hr = ind2_hr + pad_width_hr

    for i in range(len(ind1)):
        # x是打了padding的高光谱，下文的x2是打了padding的LiDAR
        # 取第i个训练patch，取一个立方体
        patch = data_HSI_pad[(ind3[i] - pad_width):(ind3[i] + pad_width),
                             (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        # conv的输入是NCHW，现在的patch是[H,W,C]，要把patch转成[C,H,W]形状
        patch = np.reshape(patch, (patchsize_HSI * patchsize_HSI, bands))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (bands, patchsize_HSI, patchsize_HSI))
        TrainPatch_HSI[i, :, :, :] = patch
        patchlabel_HSI = train_label[ind1[i], ind2[i]]
        TrainLabel_HSI[i] = patchlabel_HSI

        # MSI
        patch_msi = data_MSI_pad[(ind3_msi[i] - pad_width_msi):(ind3_msi[i] + pad_width_msi),
                    (ind4_msi[i] - pad_width_msi):(ind4_msi[i] + pad_width_msi), :]
        # conv的输入是NCHW，现在的patch是[H,W,C]，要把patch转成[C,H,W]形状
        patch_msi = np.reshape(patch_msi, (patchsize_MSI * patchsize_MSI, bands_MSI))
        patch_msi = np.transpose(patch_msi)
        patch_msi = np.reshape(patch_msi, (bands_MSI, patchsize_MSI, patchsize_MSI))
        TrainPatch_MSI[i, :, :, :] = patch_msi
        patchlabel_MSI = train_label_MSI[ind1_msi[i], ind2_msi[i]]
        TrainLabel_MSI[i] = patchlabel_MSI

        # HR
        patch_hr = data_HR_pad[(ind3_hr[i] - pad_width_hr):(ind3_hr[i] + pad_width_hr),
                    (ind4_hr[i] - pad_width_hr):(ind4_hr[i] + pad_width_hr), :]
        # conv的输入是NCHW，现在的patch是[H,W,C]，要把patch转成[C,H,W]形状
        patch_hr = np.reshape(patch_hr, (patchsize_HR * patchsize_HR, bands_HR))
        patch_hr = np.transpose(patch_hr)
        patch_hr = np.reshape(patch_hr, (bands_HR, patchsize_HR, patchsize_HR))
        TrainPatch_HR[i, :, :, :] = patch_hr
        patchlabel_HR = train_label_HR[ind1_hr[i], ind2_hr[i]]
        TrainLabel_HR[i] = patchlabel_HR

    [ind1, ind2] = np.where(test_label != 0)
    TestNum = len(ind1)
    TestPatch_HSI = np.empty((TestNum, bands, patchsize_HSI, patchsize_HSI), dtype='float16')
    TestLabel_HSI = np.empty(TestNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width

    # MSI
    test_label_MSI = np.repeat(test_label, down_ratio, axis=1)
    test_label_MSI = np.repeat(test_label_MSI, down_ratio, axis=0)
    ind1_msi = ind1 * down_ratio
    ind2_msi = ind2 * down_ratio
    TestNum_msi = len(ind1_msi)
    TestPatch_MSI = np.empty((TestNum_msi, bands_MSI, patchsize_MSI, patchsize_MSI), dtype='float16')
    TestLabel_MSI = np.empty(TestNum_msi)
    ind3_msi = ind1_msi + pad_width_msi
    ind4_msi = ind2_msi + pad_width_msi

    # HR
    test_label_HR = np.repeat(test_label, down_ratio, axis=1)
    test_label_HR = np.repeat(test_label_HR, down_ratio, axis=0)
    ind1_hr = ind1 * down_ratio
    ind2_hr = ind2 * down_ratio
    TestNum_hr = len(ind1_hr)
    TestPatch_HR = np.empty((TestNum_hr, bands_HR, patchsize_HR, patchsize_HR), dtype='float16')
    TestLabel_HR = np.empty(TestNum_hr)
    ind3_hr = ind1_hr + pad_width_hr
    ind4_hr = ind2_hr + pad_width_hr



    for i in range(len(ind1)):
        patch = data_HSI_pad[(ind3[i] - pad_width):(ind3[i] + pad_width),
                             (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch = np.reshape(patch, (patchsize_HSI * patchsize_HSI, bands))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (bands, patchsize_HSI, patchsize_HSI))
        TestPatch_HSI[i, :, :, :] = patch
        patchlabel_HSI = test_label[ind1[i], ind2[i]]
        TestLabel_HSI[i] = patchlabel_HSI

        # MSI
        patch_msi = data_MSI_pad[(ind3_msi[i] - pad_width_msi):(ind3_msi[i] + pad_width_msi),
                    (ind4_msi[i] - pad_width_msi):(ind4_msi[i] + pad_width_msi), :]
        patch_msi = np.reshape(patch_msi, (patchsize_MSI * patchsize_MSI, bands_MSI))
        patch_msi = np.transpose(patch_msi)
        patch_msi = np.reshape(patch_msi, (bands_MSI, patchsize_MSI, patchsize_MSI))
        TestPatch_MSI[i, :, :, :] = patch_msi
        patchlabel_MSI = test_label_MSI[ind1_msi[i], ind2_msi[i]]
        TestLabel_MSI[i] = patchlabel_MSI

        # HR
        patch_hr = data_HR_pad[(ind3_hr[i] - pad_width_hr):(ind3_hr[i] + pad_width_hr),
                    (ind4_hr[i] - pad_width_hr):(ind4_hr[i] + pad_width_hr), :]
        patch_hr = np.reshape(patch_hr, (patchsize_HR * patchsize_HR, bands_HR))
        patch_hr = np.transpose(patch_hr)
        patch_hr = np.reshape(patch_hr, (bands_HR, patchsize_HR, patchsize_HR))
        TestPatch_HR[i, :, :, :] = patch_hr
        patchlabel_HR = test_label_HR[ind1_hr[i], ind2_hr[i]]
        TestLabel_HR[i] = patchlabel_HR





    print('Training size and testing size of HSI are:', TrainPatch_HSI.shape, 'and', TestPatch_HSI.shape)

    # #### 数据转换以及把数据搬到GPU #####
    TrainPatch_HSI = torch.from_numpy(TrainPatch_HSI)
    TrainLabel_HSI = torch.from_numpy(TrainLabel_HSI) - 1
    TrainLabel_HSI = TrainLabel_HSI.long()

    TestPatch_HSI = torch.from_numpy(TestPatch_HSI)
    TestLabel_HSI = torch.from_numpy(TestLabel_HSI) - 1
    TestLabel_HSI = TestLabel_HSI.long()

    TrainPatch_MSI = torch.from_numpy(TrainPatch_MSI)
    TrainLabel_MSI = torch.from_numpy(TrainLabel_MSI) - 1
    TrainLabel_MSI = TrainLabel_MSI.long()

    TestPatch_MSI = torch.from_numpy(TestPatch_MSI)
    TestLabel_MSI = torch.from_numpy(TestLabel_MSI) - 1
    TestLabel_MSI = TestLabel_MSI.long()

    TrainPatch_HR = torch.from_numpy(TrainPatch_HR)
    TrainLabel_HR = torch.from_numpy(TrainLabel_HR) - 1
    TrainLabel_HR = TrainLabel_HR.long()

    TestPatch_HR = torch.from_numpy(TestPatch_HR)
    TestLabel_HR = torch.from_numpy(TestLabel_HR) - 1
    TestLabel_HR = TestLabel_HR.long()

    return TrainPatch_HSI, TrainLabel_HSI, TestPatch_HSI, TestLabel_HSI,\
        TrainPatch_MSI, TrainLabel_MSI, TestPatch_MSI, TestLabel_MSI,\
        TrainPatch_HR, TrainLabel_HR, TestPatch_HR, TestLabel_HR



def gen_cnn_data2(data_HSI, data_MSI, data_HR, patchsize_HSI, patchsize_MSI, patchsize_HR, train_label, test_label, down_ratio=4):
    height, width, bands = data_HSI.shape
    # ##### 给HSI打padding #####
    # 先给第一个维度打padding，确定打完padding的矩阵的大小后，建立一个[H,W,C]的空矩阵，再用循环给所有维度打padding
    temp = data_HSI[:, :, 0]
    pad_width = np.floor(patchsize_HSI / 2)
    pad_width = int(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [h_pad, w_pad] = temp2.shape
    data_HSI_pad = np.empty((h_pad, w_pad, bands), dtype='float16')

    for i in range(bands):
        temp = data_HSI[:, :, i]
        pad_width = np.floor(patchsize_HSI / 2)
        pad_width = int(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        data_HSI_pad[:, :, i] = temp2

    # ##### 给MSI打padding #####

    height_MSI, width_MSI, bands_MSI = data_MSI.shape
    temp_msi = data_MSI[:, :, 0]
    pad_width_msi = np.floor(patchsize_MSI / 2)
    pad_width_msi = int(pad_width_msi)
    temp2_msi = np.pad(temp_msi, pad_width_msi, 'symmetric')
    [h_pad_msi, w_pad_msi] = temp2_msi.shape
    data_MSI_pad = np.empty((h_pad_msi, w_pad_msi, bands_MSI), dtype='float16')

    for i in range(bands_MSI):
        temp = data_MSI[:, :, i]
        pad_width_msi = np.floor(patchsize_MSI / 2)
        pad_width_msi = int(pad_width_msi)
        temp2_msi = np.pad(temp, pad_width_msi, 'symmetric')
        data_MSI_pad[:, :, i] = temp2_msi

    # ##### 给HR打padding #####

    height_HR, width_HR, bands_HR = data_HR.shape
    temp_hr = data_HR[:, :, 0]
    pad_width_hr = np.floor(patchsize_HR / 2)
    pad_width_hr = int(pad_width_hr)
    temp2_hr = np.pad(temp_hr, pad_width_hr, 'symmetric')
    [h_pad_hr, w_pad_hr] = temp2_hr.shape
    data_HR_pad = np.empty((h_pad_hr, w_pad_hr, bands_HR), dtype='float16')

    for i in range(bands_HR):
        temp = data_HR[:, :, i]
        pad_width_hr = np.floor(patchsize_HR / 2)
        pad_width_hr = int(pad_width_hr)
        temp2_hr = np.pad(temp, pad_width_hr, 'symmetric')
        data_HR_pad[:, :, i] = temp2_hr


    # #### 构建MSI的训练集和测试集 #####
    [ind1, ind2] = np.where(train_label != 0)  # ind1和ind2是符合where中条件的点的横纵坐标
    TrainNum = len(ind1)
    TrainPatch_MSI = np.empty((TrainNum, bands_MSI, patchsize_MSI, patchsize_MSI), dtype='float16')
    TrainLabel_MSI = np.empty(TrainNum)
    ind3_msi = ind1 + pad_width_msi  # 因为在图像四周打了padding，所以横纵坐标要加上打padding的像素个数
    ind4_msi = ind2 + pad_width_msi

    # #### 构建HR的训练集和测试集 #####
    TrainPatch_HR = np.empty((TrainNum, bands_HR, patchsize_HR, patchsize_HR), dtype='float16')
    TrainLabel_HR = np.empty(TrainNum)
    ind3_hr = ind1 + pad_width_hr  # 因为在图像四周打了padding，所以横纵坐标要加上打padding的像素个数
    ind4_hr = ind2 + pad_width_hr

    # #### 构建高光谱的训练集和测试集 #####
    ind1_hsi = ind1 // down_ratio
    ind2_hsi = ind2 // down_ratio
    TrainPatch_HSI = np.empty((TrainNum, bands, patchsize_HSI, patchsize_HSI), dtype='float16')
    TrainLabel_HSI = np.empty(TrainNum)
    ind3_hsi = ind1_hsi + pad_width  # 因为在图像四周打了padding，所以横纵坐标要加上打padding的像素个数
    ind4_hsi = ind2_hsi + pad_width



    for i in range(len(ind1)):
        # x是打了padding的高光谱，下文的x2是打了padding的LiDAR
        # 取第i个训练patch，取一个立方体
        # MSI
        patch_msi = data_MSI_pad[(ind3_msi[i] - pad_width_msi):(ind3_msi[i] + pad_width_msi),
                    (ind4_msi[i] - pad_width_msi):(ind4_msi[i] + pad_width_msi), :]
        # conv的输入是NCHW，现在的patch是[H,W,C]，要把patch转成[C,H,W]形状
        patch_msi = np.reshape(patch_msi, (patchsize_MSI * patchsize_MSI, bands_MSI))
        patch_msi = np.transpose(patch_msi)
        patch_msi = np.reshape(patch_msi, (bands_MSI, patchsize_MSI, patchsize_MSI))
        TrainPatch_MSI[i, :, :, :] = patch_msi
        patchlabel_MSI = train_label[ind1[i], ind2[i]]
        TrainLabel_MSI[i] = patchlabel_MSI

        # HR
        patch_hr = data_HR_pad[(ind3_hr[i] - pad_width_hr):(ind3_hr[i] + pad_width_hr),
                    (ind4_hr[i] - pad_width_hr):(ind4_hr[i] + pad_width_hr), :]
        # conv的输入是NCHW，现在的patch是[H,W,C]，要把patch转成[C,H,W]形状
        patch_hr = np.reshape(patch_hr, (patchsize_HR * patchsize_HR, bands_HR))
        patch_hr = np.transpose(patch_hr)
        patch_hr = np.reshape(patch_hr, (bands_HR, patchsize_HR, patchsize_HR))
        TrainPatch_HR[i, :, :, :] = patch_hr
        patchlabel_HR = train_label[ind1[i], ind2[i]]
        TrainLabel_HR[i] = patchlabel_HR

        patch = data_HSI_pad[(ind3_hsi[i] - pad_width):(ind3_hsi[i] + pad_width),
                (ind4_hsi[i] - pad_width):(ind4_hsi[i] + pad_width), :]
        # conv的输入是NCHW，现在的patch是[H,W,C]，要把patch转成[C,H,W]形状
        patch = np.reshape(patch, (patchsize_HSI * patchsize_HSI, bands))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (bands, patchsize_HSI, patchsize_HSI))
        TrainPatch_HSI[i, :, :, :] = patch
        patchlabel_HSI = train_label[closest_smaller(ind1[i]), closest_smaller(ind2[i])]
        TrainLabel_HSI[i] = patchlabel_HSI


    # #### 构建MSI的训练集和测试集 #####
    [ind1, ind2] = np.where(test_label != 0)  # ind1和ind2是符合where中条件的点的横纵坐标
    TestNum = len(ind1)
    TestPatch_MSI = np.empty((TestNum, bands_MSI, patchsize_MSI, patchsize_MSI), dtype='float16')
    TestLabel_MSI = np.empty(TestNum)
    ind3_msi = ind1 + pad_width_msi  # 因为在图像四周打了padding，所以横纵坐标要加上打padding的像素个数
    ind4_msi = ind2 + pad_width_msi

    # #### 构建HR的训练集和测试集 #####
    TestPatch_HR = np.empty((TestNum, bands_HR, patchsize_HR, patchsize_HR), dtype='float16')
    TestLabel_HR = np.empty(TestNum)
    ind3_hr = ind1 + pad_width_hr  # 因为在图像四周打了padding，所以横纵坐标要加上打padding的像素个数
    ind4_hr = ind2 + pad_width_hr

    # #### 构建高光谱的训练集和测试集 #####
    ind1_hsi = ind1 // down_ratio
    ind2_hsi = ind2 // down_ratio
    TestPatch_HSI = np.empty((TestNum, bands, patchsize_HSI, patchsize_HSI), dtype='float16')
    TestLabel_HSI = np.empty(TestNum)
    ind3_hsi = ind1_hsi + pad_width  # 因为在图像四周打了padding，所以横纵坐标要加上打padding的像素个数
    ind4_hsi = ind2_hsi + pad_width

    for i in range(len(ind1)):
        # x是打了padding的高光谱，下文的x2是打了padding的LiDAR
        # 取第i个训练patch，取一个立方体
        # MSI
        patch_msi = data_MSI_pad[(ind3_msi[i] - pad_width_msi):(ind3_msi[i] + pad_width_msi),
                    (ind4_msi[i] - pad_width_msi):(ind4_msi[i] + pad_width_msi), :]
        # conv的输入是NCHW，现在的patch是[H,W,C]，要把patch转成[C,H,W]形状
        patch_msi = np.reshape(patch_msi, (patchsize_MSI * patchsize_MSI, bands_MSI))
        patch_msi = np.transpose(patch_msi)
        patch_msi = np.reshape(patch_msi, (bands_MSI, patchsize_MSI, patchsize_MSI))
        TestPatch_MSI[i, :, :, :] = patch_msi
        patchlabel_MSI = test_label[ind1[i], ind2[i]]
        TestLabel_MSI[i] = patchlabel_MSI

        # HR
        patch_hr = data_HR_pad[(ind3_hr[i] - pad_width_hr):(ind3_hr[i] + pad_width_hr),
                   (ind4_hr[i] - pad_width_hr):(ind4_hr[i] + pad_width_hr), :]
        # conv的输入是NCHW，现在的patch是[H,W,C]，要把patch转成[C,H,W]形状
        patch_hr = np.reshape(patch_hr, (patchsize_HR * patchsize_HR, bands_HR))
        patch_hr = np.transpose(patch_hr)
        patch_hr = np.reshape(patch_hr, (bands_HR, patchsize_HR, patchsize_HR))
        TestPatch_HR[i, :, :, :] = patch_hr
        patchlabel_HR = test_label[ind1[i], ind2[i]]
        TestLabel_HR[i] = patchlabel_HR

        patch = data_HSI_pad[(ind3_hsi[i] - pad_width):(ind3_hsi[i] + pad_width),
                (ind4_hsi[i] - pad_width):(ind4_hsi[i] + pad_width), :]
        # conv的输入是NCHW，现在的patch是[H,W,C]，要把patch转成[C,H,W]形状
        patch = np.reshape(patch, (patchsize_HSI * patchsize_HSI, bands))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (bands, patchsize_HSI, patchsize_HSI))
        TestPatch_HSI[i, :, :, :] = patch
        patchlabel_HSI = test_label[closest_smaller(ind1[i]), closest_smaller(ind2[i])]
        TestLabel_HSI[i] = patchlabel_HSI




    print('Training size and testing size of HSI are:', TrainPatch_HSI.shape, 'and', TestPatch_HSI.shape)

    # #### 数据转换以及把数据搬到GPU #####
    TrainPatch_HSI = torch.from_numpy(TrainPatch_HSI)
    TrainLabel_HSI = torch.from_numpy(TrainLabel_HSI) - 1
    TrainLabel_HSI = TrainLabel_HSI.long()

    TestPatch_HSI = torch.from_numpy(TestPatch_HSI)
    TestLabel_HSI = torch.from_numpy(TestLabel_HSI) - 1
    TestLabel_HSI = TestLabel_HSI.long()

    TrainPatch_MSI = torch.from_numpy(TrainPatch_MSI)
    TrainLabel_MSI = torch.from_numpy(TrainLabel_MSI) - 1
    TrainLabel_MSI = TrainLabel_MSI.long()

    TestPatch_MSI = torch.from_numpy(TestPatch_MSI)
    TestLabel_MSI = torch.from_numpy(TestLabel_MSI) - 1
    TestLabel_MSI = TestLabel_MSI.long()

    TrainPatch_HR = torch.from_numpy(TrainPatch_HR)
    TrainLabel_HR = torch.from_numpy(TrainLabel_HR) - 1
    TrainLabel_HR = TrainLabel_HR.long()

    TestPatch_HR = torch.from_numpy(TestPatch_HR)
    TestLabel_HR = torch.from_numpy(TestLabel_HR) - 1
    TestLabel_HR = TestLabel_HR.long()

    return TrainPatch_HSI, TrainLabel_HSI, TestPatch_HSI, TestLabel_HSI,\
        TrainPatch_MSI, TrainLabel_MSI, TestPatch_MSI, TestLabel_MSI,\
        TrainPatch_HR, TrainLabel_HR, TestPatch_HR, TestLabel_HR


