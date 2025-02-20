import numpy as np
import scipy.io as sio
import random
import time
import torch.utils.data as dataf
import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage
import dataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from model import AMFS, SCE_fusion, SCE_classification, TI_CCS
from PIL import Image
import argparse
from Fusion_metrics import psnr, CC_function, SAM, ssim1
import os.path
from thop import profile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


Seed_List = [1]   # 随机种子点

# ###################### 超参预设 ######################
curr_train_ratio1 = 0.2
patchsize_HSI = 4
patchsize_MSI = 16
batchsize = 32
batchsize_train = 32
batchsize_test = 64
BestAcc = 0     # 最优精度

# 网络的超参
growthRate = 32
BlockBatch = 100
BottomPatch = 4
hidden_ch = 128
out = 128
# ###################### 加载数据集 ######################

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--train_data', choices=['paviau', 'houston', 'salinas', 'hanchuan', 'LongKou'], default='houston', help='data')
parser.add_argument('--train_model', choices=['fusion', 'classify'], default='fusion', help='testing')
parser.add_argument('--fusion_samples_type', choices=['ratio', 'same_num'], default='ratio', help='Training ratio')
parser.add_argument('--class_samples_type', choices=['ratio', 'same_num'], default='same_num', help='Training ratio') # 训练集按照 0-按比例取训练集 1-按每类个数取训练集
parser.add_argument('--number', choices=['1', '2', '3', '4', '5', '7', '10'], default='4', help='Training number') # 训练集按照 0-按比例取训练集 1-按每类个数取训练集
args = parser.parse_args(args=[])


curr_train_ratio2 = int(args.number)  # 每类训练集占这类总样本的比例，或每类训练样本的个数
# 选择数据集
datasets = args.train_data

# 加载数据0
[data_HSI, data_MSI, data_HR, gt, val_ratio, class_count, learning_rate,
 max_epoch, dataset_name, trainloss_result] = dataset.get_dataset(datasets)

# 源域和目标域数据信息
data_HSI = np.transpose(data_HSI, (1, 2, 0))
data_MSI = np.transpose(data_MSI, (1, 2, 0))
hsiheight, hsiwidth, hsibands = data_HSI.shape
msiheight, msiwidth, msibands = data_MSI.shape
hrheight, hriwidth, hrbands = data_HR.shape

# 数据标准化
[data_HSI] = dataset.data_standard2(data_HSI)
[data_MSI] = dataset.data_standard2(data_MSI)
[data_HR] = dataset.data_standard2(data_HR)


# 打印每类样本个数
print('#####源域样本个数#####')
dataset.print_data(gt, class_count)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def normalize(data):
    # 找到数组的最小值和最大值
    min_val = np.min(data)
    max_val = np.max(data)
    # 将数组归一化到 [0, 1] 范围内
    normalized_data = (data - min_val) / (max_val - min_val)

    return normalized_data

# mask = mask(0.5)
# ---------------------- 工具函数 ----------------------

def calculate_metrics(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    OA = np.diag(cm).sum() / cm.sum()
    AA = np.mean(np.diag(cm) / (cm.sum(axis=1) + 1e-6))

    total = np.sum(cm)
    pa = np.diag(cm).sum() / total
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (total ** 2)
    Kappa = (pa - pe) / (1 - pe + 1e-6)
    return OA, AA, Kappa

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ###################### 参数初始化 ######################
train_ratio1 = curr_train_ratio1  # 训练比例
train_ratio2 = curr_train_ratio2  # 训练比例

network = SCE_fusion(hsibands, msibands, patchsize_MSI, hidden_ch, out)
network2 = AMFS(hsibands, patchsize_MSI, out)
network3 = SCE_classification(hsibands, msibands, patchsize_MSI, hidden_ch, out)
network4 = TI_CCS(hsibands, patchsize_MSI, hidden_ch, out, class_count)

###################### 划分训练测试验证集 ######################
if args.train_model == 'fusion':
    for curr_seed in Seed_List:
        torch.cuda.empty_cache()
        random.seed(curr_seed)  # 当seed()没有参数时，每次生成的随机数是不一样的，而当seed()有参数时，每次生成的随机数是一样的

        epoch = 300
        best_loss = 9999
        criteon = nn.L1Loss()
        model = network.to(device)
        model2 = network2.to(device)
        # model.load_state_dict(torch.load('net_params.pkl', map_location=device), strict=False)
        # model2.load_state_dict(torch.load('net_params2.pkl', map_location=device), strict=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.0001)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=100, gamma=0.1)
        print('start')

        Run_loss, Test_loss = np.array([]), np.array([])

        # 对源域样本进行划分，得到训练、测试、验证集
        [train_label,
         test_label,
         val_label] = dataset.data_partition(args.fusion_samples_type, class_count, gt,
                                             train_ratio1, val_ratio, msiheight, msiwidth)

        # ###################### 搭建网络 ######################
        # 搭建普通的CNN网络分别对HSI进行特征提取

        # 生成CNN需要的数据
        # 在训练的时候测试全图，则定义一个满图都有label的矩阵，作为testlabel放进去
        full_image_label = train_label + 1

        [TrainPatch_HSI, TrainLabel_HSI,
         TestPatch_HSI, TestLabel_HSI,
         TrainPatch_MSI, TrainLabel_MSI,
         TestPatch_MSI, TestLabel_MSI,
         TrainPatch_HR, TrainLabel_HR,
         TestPatch_HR, TestLabel_HR] = dataset.gen_cnn_data2(data_HSI, data_MSI, data_HR, patchsize_HSI, patchsize_MSI,
                                                             patchsize_MSI, train_label, test_label)
        show = ToPILImage()
        datasetf = dataf.TensorDataset(TrainPatch_MSI, TrainPatch_HSI, TrainPatch_HR)
        train_loader = dataf.DataLoader(datasetf, batch_size=batchsize_train, shuffle=True)

        del TrainLabel_HSI, TestLabel_HSI, TrainLabel_MSI, TestLabel_MSI, TrainLabel_HR, TestLabel_HR

        combined_data = dataf.TensorDataset(TestPatch_MSI, TestPatch_HSI, TestPatch_HR)
        test_loader = dataf.DataLoader(combined_data, batch_size=batchsize_test, shuffle=True)

        for i in range(100):
            pn, ssim, cc = 0, 0, 0
            start1 = time.time()
            temp_loss = 0
            temp_loss1 = 0
            temp_loss2 = 0
            temp_loss3 = 0
            model.train()
            model2.train()
            for step, (hrMS, lrHS, ref) in enumerate((train_loader)):
                hrMS = hrMS.type(torch.float).to(device)
                lrHS = lrHS.type(torch.float).to(device)
                ref = ref.type(torch.float).to(device)

                cl_loss, x1, a1, x2, a2 = model(lrHS, hrMS)
                hr, fu_loss = model2(x1, a1, x2, a2, ref)

                running_loss = fu_loss + cl_loss
                temp_loss += fu_loss
                temp_loss1 += cl_loss

                pn += psnr(hr.cpu().detach().numpy(), ref.cpu().detach().numpy())
                # ssim += calculate_ssim(output[0].cpu().detach().numpy(), ref[0].cpu().detach().numpy())
                # if pn_test > pn:
                cc += CC_function(hr.cpu().detach().numpy(), ref.cpu().detach().numpy())
                optimizer.zero_grad()
                optimizer2.zero_grad()
                # 反向传播
                running_loss.backward()
                # 忘记更新参数，导致loss值一直在波动而不下降##
                optimizer.step()
                optimizer2.step()

            print('epoch', i + 1, 'train_loss', (temp_loss / (step + 1)).item(),
                  'train_cl_loss', (temp_loss1 / (step + 1)).item(),
                  'psnr', pn / (step + 1), 'CC', cc / (step + 1))
            scheduler.step()
            Run_loss = np.append(Run_loss, (temp_loss / (step + 1)).cpu().detach().numpy())

            if (i + 1) % 10 == 0:
                """
                测试阶段
                """
                # torch.cuda.empty_cache()
                model.eval()
                model2.eval()
                pn, ssim, cc = 0, 0, 0
                test_loss = 0
                Test_pn = 0
                Test_cc = 0
                test_loss1 = 0
                test_loss2 = 0
                test_loss3 = 0

                # with tqdm(total=63, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
                for test_step, (test_hrMS, test_lrhs, test_ref) in enumerate((test_loader)):
                    test_hrMS = test_hrMS.type(torch.float).to(device)
                    test_lrhs = test_lrhs.type(torch.float).to(device)
                    test_ref = test_ref.type(torch.float).to(device)
                    with torch.no_grad():
                        tcl_loss, tx1, ta1, tx2, ta2 = model(test_lrhs, test_hrMS)
                        thr, tfu_loss = model2(tx1, ta1, tx2, ta2, test_ref)
                        test_loss += tfu_loss
                        test_loss1 += tcl_loss

                        pn += psnr(thr.cpu().detach().numpy(), test_ref.cpu().detach().numpy())
                        # ssim += calculate_ssim(output[0].cpu().detach().numpy(), ref[0].cpu().detach().numpy())
                        # if pn_test > pn:
                        cc += CC_function(thr.cpu().detach().numpy(), test_ref.cpu().detach().numpy())
                        del test_lrhs, test_hrMS, test_ref

                print("epoch", i + 1, "test_loss", (test_loss / (test_step + 1)).item(),
                      "train_cl_loss", (test_loss1 / (test_step + 1)).item(),
                      'psnr', pn / (test_step + 1), 'CC', cc / (test_step + 1))
                if (test_loss / (test_step + 1)) <= best_loss:
                    best_loss = (test_loss / (test_step + 1))
                    best_epoch = i + 1
                    torch.save(model.state_dict(), 'net_params.pkl')
                    torch.save(model2.state_dict(), 'net_params2.pkl')
                print("epoch", best_epoch, "TrainLoss", best_loss.item(), "lr", get_lr(optimizer))
                Test_loss = np.append(Test_loss, (test_loss / (test_step + 1)).cpu().detach().numpy())
                Test_pn = np.append(pn, (test_loss / (test_step + 1)).cpu().detach().numpy())
                Test_cc = np.append(cc, (test_loss / (test_step + 1)).cpu().detach().numpy())

                end1 = time.time()
                print('训练一轮的时间为', end1 - start1, 's')
                torch.cuda.empty_cache()

        # .detach().numpy()先转成没有梯度的tensor，即float.tensor，再转成numpy()
        # print('Finish Train')

        plt.subplot()
        plt.plot(Run_loss, label='Training_loss')
        plt.plot(Test_loss, label='Test_loss')
        # plt.plot(Test_pn, label='Test_loss')
        # plt.plot(Test_cc, label='Test_loss')
        plt.title('Training and Test loss')
        plt.legend()
        plt.savefig('Training and Test loss.jpg')
        plt.show()

elif args.train_model == 'classify':

    copy_num = 1
    LR_classify2 = 0.00001
    LR_classify3 = 0.0000
    LR_classify4 = 0.0001
    # network.load_state_dict(torch.load('net_params.pkl', map_location=device))
    network2.load_state_dict(torch.load('net_params2.pkl', map_location=device), strict=False)
    network3.load_state_dict(torch.load('net_params.pkl', map_location=device), strict=False)
    # network4.load_state_dict(torch.load('net_params4.pkl', map_location=device), strict=False)

    optimizer_classify2 = torch.optim.Adam(network2.parameters(), lr=LR_classify2)
    optimizer_classify3 = torch.optim.Adam(network3.parameters(), lr=LR_classify3)
    optimizer_classify4 = torch.optim.Adam(network4.parameters(), lr=LR_classify4)

    scheduler_classify2 = torch.optim.lr_scheduler.StepLR(optimizer_classify2, step_size=100, gamma=0.1)
    scheduler_classify3 = torch.optim.lr_scheduler.StepLR(optimizer_classify3, step_size=100, gamma=0.1)
    scheduler_classify4 = torch.optim.lr_scheduler.StepLR(optimizer_classify4, step_size=100, gamma=0.1)

    # optimizer_classify3 = torch.optim.SGD(network3.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4, nesterov=True)
    # optimizer_classify4 = torch.optim.SGD(network4.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4, nesterov=True)
    # optimizer_classify2 = torch.optim.SGD(network2.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4, nesterov=True)

    for curr_seed in Seed_List:
        setup_seed(curr_seed)  # 当seed()没有参数时，每次生成的随机数是不一样的，而当seed()有参数时，每次生成的随机数是一样的

        # 对源域样本进行划分，得到训练、测试、验证集
        [train_label,
         test_label,
         val_label] = dataset.data_partition(args.class_samples_type, class_count, gt,
                                             train_ratio2, val_ratio, msiheight, msiwidth)

        # ###################### 搭建网络 ######################
        # 搭建普通的CNN网络分别对HSI进行特征提取

        # 生成CNN需要的数据
        # 在训练的时候测试全图，则定义一个满图都有label的矩阵，作为testlabel放进去
        full_image_label = train_label + 1

        [TrainPatch_HSI, TrainLabel_HSI,
         TestPatch_HSI, TestLabel_HSI,
         TrainPatch_MSI, TrainLabel_MSI,
         TestPatch_MSI, TestLabel_MSI,
         TrainPatch_HR, TrainLabel_HR,
         TestPatch_HR, TestLabel_HR] = dataset.gen_cnn_data2(data_HSI, data_MSI, data_HR, patchsize_HSI, patchsize_MSI,
                                                             patchsize_MSI,
                                                             train_label, test_label)

        datasetf = dataf.TensorDataset(TrainPatch_MSI, TrainPatch_HSI, TrainPatch_HR, TrainLabel_MSI)
        train_loader = dataf.DataLoader(datasetf, batch_size=batchsize_train, shuffle=True)

        datasetf_test = dataf.TensorDataset(TestPatch_MSI, TestPatch_HSI, TestPatch_HR, TestLabel_HR)
        test_loader = dataf.DataLoader(datasetf_test, batch_size=batchsize_test, shuffle=False)

        # 构建CNN网络

        network2.to(device)
        network3.to(device)
        network4.to(device)
        loss_fun = torch.nn.CrossEntropyLoss()

        # # # ###################### 训练 ######################
        best_loss = 99999
        network2.eval()
        network3.train()
        network4.train()

        for epoch in range(max_epoch + 1):
            start1 = time.time()
            run_loss = 0
            temp_loss1 = 0
            temp_loss2 = 0
            for step, (batch_MSI, batch_HSI, batch_HR, batch_label) in enumerate(train_loader):
                batch_HSI = batch_HSI.type(torch.float).to(device)
                batch_MSI = batch_MSI.type(torch.float).to(device)
                batch_HR = batch_HR.type(torch.float).to(device)
                batch_label = batch_label.to(device)
                cl_loss, x1, a1, x2, a2, x6, x9 = network3(batch_HSI, batch_MSI)
                hr, fu_loss = network2(x1, a1, x2, a2, batch_HR)
                output, L = network4(hr, x6, x9)

                loss_fun = torch.nn.CrossEntropyLoss()
                loss = loss_fun(output, batch_label)
                # print('loss',loss)
                run_loss = loss + (cl_loss + L) * 0.1 + fu_loss * 10
                temp_loss1 += loss
                temp_loss2 += fu_loss

                optimizer_classify3.zero_grad()
                optimizer_classify2.zero_grad()
                optimizer_classify4.zero_grad()
                run_loss.backward()
                optimizer_classify3.step()
                optimizer_classify2.step()
                optimizer_classify4.step()
                if (epoch + 1) % 10 == 0:
                    if step % 1 == 0:
                        with torch.no_grad():
                            start = time.time()
                            network4.eval()
                            network3.eval()
                            network2.eval()
                            pn = 0
                            cc = 0
                            j = 0
                            pred_y = np.empty((len(TestLabel_HR)), dtype='float32')
                            pred_y = torch.from_numpy(pred_y).to(device)
                            for step_2, (batch_MSI, batch_HSI, batch_HR, batch_label) in enumerate(test_loader):
                                batch_HSI = batch_HSI.type(torch.float).to(device)
                                batch_MSI = batch_MSI.type(torch.float).to(device)
                                batch_HR = batch_HR.type(torch.float).to(device)
                                batch_label = batch_label.to(device)

                                cl_loss, x1, a1, x2, a2, x6, x9 = network3(batch_HSI, batch_MSI)
                                hr, fu_loss = network2(x1, a1, x2, a2, batch_HR)
                                temp, L = network4(hr, x6, x9)
                                TEMPloss = loss_fun(temp, batch_label)
                                temp = torch.max(temp, 1)[1].squeeze()
                                if step_2 == len(test_loader)-1:
                                    pred_y[j * batchsize_test:len(TestLabel_HR)] = temp.cpu()
                                else:
                                    pred_y[j * batchsize_test:(j + 1) * batchsize_test] = temp
                                j += 1

                            accuracy = torch.sum(pred_y.cpu() == TestLabel_HR.cpu()).type(
                                torch.FloatTensor) / TestLabel_HR.cpu().size(0)
                            print('Epoch: ', epoch, '| test accuracy: %.4f' % accuracy)
                            # print("clloss", (temp_loss1 / (step + 1)).item(),
                            #       "TEMPloss", (temp_loss2 / (step + 1)).item(),
                            #       'psnr', pn / number, 'CC', cc / number)
                            # end = time.time()
                            # print('每轮测试需要的时间为：', end-start, 's')
                            # 保存最优的网络结果
                            if accuracy > BestAcc:
                                torch.save(network4.state_dict(), 'net_params4.pkl')
                                torch.save(network3.state_dict(), 'net_params3.pkl')
                                torch.save(network2.state_dict(), 'net_params5.pkl')
                                BestAcc = accuracy

                            network4.train()
                            network3.train()
                            network2.train()
            end1 = time.time()
            # print('训练一轮的时间为', end1 - start1, 's')
        torch.cuda.synchronize()

        # ###################### 测试 ######################

        torch.cuda.empty_cache()
        # 加载网络模型
        with torch.no_grad():
            network2.load_state_dict(torch.load('net_params5.pkl', map_location=device))
            network3.load_state_dict(torch.load('net_params3.pkl', map_location=device))
            network4.load_state_dict(torch.load('net_params4.pkl', map_location=device))
            network2.eval().to(device)
            network3.eval().to(device)
            network4.eval().to(device)
            torch.cuda.synchronize()

            pn, ssim, cc, sam = 0, 0, 0, 0

            f_datasetf = dataf.TensorDataset(TestPatch_MSI, TestPatch_HSI, TestPatch_HR)
            test_loader = dataf.DataLoader(f_datasetf, batch_size=batchsize_train, shuffle=True)

            # with tqdm(total=63, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
            for test_step, (test_hrMS, test_lrhs, test_ref) in enumerate(test_loader):
                test_hrMS = test_hrMS.type(torch.float).to(device)
                test_lrhs = test_lrhs.type(torch.float).to(device)
                test_ref = test_ref.type(torch.float).to(device)
                with torch.no_grad():
                    cl_loss, x1, a1, x2, a2, x6, x9 = network3(test_lrhs, test_hrMS)
                    hr, fu_loss = network2(x1, a1, x2, a2, test_ref)

                    pn += psnr(hr.cpu().detach().numpy(), test_ref.cpu().detach().numpy())
                    sam += SAM(hr.cpu().detach().numpy(), test_ref.cpu().detach().numpy())
                    ssim += ssim1(hr.cpu().detach().numpy(), test_ref.cpu().detach().numpy())
                    cc += CC_function(hr.cpu().detach().numpy(), test_ref.cpu().detach().numpy())

            print('psnr', pn / (test_step + 1), 'CC', cc / (test_step + 1),
                  "ssim", (ssim / (test_step + 1)).item(), "sam", (sam / (test_step + 1)).item())

            pred_y_entropy = np.empty((len(TestLabel_HR), class_count), dtype='float32')
            pred_y_entropy = torch.from_numpy(pred_y_entropy).to(device)
            pred_y = np.empty((len(TestLabel_HR)), dtype='float32')
            pred_y = torch.from_numpy(pred_y).to(device)
            number = len(TestLabel_HR) // BlockBatch  # //是除以某个数取整，进行分批操作
            j = 0
            for j in range(number + 1):
                temp1 = TestPatch_HSI[j * BlockBatch:(j + 1) * BlockBatch, :, :, :].type(torch.float).to(device)
                temp2 = TestPatch_MSI[j * BlockBatch:(j + 1) * BlockBatch, :, :, :].type(torch.float).to(device)
                temp3 = TestPatch_HR[j * BlockBatch:(j + 1) * BlockBatch, :, :, :].type(torch.float).to(device)
                cl_loss, x1, a1, x2, a2, x6, x9 = network3(temp1, temp2)
                hr, fu_loss = network2(x1, a1, x2, a2, temp3)
                temp, L = network4(hr, x6, x9)

                pred_y_entropy[j * BlockBatch:(j + 1) * BlockBatch] = temp

                temp = torch.max(temp, 1)[1].squeeze()
                pred_y[j * BlockBatch:(j + 1) * BlockBatch] = temp

            if (j + 1) * BlockBatch < len(TestLabel_HR):
                temp1 = TestPatch_HSI[(j + 1) * BlockBatch:len(TestLabel_HSI), :, :, :].type(torch.float).to(device)
                temp2 = TestPatch_MSI[(j + 1) * BlockBatch:len(TestLabel_MSI), :, :, :].type(torch.float).to(device)
                temp3 = TestPatch_HR[(j + 1) * BlockBatch:len(TestLabel_HR), :, :, :].type(torch.float).to(device)
                cl_loss, x1, a1, x2, a2, x6, x9 = network3(temp1, temp2)
                hr, fu_loss = network2(x1, a1, x2, a2, temp3)
                temp, L = network4(hr, x6, x9)

                pred_y_entropy[(j + 1) * BlockBatch:len(TestLabel_HR)] = temp.cpu()

                temp = torch.max(temp, 1)[1].squeeze()
                pred_y[(j + 1) * BlockBatch:len(TestLabel_HR)] = temp.cpu()

            # 将输入数据转换到设备上
            temp1 = TestPatch_HSI[0].type(torch.float).to(device).unsqueeze(0)
            temp2 = TestPatch_MSI[0].type(torch.float).to(device).unsqueeze(0)
            temp3 = TestPatch_HR[0].type(torch.float).to(device).unsqueeze(0)

            # 测量运行时间
            start_time = time.time()

            # 计算网络的输出
            cl_loss, x1, a1, x2, a2, x6, x9 = network3(temp1, temp2)
            hr, fu_loss = network2(x1, a1, x2, a2, temp3)
            temp, L = network4(hr, x6, x9)
            end_time = time.time()

            # 计算总参数量
            total_params_1 = sum(p.numel() for p in network3.parameters())
            total_params_2 = sum(p.numel() for p in network2.parameters())
            total_params_3 = sum(p.numel() for p in network4.parameters())

            total_params = total_params_1 + total_params_2 + total_params_3
            # 计算总FLOPs
            flops_network3, _ = profile(network3, inputs=(temp1, temp2))
            flops_network2, _ = profile(network2, inputs=(x1, a1, x2, a2, temp3))
            flops_network4, _ = profile(network4, inputs=(hr, x6, x9))

            total_flops = flops_network3 + flops_network2 + flops_network4

            # 计算总运行时间
            elapsed_time = end_time - start_time

            # 输出总结果
            print("\nTotal Parameter3 Count:", total_params_1)
            print("\nTotal Parameter2 Count:", total_params_2)
            print("\nTotal Parameter4 Count:", total_params_3)
            print("\nTotal Parameter Count:", total_params)
            print("Total FLOPs:", total_flops)
            print(f"Total execution time: {elapsed_time:.4f} seconds")

        # 将结果按照testlabel的index转为全图尺寸
        temp = np.reshape(test_label, (msiheight * msiwidth))
        index = np.where(temp)[0]

        temp_zero = np.zeros(len(temp))
        pred_y = pred_y + 1
        temp_zero[index] = pred_y.cpu().numpy()
        result = np.reshape(temp_zero, [msiheight, msiwidth])

        temp_zero = np.zeros((len(temp), class_count))
        temp_zero[index] = pred_y_entropy.detach().cpu().numpy()
        result_entropy = np.reshape(temp_zero, [msiheight, msiwidth, class_count])

        sio.savemat("output.mat", {'output': result})
        sio.savemat("result_entropy.mat", {'result_entropy': result_entropy})

        # 精度计算
        pred_y = (pred_y - 1)
        OA_temp = torch.sum(pred_y.cpu() == TestLabel_HR.cpu()).type(torch.FloatTensor) / TestLabel_HR.cpu().size(0)
        OA = torch.sum(pred_y.cpu() == TestLabel_HR.cpu()).type(torch.FloatTensor) / TestLabel_HR.cpu().size(0)

        Classes = class_count
        EachAcc = np.empty(Classes)

        for i in range(Classes):
            cla = i
            right = 0
            sum = 0

            for j in range(len(TestLabel_HR)):
                if TestLabel_HR[j] == cla:
                    sum += 1
                if TestLabel_HR[j] == cla and pred_y[j] == cla:
                    right += 1

            EachAcc[i] = right.__float__() / sum.__float__()

        print(OA)
        print(EachAcc)

        torch.cuda.synchronize()

        cm = confusion_matrix(TestLabel_HR.cpu(), pred_y.cpu())

        # 计算Average Accuracy（AA）
        AA = np.mean(np.diag(cm) / np.sum(cm, axis=1))

        # 计算Kappa系数
        total = np.sum(cm)
        pa = np.trace(cm) / float(total)
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total ** 2)
        Kappa = (pa - pe) / (1 - pe)

        Final_OA = OA

        print('The OA is: ', Final_OA)
        print('The AA is: ', AA)
        print('The Kappa is: ', Kappa)

        plt.subplot(1, 1, 1)
        plt.imshow(result)
        # colors.ListedColormap(color_matrix))
        plt.xticks([])
        plt.yticks([])
        plt.show()

        im = Image.fromarray((1 - (result - 1)) * 255)