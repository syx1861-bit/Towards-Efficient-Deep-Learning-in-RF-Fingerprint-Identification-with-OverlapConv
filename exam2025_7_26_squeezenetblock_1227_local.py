import importlib.util
import os
import csv
import copy
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from dataset_preparation import awgn, LoadDataset, ChannelIndSpectrogram
from sklearn.model_selection import KFold
from utils.seed import set_seed
from sklearn.metrics import accuracy_score, confusion_matrix
from models.generate_model_structure_description import (
    generate_model_structure_description,
    build_model_from_description,
)
# from models.generate_model_structure_description import SegmentedSoftGroupConv
import matplotlib.pyplot as plt
import seaborn as sns

# from model_7_8.resnet8 import ResNet8_hard, ResNet8_soft, SegmentedSoftGroupConv
from model_7_8_2.mobilenet import mobilenet_v1, mobilenet_v1_soft
from model_7_8_2.squeezenet import squeezenet, squeezenet_soft
from model_7_8_2.ghostnet import ghost_net, ghost_net_soft
from model_7_8_2.simplecnn2 import MobileNetV2CNN_spec, MobileNetV2CNN_spec_soft, GhostCNN_spec, GhostCNN_spec_soft
# from model_7_8_2.simplecnn import SegmentedSoftGroupConv
# from model_7_8_2.LMSC import SegmentedSoftGroupConv
from model_7_8_2.simplecnn2 import SqueezeCNN_spec, SqueezeCNN_spec_soft
from model_7_8_2.simplecnn2 import ShuffleNetV2CNN_spec, ShuffleNetV2CNN_spec_soft, LMSC_spec, LMSC_spec_hard
from model_7_8_2.LMSC import my_resnet_short_soft, my_resnet, my_resnet_short
from model_7_8_2.fasternet_shorter import FasterNet_shorter
from model_7_8_2.lsnet.lsnet import LSNet
#from model_7_8_2.fasternet_shorter_soft import FasterNet_shorter_soft, SegmentedSoftGroupConv
from model_7_8_2.fasternet import FasterNet
from model_7_8_2.fasternet_shorter_hard import FasterNet_shorter_hard
from model_7_8_2.mobilenetv4 import mobilenetv4_conv_small
from model_7_8_2.mobilenetv4_short import mobilenetv4_conv_small_short
from model_7_8_2.mobilenetv4_short_soft import mobilenetv4_conv_small_short_soft,SegmentedSoftGroupConv
# from model_7_8_2.simplecnn_raw import SimpleCNN, SimpleCNN_group2, SimpleCNN_group4, SimpleCNN_group8, \
#     SimpleCNN_group16, SimpleCNN_soft, SimpleCNN_hard, SegmentedSoftGroupConv, SimpleCNN_group32, SimpleCNN_group64
from model_7_8_2.deep_learning_models import ClassificationNet,ClassificationNet_short,ClassificationNet_short2,ClassificationNet_short3
# from model_7_8_2.simplecnn_raw import SimpleCNN,SimpleCNN_group2,SegmentedSoftGroupConv
# 全局配置
MAX_EPOCHS = 350
MAX_EPOCHS_HARD = 100  # Hard模型的最大训练轮次
PKT_RANGE = range(0, 800)
PKT_RANGE_test = range(0, 400)
DEV_RANGE = range(0, 8)
TRAIN_FILE = 'Closed_set_RFFI/Train/dataset_training_aug.h5'
TEST_FILE = 'Closed_set_RFFI/Test/dataset_seen_devices.h5'
TRAIN_NOISE_RANGE = range(20, 80)  # 训练噪声范围
# TRAIN_NOISE_RANGE = range(0, 10)  # 训练噪声范围

# 统一优化参数
INIT_LR = 0.001  # 初始学习率
WEIGHT_DECAY = 0.01  # 权重衰减
T_MAX = 50  # Cosine调度周期
ETA_MIN = 1e-8  # 最小学习率


# def update_alpha(model, epoch, explore_epochs=50, step_size=50, alpha_min=2.0, alpha_max=6.0, step=1):
#     """动态调整alpha值控制软分组行为"""
#     if epoch < explore_epochs:
#         alpha = alpha_min + 3
#     elif explore_epochs <= epoch < (explore_epochs + 50):
#         alpha = alpha_min + 1
#     elif (explore_epochs + 50) <= epoch < (explore_epochs + 100):
#         alpha = alpha_min + 2
#     elif (explore_epochs + 100) <= epoch < (explore_epochs + 150):
#         alpha = alpha_min + 2.5
#     elif (explore_epochs + 150) <= epoch < (explore_epochs + 200):
#         alpha = alpha_min + 3
#     elif (explore_epochs + 200) <= epoch < (explore_epochs + 250):
#         alpha = alpha_min + 3.5
#     elif (explore_epochs + 250) <= epoch <= (explore_epochs + 300):
#         alpha = alpha_min + 4
#
#     # 设置模型中所有SegmentedSoftGroupConv的alpha值
#     for module in model.modules():
#         # for module in model._modules.values():
#         if isinstance(module, SegmentedSoftGroupConv):
#             module.alpha = alpha
#             flag = True
#
#     if flag == False:
#         print("未找到alpha")


def update_alpha(model, epoch, explore_epochs=50, alpha_min=2.0):
    """动态调整alpha值控制软分组行为"""
    # 1. 计算 alpha
    if epoch < explore_epochs:
        alpha = alpha_min
    elif explore_epochs <= epoch < (explore_epochs + 50):
        alpha = alpha_min + 1
    elif (explore_epochs + 50) <= epoch < (explore_epochs + 100):
        alpha = alpha_min + 2
    elif (explore_epochs + 100) <= epoch < (explore_epochs + 150):
        alpha = alpha_min + 2.5
    elif (explore_epochs + 150) <= epoch < (explore_epochs + 200):
        alpha = alpha_min + 3
    elif (explore_epochs + 200) <= epoch < (explore_epochs + 250):
        alpha = alpha_min + 3.5
    elif (explore_epochs + 250) <= epoch <= (explore_epochs + 300):
        alpha = alpha_min + 4
    else:
        alpha = alpha_min + 4

    # 2. 遍历所有子模块，递归查找 SegmentedSoftGroupConv
    flag = False
    for module in model.modules():  # ✅ modules() 会递归
        if isinstance(module, SegmentedSoftGroupConv):
            module.alpha = alpha
            flag = True

    # 3. 检查是否找到
    if not flag:
        print("未找到任何 SegmentedSoftGroupConv 模块")


def add_noise_to_overlap_rate(model, noise_scale=0.01, min_val=0.1, max_val=0.9):
    """为重叠率参数添加噪声促进探索"""
    for module in model.modules():
        if hasattr(module, 'overlap_rate'):  # 检查是否有重叠率参数
            with torch.no_grad():
                # 添加高斯噪声并限制在合理范围
                noise = torch.randn_like(module.overlap_rate) * noise_scale
                module.overlap_rate += noise
                module.overlap_rate.clamp_(min_val, max_val)


def save_results_incremental(csv_path, rows):
    """增量保存结果到CSV文件"""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Structure_ID", "Seed", "Model", "OverlapRate",
                "Train_Loss", "Train_Acc",
                "Val_Loss", "Val_Acc",
                "External_Test_Acc"  # 只保留外部测试集准确率
            ])
        writer.writerows(rows)


def load_model_hard(model_id, model_dir="generated_models", overlaprate=0.2):
    fname = f"model_{model_id}_hard_soft.py"
    module_path = os.path.join(model_dir, fname)
    module_name = f"model_module_{model_id}"

    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Model file not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    hard_class = getattr(module, f"Model_Hard_{model_id}")
    hard_model = hard_class(overlaprate)
    return hard_model


def load_model_soft(model_id, model_dir="generated_models"):
    fname = f"model_{model_id}_hard_soft.py"
    module_path = os.path.join(model_dir, fname)
    module_name = f"model_module_{model_id}"

    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Model file not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    soft_class = getattr(module, f"Model_Soft_{model_id}")
    soft_model = soft_class()
    return soft_model


def load_and_preprocess_data(file_path, dev_range, pkt_range, force_reload=False, add_noise=True):
    """加载并预处理数据，支持噪声选项"""
    # 在文件名中加入噪声标记
    noise_flag = "noisy" if add_noise else "clean"
    cache_path = f"preprocessed_data_{pkt_range.start}_{pkt_range.stop}_{noise_flag}.pt"

    # if not force_reload and os.path.exists(cache_path):
    #     print(f"Loading preprocessed data from cache: {cache_path}")
    #     data, labels = torch.load(cache_path)
    #     return data, labels

    print("Preprocessing data...")
    loader = LoadDataset()
    extractor = ChannelIndSpectrogram()

    # 加载原始IQ样本和标签
    data, labels = loader.load_iq_samples(file_path, dev_range, pkt_range)

    # 添加AWGN噪声（可选）
    if add_noise:
        data = awgn(data, TRAIN_NOISE_RANGE)

    # 标签偏移处理
    labels = labels - dev_range[0]

    # 计算频谱图
    data = extractor.channel_ind_spectrogram(data)

    # 转成tensor并调整维度
    data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)
    labels = torch.tensor(labels, dtype=torch.long)

    # 处理标签维度
    if labels.ndim > 1:
        labels = labels.squeeze(1) if labels.shape[1] == 1 else torch.argmax(labels, dim=1)
        labels = labels.long()
    # 保存缓存
    torch.save((data, labels), cache_path)
    print(f"Saved preprocessed data to cache: {cache_path}")

    return data, labels


def load_and_preprocess_data_test(file_path, test_snr, dev_range, pkt_range, force_reload=False, add_noise=True):
    """加载并预处理数据，支持噪声选项"""
    # 在文件名中加入噪声标记
    noise_flag = "noisy" if add_noise else "clean"
    cache_path = f"preprocessed_data_{pkt_range.start}_{pkt_range.stop}_{noise_flag}.pt"

    # if not force_reload and os.path.exists(cache_path):
    #     print(f"Loading preprocessed data from cache: {cache_path}")
    #     data, labels = torch.load(cache_path)
    #     return data, labels

    print("Preprocessing data...")
    loader = LoadDataset()
    extractor = ChannelIndSpectrogram()

    # 加载原始IQ样本和标签
    data, labels = loader.load_iq_samples(file_path, dev_range, pkt_range)

    # 添加AWGN噪声（可选）
    if add_noise:
        data = awgn(data, test_snr)

    # 标签偏移处理
    labels = labels - dev_range[0]

    # 计算频谱图
    data = extractor.channel_ind_spectrogram(data)

    # 转成tensor并调整维度
    data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)
    labels = torch.tensor(labels, dtype=torch.long)

    # 处理标签维度
    if labels.ndim > 1:
        labels = labels.squeeze(1) if labels.shape[1] == 1 else torch.argmax(labels, dim=1)
        labels = labels.long()
    # 保存缓存
    torch.save((data, labels), cache_path)
    print(f"Saved preprocessed data to cache: {cache_path}")

    return data, labels


def train_hard_model(model, train_loader, val_loader, criterion, optimizer, model_type="Hard", oc=None):
    """Hard模型的训练函数 - 统一优化方案"""
    best_val_loss = float('inf')
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_model_state = None

    # 创建Cosine调度器
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=T_MAX, eta_min=ETA_MIN
    # )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100
    )

    for epoch in range(1, MAX_EPOCHS_HARD + 1):
        # 训练阶段
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # 验证阶段
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)

                _, predicted = torch.max(outputs, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 打印进度
        prefix = f"{model_type} (Oc={oc})" if oc else model_type
        print(f"Epoch {epoch:03d}/{MAX_EPOCHS_HARD}: {prefix} | LR={current_lr:.2e} | "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            print(
                f"New best model saved at epoch {epoch} with val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}")

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}")
    return model, history


def train_raw_model(seed, model, train_loader, val_loader, criterion, optimizer, model_type="raw", ckpt_dir=None):
    """Hard模型的训练函数 - 统一优化方案"""
    best_val_loss = float('inf')
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_model_state = None

    # 创建Cosine调度器
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=T_MAX, eta_min=ETA_MIN
    # )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100
    )

    for epoch in range(1, MAX_EPOCHS_HARD + 1):
        # 训练阶段
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # 验证阶段
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)

                _, predicted = torch.max(outputs, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 打印进度

        print(f"Epoch {epoch:03d} | LR={current_lr:.2e} | "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        # 保存最佳模型
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_val_acc = val_acc
        #     best_model_state = copy.deepcopy(model.state_dict())
        #     torch.save(best_model_state,
        #                os.path.join(ckpt_dir, f"{model_name}_best_raw.pth"))
        #     print(
        #         f"New best model saved at epoch {epoch} with val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}")
        # 保存最佳模型（基于 val_acc）
        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state,
                       os.path.join(ckpt_dir, f"{model_name}_best_raw_seed{seed}.pth"))
            print(
                f"New best model saved at epoch {epoch} with val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}")
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}")
    return model, history


def train_soft_model(seed_dataset,model, train_loader, val_loader, criterion, optimizer, device, ckpt_dir):
    """Soft模型的训练函数 - 统一优化方案"""
    best_val_loss_last50 = float('inf')
    best_val_acc_last50 = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_model_state_last50 = None

    # 创建Cosine调度器
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=T_MAX, eta_min=ETA_MIN
    # )
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=350
    # )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=0
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        # 动态调整alpha
        update_alpha(model, epoch)

        # 训练阶段
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()

        # 计算训练指标
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Step 2: 输出 SegmentedSoftGroupConv 中的调度信息
        # === 记录 SegmentedSoftGroupConv 参数 ===
        print(f"[Epoch {epoch + 1}] SegmentedSoftGroupConv stats:")
        for name, module in model.named_modules():
            if isinstance(module, SegmentedSoftGroupConv):
                overlap_param = module.overlap_rate.item()
                overlap = 1.7 * torch.sigmoid(module.overlap_rate).item()
                grad = module.overlap_rate.grad.item() if module.overlap_rate.grad is not None else 0.0
                print(
                    f"[Epoch {epoch}] {name:<30} | param = {overlap_param:+.4f} | overlap = {overlap:.4f} | grad = {grad:.4e} | alpha = {module.alpha:.4f}")

        # for name, module in model.named_modules():
        #     if isinstance(module, SegmentedSoftGroupConv):
        #         print(f"  {name:<30} | overlap = {2.5 * torch.sigmoid(module.overlap_rate).item()} | "
        #               f"overlap_grad = {2.5 * torch.sigmoid(module.overlap_rate.grad).item() if module.overlap_rate.grad is not None else 0.0:.4e} | "
        #               f"alpha = {module.alpha:.4f}")
        #         print(f"[Epoch {epoch}] overlap_param = {module.overlap_rate.grad.item():.4f}")
        # overlap = 2.0 * torch.sigmoid(self.overlap_param)

        # 验证阶段
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']  # 获取基础参数的学习率

        # 打印进度
        print(f"Epoch {epoch:03d}/{MAX_EPOCHS}: Soft | LR={current_lr:.2e} | "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        # 在最后50轮中选择最佳模型
        if epoch > (MAX_EPOCHS - 50):
            if val_loss < best_val_loss_last50:
                best_val_loss_last50 = val_loss
                best_val_acc_last50 = val_acc
                best_model_state_last50 = copy.deepcopy(model.state_dict())
                torch.save(best_model_state_last50,
                           os.path.join(ckpt_dir, f"{model_name}_best_last50_seed{seed_dataset}.pth"))
                print(
                    f"New best model in last 50 epochs saved at epoch {epoch} with val_loss={best_val_loss_last50:.4f}, val_acc={best_val_acc_last50:.4f}")

    # 加载最后50轮中的最佳模型
    if best_model_state_last50 is not None:
        model.load_state_dict(best_model_state_last50)
        print(
            f"Loaded best model from last 50 epochs with val_loss={best_val_loss_last50:.4f}, val_acc={best_val_acc_last50:.4f}")
    return model, history


def evaluate_model(model, test_loader, criterion):
    """评估模型在测试集上的性能"""
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            test_loss += loss.item() * x.size(0)

            _, predicted = torch.max(outputs, 1)
            test_total += y.size(0)
            test_correct += (predicted == y).sum().item()

            # 收集预测和标签用于后续分析
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_acc = test_correct / test_total

    # 计算混淆矩阵
    conf_mat = confusion_matrix(all_labels, all_preds)
    return test_loss, test_acc, conf_mat


def run_test(model, file_path, dev_range, pkt_range, test_snr, model_name="model"):
    """使用独立测试集进行评估（不加噪声）"""
    # set_seed(42)

    # 加载并预处理测试数据（不加噪声）
    test_data, test_labels = load_and_preprocess_data_test(
        file_path, test_snr, dev_range, pkt_range, add_noise=True
    )

    # 创建测试数据集（使用固定随机种子确保顺序一致）
    test_dataset = TensorDataset(test_data, test_labels)
    # test_loader = DataLoader(test_dataset, batch_size=32,
    #                         generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    # 评估模型
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, conf_mat = evaluate_model(model, test_loader, criterion)

    # 绘制混淆矩阵（可选）
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
    #             xticklabels=range(len(DEV_RANGE)),
    #             yticklabels=range(len(DEV_RANGE)))
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.title(f"Confusion Matrix - {model_name}")
    # os.makedirs("results2/confusion_matrices", exist_ok=True)
    # plt.savefig(f"results2/confusion_matrices/confusion_matrix_{model_name}.pdf", bbox_inches="tight")
    # plt.savefig(f"results2/confusion_matrices/confusion_matrix_{model_name}.png")
    # plt.close()

    print(f"External Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    return test_acc


if __name__ == "__main__":
    model_list_summary = [['squeezenet']]
    for model_sub in model_list_summary:
        global_seed = 2025
        set_seed(global_seed)  # 这里先全局设置一次种子，确保数据划分和其他全局随机稳定
        # 配置
        # seed_list = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
        # seed_list = [52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
        seed_list = [42]
        seed_list_dataset = [47, 48, 49, 50, 51]
        # overlap_c_list = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0]
        # num_structures = [0, 1, 2, 3, 4]
        # num_structures = [0]\
        # test_snr_list = [-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        test_snr_list = [(30, 30)]
        # model_list = ['mobilenet', 'squeezenet','ghostnet','shufflenet','LMSC'
        # ,'LMSC_short','fasternet','fasternet_raw','fasternet_soft',‘fasternet_hard’,
        # 'LMSC_short_soft','lsnet','mobilenetv4_raw','mobilenetv4_short','mobilenetv4_short_soft',\
        # 'simplecnn','simplecnn_2g','simplecnn_4g','simplecnn_8g','simplecnn_16g',
        # 'simplecnn_32g','simplecnn_64g','simplecnn_soft','fasternet_']
        model_list = model_sub
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # 确保结果目录存在
        os.makedirs("compare_results_12_27_squeezenet_block_hard_lora_local", exist_ok=True)
        # os.makedirs("results6_25", exist_ok=True)

        # 加载并预处理训练数据（添加噪声）
        train_data, train_labels = load_and_preprocess_data(
            TRAIN_FILE, DEV_RANGE, PKT_RANGE, add_noise=True
        )
        # 假设 train_labels 是一个 NumPy 数组
        unique_labels, counts = np.unique(train_labels, return_counts=True)

        # 输出结果
        for label, count in zip(unique_labels, counts):
            print(f"类别 {label} 的数量是 {count}")


        summary_acc = []
        for seed_dataset in seed_list_dataset:
            set_seed(seed_dataset)
            # 创建数据集并划分 (90%训练, 10%验证)
            dataset = TensorDataset(train_data, train_labels)
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            # 固定数据划分
            generator = torch.Generator().manual_seed(seed_dataset)
            train_data_2, val_data_2 = random_split(
                dataset, [train_size, val_size], generator=generator
            )

            # 创建数据加载器（训练集使用随机打乱）
            train_loader = DataLoader(train_data_2, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_data_2, batch_size=32)

            # 主实验循环

            # # 生成模型描述并构建模型
            # desc = generate_model_structure_description(
            #     seed=2025 + model_id, model_id=model_id, use_residual=True,
            #     num_classes=len(DEV_RANGE), save_path=f'./model_desc_{model_id}.json'
            # )
            # # 关键修改：shared_overlap_param=False 确保每个Hard模型独立参数
            # build_model_from_description(
            #     desc_path=f'./model_desc_{model_id}.json',
            #     Oc_list=overlap_c_list,
            #     shared_overlap_param=False
            # )
            for model_name in model_list:
                # 为Soft和Hard模型分别创建CSV文件
                soft_csv_path = f"compare_results_12_27_squeezenet_block_hard_lora_local/{model_name}_soft.csv"
                # hard_csv_path = f"results7_8/structure_{model_name}_hard.csv"
                raw_csv_path = f"compare_results_12_27_squeezenet_block_hard_lora_local/{model_name}_raw.csv"
                os.makedirs(os.path.dirname(soft_csv_path), exist_ok=True)

                # 初始化CSV文件
                for csv_path in [soft_csv_path, raw_csv_path]:
                    if not os.path.exists(csv_path):
                        with open(csv_path, "w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                "model_name", "Seed", "groups", "OverlapRate",
                                "Train_Loss", "Train_Acc", "Val_Loss", "Val_Acc", "ext_test_acc"
                            ])
                # for csv_path in [soft_csv_path]:
                #     if not os.path.exists(csv_path):
                #         with open(csv_path, "w", newline="") as f:
                #             writer = csv.writer(f)
                #             writer.writerow([
                #                 "Structure_ID", "Seed", "Model", "OverlapRate",
                #                 "Train_Loss", "Train_Acc", "Val_Loss", "Val_Acc",
                #                 "External_Test_Acc"
                #             ])

                for seed in seed_list:
                    print(f"\n{'=' * 20} Seed {seed} {'=' * 20}")
                    set_seed(seed)
                    torch.cuda.empty_cache()  # 清理GPU缓存防止内存泄漏

                    # # ========== Soft模型训练和评估 ==========
                    # print("\n[Training Soft Model]")
                    # # soft_model = load_model_soft(model_id).to(device)
                    # if model_name == 'mobilenet':
                    #
                    #     soft_model = MobileNetV2CNN_spec_soft().to(device)
                    #     # print(soft_model)
                    # elif model_name == 'squeezenet':
                    #     soft_model = SqueezeCNN_spec_soft().to(device)
                    #     # soft_model = squeezenet_soft().to(device)
                    #     print(soft_model)
                    # elif model_name == 'ghostnet':
                    #
                    #     soft_model = GhostCNN_spec_soft().to(device)
                    # elif model_name == 'shufflenet':
                    #     soft_model = ShuffleNetV2CNN_spec_soft().to(device)
                    # elif model_name == 'LMSC_short_soft':
                    #     soft_model = my_resnet_short_soft().to(device)
                    #     print(soft_model)
                    # elif model_name=='fasternet_soft':
                    #     soft_model= FasterNet_shorter_soft().to(device)
                    #     print(soft_model)
                    # elif model_name=='mobilenetv4_short_soft':
                    #     soft_model=mobilenetv4_conv_small_short_soft().to(device)
                    # elif model_name=='simplecnn_soft':
                    #     soft_model=SimpleCNN_soft().to(device)
                    #
                    # # soft_model = ResNet8_soft().to(device)
                    # criterion = nn.CrossEntropyLoss()
                    #
                    # # 创建分组优化器 (AdamW统一)
                    # base_params = []
                    # overlap_params = []
                    # for name, param in soft_model.named_parameters():
                    #     if 'overlaprate' in name:
                    #         overlap_params.append(param)
                    #     else:
                    #         base_params.append(param)
                    #
                    # optimizer_soft = optim.Adam([
                    #     {'params': base_params, 'lr': INIT_LR, 'weight_decay': 0.0},
                    #     {'params': overlap_params, 'lr': 0.007, 'weight_decay': 0.0}
                    # ])
                    #
                    # # 使用新的训练函数训练原始模型
                    # soft_model, soft_history = train_soft_model(
                    #     seed_dataset,soft_model, train_loader, val_loader, criterion, optimizer_soft, device, 'compare_results_12_27_squeezenet_block_hard_lora_local'
                    # )
                    #
                    # # 提取学习到的实际重叠率（经过 sigmoid 和缩放），同时统计有多少层
                    # learned_overlap = None
                    # try:
                    #     overlap_rates = []
                    #     overlap_param_names = []
                    #
                    #     for name, param in soft_model.named_parameters():
                    #         if 'overlaprate' in name.lower():  # 更鲁棒的匹配
                    #             if param.numel() == 1:
                    #                 with torch.no_grad():
                    #                     sigmoid_val = torch.sigmoid(param).item()
                    #                     actual_overlap = 1.7 * sigmoid_val  # 根据模型内部实现
                    #                     overlap_rates.append(actual_overlap)
                    #                     overlap_param_names.append(name)
                    #             else:
                    #                 print(f"⚠️ Overlap rate parameter {name} is not scalar")
                    #
                    #     if overlap_rates:
                    #         learned_overlap = np.mean(overlap_rates)
                    #         print(f"✅ Learned **effective** overlap rate: {learned_overlap:.4f}")
                    #         print(f"ℹ️ Found {len(overlap_rates)} overlap layer(s): {overlap_param_names}")
                    #         if len(overlap_rates) > 1:
                    #             print("⚠️ Warning: More than one overlap parameter found! Is this expected?")
                    #     else:
                    #         print("⚠️ No overlap rate parameters found in the model")
                    # except Exception as e:
                    #     print(f"⚠️ Failed to extract overlap rate: {str(e)}")
                    #
                    # # 保存结果
                    # ext_test_acc = 0
                    # save_results_incremental(soft_csv_path, [[
                    #     model_name, seed, 4, learned_overlap,  # 保存学习到的重叠率
                    #     soft_history['train_loss'][-1], soft_history['train_acc'][-1],
                    #     soft_history['val_loss'][-1], soft_history['val_acc'][-1], ext_test_acc
                    #
                    # ]])
                    #
                    # # 释放Soft模型内存
                    # del soft_model
                    # torch.cuda.empty_cache()

                    # ========== raw模型训练和评估 ==========

                    criterion_hard = nn.CrossEntropyLoss()
                    # print(f"\n[Training Hard Model with Oc={oc:.1f}]")
                    # hard_model = load_model_hard(model_id, overlaprate=oc).to(device)
                    if model_name == 'mobilenet':
                        raw_model = MobileNetV2CNN_spec().to(device)
                        print(raw_model)
                    elif model_name == "squeezenet":
                        raw_model = SqueezeCNN_spec().to(device)
                        # raw_model = squeezenet().to(device)
                    elif model_name == 'ghostnet':
                        raw_model = GhostCNN_spec().to(device)
                    # raw_model = (overlaprate=oc).to(device)
                    elif model_name == 'shufflenet':
                        raw_model = ShuffleNetV2CNN_spec().to(device)
                    elif model_name == 'LMSC':
                        #raw_model = LMSC_spec().to(device)
                        raw_model = my_resnet().to(device)
                    elif model_name == 'LMSC_short':
                        raw_model = my_resnet_short().to(device)
                    elif model_name == 'LMSC_short_hard':
                        raw_model = my_resnet_short_soft().to(device)
                    elif model_name == 'fasternet':
                        raw_model = FasterNet_shorter().to(device)
                    elif model_name == 'fasternet_raw':
                        raw_model = FasterNet().to(device)
                    elif model_name == 'lsnet':
                        raw_model = LSNet().to(device)
                    elif model_name == 'fasternet_hard':
                        raw_model = FasterNet_shorter_hard(overlaprate=0.4946154564619064).to(device)
                    elif model_name == 'mobilenetv4_raw':
                        raw_model = mobilenetv4_conv_small().to(device)
                    elif model_name == 'mobilenetv4_short':
                        raw_model = mobilenetv4_conv_small_short().to(device)
                    elif model_name=='extractor':
                        raw_model= ClassificationNet().to(device)
                    elif model_name=='extractor_short':
                        raw_model=ClassificationNet_short().to(device)
                    elif model_name=='extractor_short2':
                        raw_model=ClassificationNet_short2().to(device)
                    elif model_name=='extractor_short3':
                        raw_model=ClassificationNet_short3().to(device)
                    elif model_name == 'simplecnn':
                        raw_model = SimpleCNN().to(device)
                        print(raw_model)
                    elif model_name == 'simplecnn_2g':
                        raw_model = SimpleCNN_group2().to(device)
                    elif model_name == 'simplecnn_4g':
                        raw_model = SimpleCNN_group4().to(device)
                    elif model_name == 'simplecnn_8g':
                        raw_model = SimpleCNN_group8().to(device)
                    elif model_name == 'simplecnn_16g':
                        raw_model = SimpleCNN_group16().to(device)
                    elif model_name == 'simplecnn_32g':
                        raw_model = SimpleCNN_group32().to(device)
                    elif model_name == 'simplecnn_64g':
                        raw_model = SimpleCNN_group64().to(device)

                    # 统一使用AdamW优化器
                    optimizer_hard = optim.Adam(
                        raw_model.parameters(),
                        lr=INIT_LR,
                        weight_decay=0.0
                    )

                    # 使用新的训练函数
                    raw_model2, hard_history = train_raw_model(
                        seed_dataset, raw_model, train_loader, val_loader, criterion_hard, optimizer_hard, "Hard", 'compare_results_12_27_squeezenet_block_hard_lora_local'
                    )
                    for test_snr in test_snr_list:
                        # 评估外部测试集（不加噪声）
                        ext_test_acc = run_test(raw_model2, TEST_FILE, DEV_RANGE, PKT_RANGE_test, test_snr=test_snr,
                                                model_name=f"hard_model_{model_name}_raw_seed_{seed_dataset}")

                        # 保存结果
                        save_results_incremental(raw_csv_path, [[
                            model_name, seed_dataset, 1, 0,
                            hard_history['train_loss'][-1], hard_history['train_acc'][-1],
                            hard_history['val_loss'][-1], hard_history['val_acc'][-1],
                            ext_test_acc
                        ]])
                        summary_acc.append(ext_test_acc)

                    # 释放Hard模型内存
                    del raw_model, raw_model2
                    torch.cuda.empty_cache()

        print("\nExperiment completed successfully!")
        if summary_acc:
            mean_acc = sum(summary_acc) / len(summary_acc)
        else:
            mean_acc = 0
        print('summary_acc=', summary_acc)
        print("mean_acc=", mean_acc)
