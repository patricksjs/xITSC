import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from data.data_loader import load_data
from model.Model import Model

# 初始化 wandb（如果需要）
os.environ["WANDB_MODE"] = "offline" # 如果想离线运行


parser = argparse.ArgumentParser()
# 模型参数（补充缺失的4个核心参数）
parser.add_argument('--seq_len', type=int, default=637, help='输入序列长度')
parser.add_argument('--label_len', type=int, default=36, help='label长度（分类任务无实际作用，需占位）')
parser.add_argument('--pred_len', type=int, default=72, help='预测长度（分类任务无实际作用，需占位）')
parser.add_argument('--enc_in', type=int, default=1, help='输入特征数（LKA数据集是单特征时序数据，设为1）')
parser.add_argument('--c_out', type=int, default=3, help='输出特征数（分类任务=类别数）')
parser.add_argument('--d_model', type=int, default=64, help='嵌入维度')
parser.add_argument('--d_ff', type=int, default=256, help='FeedForward 隐藏层维度')
parser.add_argument('--num_kernels', type=int, default=6, help='Inception 核数量')
parser.add_argument('--top_k', type=int, default=5, help='FFT Top-k 周期')
parser.add_argument('--e_layers', type=int, default=2, help='TimesBlock 层数')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout 率')
parser.add_argument('--embed', type=str, default='fixed', help='嵌入类型')
parser.add_argument('--freq', type=str, default='h', help='时间频率')

# spectrogram 相关参数
parser.add_argument('--fs', type=int, default=1)
parser.add_argument('--noverlap', type=int, default=8)
parser.add_argument('--nperseg', type=int, default=16)

# 数据参数
parser.add_argument('--dataset', type=str, default='Lightning', help="Dataset to train on")
parser.add_argument('--num_class', type=int, default=2, help='类别数')
parser.add_argument('--batch_size', type=int, default=16, help='批次大小')

# 训练参数
parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
parser.add_argument('--n_epochs', type=int, default=200, help='训练轮数')
parser.add_argument('--savedir', type=str, default='timesnet_models', help='模型保存路径')
parser.add_argument('--patience', type=int, default=30, help='早停耐心值')
parser.add_argument('--mode', type=str, default='train', choices=['test', 'train'])
parser.add_argument('--classification_model', type=str, default="timesnet_models/best_timesnet.pth",
                    help='trained classifier model for testing')

# 任务参数
parser.add_argument('--task_name', type=str, default='classification', help='任务类型')
args = parser.parse_args()

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'使用设备：{device}')

# 创建模型保存目录
os.makedirs(args.savedir, exist_ok=True)


ds, num_features, _ = load_data(args.dataset, args)

train_size = int(0.7 * len(ds))
val_size = int(0.1 * len(ds))
test_size = len(ds) - train_size - val_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


model = Model(args).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for data, labels, spectrogram, background_spectrogram, xrec in loader:
        data = data.unsqueeze(-1).to(device)  # 新增：添加特征维度，变为 [B, 720, 1]
        labels = labels.to(device)

        # 生成 x_mark_enc：全1张量 [B, T]，表示所有序列位置有效
        x_mark_enc = torch.ones(data.shape[0], data.shape[1]).to(device)

        optimizer.zero_grad()
        # 模型需要传入 x_enc 和 x_mark_enc 两个参数（分类任务）
        outputs = model(x_enc=data, x_mark_enc=x_mark_enc, x_dec=None, x_mark_dec=None)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, labels, spectrogram, background_spectrogram, xrec in loader:
            data = data.unsqueeze(-1).to(device)  # 新增：添加特征维度
            labels = labels.to(device)
            x_mark_enc = torch.ones(data.shape[0], data.shape[1]).to(device)  # 新增：时间标记

            outputs = model(x_enc=data, x_mark_enc=x_mark_enc, x_dec=None, x_mark_dec=None)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels


# -------------------------- 5. 主训练循环 --------------------------
best_val_acc = 0.0
patience_counter = 0

for epoch in range(args.n_epochs):
    print(f'Epoch [{epoch + 1}/{args.n_epochs}]')

    # 训练
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f'Train Loss: {train_loss:.4f}')

    # 验证
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # 记录 wandb


    # 保存最优模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
        }, os.path.join(args.savedir, 'best_timesnet.pth'))
        print(f'保存最优模型 (Val Acc: {best_val_acc:.4f})')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= args.patience:
            print(f'早停触发！已连续 {args.patience} 轮无提升。')
            break

# -------------------------- 6. 测试 --------------------------
print('\n=== 最终测试 ===')
checkpoint = torch.load(os.path.join(args.savedir, 'best_timesnet.pth'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_loss, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device)

final_precision = precision_score(all_labels, all_preds, average='macro')
final_recall = recall_score(all_labels, all_preds, average='macro')
final_f1 = f1_score(all_labels, all_preds, average='macro')

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test Precision (macro): {final_precision:.4f}')
print(f'Test Recall (macro): {final_recall:.4f}')
print(f'Test F1-Score (macro): {final_f1:.4f}')


print("Finished training / testing")