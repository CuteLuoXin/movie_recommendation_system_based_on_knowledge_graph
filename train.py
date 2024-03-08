import pandas as pd
import time
from utils import fix_seed_torch, draw_loss_pic
import argparse
from model import GCN
from Logger import Logger
from mydataset import MyDataset
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import sys

# 固定随机数种子
fix_seed_torch(seed=2021)
# 设置训练的超参数
parser = argparse.ArgumentParser()
parser.add_argument('--gcn_layers', type=int, default=2, help='the number of gcn layers')
parser.add_argument('--n_epochs', type=int, default=30, help='the number of epochs')
parser.add_argument('--embedSize', type=int, default=64, help='dimension of user and entity embeddings')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')
args = parser.parse_args()
# 设备是否支持cuda
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
args.device = device
# 读取用户特征、天气特征、评分
user_feature = pd.read_csv('./data/user.txt', encoding='utf-8', sep='\t')
item_feature = pd.read_csv('./data/weather.txt', encoding='utf-8', sep='\t')
rating = pd.read_csv('./data/rating.txt', encoding='utf-8', sep='\t')
# 构建数据集
dataset = MyDataset(rating)
trainLen = int(args.ratio * len(dataset))
train, test = random_split(dataset, [trainLen, len(dataset) - trainLen])
train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test, batch_size=len(test))
# 记录训练的超参数
start_time = '{}'.format(time.strftime("%m-%d-%H-%M", time.localtime()))
logger = Logger('./log/log-{}.txt'.format(start_time))
logger.info(' '.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
# 定义模型
model = GCN(args, user_feature, item_feature, rating)
model.to(device)
# 定义优化器
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
# 定义损失函数
loss_function = MSELoss()
train_result = []
test_result = []
# 最好的epoch
best_loss = sys.float_info.max
# 训练
for i in range(args.n_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        prediction = model(batch[0].to(device), batch[1].to(device))
        train_loss = torch.sqrt(loss_function(batch[2].float().to(device), prediction))
        train_loss.backward()
        optimizer.step()
    train_result.append(train_loss.item())
    for data in test_loader:
        model.eval()
        prediction=model(data[0].to(device), data[1].to(device))
        test_loss = torch.sqrt(loss_function(data[2].float().to(device), prediction))
        test_loss = test_loss.item()
        if best_loss > test_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), './model/model-{}.pth'.format(start_time))
    test_result.append(test_loss)
    logger.info("Epoch {:d}: TrainLoss {:.4f}, TestLoss {:.4f}".format(i, train_loss, test_loss))
best_epoch, RMSE = sorted(list(enumerate(train_result)), key=lambda x: x[1])[0]
logger.info("Epoch {:d}: bestTrainLoss {:.4f}".format(best_epoch, RMSE))
best_epoch, RMSE = sorted(list(enumerate(test_result)), key=lambda x: x[1])[0]
logger.info("Epoch {:d}: bestTestLoss {:.4f}".format(best_epoch, RMSE))
# 画图
draw_loss_pic(train_result, test_result)
