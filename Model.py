import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn import metrics
from tqdm import tqdm
from sklearn.model_selection import KFold
from visdom import Visdom

# # visdom画图
viz = Visdom()
viz.line([[0.3, 0.3]], [0.], win='loss', opts=dict(
    title='loss', legend=['train_loss', 'test_loss']))
viz.line([[0.2, 0.2, 0.2, 0.2]], [0.], win='rmse&mae', opts=dict(
    title='rmse&mae', legend=['train_rmse', 'test_rmse', 'train_mae', 'test_mae']))


def evaluate(pred, label):
    mae = metrics.mean_absolute_error(label, pred)
    rmse = metrics.mean_squared_error(label, pred) ** 0.5
    return rmse, mae


# 训练集
def format_data(train_record, n_splits=3):

    train = [[], []]  # 课程id，参与意愿
    label = [[], [], []]  # 学生, 课程id，参与意愿  (用来计算loss)
    stu_list = set(train_record.index)

    KF = KFold(n_splits=n_splits, shuffle=True)
    count = 0
    for stu in stu_list:
        stu_cour = train_record.loc[[stu], 'course_id'].values
        stu_willingness = train_record.loc[[stu], 'willingness'].values
        if len(stu_cour) >= n_splits:
            for train_idx, label_idx in KF.split(stu_cour):
                train[0].append(stu_cour[train_idx])
                train[1].append(stu_willingness[train_idx])

                label[0].extend([count] * len(label_idx))
                label[1].extend(stu_cour[label_idx])
                label[2].extend(stu_willingness[label_idx])
                count += 1
    return train, label


# 测试集
def format_test_data(train_record, test_record, ):
    train = [[], [], []]  # 学生ID,课程，参与意愿
    test = [[], [], []]  # 学生ID,课程，参与意愿
    stu_list = set(train_record.index)
    count = 0
    for stu in stu_list:
        stu_cour = train_record.loc[[stu], 'course_id'].values
        stu_willingness = train_record.loc[[stu], 'willingness'].values
        test_cour = test_record.loc[[stu], 'course_id'].values
        test_activity = test_record.loc[[stu], 'willingness'].values

        train[0].append(stu)
        train[1].append(stu_cour)
        train[2].append(stu_willingness)

        # 读取测试集的真值
        test[0].extend([count] * len(test_cour))
        test[1].extend(test_cour)
        test[2].extend(test_activity)
        count += 1
    return train, test


# sigmoid是激活函数的一种，它会将样本值映射到0到1之间。
def sigmoid(x):
    return torch.sigmoid(x)


class Net(nn.Module):
    # 模型初始化
    def __init__(self, CA, UA, UC, train_cour_num, behavior_num, device='cpu'):
        super(Net, self).__init__()
        self.train_cour_num = train_cour_num
        self.behavior_num = behavior_num
        CA = CA.to(device)
        UA = UA.to(device)
        UC = UC.to(device)
        CA_ = CA.clone()
        UA_ = UA.clone()
        UC_ = UC.clone()
        self.device = device
        # --------------模型参数---------------------
        guess_ = nn.Parameter(torch.ones(1, train_cour_num).double() * -2)
        slide_ = nn.Parameter(torch.ones(1, train_cour_num).double() * -2)
        # ------------------------------------------
        self.CA_ = CA_
        self.UA_ = UA_
        self.UC_ = UC_
        self.guess_ = guess_
        self.slide_ = slide_

    def forward(self, cour_list, acti_list):  # 前向传播,传入活跃度列表和课程索引列表
        k = self.behavior_num
        CA_ = self.CA_
        UA_ = self.UA_
        UC_ = self.UC_
        guess_ = self.guess_
        slide_ = self.slide_


        A = torch.empty(len(acti_list), k).double().to(self.device)
        for i, X_i in enumerate(acti_list):
            X_i = torch.tensor(X_i).double().to(self.device).reshape(1, -1)
            W_i = torch.softmax(CA_[cour_list[i]], dim=0)
            A[i] = X_i @ W_i
        UC_ = torch.softmax(UC_, dim=1)
        UA_ = torch.softmax(UA_, dim=0)
        Q = UC_.T @ UA_
        Y_ = A @ Q.T
        # 激活
        slide = sigmoid(slide_)
        guess = sigmoid(guess_)
        Y = (1 - slide) * Y_ + guess * (1 - Y_)
        return  Y


# epoch_step = 0
# 定义模型
class Model():
    def __init__(self, CA, UA, UC, train_cour_num, behavior_num, lr=1e-3, device='cpu'):
        net = Net(CA, UA, UC, train_cour_num=train_cour_num, behavior_num=behavior_num, device=device).to(device)
        self.CA = CA
        self.UA = UA
        self.UC = UC
        self.device = device
        self.net = net
        # 需要更新的参数有guess，slide
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        self.loss_function = torch.nn.BCELoss(reduction='mean')  # 交叉熵损失函数
        self.all_pred = pd.DataFrame()

    # 训练
    def fit(self, index_loader, train_data, test_data,
            epochs,
            save_dir=None, save_size=None):
        for epoch in range(epochs):
            loss_list = [[]]  # [[train]]
            label_list, pred_list = [[]], [[]]  # [[train]]
            for batch_data in tqdm(index_loader, "[Epoch:%s]" % epoch):
                stu_list = np.array([x.numpy() for x in batch_data], dtype='int').reshape(-1)
                train_pred, label_data = format_data(train_data.loc[stu_list, :], )


                # -----start training-------------------
                all_pred = self.net(train_pred[0], train_pred[1])
                pred = all_pred[label_data[0], label_data[1]]

                label = torch.DoubleTensor(label_data[2]).to(self.device)
                loss = self.loss_function(pred, label)
                # ------end training--------------------

                loss_list[0].append(loss.item())
                pred_list[0].extend(pred.clone().to('cpu').detach().tolist())
                label_list[0].extend(label_data[2])

                # ------start update parameters----------
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # ------ end update parameters-----------

            # -------start evaluate and drawing-----------
            epoch_loss = np.nanmean(loss_list[0])
            rmse, mae = evaluate(pred_list[0], label_list[0])


            if save_size is not None:
                test_rmse, test_mae, test_loss = self.test(index_loader, train_data, test_data, save_dir=save_dir, save_size=save_size)
                viz.line([[epoch_loss, test_loss]], [epoch], win='loss', update='append')
                viz.line([[rmse, test_rmse, mae, test_mae]], [epoch], win='rmse&mae', update='append')
            # -------end evaluate and drawing-----------

    def test(self, index_loader, train_data, test_data, save_dir=None, save_size=None):
        test_loss_list = [[]]
        test_pred_list, test_label_list = [], []
        if save_size is not None:
            all_pred_score = torch.empty(save_size, dtype=torch.double).to('cpu')
        for batch_data in tqdm(index_loader, "[Testing:]"):
            stu_list = np.array([x.numpy() for x in batch_data], dtype='int').reshape(-1)
            train, test = format_test_data(train_data.loc[stu_list, :],
                                           test_data.loc[stu_list, :])

            with torch.no_grad():
                all_pred = self.net(train[1], train[2])
                test_pred = all_pred[test[0], test[1]].clone().to('cpu').detach()
                test_label = torch.DoubleTensor(test[2])
                test_loss = self.loss_function(test_pred, test_label)
                test_pred_list.extend(test_pred.tolist())
                test_label_list.extend(test[2])

                test_loss_list[0].append(test_loss.item())
                epoch_test_loss = np.mean(test_loss_list[0])

                # -------record -----------
                if save_size is not None:
                    all_pred_score[torch.LongTensor(train[0])] = all_pred.cpu().detach()
        rmse, mae = evaluate(test_pred_list, test_label_list)
        print("\t test_result: rmse:%.6f, mae:%.6f" % (rmse, mae))
        if save_size is not None:
            np.savetxt(save_dir + 'pred_data.csv', all_pred_score.numpy(),
                       fmt='%.6f', delimiter=',')
        return rmse, mae, epoch_test_loss

    def save_parameter(self, save_dir):
        np.savetxt(save_dir + 'slide.txt', self.net.slide_.cpu().detach().numpy())
        np.savetxt(save_dir + 'guess.txt', self.net.guess_.cpu().detach().numpy())

        print('模型参数已成功保存！')
