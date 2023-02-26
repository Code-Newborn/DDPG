# 网络脚本，包括演员网络和评论家网络

import torch as T
import torch.nn as nn
import torch.optim as optim

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


def weight_init(m):
    if isinstance(m, nn.Linear):  # 若存在线性全连接
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):  # 若存在线性全连接
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)  # 定义神经网络的线性全连接层，输入维度state_dim
        self.ln1 = nn.LayerNorm(fc1_dim)  # 中间层归一化
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)  # 定义神经网络的线性全连接层
        self.ln2 = nn.LayerNorm(fc2_dim)  # 中间层归一化
        # 定义神经网络的线性全连接层，输出维度action_dim
        self.action = nn.Linear(fc2_dim, action_dim)

        # Adam(Adaptive Moment Estimation) 优化模型参数的方法，随机梯度的一种
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.apply(weight_init)  # 使每个子模块应用参数初始化函数weight_init()
        self.to(device)

    '''
    brief      前向传播
    msg        描述前向传播的过程
    param      {*} self 本类对象
    param      {*} state 状态
    return     {*} 动作
    '''

    def forward(self, state):
        x = T.relu(self.ln1(self.fc1(state)))
        x = T.relu(self.ln2(self.fc2(x)))
        action = T.tanh(self.action(x))
        return action

    def save_checkpoint(self, checkpoint_file):
        # 以字典形式保存模块的整个状态到路径checkpoint_file
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        # 以字典形式加载模块的整个状态到路径checkpoint_file
        self.load_state_dict(T.load(checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)  # 定义神经网络的线性全连接层，输入维度state_dim
        self.ln1 = nn.LayerNorm(fc1_dim)  # 中间层归一化
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)  # 定义神经网络的线性全连接层
        self.ln2 = nn.LayerNorm(fc2_dim)  # 中间层归一化
        self.fc3 = nn.Linear(action_dim, fc2_dim)  # 定义神经网络的线性全连接层，比演员网络多2层连接
        self.q = nn.Linear(fc2_dim, 1)  # 定义神经网络的线性全连接层，输出维度1

        self.optimizer = optim.Adam(
            self.parameters(), lr=beta, weight_decay=0.001)  # 根据参数大小进行权值衰减weight_decay
        self.apply(weight_init)
        self.to(device)

    '''
    brief      前向传播
    msg        网络模块子类重写
    param      {*} self 本类对象
    param      {*} state 状态Tensor
    param      {*} action 动作Tensor
    return     {*}
    '''

    def forward(self, state, action):
        x_s = T.relu(self.ln1(self.fc1(state)))
        x_s = self.ln2(self.fc2(x_s))
        x_a = self.fc3(action)
        x = T.relu(x_s + x_a)
        q = self.q(x)
        return q

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))
