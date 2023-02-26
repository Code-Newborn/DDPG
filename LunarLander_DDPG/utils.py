# 工具箱脚本，主要放置一些工具函数例如【创建文件夹】、【绘制学习曲线】、【动作缩放】
import os
import numpy as np
import matplotlib.pyplot as plt

# 利用Ornstein-Uhlenbeck过程产生时序相关的噪声，以提高在惯性系统（环境）中的控制任务的探索效率。


class OUActionNoise:  # OU噪声Ornstein-Uhlenbeck
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta  # theta就是均值回归的速率（theta越大，干扰越小）
        self.mu = mu  # mu是均值
        self.sigma = sigma  # sigma是波动率（扰动的程度）
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):  # 调用生成OU噪声
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros_like(self.mu)


'''
brief      创建文件夹
msg        利用os模块创建文件夹
param      {str} path 文件夹路径
param      {list} sub_paths 子文件夹路径
return     {*}
'''


def create_directory(path: str, sub_paths: list):
    for sub_path in sub_paths:
        if not os.path.exists(path + sub_path):
            os.makedirs(path + sub_path, exist_ok=True)
            print('Create path: {} successfully'.format(path+sub_path))
        else:
            print('Path: {} is already existence'.format(path+sub_path))


'''
brief      绘制学习曲线
msg        matplotlib绘制学习曲线
param      {*} episodes 模型验证次数
param      {*} records
param      {*} title 标题
param      {*} ylabel y轴标签
param      {*} figure_file 绘图文件
return     {*} 无
'''


def plot_learning_curve(episodes, records, title, ylabel, figure_file):
    plt.figure()
    plt.plot(episodes, records, color='r', linestyle='-')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)

    plt.show()
    plt.savefig(figure_file)


'''
brief      缩放动作向量
msg        尺度缩放动作向量
param      {*} action 动作向量
param      {*} high 
param      {*} low
return     {*} 动作值
'''


def scale_action(action, high, low):
    action = np.clip(action, -1, 1)  # 限制action的值
    weight = (high - low) / 2
    bias = (high + low) / 2
    action_ = action * weight + bias

    return action_
