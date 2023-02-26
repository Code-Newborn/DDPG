# 经验回放池脚本

import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, batch_size):
        self.mem_size = max_size
        self.batch_size = batch_size
        self.mem_cnt = 0

        self.state_memory = np.zeros((self.mem_size, state_dim))
        self.action_memory = np.zeros((self.mem_size, action_dim))
        self.reward_memory = np.zeros((self.mem_size, ))
        self.next_state_memory = np.zeros((self.mem_size, state_dim))
        self.terminal_memory = np.zeros((self.mem_size, ), dtype=np.bool_)

    '''
    brief      存储状态转移数据
    msg        存储状态转移数据到经验池
    param      {*} self 经验回放类
    param      {*} state 当前状态
    param      {*} action 当前状态下的动作
    param      {*} reward 执行动作后的奖励
    param      {*} state_ 下一状态
    param      {*} done 终止标志
    return     {*}
    '''

    def store_transition(self, state, action, reward, state_, done):
        mem_idx = self.mem_cnt % self.mem_size  # 经验覆盖

        self.state_memory[mem_idx] = state
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.next_state_memory[mem_idx] = state_
        self.terminal_memory[mem_idx] = done

        self.mem_cnt += 1

    '''
    brief      经验回放
    msg        均匀回放：随机等概率从经验池中采样经验。
    param      {*} self 经验池
    return     {*} 返回经验数据
    '''

    def sample_buffer(self):
        mem_len = min(self.mem_size, self.mem_cnt)
        batch = np.random.choice(mem_len, self.batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminals

    def ready(self):
        return self.mem_cnt >= self.batch_size
