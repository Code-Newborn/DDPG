# DDPG算法的实现脚本

# 在DDPG算法之前，我们在求解连续动作空间问题时，主要有两种方式：
# 一是对连续动作做离散化处理，然后再利用强化学习算法（例如DQN）进行求解。
# 二是使用Policy Gradient (PG)算法 (例如Reinforce) 直接求解。
# 对于方式一，离散化处理在一定程度上脱离了工程实际；
# 对于方式二，PG算法在求解连续控制问题时效果往往不尽人意。
# 为此，DDPG算法横空出世，在许多连续控制问题上取得了非常不错的效果。


import torch as T
import torch.nn.functional as F
import numpy as np
from networks import ActorNetwork, CriticNetwork
from buffer import ReplayBuffer

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class DDPG:
    def __init__(self, alpha, beta,
                 state_dim, action_dim,
                 actor_fc1_dim, actor_fc2_dim, critic_fc1_dim, critic_fc2_dim,
                 ckpt_dir,
                 gamma=0.99, tau=0.005,
                 action_noise=0.1,
                 max_size=1000000,
                 batch_size=256):

        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.checkpoint_dir = ckpt_dir

        # 演员网络 初始化
        self.actor = ActorNetwork(alpha=alpha,
                                  state_dim=state_dim,
                                  action_dim=action_dim,
                                  fc1_dim=actor_fc1_dim,
                                  fc2_dim=actor_fc2_dim)

        # 目标演员网络 初始化
        self.target_actor = ActorNetwork(alpha=alpha,
                                         state_dim=state_dim,
                                         action_dim=action_dim,
                                         fc1_dim=actor_fc1_dim,
                                         fc2_dim=actor_fc2_dim)

        # 评委网络 初始化
        self.critic = CriticNetwork(beta=beta,
                                    state_dim=state_dim,
                                    action_dim=action_dim,
                                    fc1_dim=critic_fc1_dim,
                                    fc2_dim=critic_fc2_dim)

        # 目标评委网络 初始化
        self.target_critic = CriticNetwork(beta=beta,
                                           state_dim=state_dim,
                                           action_dim=action_dim,
                                           fc1_dim=critic_fc1_dim,
                                           fc2_dim=critic_fc2_dim)

        # 经验回放池 初始化
        self.memory = ReplayBuffer(max_size=max_size,
                                   state_dim=state_dim,
                                   action_dim=action_dim,
                                   batch_size=batch_size)

        self.update_network_parameters(tau=1.0)  # 更新网络参数

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # 通过指数平滑方法而不是直接替换参数来更新目标网络，
        # 超参数 tau ≪ 1, 目标网络的更新缓慢且平稳, 这种方式提高了学习的稳定性。

        # 目标演员网络的参数更新
        for actor_params, target_actor_params in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_actor_params.data.copy_(
                tau * actor_params + (1 - tau) * target_actor_params)
        # 目标评委网络的参数更新
        for critic_params, target_critic_params in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_critic_params.data.copy_(
                tau * critic_params + (1 - tau) * target_critic_params)

    '''
    brief      经验记忆
    msg        记录一次动作过程中的状态和动作信息
    param      {*} self 本类对象
    param      {*} state 当前状态
    param      {*} action 动作
    param      {*} reward 奖励
    param      {*} state_ 下一状态
    param      {*} done 终止标志
    return     {*}
    '''

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation, train=True):
        self.actor.eval()  # 进入评估
        state = T.as_tensor(T.from_numpy(observation).float(),
                            dtype=T.float32).to(device)
        action = self.actor.forward(state).squeeze()

        if train:
            noise = T.tensor(np.random.normal(
                loc=0.0, scale=self.action_noise), dtype=T.float32).to(device)
            action = T.clamp(action+noise, -1, 1)  # 限制在(-1,1)
        self.actor.train()

        return action.detach().cpu().numpy()

    '''
    brief      学习
    msg        更新目标AC网络结构参数
    param      {*} self DDPG算法对象
    return     {*} 无
    '''

    def learn(self):

        if not self.memory.ready():
            return

        # 回放经验
        states, actions, reward, states_, terminals = self.memory.sample_buffer()  # 经验采样
        # 转换成Tensor处，在GPU中处理
        states_tensor = T.tensor(states, dtype=T.float32).to(device)
        actions_tensor = T.tensor(actions, dtype=T.float32).to(device)
        rewards_tensor = T.tensor(reward, dtype=T.float32).to(device)
        next_states_tensor = T.tensor(states_, dtype=T.float32).to(device)
        terminals_tensor = T.tensor(terminals).to(device)

        # 不希望将以下参数进行梯度下降更新
        with T.no_grad():
            next_actions_tensor = self.target_actor.forward(next_states_tensor)
            q_ = self.target_critic.forward(
                next_states_tensor, next_actions_tensor).view(-1)
            q_[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * q_

        q = self.critic.forward(states_tensor, actions_tensor).view(-1)

        critic_loss = F.mse_loss(q, target.detach())
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        new_actions_tensor = self.actor.forward(states_tensor)
        actor_loss = -T.mean(self.critic(states_tensor, new_actions_tensor))
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def save_models(self, episode):
        self.actor.save_checkpoint(
            self.checkpoint_dir + 'Actor/DDPG_actor_{}.pth'.format(episode))
        print('Saving actor network successfully!')

        self.target_actor.save_checkpoint(
            self.checkpoint_dir + 'Target_actor/DDPG_target_actor_{}.pth'.format(episode))
        print('Saving target_actor network successfully!')

        self.critic.save_checkpoint(
            self.checkpoint_dir + 'Critic/DDPG_critic_{}'.format(episode))
        print('Saving critic network successfully!')

        self.target_critic.save_checkpoint(
            self.checkpoint_dir + 'Target_critic/DDPG_target_critic_{}'.format(episode))
        print('Saving target critic network successfully!')

    def load_models(self, episode):
        self.actor.load_checkpoint(
            self.checkpoint_dir + 'Actor/DDPG_actor_{}.pth'.format(episode))
        print('Loading actor network successfully!')

        self.target_actor.load_checkpoint(
            self.checkpoint_dir + 'Target_actor/DDPG_target_actor_{}.pth'.format(episode))
        print('Loading target_actor network successfully!')

        self.critic.load_checkpoint(
            self.checkpoint_dir + 'Critic/DDPG_critic_{}'.format(episode))
        print('Loading critic network successfully!')

        self.target_critic.load_checkpoint(
            self.checkpoint_dir + 'Target_critic/DDPG_target_critic_{}'.format(episode))
        print('Loading target critic network successfully!')
