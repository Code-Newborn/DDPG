import gym
import imageio
import argparse
from DDPG import DDPG
from utils import scale_action

parser = argparse.ArgumentParser()  # 命令行参数解释器
parser.add_argument('--filename', type=str,
                    default='./output_images/LunarLander.gif')
parser.add_argument('--checkpoint_dir', type=str,
                    default='./checkpoints/DDPG/')
parser.add_argument('--save_video', type=bool,
                    default=True)
parser.add_argument('--fps', type=int,
                    default=30)
parser.add_argument('--render', type=bool,
                    default=True)

args = parser.parse_args()


def main():
    env = gym.make('LunarLanderContinuous-v2')  # 创建环境
    '''
    参考：https://github.com/openai/gym/wiki/Leaderboard#lunarlander-v2
    观测空间中的状态参数有：
    1. 着陆器的x位置
    2. 着陆器的y位置
    3. 着陆器的x水平速度
    4. 着陆器的y垂直速度
    5. 着陆器的θ姿态角度
    6. 着陆器的θ姿态角速度
    7. 着陆器的左腿接地(Boolean)
    8. 着陆器的右腿接地(Boolean)
    动作空间中的控制参数有：
    1. 主引擎控制：
        (1) =-1.0 时：关闭主引擎
        (2) >-1.0 && <=0 时：主引擎动力不足以支持工作
        (3) >0 && <=1.0 时：主引擎动力从50%到100%输出
    2. 左右引擎控制
        (1) >=-1.0 && <-0.5 时：右引擎点火输出
        (2) >=-0.5 && <=0.5 时：关闭主引擎
        (3) >0.5 && <=1.0 时：左引擎点火输出
    '''
    agent = DDPG(alpha=0.0003,
                 beta=0.0003,
                 state_dim=env.observation_space.shape[0],  # 状态空间维度 8
                 action_dim=env.action_space.shape[0],  # 动作空间维度 2
                 actor_fc1_dim=400,  # 演员网络中间层神经元
                 actor_fc2_dim=300,  # 演员网络中间层神经元
                 critic_fc1_dim=400,  # 评委网络中间层神经元
                 critic_fc2_dim=300,  # 评委网络中间层神经元
                 ckpt_dir=args.checkpoint_dir,  # 模型数据的保存路径
                 batch_size=256)
    agent.load_models(1000)
    video = imageio.get_writer(args.filename, fps=args.fps)

    done = False
    observation = env.reset()
    while not done:
        if args.render:
            env.render()
        action = agent.choose_action(observation, train=True)
        action_ = scale_action(
            action.copy(), env.action_space.high, env.action_space.low)
        observation_, reward, done, info = env.step(action_)
        observation = observation_
        if args.save_video:
            video.append_data(env.render(mode='rgb_array'))


if __name__ == '__main__':
    main()
