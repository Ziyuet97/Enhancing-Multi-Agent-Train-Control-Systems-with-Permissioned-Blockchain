import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
from sumolib import checkBinary
import gymnasium as gym
import numpy as np
import pandas as pd
import sumolib
import traci
from gymnasium.utils import EzPickle, seeding
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
import pygame
import pickle

import torch
from torch import nn
import torch.nn.functional as F
import collections
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ

# --- PettingZoo 环境封装 ---
# 提供了创建 PettingZoo 环境的函数
def env(**kwargs):
    """Instantiate a PettingoZoo environment."""
    env = SumoEnvironmentPZ(**kwargs) # 注意：SumoEnvironmentPZ 类未在此代码中定义，但 env 和 parallel_env 未在 __main__ 中使用
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

parallel_env = parallel_wrapper_fn(env)

# --- SUMO 配置 ---
sumoBinary = checkBinary('sumo-gui') # 检查 'sumo-gui' (带界面的SUMO)
sumoCmd = [sumoBinary, "-c", "osm.sumocfg"] # 准备SUMO启动命令

# 检查 SUMO_HOME 环境变量，这对于 traci 导入工具至关重要
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# --- SUMO 强化学习环境 (Gym 风格) ---
class SumoEnvironment(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(
            self,
            net_file,
            route_file,
            osmcfg_file,
            use_gui,
            num_seconds,
            render_mode,
            width,
            height,
            zoom_size,
            ):
        
        # --- 1. 初始化环境参数 ---
        self._osmcfg = osmcfg_file      # SUMO 配置文件路径
        self._net = net_file            # SUMO 路网文件路径
        self._route = route_file        # SUMO 路由文件路径
        self.use_gui = use_gui          # 是否使用 GUI
        self.num_seconds = num_seconds  # 模拟总秒数
        self.render_mode = render_mode  # 渲染模式
        
        # --- 2. 关键：运行一次模拟以收集所有“行程”数据 ---
        # 启动一个临时的 SUMO 实例，专门用于收集数据
        traci.start([sumolib.checkBinary("sumo"), "-c", self._osmcfg,'--no-warnings'])
        conn = traci
        
        self.vs = []            # 存储所有出现过的车辆ID
        self.busstops = []      # 存储所有唯一的公交站ID
        self.vlane = {}         # 字典：{车辆ID: [该车的站点路径]}
        
        # 运行一个很长的模拟 (14856 步)，确保所有公交车都出现
        # 这是你提到的“首先运行一轮之后记录全部trips的数据”
        for i in range(14856):
            conn.simulationStep()
            vs = conn.vehicle.getIDList()
            # 找到新出现的车辆
            vs_extend = [item for item in vs if item not in self.vs]
            for item in vs_extend:
                # 获取新车辆的完整站点停靠列表
                vs_extend_busstop_item = conn.vehicle.getNextStops(item)
                vlane = []
                for busstop in vs_extend_busstop_item:
                    vlane.append(busstop[0]) # busstop[0] 是站点的 lane ID
                    # 如果这个站点是第一次见，就加入总列表
                    if busstop[0] not in self.busstops:
                        self.busstops.append(busstop[0])
                # 存储这辆车的站点路径
                self.vlane.setdefault(item, vlane)
            # 将新车辆加入总车辆列表
            self.vs.extend(vs_extend)

        # --- 3. 生成“行程对” (stop_peers) ---
        # self.stop_peers 存储了所有[起点站, 终点站]对
        # 这代表了公交网络中的所有“边”
        # 奖励将基于有多少“边”跨越了两个区域
        self.stop_peers = []
        for v, stops in self.vlane.items():
            for index in range(len(stops)-1):
                self.stop_peers.append([stops[index],stops[index+1]])

        self.lane_ids = list(conn.lane.getIDList())
        self.edge_ids = conn.edge.getIDList()

        # --- 4. 定义动作空间和观测空间 ---
        # 动作空间：离散，大小=公交站总数 (选择一个公交站)
        self.action_space = gym.spaces.Discrete(len(self.busstops))
        # 观测空间：离散，大小=公交站总数 (每个公交站属于哪个区域)
        self.observation_space = gym.spaces.Discrete(len(self.busstops))
        # 动作掩码 (Mask)：用于DQN，确保不会重复选择已分配的站点
        self.mask = [1 for i in range(len(self.busstops))]

        # 动作 -> 站点ID 的映射
        self._action_to_direction = {}
        for index in range(len(self.busstops)):
            self._action_to_direction[index] = self.busstops[index]
    
        # --- 5. 渲染参数 ---
        self.width = width
        self.height = height
        self.zoom_size = zoom_size

        # --- 6. 关键：关闭临时连接并设置 conn 为 None ---
        conn.close() 
        self.conn = None # 修复 TraCIException 的关键：确保 conn 属性存在并初始化为 None

    def reset(self, seed=None):
        super().reset(seed=seed) # 重置 Gym 的随机种子

        # --- 关键：修复 TraCIException ---
        # 如果上一个 episode 的连接 (self.conn) 还存在，
        # 必须先关闭它，才能启动新连接。
        if hasattr(self, 'conn') and self.conn is not None:
            try:
                self.conn.close()
            except traci.exceptions.TraCIException:
                pass # 有可能连接已经意外关闭
            self.conn = None
        # --- 修复结束 ---

        # 为这个新的 episode 启动一个新的 SUMO 实例
        traci.start([sumolib.checkBinary("sumo"), "-c", self._osmcfg,'--no-warnings']) 
        self.conn = traci
        self.conn.simulationStep()
        
        # --- 初始化状态 (self.classfication) ---
        self.classfication = {}
        for stop in self.busstops:
            self.classfication[stop] = 0 # 所有节点都属于区域 0
        
        # 随机选取一个节点作为初始区域 zone 1
        starter = np.random.choice(self.busstops, size=1, replace=False)
        self.classfication[starter[0]] = 1
        
        obs = self._get_obs() # 获取初始观测
        info = {"reset": self.classfication}

        # 初始化奖励计算
        self.reward_before = 0
        self.reward_after = 0

        return obs, info
    
    def step(self, action):
        # 1. 执行动作：将选定的站点（action）分配给 zone 1
        lane_choice = self._action_to_direction[action] # 动作ID -> 站点ID
        self.classfication[lane_choice] = 1 # 将该站点标记为 1

        # 2. 计算奖励 (Reward)
        # 奖励 = -(穿越两个区域的行程数量)
        # 我们计算的是奖励的“变化量”
        for stop_peer in self.stop_peers:
            # 检查这个行程 (stop_peer) 的起点和终点是否在不同区域
            if self.classfication[stop_peer[0]] != self.classfication[stop_peer[1]]:
                # 如果是，则总 "切割数" 增加 1 (即奖励 -1)
                self.reward_after = self.reward_after - 1

        reward = self.reward_after - self.reward_before # 奖励 = 新的切割数 - 旧的切割数
        self.reward_before = self.reward_after # 更新旧值
        self.reward_after = 0 # 重置计数器

        # 3. 检查终止条件 (Terminated)
        # 当一半的站点被分配到 zone 1 时，回合结束
        terminated = len(self.filter_key_by_value(self.classfication, 1)) == len(self.classfication)//2 

        info = {"step" : reward}
        obs = self._get_obs() # 获取新状态
        observation=np.array(obs, dtype=np.int8)

        # return: 观测, 奖励, 是否终止, 是否截断 (固定为False), 信息
        return observation, reward, terminated, False, info
    
    def find_num_of_lane(self,lane):
        # 辅助函数：通过站点ID找到对应的动作ID
        for key, value in self._action_to_direction.items():
            if value == lane:
                return key

    
    def _get_obs(self):
        # 观测（状态）就是所有站点的区域分配列表
        observation = []
        for _, zone in self.classfication.items():
            observation.append(zone) 
        return observation

    def render_frame(self,conn):
        # 使用 Pygame 进行可视化 (未在DQN训练中调用)
        DISPLAYSURF = pygame.display.set_mode((self.width, self.height), 0, 32)
        pygame.display.set_caption("Drawing")
        DISPLAYSURF.fill((255, 255, 255))
        pixels = pygame.PixelArray(DISPLAYSURF)

        self.lane_position_0 = {}
        self.lane_ids_0 = self.filter_key_by_value(self.classfication, 0) # zone 0 站点
        self.lane_position_1 = {}
        self.lane_ids_1 = self.filter_key_by_value(self.classfication, 1) # zone 1 站点

        for lane in self.lane_ids_0:
            self.lane_position_0[lane] = conn.lane.getShape(lane)[0]
            pixels[int(self.lane_position_0[lane][0]*self.zoom_size)][int(self.lane_position_0[lane][1]*self.zoom_size)] = 255,0,0 # 红色
        for lane in self.lane_ids_1:
            self.lane_position_1[lane] = conn.lane.getShape(lane)[0]
            pixels[int(self.lane_position_1[lane][0]*self.zoom_size)][int(self.lane_position_1[lane][1]*self.zoom_size)] = 0,255,0 # 绿色
            
        while True:
            pygame.display.update()

    def filter_key_by_value(self,d, value):
        # 辅助函数：根据值(value)过滤字典(d)中的键(key)
        result = []
        for key, val in d.items():
            if val == value:
                result.append(key)
        return result
    
    def close(self):
        # 关闭环境时，确保 SUMO 连接已关闭
        if hasattr(self, 'conn') and self.conn is not None:
            self.conn.close()


# --- 经验回放池 ---
class ReplayBuffer():
    def __init__(self, capacity):
        # 使用 collections.deque 创建一个固定容量的双端队列
        self.buffer = collections.deque(maxlen=capacity)
    
    # 向经验池中添加一个“五元组” (s, a, r, s', done)
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    # 从经验池中随机采样一个批量 (batch)
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # 将 (s, a, r, s', done) 五元组解压
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    
    # 返回当前经验池的大小
    def size(self):
        return len(self.buffer)
    
# --- Q 网络 (PyTorch) ---
class Net(nn.Module):
    # 定义一个简单的全连接神经网络 (MLP)
    def __init__(self, n_states, n_hidden, n_actions):
        super(Net, self).__init__()
        # 隐藏层
        self.fc1 = nn.Linear(n_states, n_hidden)
        # 输出层 (输出每个动作的 Q 值)
        self.fc2 = nn.Linear(n_hidden, n_actions)
    
    # 定义前向传播
    def forward(self, x):
        x = self.fc1(x)   # [b, n_states] -> [b, n_hidden]
        # 注：你原始代码中缺少激活函数，这里可以加上
        x = F.relu(x) 
        x = self.fc2(x)   # [b, n_hidden] -> [b, n_actions]
        return x

# --- DDQN 智能体 ---
class DDQN: # <--- 已修改：类名改为 DDQN
    # (1) 初始化
    def __init__(self, n_states, n_hidden, n_actions,
                 learning_rate, gamma, epsilon,
                 target_update, device):
        # 属性分配
        self.n_states = n_states      # 状态特征数
        self.n_hidden = n_hidden      # 隐藏层神经元数
        self.n_actions = n_actions    # 动作数
        self.learning_rate = learning_rate # 学习率
        self.gamma = gamma            # 折扣因子
        self.epsilon = epsilon        # 贪婪策略参数
        self.target_update = target_update # 目标网络更新频率
        self.device = device          # 设备 (CPU/GPU)
        self.count = 0                # 迭代次数计数器

        # 实例化 Q 网络 (q_net) 和目标网络 (target_q_net)
        self.q_net = Net(self.n_states, self.n_hidden, self.n_actions).to(self.device)
        self.target_q_net = Net(self.n_states, self.n_hidden, self.n_actions).to(self.device)
        
        # 定义优化器 (Adam)，只优化 q_net
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    # (2) 动作选择 (带掩码的 Epsilon-Greedy)
    def take_action(self, env, state):
        # 状态转为张量
        state = np.array(state, dtype=np.int8)
        state = torch.Tensor(state[np.newaxis, :]).to(self.device)

        # --- 关键：构建动作掩码 (Mask) ---
        # 掩码的目的是确保智能体不会选择一个已经被分配到 zone 1 的站点
        for index, classfication in enumerate(list(env.classfication.values())):
            if classfication == 0:
                env.mask[index] = 1 # 1 = 可以选择
            else:
                env.mask[index] = 0 # 0 = 不可选 (已被分配)

        # Epsilon-Greedy 策略
        # epsilon 会随着训练 (TERM/episode_num) 动态增加，即利用 (exploitation) 的概率增加
        if np.random.random() < self.epsilon + 0.09 * TERM / episode_num:
            # --- 利用 (Exploitation) ---
            self.q_net.eval() # 设置为评估模式
            with torch.no_grad(): # 不计算梯度
                actions_value = self.q_net(state) # [1, n_actions]
            self.q_net.train() # 切换回训练模式
            
            i = -1 # 用于从 Q 值最高的动作开始索引

            # 获取 Q 值最高的动作
            action = actions_value.argmax().item()
            
            # --- 关键：应用掩码 ---
            # 检查 Q 值最高的动作是否“可选” (mask == 1)
            # 如果不可选，就选择 Q 值第二高、第三高...的动作，直到找到一个可选的
            while env.mask[action] == 0:
                # 获取排序后的 Q 值索引
                action = actions_value.argsort()[0][i].item()
                i = i - 1

        else:
            # --- 探索 (Exploration) ---
            # 随机选择一个动作
            # 我们使用 env.action_space.sample(mask_tuple) 来确保
            # 随机选择的动作也是“可选”的 (mask == 1)
            a = np.array(env.mask, dtype=np.int8)
            mask_tuple = (a)
            action = env.action_space.sample(mask_tuple)

        return action

    # (3) 网络训练 (DDQN 更新)
    def update(self, transition_dict):
        # --- 1. 从 transition_dict 中提取数据并转为张量 ---
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)

        # --- 2. 计算 Q(s, a) ---
        # q_net(states) 输出 [b, n_actions]
        # .gather(1, actions) 会根据 actions [b, 1] 中的索引，提取出对应动作的 Q 值
        q_values = self.q_net(states).gather(1, actions) # [b, 1]

        # --- 3. 计算 Target Q 值 (!!! DDQN 核心修改 !!!) ---
        # DDQN: Q_target = r + γ * Q_target(s', argmax_a' Q_current(s', a'))
        
        with torch.no_grad(): # 目标网络不计算梯度
            # 3.1. 使用 q_net (当前网络) 选择 s' 中的最佳动作
            # q_net(next_states) -> [b, n_actions]
            # .max(1)[1] -> argmax (最佳动作的索引), 形状为 [b]
            # .view(-1, 1) -> 形状变为 [b, 1]，以便用于 gather
            best_next_actions = self.q_net(next_states).max(1)[1].view(-1, 1)
            
            # 3.2. 使用 target_q_net (目标网络) 评估 3.1. 中所选动作的 Q 值
            # .gather(1, best_next_actions) 
            #   从 target_q_net(next_states) [b, n_actions] 中，
            #   根据 best_next_actions [b, 1] 提供的索引，
            #   提取 Q 值。
            next_q_values = self.target_q_net(next_states).gather(1, best_next_actions) # [b, 1]

        # 3.3. 计算 DDQN 目标
        # (1-dones) 的作用是：如果 s' 是终止状态 (done=1)，则未来价值为 0
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        # --- DDQN 修改结束 ---
        
        # --- 4. 计算损失函数 (MSE) ---
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))

        # --- 5. 反向传播和参数更新 ---
        self.optimizer.zero_grad() # 梯度清零
        dqn_loss.backward()        # 反向传播
        self.optimizer.step()      # 更新 q_net 的参数

        # --- 6. 更新目标网络 (target_q_net) ---
        if self.count % self.target_update == 0:
            # 硬更新：直接复制 q_net 的参数
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())
        
        self.count += 1

# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 1. 初始化 SUMO 环境 ---
    env = SumoEnvironment(
        net_file="./osm.net.xml",
        route_file="./osm_pt.rou.xml",
        osmcfg_file="./osm.sumocfg",
        use_gui=True,
        num_seconds=8000,
        render_mode="human",
        width = 815,
        height = 704,
        zoom_size = 1/100,
    )

    # --- 2. 定义 DDQN 训练超参数 ---
    lr = 1e-3
    n_hidden = 128
    gamma = 0.01
    epsilon = 0.9
    target_update = 20
    buffer_size = 10000
    minimal_size = 500  # 经验池中至少有 500 条数据时才开始训练
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    global episode_num
    episode_num = 100 # 总训练回合数
    global TERM
    TERM = 0 # 当前回合数计数器 (用于 epsilon 衰减)

    # --- 3. 获取环境维度 ---
    n_states = len(env.busstops)
    n_actions = env.action_space.n

    # --- 4. 实例化 DDQN 智能体和经验回放池 ---
    replay_buffer = ReplayBuffer(buffer_size)
    agent = DDQN(n_states, n_hidden, n_actions, lr, gamma, epsilon, target_update, device) # <--- 已修改：使用 DDQN

    # --- 5. 检查并加载已保存的模型 ---
    MODEL_PATH = "ddqn_partition_model.pth" # <--- 已修改：模型文件名
    
    if os.path.exists(MODEL_PATH):
        print(f"--- 发现已存在的模型！正在从 {MODEL_PATH} 加载权重... ---")
        # 加载权重文件
        saved_state_dict = torch.load(MODEL_PATH, map_location=device)
        
        # 将权重加载到 Q 网络和目标网络
        agent.q_net.load_state_dict(saved_state_dict)
        agent.target_q_net.load_state_dict(saved_state_dict)
        
        # 确保网络在正确的设备上
        agent.q_net.to(device)
        agent.target_q_net.to(device)
        
        print("--- 模型加载完毕。继续训练... ---")
    else:
        print(f"--- 未在 {MODEL_PATH} 找到模型。开始新的训练... ---")

    # --- 6. 开始主训练循环 ---
    return_list = [] # 存储每个回合的总奖励
    print(f"开始 DDQN 训练，共 {episode_num} 回合...") # <--- 已修改：打印 DDQN
    print(f"目标：最小化两个区域间的'切割'数量 (即最大化负奖励值，使其接近0)")

    for i in range(episode_num):
        TERM = i  # 更新全局回合计数器
        episode_return = 0
        
        # 重置环境，获取初始状态
        state, _ = env.reset()
        terminated = False

        # 内部循环：执行一个完整的 回合 (episode)
        while not terminated:
            # 6.1 智能体选择动作 (已包含掩码逻辑)
            action = agent.take_action(env, state)
            
            # 6.2 环境执行动作，获取 (s', r, done)
            next_state, reward, terminated, _, info = env.step(action)
            
            # 6.3 存入经验池
            replay_buffer.add(state, action, reward, next_state, terminated)
            
            # 6.4 更新状态
            state = next_state
            
            # 6.5 累积奖励
            episode_return += reward
            
            # 6.6 如果经验池够大，则开始训练
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                # 更新 Q 网络
                agent.update(transition_dict)
        
        # --- 回合结束 ---
        return_list.append(episode_return)
        if (i + 1) % 10 == 0:
            print(f"回合: {i+1}/{episode_num}, 回合总奖励: {episode_return}")

    # --- 训练结束 ---
    env.close() # 关闭 SUMO 环境
    print("训练完成。")

    # --- 7. 保存训练好的模型 ---
    print(f"--- 正在保存模型到 {MODEL_PATH}... ---")
    torch.save(agent.q_net.state_dict(), MODEL_PATH) # 只保存 q_net 即可
    print("--- 模型保存成功！ ---")

    # --- 8. 绘制训练结果 ---
    plt.figure(figsize=(12, 6))
    plt.plot(return_list)
    plt.xlabel("Episode (回合数)")
    plt.ylabel("Total Reward per Episode (每回合总奖励)")
    plt.title("DDQN Training Progress (Stop Partitioning)") # <--- 已修改：标题
    plt.grid(True)
    plt.show()