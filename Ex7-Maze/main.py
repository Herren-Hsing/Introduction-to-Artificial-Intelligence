# 导入相关包 
import os
import random
import numpy as np
from Maze import Maze
from Runner import Runner
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot # PyTorch版本
import matplotlib.pyplot as plt


import numpy as np

# 机器人移动方向
move_map = {
    'u': (-1, 0), # up
    'r': (0, +1), # right
    'd': (+1, 0), # down
    'l': (0, -1), # left
}


# 迷宫路径搜索树
class SearchTree(object):


    def __init__(self, loc=(), action='', parent=None):
        """
        初始化搜索树节点对象
        :param loc: 新节点的机器人所处位置
        :param action: 新节点的对应的移动方向
        :param parent: 新节点的父辈节点
        """

        self.loc = loc  # 当前节点位置
        self.to_this_action = action  # 到达当前节点的动作
        self.parent = parent  # 当前节点的父节点
        self.children = []  # 当前节点的子节点

    def add_child(self, child):
        """
        添加子节点
        :param child:待添加的子节点
        """
        self.children.append(child)

    def is_leaf(self):
        """
        判断当前节点是否是叶子节点
        """
        return len(self.children) == 0


def expand(maze, is_visit_m, node):
    """
    拓展叶子节点，即为当前的叶子节点添加执行合法动作后到达的子节点
    :param maze: 迷宫对象
    :param is_visit_m: 记录迷宫每个位置是否访问的矩阵
    :param node: 待拓展的叶子节点
    """
    can_move = maze.can_move_actions(node.loc)
    for a in can_move:
        new_loc = tuple(node.loc[i] + move_map[a][i] for i in range(2))
        if not is_visit_m[new_loc]:
            child = SearchTree(loc=new_loc, action=a, parent=node)
            node.add_child(child)


def back_propagation(node):
    """
    回溯并记录节点路径
    :param node: 待回溯节点
    :return: 回溯路径
    """
    path = []
    while node.parent is not None:
        path.insert(0, node.to_this_action)
        node = node.parent
    return path


def dfs(maze,current_node,is_visit_m, path):
    is_visit_m[current_node.loc] = 1
    if current_node.loc == maze.destination:
        # 回溯并记录节点路径
        res = back_propagation(current_node)
        for items in res:
            path.append(items)
        return
    if current_node.is_leaf():
        # 拓展叶子节点
        expand(maze, is_visit_m, current_node)
    for child in current_node.children:
        dfs(maze,child,is_visit_m, path)
    is_visit_m[current_node.loc] = 0


    
def depth_first_search(maze):
    # 对迷宫进行深度优先搜索
    start = maze.sense_robot()
    root = SearchTree(loc=start)
    h, w, _ = maze.maze_data.shape
    is_visit_m = np.zeros((h, w), dtype=np.int)  # 标记迷宫的各个位置是否被访问过
    path = []  # 记录路径
    dfs(maze,root,is_visit_m,path)
    return path


def my_search(maze):
    """
    任选深度优先搜索算法、最佳优先搜索（A*)算法实现其中一种
    :param maze: 迷宫对象
    :return :到达目标点的路径 如：["u","u","r",...]
    """

    path = depth_first_search(maze)
    
    return path

import os
import time
import random
import numpy as np
import torch
from ReplayDataSet import ReplayDataSet
import matplotlib.pyplot as plt
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot # PyTorch版本
from QRobot import QRobot
from Maze import Maze
from Runner import Runner

class Robot(TorchRobot):
    def __init__(self, maze):
        # 初始化Robot对象，传入迷宫对象maze作为参数
        super(Robot, self).__init__(maze)
        maze.set_reward(reward={
            "hit_wall": 50,
            "destination": -1000,
            "default": 1,
        })
        # 设置迷宫的奖励值
        self.epsilon = 0.1
        # 初始化epsilon值，用于epsilon-greedy策略中的随机探索概率
        self.maze = maze
        # 将迷宫对象赋值给Robot对象的属性maze
        self.memory.build_full_view(maze=maze)
        # 在内存中建立完整的迷宫视图，即将迷宫的状态存储在内存中，以便智能体进行学习和决策
        self.loss_list = self.train()
        # 调用train()方法进行训练，并将返回的损失值列表赋值给Robot对象的属性loss_list

    def train(self):
        # 训练方法
        loss_list = []  # 存储每次迭代的损失值的列表
        maze_size_squared = self.maze.maze_size ** 2

        while True:
            loss = self._learn(batch=len(self.memory))
            # 调用_learn()方法进行批量训练，并返回损失值
            loss_list.append(loss)
            # 将损失值添加到损失值列表中
            self.reset()
            # 重置智能体的状态和环境

            for step in range(maze_size_squared - 1):
                action, reward = self.test_update()
                # 执行测试更新，获取动作和奖励值

                if reward == self.maze.reward["destination"]:
                    # 如果奖励值等于目标奖励值
                    return loss_list
                    # 返回损失值列表作为训练过程中的损失值记录

    def train_update(self):
        # 训练更新方法
        state = self.sense_state()
        # 获取当前状态
        action = self._choose_action(state)
        # 根据当前状态选择动作
        reward = self.maze.move_robot(action)
        # 根据动作移动机器人并计算奖励
        return action, reward
        # 返回动作和奖励值

    def test_update(self):
        # 测试更新方法
        state = self.get_state()
        # 获取当前状态
        q_value = self.eval_model(state).cpu().data.numpy()
        # 使用评估模型获取当前状态的Q值，并将结果转换为numpy数组
        action = self.get_action(q_value)
        # 根据Q值选择动作
        reward = self.get_reward(action)
        # 根据动作获取奖励值
        return action, reward
        # 返回动作和奖励值

    def get_state(self):
        # 获取状态方法
        state = self.sense_state()
        # 获取当前状态
        return torch.from_numpy(np.array(state, dtype=np.int16)).float().to(self.device)
        # 将状态转换为torch张量，并将其放置在指定设备上

    def get_action(self, q_value):
        # 获取动作方法
        action = self.valid_action[np.argmin(q_value).item()]
        # 根据Q值选择动作
        return action
        # 返回动作

    def get_reward(self, action):
        # 获取奖励值方法
        reward = self.maze.move_robot(action)
        # 根据动作移动机器人并计算奖励
        return reward
        # 返回奖励值