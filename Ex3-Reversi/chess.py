import math
import random
import datetime
import copy
from game import Game

class HumanPlayer:
    """
    人类玩家
    """

    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        self.color = color
    

    def get_move(self, board):
        """
        根据当前棋盘输入人类合法落子位置
        :param board: 棋盘
        :return: 人类下棋落子位置
        """
        # 如果 self.color 是黑棋 "X",则 player 是 "黑棋"，否则是 "白棋"
        if self.color == "X":
            player = "黑棋"
        else:
            player = "白棋"

        # 人类玩家输入落子位置，如果输入 'Q', 则返回 'Q'并结束比赛。
        # 如果人类玩家输入棋盘位置，e.g. 'A1'，
        # 首先判断输入是否正确，然后再判断是否符合黑白棋规则的落子位置
        while True:
            action = input(
                    "请'{}-{}'方输入一个合法的坐标(e.g. 'D3'，若不想进行，请务必输入'Q'结束游戏。): ".format(player,
                                                                                 self.color))

            # 如果人类玩家输入 Q 则表示想结束比赛
            if action == "Q" or action == 'q':
                return "Q"
            else:
                row, col = action[1].upper(), action[0].upper()

                # 检查人类输入是否正确
                if row in '12345678' and col in 'ABCDEFGH':
                    # 检查人类输入是否为符合规则的可落子位置
                    if action in board.get_legal_actions(self.color):
                        return action
                else:
                    print("你的输入不合法，请重新输入!")

class Node:

    def __init__(self, parent, state):  # 节点初始化
        self.visits = 0  # 每个节点被访问次数
        self.score = 0  # 每个节点得分
        self.state = state  # 棋子的左边
        self.unvisited = []  # 未扩展的子节点
        self.next_nodes = []  # 子节点
        self.parent = parent  # 父节点

    def calUCB1(self, c=1):  # UCB1计算
        if self.visits == 0 or self.parent.visits == 0:
            return 0
        return self.score / self.visits + c * (math.sqrt(
            2 * math.log(self.parent.visits / self.visits)))


class AIPlayer:

    def __init__(self, color):
        # 玩家初始化:'X' - 黑棋，'O' - 白棋
        self.color = color
        self.select_color = color  # 选择过程所用棋子的颜色
        self.sim_color = ""  # 模拟过程所用棋子的颜色
        self.root = Node(None, (-1, -1))  # 根节点初始化

    def changeColor(self, cur_color):
        # 交换下棋
        return "X" if cur_color == 'O' else 'O'

    def game_over(self, board):
        # 判断游戏是否结束，双方都无棋子可下游戏结束
        return len(list(board.get_legal_actions('X'))) == 0 and len(
            list(board.get_legal_actions('O'))) == 0

    def Select(self, board):
        # 选择过程
        cur_node = self.root
        while not self.game_over(board):
            #
            if cur_node.unvisited:
                #如果有未拓展过的节点，则随机扩展一个节点
                action = cur_node.unvisited.pop(
                    random.randrange(len(cur_node.unvisited)))
                # 随机选取一个未扩展的子结点
                cur_node.next_nodes.append(Node(cur_node, action))
                # 加入到当前节点的子节点的子结点列表中
                board._move(action, self.select_color)
                # 在棋盘上移动
                self.select_color = self.changeColor(self.select_color)
                # 更换颜色
                cur_node.next_nodes[-1].unvisited = list(
                    board.get_legal_actions(self.select_color))
                # 初始化新节点的未扩展子结点列表
                return cur_node.next_nodes[-1]
                # 返回新节点
            else:
                best_node = max(cur_node.next_nodes,
                                key=lambda x: x.calUCB1(),
                                default=None)
                if best_node == None:
                    # 如果当前节点没有任何子节点，创建一个虚拟节点并继续搜索
                    cur_node.next_nodes.append(Node(cur_node, (-1, -1)))
                    best_node = cur_node.next_nodes[0]
                    cur_node.next_nodes[0].unvisited = list(
                        board.get_legal_actions(
                            self.changeColor(self.select_color)))
                    # 虚拟节点的 unvisited 列表中包含了所有可能的动作
                    # 更换颜色继续搜索节点
                elif best_node.state != (-1, -1):
                    # 如果选择的不是占位节点，说明有棋可下
                    board._move(best_node.state, self.select_color)
                self.select_color = self.changeColor(self.select_color)
                cur_node = best_node  # 将当前节点设为选择的子结点
        return cur_node

    def Simulate(self, board):
        # 模拟过程
        while not self.game_over(board):
            action_list = list(board.get_legal_actions(self.sim_color))
            if action_list:
                # 有合法落子位置，下棋
                action = random.choice(action_list)
                board._move(action, self.sim_color)
            self.sim_color = self.changeColor(self.sim_color)
            # 即使当前玩家没有合法落子位置，也需要交换颜色
        winner, win_score = board.get_winner()
        # 获得获胜方和获胜子数
        if winner == 0:
            return "X", win_score
        elif winner == 1:
            return "O", win_score
        else:
            return "-", win_score

    def Backpropagation(self, node, winner, win_score):
        # 反向传播过程
        while node != None:
            if self.select_color == winner:
                node.score -= win_score
            # 如果胜利者与选择过程所用棋子颜色相同，过程中所有节点减去赢的得分
            elif self.changeColor(self.select_color) == winner:
                node.score += win_score
            # 如果胜利者与选择过程所用棋子颜色不同，过程中所有节点加上赢的得分
            node.visits += 1  # 该节点访问次数加1
            self.select_color = self.changeColor(self.select_color)
            node = node.parent  # 移动到父节点进行下一次迭代

    def MCTSmain(self, board):
        start_time = datetime.datetime.now()
        self.root = Node(None, (-1, -1))  # 创建根节点
        self.root.unvisited = list(board.get_legal_actions(
            self.color))  #初始根节点未访问节点列表
        while ((datetime.datetime.now() - start_time).seconds <
               60):  # 时间限制在1分钟内
            new_board = copy.deepcopy(board)
            self.select_color = copy.deepcopy(self.color)
            start_node = self.Select(new_board)  #选择
            self.sim_color = copy.deepcopy(self.select_color)  #赋值模拟选手颜色
            winner, win_score = self.Simulate(new_board)  #模拟走棋
            self.Backpropagation(start_node, winner, win_score)  #反向传播
            # 判断是否有多次重复访问的点
            flag = False
            for n in self.root.next_nodes:
                if n.visits > 1000:
                    flag = True
                    break
            if flag:  # 如果存在多次访问的节点，则停止搜索
                break
        if (self.root.next_nodes):
            # 选择UCB1值最大的节点
            tmp = max(self.root.next_nodes,
                      key=lambda x: x.calUCB1(),
                      default=None)
        return tmp.state

    def get_move(self, board):
        # 根据当前棋盘状态获取最佳落子位置
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))
        action = self.MCTSmain(board)
        return action

# 人类玩家黑棋初始化
black_player =  HumanPlayer("X")

# AI 玩家 白棋初始化
white_player = AIPlayer("O")

# 游戏初始化，第一个玩家是黑棋，第二个玩家是白棋
game = Game(black_player, white_player)

# 开始下棋
game.run()