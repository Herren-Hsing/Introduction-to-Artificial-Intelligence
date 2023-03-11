from kanren import run, eq, membero, var, conde        # kanren一个描述性Python逻辑编程系统
from kanren.core import lall                           # lall包用于定义规则
import time

def left(m, n, list):
    return membero((m, n), zip(list, list[1:]))


def next(m, n, list):
    return conde([left(m, n, list)], [left(n, m, list)])


class Agent:
    """
    推理智能体.
    """

    def __init__(self):
        """
        智能体初始化.
        """

        self.units = var()              # 单个unit变量指代一座房子的信息(国家，工作，饮料，宠物，颜色)
        # 例如('英国人', '油漆工', '茶', '狗', '红色')即为正确格式，但不是本题答案
        # 请基于给定的逻辑提示求解五条正确的答案
        self.rules_zebraproblem = None  # 用lall包定义逻辑规则
        self.solutions = None           # 存储结果

    def define_rules(self):
        """
        定义逻辑规则.
        """

        self.rules_zebraproblem = lall(
            # self.units共包含五个unit成员，即每一个unit对应的var都指代一座房子(国家，工作，饮料，宠物，颜色)
            (eq, (var(), var(), var(), var(), var()), self.units),
            # 各个unit房子又包含五个成员属性: (国家，工作，饮料，宠物，颜色)

            # 国家、工作、饮料、宠物、颜色  这是self.units的成员之一
            (membero, ('英国人', var(), var(), var(), '红色'), self.units),
            (membero, ('西班牙人', var(), var(), '狗', var()), self.units),
            (membero, ('日本人', '油漆工', var(), var(), var()), self.units),
            (membero, ('意大利人', var(), '茶', var(), var()), self.units),
            (membero, (var(), '外交官', var(), var(), '黄色'), self.units),
            (membero, (var(), '摄影师', var(), '蜗牛', var()), self.units),
            (membero, (var(), var(), '咖啡', var(), '绿色'), self.units),
            (membero, (var(), '小提琴家', '橘子汁', var(), var()), self.units),
            # 中间那个房子的人喜欢喝牛奶  喝牛奶的这个房子在所有的unit成员中居中
            # self.units与这样描述的五个unit成员等价
            (eq, (var(), var(), (var(), var(), '牛奶',
             var(), var()), var(), var()), self.units),
            # 挪威人住在左边的第一个房子里
            (eq, (('挪威人', var(), var(), var(), var()),
             var(), var(), var(), var()), self.units),
            # 绿房子在白房子的右边
            (left, (var(), var(), var(), var(), '白色'),
             (var(), var(), var(), var(), '绿色'), self.units),
            # 挪威人住在蓝色的房子旁边
            (next, ('挪威人', var(), var(), var(), var()),
             (var(), var(), var(), var(), '蓝色'), self.units),
            # 养狐狸的人与医生房子相邻
            (next, (var(), '医生', var(), var(), var()),
             (var(), var(), var(), '狐狸', var()), self.units),
            # 养马的人与外交官房子相邻
            (next, (var(), '外交官', var(), var(), var()),
             (var(), var(), var(), '马', var()), self.units),
            # 未列出的：
            (membero, (var(), var(), var(), '斑马', var()), self.units),
            (membero, (var(), var(), '矿泉水', var(), var()), self.units)

        )

    def solve(self):
        """
        规则求解器(请勿修改此函数).
        return: 斑马规则求解器给出的答案，共包含五条匹配信息，解唯一.
        """

        self.define_rules()
        self.solutions = run(0, self.units, self.rules_zebraproblem)
        return self.solutions
