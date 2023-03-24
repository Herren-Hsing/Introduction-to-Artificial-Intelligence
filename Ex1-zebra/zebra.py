from kanren import run, eq, membero, var, conde        # kanren一个描述性Python逻辑编程系统
from kanren.core import lall                           # lall包用于定义规则
import time

def left(m, n, list):
    return membero((m, n), zip(list, list[1:]))

def next(m, n, list):
    return conde([left(m, n, list)], [left(n, m, list)])

class Agent:
    def __init__(self):
        self.units = var()              
        self.rules_zebraproblem = None  # 用lall包定义逻辑规则
        self.solutions = None           # 存储结果

    def define_rules(self):
        self.rules_zebraproblem = lall(
            (eq, (var(), var(), var(), var(), var()), self.units),
            (membero, ('英国人', var(), var(), var(), '红色'), self.units),
            (membero, ('西班牙人', var(), var(), '狗', var()), self.units),
            (membero, ('日本人', '油漆工', var(), var(), var()), self.units),
            (membero, ('意大利人', var(), '茶', var(), var()), self.units),
            (membero, (var(), '外交官', var(), var(), '黄色'), self.units),
            (membero, (var(), '摄影师', var(), '蜗牛', var()), self.units),
            (membero, (var(), var(), '咖啡', var(), '绿色'), self.units),
            (membero, (var(), '小提琴家', '橘子汁', var(), var()), self.units),
            (eq, (var(), var(), (var(), var(), '牛奶', var(), var()), var(), var()), self.units),
            (eq, (('挪威人', var(), var(), var(), var()),var(), var(), var(), var()), self.units),
            (left, (var(), var(), var(), var(), '白色'),(var(), var(), var(), var(), '绿色'), self.units),
            (next, ('挪威人', var(), var(), var(), var()),(var(), var(), var(), var(), '蓝色'), self.units),
            (next, (var(), '医生', var(), var(), var()),(var(), var(), var(), '狐狸', var()), self.units),
            (next, (var(), '外交官', var(), var(), var()),(var(), var(), var(), '马', var()), self.units),
            (membero, (var(), var(), var(), '斑马', var()), self.units),
            (membero, (var(), var(), '矿泉水', var(), var()), self.units)
        )

    def solve(self):
        self.define_rules()
        self.solutions = run(0, self.units, self.rules_zebraproblem)
        return self.solutions
