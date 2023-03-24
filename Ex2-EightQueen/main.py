import numpy as np           # 提供维度数组与矩阵运算
import copy                  # 从copy模块导入深度拷贝方法
from board import Chessboard

# 基于棋盘类，设计搜索策略
class Game:
    def __init__(self, show = True):
        self.chessBoard = Chessboard(show)
        self.solves = []
        self.gameInit()
        
    def gameInit(self, show = True):
        self.Queen_setRow = [-1] * 8
        self.chessBoard.boardInit(False)
        
    def conflict(self, row, list):
        for i in range(row):
            if list[i] == list[row] or abs(list[row] - list[i]) == abs(row - i):
                return True
        return False
    
    def queens(self, row, list):
        solutions = []
        if row == 8 :
            solutions.extend(list)
            self.solves.append(solutions)
            return
        for i in range (8):
            list[row] = i
            if not self.conflict(row, list):
                self.queens(row + 1, list)
        
    def run(self, row = 0 ):
        list = [0] * 8
        self.queens(row, list)
    
    def showResults(self, result):
        self.chessBoard.boardInit(False)
        for i,item in enumerate(result):
            if item >= 0:
                self.chessBoard.setQueen(i,item,False)
        
        self.chessBoard.printChessboard(False)
    
    def get_results(self):
        self.run()
        return self.solves
   
game = Game()
solutions = game.get_results()
print('There are {} results.'.format(len(solutions)))
print(solutions)