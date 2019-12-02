# -*- coding: utf-8 -*-
import time

import numpy as np

# from boad import Board


class Game(object):
    # 提供了棋局命令行可视化、自博弈、 人机对弈等接口？？

    def __init__(self, board):
        self.board = board

    def start_play(self, mc_player1, mc_player2, start_player=0, is_shown=False):
        is_shown = True
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 \'B\' first) '
                            'or 1 (player2 \'W\' first)')
        self.board.init_board()

        p1, p2 = self.board.players

        players = {p1: [mc_player1, mc_player2][start_player],
                   p2: [mc_player1, mc_player2][1 - start_player]}
        players[p1].set_player_ind(p1)
        players[p2].set_player_ind(p2)

        # 初始化前三子
        self.board.set_start_board()  # BWW

        if is_shown:
            self.board.show_board()

        if is_shown:
            self.board.show_board()
        while True:
            players[p1].reset_player(p1)
            players[p2].reset_player(p2)
            current_player = self.board.current_player()
            player_in_turn = players[current_player]

            # 下第一子
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            print(self.board.str_states)
            if is_shown:
                self.board.show_board()

            end, winner = self.board.game_end()  # winner = 'B' 'W' 'O'
            if end:
                if winner != 'NULL':  # -1:  # -1 是和棋
                    print("Game.start_play : Game end. Winner is", players[winner])
                else:
                    print("Game.start_play : Game end. Tie")
                return winner

            # 下第二子
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            print(self.board.str_states)
            if is_shown:
                self.board.show_board()
            # if is_shown:
            #     self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()  # winner = 'B' 'W' 'O'
            if end:
                if True:  # is_shown:
                    if winner != 'NULL':  # -1:  # -1 是和棋
                        print("Game.start_play : Game end. Winner is", players[winner])
                    else:
                        print("Game.start_play : Game end. Tie")
                return winner

    def start_self_play(self, mc_player, is_shown=False, temp1=1e-3, temp2=1e-3):
        """ 自博弈产生训练数据，储存格式（s, pi, z）  """
        self.board.init_board()
        p1, p2 = self.board.players  # B W

        # 初始化前三子
        self.board.set_start_board()  # BWW
        # self.board.show_board()

        # 初始化MCT
        mc_player.reset_player('B')

        # 第一子,二子 单局经验池
        first_states, first_mcts_probs, first_current_players = [], [], []
        second_states, second_mcts_probs, second_current_players = [], [], []
        while True:
            time1 = time.time()
            # ##############################第一子:#######################
            move, move_probs = mc_player.get_action(self.board, temp=temp1, return_prob=True)
            # print("think first step :{:.3f}s".format(time.time() - time1))
            # print(type(move_probs))
            first_states.append(self.board.current_player() + self.board.current_str_state())
            first_mcts_probs.append(move_probs)
            first_current_players.append(self.board.current_player())

            self.board.do_move(move)
            # 是否在命令行绘制棋盘
            if is_shown:
                self.board.show_board()

            # 检查是否游戏结束（每次board.do_move之后都检查一次）
            end, winner = self.board.game_end()
            if end:
                # print(f"Game.start_self_play : 游戏结束,收集z的信息...")
                # 如果游戏结束
                first_winners_z = np.zeros(len(first_current_players))
                second_winners_z = np.zeros(len(second_current_players))
                if winner != 'NULL':
                    first_winners_z[np.array(first_current_players) == winner] = 1.0
                    first_winners_z[np.array(first_current_players) != winner] = -1.0

                    second_winners_z[np.array(second_current_players) == winner] = 1.0
                    second_winners_z[np.array(second_current_players) != winner] = -1.0
                    # print(f"Game.start_self_play : Winner is player:{winner}")
                else:
                    print(f"Game.start_self_play : 平局和棋, 不存在这种情况...")

                # reset MCTS root node
                mc_player.reset_player('B')
                # print(f"Game.start_self_play : 重置棋盘...")
                # 是否在命令行绘制棋盘
                if is_shown:
                    if winner != "NULL":
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")

                # 一局结束后，将本局的经验回放数据打包return
                return (winner,
                        zip(first_states, first_mcts_probs, first_winners_z),
                        zip(second_states, second_mcts_probs, second_winners_z),)

            # ########################第二子######################
            time2 = time.time()
            # print(f"Game.start_self_play : 不分胜负,下第二子.......")
            move, move_probs = mc_player.get_action(self.board, temp=temp2, return_prob=True)
            # print("think second step :{:.3f}s".format(time.time() - time2))

            second_states.append(self.board.current_player() + self.board.current_str_state())
            second_mcts_probs.append(move_probs)
            second_current_players.append(self.board.current_player())

            self.board.do_move(move, )

            # 是否在命令行绘制棋盘
            if is_shown:
                self.board.show_board()

            # 检查是否游戏结束（每次board.do_move之后都检查一次）
            end, winner = self.board.game_end()
            if end:
                # print(f"Game.start_self_play : 游戏结束,收集z的信息...")
                # 如果游戏结束
                # 给z赋值(只有在一局游戏结束分出胜负和之后才能给之前的每个（s, pi, z）中的z赋值，结束前并不知道)
                # 若为和棋（winner = -1）：所有z值 = 0
                # 若有winner（winner = 1 或 2）：胜负双方的z分别赋值为 1，-1
                first_winners_z = np.zeros(len(first_current_players))
                second_winners_z = np.zeros(len(second_current_players))
                if winner != 'NULL':
                    first_winners_z[np.array(first_current_players) == winner] = 1.0
                    first_winners_z[np.array(first_current_players) != winner] = -1.0

                    second_winners_z[np.array(second_current_players) == winner] = 1.0
                    second_winners_z[np.array(second_current_players) != winner] = -1.0
                    # print(f"Game.start_self_play :  Winner is player:{winner}")
                else:
                    print(f"Game.start_self_play : 平局和棋...")

                # reset MCTS root node
                mc_player.reset_player('B')

                # 是否在命令行绘制棋盘
                if is_shown:
                    if winner != "NULL":
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")

                # 一局结束后，将本局的经验回放数据打包return 打包格式row行（s, pi, z）
                return (winner,
                        zip(first_states, first_mcts_probs, first_winners_z),
                        zip(second_states, second_mcts_probs, second_winners_z),)
