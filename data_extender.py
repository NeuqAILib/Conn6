# -*- coding: utf-8 -*-
import numpy as np


class ChessExtender():

    def __init__(self,):
        self.board_x = 19
        self.board_y = 19
        self.rev = {i: r for i, r in zip("abcdefghijklmnopqrs".upper(), reversed("abcdefghijklmnopqrs".upper()))}
        self.s2d = {s: d for d, s in enumerate("abcdefghijklmnopqrs".upper())}
        self.d2s = {d: s for d, s in enumerate("abcdefghijklmnopqrs".upper())}
        pass

    def str_state_rot(self, str_state, angle):
        angle = angle % 4
        str_state = str_state.upper()
        result = ''
        if angle == 1:
            for i in range(len(str_state) // 2):
                str_pos = str_state[i * 2:(i + 1) * 2]
                x, y = self.s2d[str_pos[0]] - 9, self.s2d[str_pos[1]] - 9
                x, y = y, -x
                x, y = self.d2s[x + 9], self.d2s[y + 9]
                result += x + y
        elif angle == 2:
            for i in range(len(str_state) // 2):
                str_pos = str_state[i * 2:(i + 1) * 2]
                x, y = self.s2d[str_pos[0]] - 9, self.s2d[str_pos[1]] - 9
                x, y = -x, -y
                x, y = self.d2s[x + 9], self.d2s[y + 9]
                result += x + y
        elif angle == 3:
            for i in range(len(str_state) // 2):
                str_pos = str_state[i * 2:(i + 1) * 2]
                x, y = self.s2d[str_pos[0]] - 9, self.s2d[str_pos[1]] - 9
                x, y = -y, x
                x, y = self.d2s[x + 9], self.d2s[y + 9]
                result += x + y
        else:
            result += str_state

        return result

    def str_state_fliplr(self, str_state):
        result = ''
        str_state = str_state.upper()
        for i in range(len(str_state) // 2):
            str_pos = str_state[i * 2:(i + 1) * 2]
            x = self.rev[str_pos[0]]
            str_pos = x + str_pos[1]
            result += str_pos
        return result

    def get_equi_data(self, play_data):
        # 数据格式:
        # [(state, mcts_porb, v), (state, mcts_porb, v), ..]
        # state: str, exp: "BJJJKKJ", "WJJJK"
        # mcts_porb, ndarray, exp: [.1, .1, .2, ....] 包含19 * 19 个浮点数的numpy数组, shape:(361, )
        # v: int, 1 or -1
        # 返回值:
        # list of ndarray, 与原格式一致(或类似格式)的ndarray组成的python数组  # TODO 能否序列化传输 ??
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                # print(f"TrainPipeline.get_equi_data?:state:{state}")
                equi_state = state[0] + self.str_state_rot(state[1:], i)
                equi_mcts_prob = np.rot90(mcts_porb.reshape(self.board_x, self.board_x), i)
                # tmp_prob = equi_mcts_prob.flatten()
                # print(f"state:{equi_state}\nprob:{tmp_prob}\nwinner:{winner}")
                extend_data.append((equi_state, equi_mcts_prob.flatten(),  winner))

                # flip horizontally
                equi_state = equi_state[0] + self.str_state_fliplr(equi_state[1:])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                # tmp_prob = equi_mcts_prob.flatten()
                # print(f"state:{equi_state}\nprob:{tmp_prob}\nwinner:{winner}")
                extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))

        return extend_data