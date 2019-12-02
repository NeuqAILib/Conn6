import copy
from typing import List, Tuple

from environment import Environment


class Board(object):
    """board for the game
    /*
     * /////////////////////////////////////////////////
     * // //////////////////////// 棋盘表示， 坐标， 一维坐标， 字母
     * //       Y
     * //       ^
     * // S 18  |
     * // R 17  |
     * // Q 16  |
     * // P 15  |
     * // O 14  |
     * // N 13  |
     * // M 12  |
     * // L 11  |
     * // K 10  |
     * // J  9  |
     * // I  8  |
     * // H  7  |
     * // G  6  |
     * // F  5  |
     * // E  4  |
     * // D  3  |
     * // C  2  |  38
     * // B  1  |  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37
     * // A  0  |   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
     * //       |-------------------------------------------------------------------------------> X
     * //          0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
     * //          A   B   C   D   E   F   G   H   I   J   K   L   M   N   O   P   Q   R   S
     */

    """
    env = None

    def __init__(self, ):
        self.width = 19
        self.height = 19

        self.states = {}  # 180:"B", 181:"w"
        self.str_states = ''  # 当前状态的String表示，第二种表示方式(大写字母组合表示)
        self.stone_num = 0

        if not Board.env:
            Board.env = Environment()
        self.env = Board.env  # 新的棋盘和CPP估值函数
        self.character = None
        self.availables = None
        self.CPP_availables = None
        self.is_end = None
        self.winner = None

        self.players = ['B', 'W']  # 进攻方(黑色) 1, 防守方(白色) -1
        self.turn = tuple(['B' if (i % 4 == 0 or i % 4 == 3) else 'W' for i in range(19 * 19)])  # B W W B B W W
        self.first_hand = tuple([True if (i % 4 == 1 or i % 4 == 3) else False for i in range(19 * 19)])

        self.pos2str = {}
        self.str2pos = {}
        for y_index, y in enumerate("abcdefghijklmnopqrs".upper()):
            for x_index, x in enumerate("abcdefghijklmnopqrs".upper()):
                pos = y_index * 19 + x_index
                self.str2pos[f"{x}{y}"] = pos
                self.pos2str[pos] = f"{x}{y}"

    def init_board(self, ):
        """
        重新初始化棋盘
        """
        self.states = {}
        self.str_states = ''  # 表示当前的状态表示，第二种表示方式(大写字母组合表示)
        self.stone_num = 0

        self.character = None
        # keep available moves in a list 可以落子的位置
        self.availables = list(range(self.width * self.height))  # 是否缩小范围?
        self.CPP_availables = list(range(self.width * self.height))  # c++给出的可选动作范围?

        self.is_end = False
        self.winner = 'NULL'

    def next_player(self) -> str:
        assert 0 < self.stone_num + 1 < 19 * 19, f"'{self.stone_num + 1}'超出合法范围"
        # 下一个玩家
        return self.turn[self.stone_num + 1]

    def current_player(self) -> str:
        # 当前动作的玩家
        return self.turn[self.stone_num]

    def prev_player(self) -> str:
        # 上一手动作的玩家
        assert 0 <= self.stone_num - 1 < 19 * 19, f"'{self.stone_num - 1}'超出合法范围"
        return self.turn[self.stone_num - 1]

    def is_first_hand(self) -> bool:
        # 当前玩家是否是第一手(2子分1,2手)
        return self.first_hand[self.stone_num]

    def cpp_available(self) -> list:
        return self.CPP_availables[:]

    def move_to_location(self, move: int) -> Tuple[int, int]:
        """
        一维坐标 —> 二维的坐标
        3*3 board's moves like:
          ^ y
        1 \ 3 4 5
        0 \ 0 1 2
          \————————>x
            0 1 2
        and move 5's location is (2, 1)
        """
        x = move % self.width
        y = move // self.width
        return (x, y)

    def location_to_move(self, location: Tuple[int, int]) -> int:
        """
        二维坐标 转换-> 一维坐标
        """
        if len(location) != 2:
            return -1
        x = location[0]
        y = location[1]
        move = y * self.width + x
        if move not in range(self.width * self.height):
            assert 0 <= move < self.width * self.height, "转换坐标非法"
            return -1
        return move

    def current_state(self):
        # 从当前玩家的角度返回棋盘状态
        return self.character

    def current_str_state(self) -> str:
        # 从当前玩家的角度返回棋盘状态
        return self.str_states

    def set_start_board(self, GUI_msg=None):
        self.init_board()
        if GUI_msg:
            # 处理gui传回来的局面信息
            # BJJJKKJ   WJJJK
            GUI_msg = GUI_msg.upper()
            print(GUI_msg)
            player = GUI_msg[0]
            assert player == 'B' or player == 'W', "GUI msg player ERROR!"
            board_info = GUI_msg[1:]
            # print(board_info)
            stone_num = len(board_info) // 2
            # print(stone_num)
            move_list = []
            for i in range(stone_num):
                str_move = board_info[2 * i:2 * i + 2]
                # print(str_move)
                move_list.append(self.str2pos[str_move])

            for move in move_list:
                self.states[move] = self.current_player()
                self.str_states += self.pos2str[move]
                self.stone_num += 1
                self.availables.remove(move)
            curr_player = self.current_player()
            assert curr_player == player, "GUI curr player ERROR!"
            self.character, self.is_end, self.CPP_availables, reward = self.env.tss_interact(curr_player,
                                                                                             self.str_states)
            if reward == 1:
                self.winner = self.current_player()
            else:
                self.winner = "NULL"
        else:
            # 处理开局的前三颗子,仅训练/评估模式
            w1_start, w2_Start = self.env.choice_random_opening(180)
            # print("w1:{}, w2{}".format(self.move_to_location(w1_start), self.move_to_location(w2_Start)))

            for move in [180, w1_start, w2_Start]:
                self.states[move] = self.current_player()
                self.str_states += self.pos2str[move]
                self.stone_num += 1
                self.availables.remove(move)

            # 处理棋盘特征
            curr_player = self.current_player()
            assert curr_player == 'B', "set start board error!"
            self.character, _, self.CPP_availables, reward = self.env.tss_interact(curr_player, self.str_states)
            self.is_end = False
            self.winner = "NULL"

    def do_move(self, move: int):
        # 在棋盘上落子，并切换当前活动玩家
        pos = self.move_to_location(move)
        assert 0 <= move < self.width * self.height, f"move 不合法:{move}:{pos}"
        # print(f"Board.do_move : move->{move}:{pos}")
        self.states[move] = self.current_player()  # 记录state
        if move not in self.availables:
            print(f"move:{move}")
            print(f"avlbs:{self.availables}")
            print(f"curr:{self.current_player()}")
            print(f"hand:{self.is_first_hand()}")
            print(f"state:{self.states}")
            print(f"str:{self.str_states}")
            print(f"cpp:{self.CPP_availables}")
            assert False
        self.availables.remove(move)
        # print(f"Board.do_move : {self.current_player()}->{move}:{pos}")
        self.str_states += self.pos2str[move]  # 把落子序列保存，后面copy棋盘时用到
        # print(f"Board.do_move : str_states:{self.str_states}")

        self.stone_num += 1
        # print(f"Board.do_move : stone_num:{self.stone_num}")
        # 调用C++更新局面, 把解析的state存起来,更新self.availables列表, 更新is_end值,更新has_a_winner值
        # np.array(character).reshape((18, 19, 19)), is_end, action_list, reward,
        curr_player = self.current_player()
        self.character, self.is_end, self.CPP_availables, reward = self.env.tss_interact(curr_player, self.str_states)
        if len(self.CPP_availables) == 0:
            self.CPP_availables = self.availables[:]
            print("let's random")
        if reward == 1:
            self.winner = self.current_player()
        else:
            self.winner = "NULL"
        # print(f"Board.do_move : is_end:{self.is_end}, winner:{self.winner}")

    def undo_move(self, move: int):
        # 在棋盘上落子，并切换当前活动玩家
        pos = self.move_to_location(move)
        assert 0 <= move < self.width * self.height, f"move 不合法:{move}:{pos}"
        if self.states.get(move):
            del self.states[move]
            self.availables.append(move)
            self.str_states = self.str_states[:-2]
            self.stone_num -= 1
            curr_player = self.current_player()
            # print(f"now player:{curr_player}")
            self.character, self.is_end, self.CPP_availables, reward = self.env.tss_interact(curr_player,
                                                                                             self.str_states)
            if reward == 1:
                self.winner = self.current_player()
            else:
                self.winner = "NULL"
        else:
            print(f"no move :{move} in move")

    def game_end(self) -> Tuple[bool, str]:
        # 对局是否结束，胜方是谁（1 ：player1， 2 ：player2, -1 ：未分胜负）
        if self.is_end:
            return True, self.winner
        return False, 'NULL'

    def deepcopy(self):
        copy_board = Board()
        copy_board.states = copy.deepcopy(self.states)
        copy_board.str_states = self.str_states[:]
        copy_board.stone_num = self.stone_num
        copy_board.env = self.env  # Environment()

        copy_board.character = self.character.copy()
        copy_board.availables = self.availables[:]
        copy_board.CPP_availables = self.CPP_availables[:]

        copy_board.is_end = self.is_end
        copy_board.winner = self.winner[:]

        return copy_board

    def show_board(self):
        # 棋盘可视化

        width = self.width
        height = self.height

        print("   ", end='')
        for x in range(width):
            print("{0:3}".format(x), end='')
        print()
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = self.states.get(loc, -1)
                if p == 'B':
                    print('B'.center(3), end='')
                elif p == 'W':
                    print('W'.center(3), end='')
                else:
                    print('_'.center(3), end='')
            print()

    def release(self):
        if self.env:
            self.env.close()

# if __name__ == "__main__":
#     try:
#         b = Board()
#         b.init_board()
#         b.set_start_board("BJJJKKJLKKLIJJI")
#         b.show_board()
#         b.do_move(0)
#         b.show_board()
#         b.undo_move( 0)
#         b.show_board()
#         b.do_move(7)
#         b.show_board()
#         b.do_move(8)
#         b.show_board()
#         b.undo_move(7)
#         b.show_board()
#         b.undo_move(8)
#         b.show_board()
#
#     except Exception as e:
#         print(e)
#     finally:
#         b.env.close()
