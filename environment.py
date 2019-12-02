# -*- coding: utf-8 -*-
import random
import time
import re
import subprocess

import numpy as np

class Environment:
    max_call_time = 30000
    # txt_file_path = './tmp_files/'
    board_x = 19
    board_y = 19
    character_num = 2*4*12  # 特征平面数量
    drlinfo_line = character_num + 1  # 特征和走法
    analysis_line = character_num
    log_file = f"./log/{time.strftime('%Y%m%d_%H.%M.%S', time.localtime())}_cpp.log"

    def __init__(self, obj=None):
        self.n_used = 0
        self.obj = obj if obj else Environment.get_obj()
        self.white_opening = np.array([
            #     +
            #     o +
            np.array(((+1, 0), (0, +1))),
            np.array(((0, +1), (-1, 0))),
            np.array(((-1, 0), (0, -1))),
            np.array(((0, -1), (+1, 0))),
            #       +
            #     o +
            np.array(((+1, 0), (+1, +1))),
            np.array(((+1, +1), (0, +1))),
            np.array(((0, +1), (-1, +1))),
            np.array(((-1, +1), (-1, 0))),
            np.array(((-1, 0), (-1, -1))),
            np.array(((-1, -1), (0, -1))),
            np.array(((0, -1), (+1, -1))),
            np.array(((+1, -1), (+1, 0))),
            #       +
            #     o   +
            np.array(((+2, 0), (+1, +1))),
            np.array(((+2, 0), (+1, -1))),
            np.array(((0, +2), (+1, +1))),
            np.array(((0, +2), (-1, +1))),
            np.array(((-2, 0), (-1, +1))),
            np.array(((-2, 0), (-1, -1))),
            np.array(((0, -2), (-1, -1))),
            np.array(((0, -2), (+1, -1))),
        ])
        self.pos2str = {}
        self.str2pos = {}
        for y_index, y in enumerate("abcdefghijklmnopqrs".upper()):
            for x_index, x in enumerate("abcdefghijklmnopqrs".upper()):
                pos = y_index * 19 + x_index
                self.str2pos[f"{x}{y}"] = pos
                self.pos2str[pos] = f"{x}{y}"

    @staticmethod
    def print_log(*msgs):
        pass  # 不输出了, io太慢
        with open(Environment.log_file, 'a', encoding='utf-8') as f:
            for msg in msgs:
                f.write(f"{time.strftime('[%H:%M:%S]', time.localtime())} {str(msg)}\n")
            f.close()

    @staticmethod
    def get_obj():
        # 创建一个CPP 的子进程
        print("New Subprocess!")
        Environment.print_log('New Subprocess!')
        return subprocess.Popen(["./delta_conn6_server_windows.exe"],  # ./delta_conn6_server_linux.exe
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=None,  # stderr不捕获,防止堵塞
                                universal_newlines=True,  # stdin可以为str
                                )

    def peocess_win_states(self, GUI_msg):
        time_start = time.time()
        GUI_msg = GUI_msg.upper()
        player = GUI_msg[0]
        assert  player == 'B' or player == 'W', "msg palyer ERROR"
        str_action = GUI_msg[1:]

        print('HELP' + ' ' + player + ' ' + str(str_action))
        self.obj.stdin.write('HELP' + ' ' + player + ' ' + str(str_action) + '\n')
        self.obj.stdin.flush()

        # 返回的信息格式为: action1, action2,reword,需要2个动作, 1个值
        msg = self.obj.stdout.readline()
        print(f"{msg}")

        pattern_num = r"[+-]{0,1}\d{1,3}"
        pattern = re.compile(pattern_num)
        result = pattern.findall(msg)

        # 这里面存的是里面所有数字组成的list
        result = list(map(int, result))

        if len(result) != 3:
            print(f"result:{result}")
        try:
            assert len(result) == 3, "HELP len ERROR:{}".format(len(result))
        except Exception as e:
            print(e)
            print("HELP 动作 失败, 命令为:"+'HELP' + ' ' + player + ' ' + str(str_action))
            self.print_log("HELP 动作 失败, 命令为:"+'HELP' + ' ' + player + ' ' + str(str_action))
            self.close()
            self.obj = self.get_obj()
            print('HELP' + ' ' + player + ' ' + str(str_action))
            self.obj.stdin.write('HELP' + ' ' + player + ' ' + str(str_action) + '\n')
            self.obj.stdin.flush()

            # 返回的信息格式为: action1, action2,reword,需要2个动作, 1个值
            msg = self.obj.stdout.readline()
            print(f"{msg}")

            pattern_num = r"[+-]{0,1}\d{1,3}"
            pattern = re.compile(pattern_num)
            result = pattern.findall(msg)

            # 这里面存的是里面所有数字组成的list
            result = list(map(int, result))
        finally:
            print("ERROR process done!")
            assert len(result) == 3, "HELP len ERROR:{}".format(len(result))

        move1, move2, reward = result

        assert 0<=move1<=360 and 0<=move2<=360, "help move error"

        # assert reward == 1, "竟然不是一直赢???, What??{}".format(reward) TODO

        is_end = True

        time_end = time.time()
        time_cost = time_end - time_start
        # print(f"total time cost {time_cost} s")
        if time_cost > 5:
            Environment.print_log(f"help time out: {time_cost}")

        return  move1, move2

    def tss_interact(self, player: "str 'B' or 'W'", str_action: "str record"):
        time_start = time.time()
        # if len(str_action) <= 4:
        #     return self.start_board_action(player, str_action)
        if self.n_used > Environment.max_call_time:  # 防止内存泄漏,及时重启CPP
            self.close()
            self.obj = Environment.get_obj()
            self.n_used = 1
            print("Environment.interact :  reset the obj!")
            Environment.print_log("Environment.interact :  reset the obj!")

        # DRLINFO B JJ
        # print('DRLINFO' + ' ' + player + ' ' + str(str_action))# + '\n')
        Environment.print_log('DRLINFO' + ' ' + player + ' ' + str(str_action))
        self.obj.stdin.write('DRLINFO' + ' ' + player + ' ' + str(str_action) + '\n')
        self.obj.stdin.flush()
        self.n_used += 1

        # 返回的信息格式为: action, action, action .... .... , reword,
        # print(f"Environment.tss_process : 读取全部内容 :\n")
        msg = ""
        for i in range(Environment.drlinfo_line):  # 一定记得读的行数和输出行数一致，读少了会漏数据， 读多了会等待obj向stdout输入数据
            ## #总计 12 * 4 * 2 张棋盘，每张一行 + 1行动作列表+胜负（1表示胜）
            # 所以一共97行数据
            msg += self.obj.stdout.readline()
        # print(f"{msg}")

        pattern_num = r"[+-]{0,1}\d{1,3}"
        pattern = re.compile(pattern_num)
        result = pattern.findall(msg)

        # print(f"Environment.tss_process : pattern_result :\n {result}")
        # 这里面存的是里面所有数字组成的list
        result = list(map(int, result))

        reward = result[-1]
        character = result[:Environment.character_num * Environment.board_x * Environment.board_y]
        action_list = result[Environment.character_num * Environment.board_x * Environment.board_y:-1]
        random.shuffle(action_list)

        if len(character) != Environment.character_num * Environment.board_x * Environment.board_y:
            print(f"{player} {str_action}")
            print("tss msg ERROR")
        # 此时需要 baord给出下一步指导, 因为环境只能判断胜, 若出现平局, 仍须落子到棋盘满为止,此阶段无训练意义环境不做指导
        if len(action_list) == 0:  # TODO
            print(f"{player} {str_action}")
            print(">>??????????>>")
            print("maybe draw!!")
            reward = 0

        if reward == 1:  # 1， -1 胜， 负
            is_end = True
        else:
            is_end = False  # 未知胜负

        time_end = time.time()
        time_cost = time_end - time_start
        # print(f"total time cost {time_cost} s")
        if time_cost > 1:
            Environment.print_log(f"cpp time out: {time_cost}")
        # print(f"Environment.search_process : is_end:{is_end} , action_list{action_list}")
        if len(result) <= 19 * 19 * self.character_num:
            print("info, len:{}, {}".format(len(result), result))
            print("引起管道破裂的命令:"+'DRLINFO' + ' ' + player + ' ' + str(str_action))
            # cnt = 0
            # while True:
            #     if cnt > 10:
            #         print("放弃重建管道")
            #         net_data = np.zeros((Environment.character_num, Environment.board_x, Environment.board_y))
            #         is_end = 1
            #         action_list = []
            #         reward = 0
            #         break
            #     if len(character) != Environment.character_num * Environment.board_x * Environment.board_y:
            #         print("管道破裂, 重试:{}".format(cnt))
            #     try:
            #         # self.close()
            #         self.obj = self.get_obj()
            #         return self.tss_interact(player, str_action)
            #     except Exception as e:
            #         print(e)
            #         cnt += 1
            #         print("重试失败")
        net_data = np.array(character).reshape((Environment.character_num, Environment.board_x, Environment.board_y))
        return net_data, is_end, action_list, reward,

    def analysis_interact(self, player: "str 'B' or 'W'", str_action: "str record"):
        time_start = time.time()
        # ANALYSIS B JJ
        # print('ZERO' + ' ' + player + ' ' + str(str_action))# + '\n')/home/mrloading/projects/CLionProjects/delta_conn6_server/cmake-build-release/delta_conn6_server
        Environment.print_log('ZERO' + ' ' + player + ' ' + str(str_action))
        self.obj.stdin.write('ZERO' + ' ' + player + ' ' + str(str_action) + '\n')
        self.obj.stdin.flush()

        return self.analysis_process(time_start)


    def analysis_process(self, time_start):
        # 返回的信息格式为: num, num, num .... .... , num,
        # print(f"Environment.analysis_process : 读取全部内容 :\n")
        msg = ""
        for i in range(Environment.analysis_line):  # 一定记得读的行数和输出行数一致，读少了会漏数据， 读多了会等待obj向stdout输入数据
            msg += self.obj.stdout.readline()

        pattern_num = r"[+-]{0,1}\d{1,3}"
        pattern = re.compile(pattern_num)
        result = pattern.findall(msg)

        # 这里面存的是里面所有数字组成的list
        result = list(map(int, result))
        # print(f"[Environment.analysis_process] : 正则处理结果result:")
        # for row in range(Environment.character_num):
        #     print(f"[Environment.analysis_process] :\n")
        #     for i in range(19):
        #         print(f"{result[row*19*19 + i*19:(row)*19*19 + (i+1)*19]}\n")

        time_end = time.time()
        time_cost = time_end - time_start
        # print(f"total time cost {time_cost} s")
        if time_cost > 1:
            Environment.print_log(f"cpp time out: {time_cost}")
        # print(f"Environment.file_process : is_end:{is_end} , action_list{action_list}")
        # [黑 / 白(落子方在前)][进攻 / 防守][4 方向][x][y] + 4 * [最近棋盘记录(当前视角)]
        if len(result) != 19 * 19 * self.character_num:
            print(result)
        return np.array(result).reshape((Environment.character_num, Environment.board_x, Environment.board_y))

    def choice_random_opening(self, black_stone):
        assert 0 <= black_stone < 361, "stone pos error!"

        stone = np.array(((black_stone % 19, black_stone // 19),
                          (black_stone % 19, black_stone // 19)))

        opening = self.white_opening + stone

        filtered_opening = []
        for pair in opening:
            white1, white2 = pair
            if (0 <= white1[0] < 19) and (0 <= white1[1] < 19) \
                    and (0 <= white2[0] < 19) and (0 <= white2[1] < 19):
                filtered_opening.append(pair)

        filtered = np.array(filtered_opening)

        white1, white2 = filtered[np.random.choice(filtered.shape[0])]

        return white1[0] * 19 + white1[1], white2[0] * 19 + white2[1]

    def close(self):
        if self.obj:
            # print("Environment.close : bing close obj CPP!")
            self.obj.stdin.flush()
            self.obj.stdin.write('EXIT\n')  # NEW  # 退出QUIT DRLQUIT
            self.obj.stdin.flush()
            self.obj.stdin.close()
            time.sleep(0.1)
            print("Environment.close : close obj CPP!")
            Environment.print_log("Environment.close : close obj CPP.")
        else:
            print("Environment.close : obj CPP already closed!")
#
# if __name__ == "__main__":
#     env = Environment()
#     env.tss_interact("B", "JJIKJLHJIIKKGG")
#     r = env.analysis_interact("b", "JJIKJLHJIIKKGG")
#     for i in r:
#         print("-------")
#         print(i)
#     env.close()
