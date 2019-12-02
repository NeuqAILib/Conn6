import glob
import os
import time
import random
from collections import deque

import numpy as np

from board import Board
from game import Game
from mcts_player import MCTSPlayer
from net import PolicyValueNet
# from net_curr import PolicyValueNet as CurrPolicyValueNet
# from net_curr import PolicyValueNet as BestPolicyValueNet
# from net_best import PolicyValueNet as BestPolicyValueNet
from data_extender import ChessExtender


class TrainPipeline():
    def __init__(self,
                 init_first=None,
                 init_second=None,
                 ):
        self.board_width = 19  # 棋盘宽高
        self.board_height = 19

        self.board = Board()
        self.game = Game(self.board)
        # training params
        self.learn_rate = 1e-4  # 一般在[e-3,e-6]学习率
        self.first_lr_multiplier = 1.0  # 根据KL散度自动调整学习率,调整时变化倍数,初始为1.0,程序会自动调整
        self.second_lr_multiplier = 1.0
        self.temp1 = 1  # 温度1 param
        self.temp2 = 0.9  # 温度2 param
        self.c_puct = 10  # 超参数, Cpuct越大探索越多

        self.buffer_size = 10000  # 经验池大小,这里我们需要第一子\第二子共两个经验池,大小相同(暂定)
        self.data_buffer_first = deque(maxlen=self.buffer_size)  # 第1子 双向队列 经验池
        self.data_buffer_second = deque(maxlen=self.buffer_size)  # 第2子 双向队列 经验池

        self.n_playout = 50  # 200  # 400  每一步 simulations 次数,建议在[1e2, 1e4]之间,
        self.batch_size = 64  # 512  # mini-batch size for training
        self.play_batch_size = 2  # 每次自博弈收集数据时进行的局数
        self.epochs = 1  # 一个batch train 5次
        self.game_batch_num = 15000  # 模型训练次数
        self.kl_targ = 0.02  # 离散程度,用来调整学习率,如果一次训练前后模型lk过于发散,超出阈值,我们将提前终止本次训练,并调整学习率

        self.save_freq = 1  # 5
        self.update_manager_freq = 2 * self.save_freq
        self.evaluate_freq = 2 * self.update_manager_freq
        self.saved_evaluate_time = 0
        self.evaluate_times = 2  # 00  # 用于评估模型的棋力的次数
        self.episode_len = 0  # 一次自博弈序列长度
        self.best_win_ratio = 0.0  # 当前最佳胜率,用于判断最新模型和历史最优模型的优劣

        self.current1_dir = './save/current1/'  # current1.model'
        self.current2_dir = './save/current2/'  # current2.model'
        self.best1_dir = './save/best1/'  # best1.model'
        self.best2_dir = './save/best2/'  # best2.model'
        self.data_reg_dir = r'./data/*.npy'
        self.extender = ChessExtender()
        self.wait_time = 1
        self.wait_mul = 2
        self.min_wait_time = 1
        self.max_wait_time = 8

        if init_first:
            # print(f"载入模型路径：{init_first}")
            # 从一个训练好的policy-value net继续训练
            self.current1_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_first, DEVICES="3")
            self.best1_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_first, DEVICES="3")
        else:
            # 训练一个新的first-policy-value net
            self.current1_net = PolicyValueNet(self.board_width, self.board_height, DEVICES="3")
            # self.best1_net = PolicyValueNet(self.board_width, self.board_height, DEVICES="3")
        if init_second:
            # print(f"载入模型路径：{init_second}")
            # 从一个训练好的policy-value net继续训练
            self.current2_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_second, DEVICES="3")
            self.best2_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_second, DEVICES="3")
        else:
            # 训练一个新的second-policy-value net
            self.current2_net = PolicyValueNet(self.board_width, self.board_height, DEVICES="3")
            # self.best2_net = PolicyValueNet(self.board_width, self.board_height, DEVICES="3")

        self.mcts_player = MCTSPlayer(self.current1_net.policy_value_fn, self.current2_net.policy_value_fn,
                                      c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=True)

    def load_data(self, data_reg_dir):
        file_list = sorted([i for i in glob.glob(data_reg_dir)], reverse=True)
        rec_len = 100 if len(file_list) > 100 else len(file_list)
        file_list = file_list[:rec_len]
        for i in file_list:
            data = np.load(i, allow_pickle=True)
            winner, first_data, second_data = data
            self.data_buffer_first.extend(self.extender.get_equi_data(first_data))
            self.data_buffer_second.extend(self.extender.get_equi_data(second_data))
        print(f"done load start data:{rec_len}")

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            time1 = time.time()
            winner, first_data, second_data = self.game.start_self_play(self.mcts_player, is_shown=False, temp1=self.temp1, temp2=self.temp2)
            print("finish one game cost:{:.3f}".format(time.time() - time1))
            first_play_data = list(first_data)[:]
            second_play_data = list(second_data)[:]
            self.episode_len = len(first_play_data) + len(second_play_data)

            first_play_data = self.extender.get_equi_data(first_play_data)
            second_play_data = self.extender.get_equi_data(second_play_data)

            self.data_buffer_first.extend(first_play_data)
            self.data_buffer_second.extend(second_play_data)

    def analysis_data(self, str_data_list):
        character_data = []
        for str_data in str_data_list:
            # 默认str_data 第一个字符表示当前数据下的玩家 BJJJKKJ
            character_data.append(self.board.env.analysis_interact(str_data[0], str_data[1:]))
        return character_data

    def policy_update(self, net='first'):
        net = net.upper()
        if net == "FIRST":
            data_buffer = self.data_buffer_first
            policy_value_net = self.current1_net
            lr_multiplier = self.first_lr_multiplier
        elif net == "SECOND":
            data_buffer = self.data_buffer_second
            policy_value_net = self.current2_net
            lr_multiplier = self.second_lr_multiplier
        else:
            print("Net Name ERROR!!!")
            return

        mini_batch = random.sample(data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        state_batch = self.analysis_data(state_batch)
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = policy_value_net.policy_value(state_batch, False)

        loss, entropy, new_probs, new_v, kl = None, None, None, None, None
        for i in range(self.epochs):
            loss, entropy = policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * lr_multiplier)
            new_probs, new_v = policy_value_net.policy_value(state_batch, False)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly 如果D_KL严重发散，则提前停止
                print(net, "D_KL严重发散，提前停止本次训练")
                break

        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and lr_multiplier > 0.1:
            lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and lr_multiplier < 10:
            lr_multiplier *= 1.5

        if net == "FIRST":
            self.first_lr_multiplier = lr_multiplier
        elif net == "SECOND":
            self.second_lr_multiplier = lr_multiplier

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))

        print("kl:{:.5f} | 1_lr_mul:{:.3f} | 2_lr_mul:{:.3f} | ".format(
            kl, self.first_lr_multiplier, self.second_lr_multiplier), end="")
        print("{} entropy:{:.10f} | {} LOSS:{:.10f}".format(net, loss, net, entropy))

    def policy_evaluate(self, n_games=10):

        current_mcts_player = MCTSPlayer(self.current1_net.policy_value_fn, self.current2_net.policy_value_fn,
                                         c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=True)
        best_mcts_player = MCTSPlayer(self.best1_net.policy_value_fn, self.best2_net.policy_value_fn,
                                      c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=True)

        win_cnt = {"win": 0, "lose": 0, "draw": 0}
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          best_mcts_player,
                                          start_player=i % 2,
                                          is_shown=True)
            if winner == "NULL":
                win_cnt["draw"] += 1
                print("curr draw")
            elif i % 2 == 0 and winner == "B":
                win_cnt["win"] += 1
                print("curr win")

            elif i % 2 == 1 and winner == "W":
                win_cnt["win"] += 1
                print("curr win")

            else:
                print("curr lose")

                win_cnt["lose"] += 1
        win_ratio = 1.0 * (win_cnt["win"] + 0.5 * win_cnt["draw"]) / n_games
        print(
            f"net:evaluate:{self.evaluate_times}, win: {win_cnt['win']}, lose: {win_cnt['lose']}, tie:{win_cnt['draw']}")
        return win_ratio

    def release(self):
        self.board.release()

    def run(self):
        time_start = time.time()
        try:
            # 载入起始数据
            self.load_data(self.data_reg_dir)
            # Environment.remove_txt_in_dir()
            for i in range(self.game_batch_num):
                print("开始收集自博弈数据：")
                self.collect_selfplay_data(self.play_batch_size)
                print(
                    f"数据收集结束，当前first buffer：{len(self.data_buffer_first)}, second buffer：{len(self.data_buffer_second)}")
                # 　更新第一子网络
                if len(self.data_buffer_first) > self.batch_size:
                    print("更新第一子网络")
                    print(f"first_buff:{len(self.data_buffer_first)} | "
                          f"batch_size:{self.batch_size} | train FIRST")
                    self.policy_update(net="FIRST")
                    #########################################################
                    print("#############################")
                    print("第一子网络已经更新，清空缓存字典")
                    self.mcts_player.clear_cache_macts()
                    incache, not_incache, a = self.mcts_player.in_not_in()
                    print(incache, not_incache, a)
                    ########################################################
                else:
                    print(f"first_buff:{len(self.data_buffer_first)} | not enough data")

                # 　更新第二子网络
                if len(self.data_buffer_second) > self.batch_size:
                    print("更新第二子网络")
                    print(f"second_buff:{len(self.data_buffer_second)} |"
                          f" batch_size:{self.batch_size} | train SECOND")
                    self.policy_update(net="SECOND")
                    ##########################################################
                    print("#############################")
                    print("第2子网络已经更新，清空缓存字典")
                    self.mcts_player.clear_cache_macts()
                    incache, not_incache, a = self.mcts_player.in_not_in()
                    print(incache, not_incache, a)
                    #########################################################
                else:
                    print(f"second_buff:{len(self.data_buffer_second)} | not enough data")

                # 保存最新
                if (i + 1) % self.save_freq == 0:
                    print("保存最新")
                    print(f"current self-play batch: {i + 1}, save current net")
                    self.current1_net.save_model(self.current1_dir, "current1.model")
                    self.current2_net.save_model(self.current2_dir, "current2.model")

                # 人为评估一次
                if (i + 1) % self.evaluate_freq == 0:
                    print("人为评估一次")
                    win_ratio = self.policy_evaluate()
                    self.best_win_ratio = win_ratio
                    if win_ratio > 0.55:
                        print("GREAT NET New best policy!!!!!!!!")
                    else:
                        print("BAD NET,pass!!!!!!!!")
                    self.current1_net.save_model(self.best1_dir, f"best1_{self.save_freq % 5}.model")
                    self.current2_net.save_model(self.best2_dir, f"best2_{self.save_freq % 5}.model")
                    # 更新评估用模型
                    print("更新评估用模型")
                    self.best1_net.restore_model(os.path.join(self.best1_dir, f"best1_{self.save_freq % 5}.model"))
                    self.best2_net.restore_model(os.path.join(self.best2_dir, f"best2_{self.save_freq % 5}.model"))
                    self.saved_evaluate_time += 1

        except KeyboardInterrupt:
            print('\n\rquit')
        finally:
            time_end = time.time()
            print('total train time cost {:.1f} s'.format(time_end - time_start))
            self.release()

if __name__ == '__main__':
    training_pipeline = TrainPipeline(
        # init_first='./save/current1/current1.model',
        # init_second='./save/current2/current2.model'
    )
    training_pipeline.run()

