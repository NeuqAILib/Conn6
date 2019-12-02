# -*- coding: utf-8 -*-
import numpy as np

import json


def softmax(x):
    probs = np.exp(x - np.max(x))  # (x - max(x)) 是为了防止e^出现无穷大
    probs /= np.sum(probs)
    return probs


class TreeNode(object):

    # MCTreeSearch中的节点,每个节点保存了,该节点的Q 值、先验概率P、U 值

    def __init__(self, parent, prior_p: float, player: str, ):
        self.parent = parent
        self.children = {}  # key：a, value：S'
        self.player = player  # 'B''W' 处于该状态的玩家是谁

        self.n_visits = 0  # 访问次数（用于计算U）

        self.Q = 0.0  # value
        self.u = 0.0
        self.P = prior_p

    def expand(self, action_priors, cpp_action_filter: list, player: str):
        for action, prob in action_priors:
            if (action not in self.children) and (action in cpp_action_filter):  # 通过C++list过滤生成的节点
                # print(f"TreeNode.expand?:扩展节点")
                self.children[action] = TreeNode(self, prob, player)
        # print(f"TreeNode.expand?:self.children:{self.children}")

    def select(self, c_puct: float) -> int:
        #  :return 元组: (action, next_node)
        return max(self.children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value: float):
        # 用叶节点value来更新当前节点值
        self.n_visits += 1
        # Update Q, a running average of values for all visits.
        self.Q += 1.0 * (leaf_value - self.Q) / self.n_visits  # 更新Q值

    def update_recursive(self, leaf_value: float):
        # 递归地应用于更新所有祖先的Q 值 , 如果不是根节点，应先更新其父节点,再更新自身
        if self.parent:
            if self.parent.player == self.player:  # 判断是否反转结果
                self.parent.update_recursive(leaf_value)
            else:
                self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct: float) -> float:
        # C_puct * P(s, a) * { [∑ #(s,b)]^0.5  / 1 + #(s, a)} 见论文nature2016.pdf
        self.u = (c_puct * self.P * np.sqrt(self.parent.n_visits) / (1 + self.n_visits))
        return self.Q + self.u

    def is_leaf(self) -> bool:
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self.children == {}

    def is_root(self) -> bool:
        return self.parent is None

    def to_json(self, name) -> dict:
        json_dict = {"name": str(name) if name else "root",
                     "n_visits": int(self.n_visits),
                     "q": float(self.Q),
                     "u": float(self.u),
                     "p": float(self.P),
                     "children": [item[1].to_json(item[0]) for item in self.children.items()]}
        return json_dict


class MCTS(object):
    # 蒙特卡罗树搜索
    def __init__(self,
                 first_pv_fn,
                 second_pv_fn,
                 c_puct: float = 5,  # c_puct 越高, 越相信先验
                 n_playout: int = 10000):

        self.root = TreeNode(None, 1.0, 'B')  # 根节点,先验概率为1, 第一子定义为黑棋的第2步,但实际中我们不用这一步
        self.first_policy = first_pv_fn  # 第1子 策略值函数, 是一个神经网络
        self.second_policy = second_pv_fn  # 第2子 策略值函数, 是一个神经网络

        self.c_puct = c_puct  # 超参数C
        self.n_playout = n_playout  # 深度限制
        ###########################################
        self.cache = {}  # 缓存，用来存储已经搜索过的局面
        self.incache = 0
        self.notincacahe = 0
        ###########################################

    def playout(self, state):
        """从根到叶子进行一次的playout（采样到结束）,在叶子上获取一个值，并通过其父节点将其传播回去。
        状态被就地修改(子节点状态在采样时就地修改,会影响下一次采样,
        所以应该在copy上进行模拟采样,而不是在原树结构上直接操作)，
        因此必须提供一个副本。 :type state: 残局棋盘board """
        node = self.root
        while True:
            if node.is_leaf():
                break
            # 用贪婪策略选择下一步动作
            action, node = node.select(self.c_puct)
            # print(f"MCTS.playout?:贪婪落子:{action}")
            state.do_move(action)
        is_first_hand = state.is_first_hand()
        policy = self.first_policy if is_first_hand else self.second_policy
        # print(f"MCTS.playout?:state.current_state():{state.current_state()}")
        curr_state = state.current_state()
        cpp_avlb = state.cpp_available()
        # print(f"curr_state:\n{curr_state}")
        # print(f"cpp_avlb:\n{cpp_avlb}")
        action_probs, leaf_value = policy(state.current_state(),
                                          state.cpp_available())  # 网络给出的leaf_value不一定是整数,还是得回溯value而不是winner
        #####################################################
        # state_str = state.current_str_state()
        # if state_str in self.cache:
        #     print("缓存字典中已经存在该局面，不调用网络")
        #     action_probs = self.cache[state_str][0]
        #     leaf_value = self.cache[state_str][1]
        #     self.incache = self.incache + 1
        # else:
        #     print("缓存字典中没有该局面，调用网络")
        #     action_probs, leaf_value = policy(state.current_state(),
        #                                       state.cpp_available())  # 网络给出的leaf_value不一定是整数,还是得回溯value而不是winner ###！！！！！在这里加？
        #     action_probs = list(action_probs)
        #     # print("将该局面存入缓存字典")
        #     self.cache[state_str] = (action_probs, leaf_value)
        #     assert state_str in self.cache
        #     # print("该局面已存入缓存字典")
        #     # print(self.cache)
        #     self.notincacahe += 1
        #############################################################

        # 查看是否游戏结束
        end, winner = state.game_end()
        if not end:
            # 不到游戏结束,就往下扩展一层
            # print(f"MCTS.playout?:网络策略估值:action_probs{action_probs}")
            # print(f"MCTS.playout?:游戏没有结束, next_player:{next_player}， next_is_first：{next_is_first}")

            cpp_action_filter = state.CPP_availables[:]  # 过滤列表
            # print(f"MCTS:playout: C++ 过滤列表:{cpp_action_filter}")
            next_player = state.next_player()
            node.expand(action_probs, cpp_action_filter, next_player)
            # print(f"MCTS.playout?:扩展一层:")

        else:
            # 游戏结束
            if winner == 'NULL':  # tie 和棋
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == node.player else -1.0
            # print(f"MCTS.playout?:游戏结束, leaf_value:{leaf_value}, winner:{winner}")
        node.update_recursive(-leaf_value)  # 正负号

    def get_move_probs(self, state, temp=1e-3):  # TODO temp 多少合适??
        """ 依次运行所有的Simulation(模拟采样到结束)，并返回可用的action及其概率。 """
        # if state.stone_num > 10:  # 数搜索后期鼓励利用
        #     temp /= np.log(state.stone_num+1)
        for n in range(self.n_playout):  # _n_playout是Simulation(模拟采样)的次数
            # print(f"MCTS.get_move_probs?:开始第{n}次模拟.......")
            state_copy = state.deepcopy()  # 深度copy整个棋盘
            self.playout(state_copy)
        # print(f"MCTS.get_move_probs?:{self.n_playout}次模拟全部结束.......")

        if len(self.root.children.items()) == 0:
            input("root children == 0")
            act_visits = [(i, 0) for i in state.availables]

        act_visits = [(act, node.n_visits) for act, node in self.root.children.items()]
        # print(f"MCTS.get_move_probs?:act_visits{act_visits}")
        acts, visits = zip(*act_visits)
        # 看不懂这个公式,与AlphaZero不同,但效果一样,都是温度高时探索变多,温度低时利用变多,log会抑制访问次数多的点
        act_probs = softmax((1.0 / temp) * np.log(np.array(visits) + 1e-10))
        # act_probs = softmax(1.0 / temp * np.array(visits) + 1e-10)

        return acts, act_probs

    def update_with_move(self, last_move: int):
        """先前走一步棋,即root节点下移一层,并尽量利用内存中已知的搜索树部分,如果实在没用,则清空整棵树"""
        self.root = self.root.children[last_move]
        self.root._parent = None

    def reset_mct(self, player):
        # print(f"MCTS.update_with_move?:重置MCTree为:{player}")
        self.root = TreeNode(None, 1.0, player)

    def __str__(self) -> str:
        return "MCTS"

    ##################################################################################
    def clear_cache(self):
        print("在MCTSPLAYER中，网络已更新，清楚缓存字典")
        self.cache.clear()
        assert not self.cache
        print("缓存字典已清空")

    def in_or_not_in(self):
        return self.incache, self.notincacahe, self.incache / (self.incache + self.notincacahe)
    ############################################################################################


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self,
                 first_pv_function,
                 second_pv_function,
                 c_puct: float = 5,
                 n_playout: int = 2000,
                 is_selfplay: bool = False):

        self.mcts = MCTS(first_pv_function,
                         second_pv_function,
                         c_puct, n_playout)
        self.is_selfplay = is_selfplay
        self.player = "NULL"

    def set_player_ind(self, player: str):  # TODO 对弈的时候需要指定
        """ 设置先后手 B W """
        assert player == 'B' or player == 'W', "落子方不合法"
        self.player = player

    def reset_player(self, player: str):  # TODO 重开每一局需要指定
        # 重置整棵树
        assert player == 'B' or player == 'W', "落子方不合法"
        self.mcts.reset_mct(player)

    def get_action(self, board, temp=1e-3, return_prob=False, acts_and_probs=False):
        sensible_moves = board.cpp_available()  # 可选动作C++给出
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)

            ################################################
            print(json.dumps(self.mcts.root.to_json(None)))
            ################################################

            move_probs[list(acts)] = probs
            if self.is_selfplay:
                # 添加25%狄利克雷噪音用于鼓励探索
                move = np.random.choice(
                    acts,  # 是否关心抽出来的action是否在可选action中???
                    p=probs,
                    # 0.8 * probs + 0.2 * np.random.dirichlet(0.3 * np.ones(len(probs)))  # max = 1, mean = 1/n
                )
                # update the root node and reuse the search tree
                # move = max(acts, key=lambda x: move_probs[x])  # 确定性策略, 完全根据MCTS结果贪婪选取
                self.mcts.update_with_move(move)
            else:
                # else是人机对战部分,贪婪
                move = max(acts, key=lambda x: move_probs[x])  # np.random.choice(acts, p=probs)
                # reset the root node
                self.reset_player(self.player)
            if acts_and_probs:  # 仅仅用于GUI
                return acts, probs
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self) -> str:
        return "MCTS {}".format(self.mcts.root.player)

    #########################################################################################
    def clear_cache_macts(self):
        self.mcts.clear_cache()

    def in_not_in(self):
        return self.mcts.incache, self.mcts.notincacahe, self.mcts.incache / (
                self.mcts.incache + self.mcts.notincacahe + 1)
    #############################################################################################
