# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf


class PolicyValueNet:
    is_server_mod = False

    def __init__(self, board_x, board_y, model_file=None, **kwargs):
        self.graph = tf.Graph()
        self.config = self.generate_config(**kwargs)  # tensorflow自定义配置选项
        self.board_x = board_x
        self.board_y = board_y
        self.resnet_channels = 32  # resnet 的通道数
        self.character_num = 2 * 4 * 12  # 20  #
        self.residual_tower_input_channels = 24
        self.action_size = self.board_x * self.board_y
        self.channels_axis = -1  # 通道
        self.batch_axis = 0  # batch
        self.train_time = 0
        self.l2_beta = 1e-4

        with self.graph.as_default():
            # 1. Input: 输入
            self.input_states = tf.placeholder(tf.float32, shape=[None, self.character_num, board_x, board_y])
            self.input_features = tf.transpose(self.input_states,
                                               [0, 2, 3, 1])  # NCHW -> NHWC高维矩阵的转置(维度调整) "channels_last"

            # self.curr_line  = tf.slice(self.input_features, [0, 0, 0, 0],[-1, -1, -1, 8])
            # self.opp_line  = tf.slice(self.input_features, [0, 0, 0, 8],[-1, -1, -1, 8])
            # self.history_board = tf.slice(self.input_features, [0, 0, 0, 16],[-1, -1, -1, 4])
            # self.type_angle_list = []
            # for i in range(12):
            #     self.type_angle_list.append(tf.slice(self.input_features, [0, 0, 0, 4 * i], [-1, -1, -1, 4 * i + 4], name=f'angle_type{i}'))

            self.is_training = tf.placeholder(tf.bool, name="is_training")
            self.learning_rate = tf.placeholder(tf.float32)
            self.target_probs = tf.placeholder(tf.float32, shape=[None, self.action_size])
            self.target_vs = tf.placeholder(tf.float32, shape=[None, 1])

            # 分别把line方向卷没先
            self.slice = tf.split(self.input_features, num_or_size_splits=self.residual_tower_input_channels, axis=-1)
            self.conv_slice = [self.pre_conv_block(input_layer=image, kernel_size=3, filters=1, stage=0, block=f"{i}")
                               for i, image in enumerate(self.slice)]

            # 叠加在一起
            # x_image = tf.concat([self.merge_curr, self.merge_opp, self.history_board], axis=self.channels_axis)
            x_image = tf.concat(self.conv_slice, axis=self.channels_axis)

            # 1. 先卷积一层
            # batch_size  x board_x x board_y x 1
            x_image = tf.reshape(x_image, [-1, self.board_x, self.board_y, self.residual_tower_input_channels])
            print(x_image.shape)
            x_image = tf.layers.conv2d(x_image, self.resnet_channels, kernel_size=(3, 3), strides=(1, 1), name='conv',
                                       data_format='channels_last', padding='same', use_bias=False,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_beta))
            x_image = tf.layers.batch_normalization(x_image, axis=self.channels_axis, name='conv_bn',
                                                    fused=True, training=self.is_training)
            x_image = tf.nn.relu(x_image)

            # 2. 残差塔部分
            residual_tower = self.residual_block(input_layer=x_image, kernel_size=3, filters=self.resnet_channels,
                                                 stage=1, block='a')
            # residual_tower = self.residual_block(input_layer=residual_tower, kernel_size=3,
            #                                      filters=self.resnet_channels,
            #                                      stage=2, block='b')
            # residual_tower = self.residual_block(input_layer=residual_tower, kernel_size=3,
            #                                      filters=self.resnet_channels,
            #                                      stage=3, block='c')
            # residual_tower = self.residual_block(input_layer=residual_tower, kernel_size=3,
            #                                      filters=self.resnet_channels,
            #                                      stage=4, block='d')
            # residual_tower = self.residual_block(input_layer=residual_tower, kernel_size=3,
            #                                      filters=self.resnet_channels,
            #                                      stage=5, block='e')
            # residual_tower = self.residual_block(input_layer=residual_tower, kernel_size=3,
            #                                      filters=self.resnet_channels,
            #                                      stage=6, block='g')
            # residual_tower = self.residual_block(input_layer=residual_tower, kernel_size=3,
            #                                      filters=self.resnet_channels,
            #                                      stage=7, block='h')
            # residual_tower = self.residual_block(input_layer=residual_tower, kernel_size=3,
            #                                      filters=self.resnet_channels,
            #                                      stage=8, block='i')

            # residual_tower = self.residual_block(input_layer=residual_tower, kernel_size=3, filters=self.num_channels,
            #                                      stage=9, block='j')
            # residual_tower = self.residual_block(input_layer=residual_tower, kernel_size=3, filters=self.num_channels,
            #                                      stage=10, block='k')
            # residual_tower = self.residual_block(input_layer=residual_tower, kernel_size=3, filters=self.num_channels,
            #                                      stage=11, block='m')
            # residual_tower = self.residual_block(input_layer=residual_tower, kernel_size=3, filters=self.num_channels,
            #                                      stage=12, block='n')

            # 3.策略网络 PI Head
            policy = tf.layers.conv2d(residual_tower, 2, kernel_size=(1, 1), strides=(1, 1), name='pi', padding='same',
                                      data_format='channels_last', use_bias=False,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_beta))
            policy = tf.layers.batch_normalization(policy, axis=self.channels_axis, name='bn_pi',
                                                   fused=True, training=self.is_training)
            policy = tf.nn.relu(policy)
            policy = tf.layers.flatten(policy, name='p_flatten')
            self.pi = tf.layers.dense(policy, self.action_size)
            self.act_probs = tf.nn.softmax(self.pi)

            # 4. 值网络 Value Head
            value = tf.layers.conv2d(residual_tower, 1, kernel_size=(1, 1), strides=(1, 1), name='v', padding='same',
                                     data_format='channels_last', use_bias=False,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_beta))
            value = tf.layers.batch_normalization(value, axis=self.channels_axis, name='bn_v',
                                                  fused=True, training=self.is_training)
            value = tf.nn.relu(value)
            value = tf.layers.flatten(value, name='v_flatten')
            value = tf.layers.dense(value, units=256)
            value = tf.nn.relu(value)
            value = tf.layers.dense(value, 1)
            self.v = tf.nn.tanh(value)

            # 损失
            self.loss_pi = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target_probs,
                                                                                     logits=self.pi))
            # self.loss_pi = tf.losses.softmax_cross_entropy(onehot_labels =self.target_probs, logits=self.pi)
            self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1, 1]))
            self.loss_reg = tf.reduce_mean(tf.losses.get_regularization_losses())  # mean 使得正则约束较小,加不加作用域用于筛选?
            self.total_loss = self.loss_pi + self.loss_v + self.loss_reg

            tf.summary.scalar("loss_pi", self.loss_pi)
            tf.summary.scalar("loss_v", self.loss_v)
            tf.summary.scalar("loss_reg", self.loss_reg)
            tf.summary.scalar("loss", self.total_loss)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

            # Make a session
            self.session = tf.Session(graph=self.graph, config=self.config)
            self.merged_summary = tf.summary.merge_all()

        # 加载模型
        with self.session.as_default():
            with self.graph.as_default():
                self.train_writer = tf.summary.FileWriter("./summary/", self.session.graph)
                # Initialize variables
                init = tf.global_variables_initializer()
                self.session.run(init)
                # For saving and restoring
                self.saver = tf.train.Saver(tf.global_variables())
                if model_file is not None:
                    print("载入模型:")
                    self.restore_model(model_file)

    def pre_conv_block(self, input_layer, filters, kernel_size, stage, block):
        conv_name = 'pre' + str(stage) + block + '_branch'
        bn_name = 'bn' + str(stage) + block + '_branch'
        x = input_layer
        x = tf.layers.conv2d(x, filters, kernel_size=(kernel_size, kernel_size), strides=(1, 1),
                                          name=conv_name, padding='same', use_bias=False,
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_beta))
        x = tf.layers.batch_normalization(x, axis=self.channels_axis, name=bn_name,
                                                       fused=True, training=self.is_training)
        x = tf.nn.relu(x)

        return x

    def residual_block(self, input_layer, filters, kernel_size, stage, block):
        conv_name = 'res' + str(stage) + block + '_branch'
        bn_name = 'bn' + str(stage) + block + '_branch'
        shortcut = input_layer
        residual_layer = tf.layers.conv2d(input_layer, filters, kernel_size=(kernel_size, kernel_size), strides=(1, 1),
                                          name=conv_name + '2a', padding='same', use_bias=False,
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_beta))
        residual_layer = tf.layers.batch_normalization(residual_layer, axis=self.channels_axis, name=bn_name + '2a',
                                                       fused=True, training=self.is_training)
        residual_layer = tf.nn.relu(residual_layer)
        residual_layer = tf.layers.conv2d(residual_layer, filters, kernel_size=(kernel_size, kernel_size),
                                          strides=(1, 1), name=conv_name + '2b', padding='same', use_bias=False,
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_beta))
        residual_layer = tf.layers.batch_normalization(residual_layer, axis=self.channels_axis, name=bn_name + '2b',
                                                       fused=True, training=self.is_training)
        add_shortcut = tf.add(residual_layer, shortcut)
        residual_result = tf.nn.relu(add_shortcut)

        return residual_result

    def generate_config(self, **kwargs):
        DEVICES = kwargs.get("DEVICES")
        # 限制计算卡占用
        if PolicyValueNet.is_server_mod:
            os.environ['CUDA_VISIBLE_DEVICES'] = DEVICES if DEVICES else "0"
        else:
            pass
            # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        # 限制显存占用
        tf_config = tf.ConfigProto()
        # 开启TensorFlow显存占用按需求增长
        tf_config.gpu_options.allow_growth = True
        # 设定每块GPU上使用的显存上限（0.4即40%）
        # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
        return tf_config

    def policy_value(self, state_batch, is_train):
        with self.session.as_default():
            with self.graph.as_default():
                act_probs, value = self.session.run(
                    [self.act_probs, self.v],
                    feed_dict={self.input_states: state_batch,
                               self.is_training: is_train}
                )
                return act_probs, value

    def policy_value_fn(self, current_state, CPP_availables):
        with self.session.as_default():
            with self.graph.as_default():
                legal_positions = CPP_availables  # 过滤CPP给出的可选动作
                assert legal_positions, "PolicyValueNet.policy_value_fn?:legal_positions is None!!"
                # 转化为列优先内存布局的ndarray
                current_state = np.ascontiguousarray(
                    current_state.reshape(-1, self.character_num, self.board_x, self.board_y))
                # print(current_state.shape)
                act_probs, value = self.policy_value(current_state, False)
                # print(value)
                r_act_probs = zip(legal_positions, act_probs[0][legal_positions])
                # for act, prob in r_act_probs:
                    # print(f"act:{act}, prob:{prob}")
                # r_act_probs = zip(legal_positions, act_probs[0][legal_positions])
                return r_act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        with self.session.as_default():
            with self.graph.as_default():
                winner_batch = np.reshape(winner_batch, (-1, 1))
                mcts_probs = np.reshape(mcts_probs, (-1, self.action_size))
                loss, entropy, merged_summary, _ = self.session.run([self.total_loss,
                                                                     self.loss_pi,
                                                                     self.merged_summary,
                                                                     self.optimizer],
                                                                    feed_dict={self.input_states: state_batch,
                                                                               self.target_probs: mcts_probs,
                                                                               self.target_vs: winner_batch,
                                                                               self.learning_rate: lr,
                                                                               self.is_training: True})
                self.train_writer.add_summary(merged_summary, self.train_time)
                self.train_time += 1
                return loss, entropy

    def save_model(self, folder, file_name):
        file_path = os.path.join(folder, file_name)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            # print("Checkpoint Directory exists! ")
            pass
        with self.session.as_default():
            with self.graph.as_default():
                self.saver.save(self.session, file_path)
                print(f"model saved:{file_name}.")

    def restore_model(self, model_path):
        if not os.path.exists(model_path + '.meta'):
            # print(f"model not exists, ERROR!{model_path}")
            raise ("No model in path {}".format(model_path))
        with self.session.as_default():
            with self.graph.as_default():
                self.saver.restore(self.session, model_path)
                print(f"model loaded:{model_path}.")

# if __name__ == '__main__':
#     net1 = PolicyValueNet(19,19)
#     net2 = PolicyValueNet(19,19)
#
#     net1.save_model('./save/current1', '1.model')
#     net1.save_model('./save/current2', '2.model')
#     net1.restore_model('./save/current1/1.model')
#     net1.restore_model('./save/current2/2.model')