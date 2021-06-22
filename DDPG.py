import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import tensorlayer as tl
import csv

# Define hyperparameter
MEMORY_CAPACITY = 10000
LR_A = 0.001                # learning rate for actor
LR_C = 0.0001             # learning rate for critic
BATCH_SIZE = 32
GAMMA = 0.99                 # reward discount
TAU = 0.005                 # soft replacement
L2_DECAY = 0.01

BIT_RATE      = [500.0,850.0,1200.0,1850.0]
TARGET_BUFFER = [0.5,1.0]
LATENCY_THRESHOLD = [0.3, 2.5]

class DDPG(object):
    """
    DDPG class
    """
    def __init__(self, past_frame_num, a_dim, s_dim_1,s_dim_2, a_bound):
        # memory用于储存跑的数据的数组：
        # 保存个数MEMORY_CAPACITY，s_dim * 2 + a_dim + 1：分别是两个state，一个action，和一个reward
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim_1*past_frame_num*2 +  s_dim_2*2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.past_frame_num, self.a_dim, self.s_dim_1, self.s_dim_2, self.a_bound = past_frame_num, a_dim, s_dim_1, s_dim_2, a_bound
        self.kernel_size = 4
        self.neuron = 128

        # W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        # b_init = tf.constant_initializer(0.1)

        # 建立actor网络，输入s，输出a
        def get_actor(input_state_shape_1, input_state_shape_2, name=''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: act
            """
            last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

            s1 = tf.keras.Input(input_state_shape_1, name='A_input1')
            x1 = tf.keras.layers.Conv1D(filters=self.neuron, kernel_size=self.kernel_size, activation='relu', padding='same', name='A_l1')(s1)
            x1 = tf.keras.layers.Dense(units=1, activation='relu', name='A_l2')(x1)
            x1 = tf.keras.layers.Reshape((self.s_dim_1,))(x1)

            s2 = tf.keras.Input(input_state_shape_2, name='A_input2')
            x = tf.keras.layers.concatenate(inputs=[x1, s2], axis=1)

            x = tf.keras.layers.Dense(units=self.neuron, activation='relu', name='A_l3')(x)
            # Problem: always 輸出 1 或 0
            x = tf.keras.layers.Dense(units=a_dim, activation='tanh' , kernel_initializer=last_init, name='A_a')(x)
            # x = tf.keras.layers.Lambda(lambda x: np.array(a_bound) * x)(x)            #注意这里，先用tanh把范围限定在[-1,1]之间，再进行映射
            return tf.keras.models.Model(inputs=[s1, s2], outputs=x, name='Actor' + name)

        #建立Critic网络，输入s，a。输出Q值
        def get_critic(input_state_shape_1, input_state_shape_2, input_action_shape, name=''):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            s1 = tf.keras.Input(input_state_shape_1, name='C_s_input1')
            s1_ = tf.keras.layers.Conv1D(filters=self.neuron, kernel_size=self.kernel_size, activation='relu', padding='same', name='C_l1')(s1)
            s1_ = tf.keras.layers.Dense(units=1, activation='relu', name='C_l2')(s1_)
            s1_ = tf.keras.layers.Reshape((self.s_dim_1,))(s1_)

            s2 = tf.keras.Input(input_state_shape_2, name='C_s_input2')
            s_ = tf.keras.layers.concatenate(inputs=[s1_, s2], axis=1)

            a = tf.keras.Input(input_action_shape, name='C_a_input')
            a_ = tf.keras.layers.Dense(units=a_dim, activation='relu', name='C_l3')(a)
            
            x = tf.keras.layers.concatenate(inputs=[s_, a_], axis=1)
            x = tf.keras.layers.Dense(units=self.neuron, activation='relu', name='C_l4')(x)
            x = tf.keras.layers.Dense(units=1, name='C_out')(x)
            return tf.keras.models.Model(inputs=[s1, s2, a], outputs=x, name='Critic' + name)

        self.actor = get_actor([s_dim_1, past_frame_num], [s_dim_2])
        self.critic = get_critic([s_dim_1, past_frame_num], [s_dim_2], [a_dim])
        # self.actor.train()
        # self.critic.train()

        #更新参数，只用于首次赋值，之后就没用了
        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        #建立actor_target网络，并和actor参数一致，不能训练
        self.actor_target = get_actor([s_dim_1, past_frame_num], [s_dim_2], name='_target')
        copy_para(self.actor, self.actor_target)
        # self.actor_target.eval()

        #建立critic_target网络，并和actor参数一致，不能训练
        self.critic_target = get_critic([s_dim_1, past_frame_num], [s_dim_2], [a_dim], name='_target')
        copy_para(self.critic, self.critic_target)
        # self.critic_target.eval()

        # self.R = tl.layers.Input([None, 1], tf.float32, 'r')

        #建立ema，滑动平均值
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)


    def ema_update(self):
        """
        滑动平均更新
        """
        # 其实和之前的硬更新类似，不过在更新赋值之前，用一个ema.average。
        paras = self.actor.trainable_weights + self.critic.trainable_weights    #获取要更新的参数包括actor和critic的
        self.ema.apply(paras)                                                   #主要是建立影子参数
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))                                       # 用滑动平均赋值

    # def map_bit_rate(self, bit_rate):
    #     if bit_rate >= -1 and bit_rate < -0.5:
    #         bit_rate = 0
    #     elif bit_rate >= -0.5 and bit_rate < 0:
    #         bit_rate = 1
    #     elif bit_rate >= 0 and bit_rate < 0.5:
    #         bit_rate = 2
    #     else:
    #         bit_rate = 3
    #     return bit_rate
    # def map_target_buffer(self, target_buffer):
    #     if target_buffer < 0:
    #         target_buffer = 0
    #     else:
    #         target_buffer = 1
    #     return target_buffer
    # def map_latency_limit(self, latency_limit):
    #     latency_limit = LATENCY_THRESHOLD[0] +((LATENCY_THRESHOLD[1]-LATENCY_THRESHOLD[0])/(1-(-1))) * (latency_limit - (-1))
    #     return latency_limit
    # 选择动作，把s带进入，输出a
    def choose_action(self, s1, s2, noise_obj=None):
        """
        Choose action
        :param s: state
        :return: act(bit_rate, target_buffer, latency_limit)
        """
        # Mapping
        s1 = np.array([s1], dtype=np.float32)
        s2 = np.array([s2], dtype=np.float32)
        action = self.actor([s1, s2])[0]
        
        if noise_obj:
            noise = noise_obj()
            action = action.numpy() + noise
        else:
            action = action.numpy()
        # bit_rate = self.map_bit_rate(action[0])
        # target_buffer = self.map_target_buffer(action[1])
        # latency_limit = self.map_latency_limit(action[2])
        # action = np.array([bit_rate, target_buffer, latency_limit])
        return action

    def learn(self):
        """
        Update parameters
        :return: None
        """
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)    #随机BATCH_SIZE个随机数
        s1_range  = self.s_dim_1*self.past_frame_num
        s2_range = s1_range+self.s_dim_2+self.a_dim+1
        bt = self.memory[indices, :]                    #根据indices，选取数据bt，相当于随机
        bs1 = bt[:, :s1_range]
        bs2 = bt[:, s1_range:s1_range+self.s_dim_2]                          #从bt获得数据s

        ba = bt[:, s1_range+self.s_dim_2: s1_range+self.s_dim_2+self.a_dim]  #从bt获得数据a
        
        br = bt[:, s1_range+self.s_dim_2+self.a_dim: s1_range+self.s_dim_2+self.a_dim + 1]         #从bt获得数据r
        bs1_ = bt[:, s2_range:s2_range+s1_range]   
        bs2_ = bt[:, s2_range+s1_range:s2_range+s1_range+self.s_dim_2]                    #从bt获得数据s'

        # Reshape State
        bs1 = np.reshape(bs1, (bs1.shape[0], self.s_dim_1, self.past_frame_num))
        bs1_ = np.reshape(bs1_, (bs1_.shape[0], self.s_dim_1, self.past_frame_num))

        # Critic：
        # Critic更新和DQN很像，不过target不是argmax了，是用critic_target计算出来的。
        # br + GAMMA * q_
        record = []
        with tf.GradientTape() as tape:
            a_ = self.actor_target([bs1_, bs2_], training=True)
            q_ = self.critic_target([[bs1_, bs2_], a_], training=True)
            y = br + GAMMA * q_
            q = self.critic([[bs1, bs2], ba], training=True)
            L2 = np.sum([tf.nn.l2_loss(v) for v in self.critic.trainable_weights if "W" in v.name])
            td_error = tf.losses.mean_squared_error(y, q) + L2_DECAY * L2
            record.append(np.sum(td_error))
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

        # Actor：
        # Actor的目标就是获取最多Q值的。
        with tf.GradientTape() as tape:
            a = self.actor([bs1, bs2], training=True)
            q = self.critic([[bs1, bs2], a], training=True)
            a_loss = -tf.reduce_mean(q)  # 【敲黑板】：注意这里用负号，是梯度上升！也就是离目标会越来越远的，就是越来越大。
            record.append(np.sum(a_loss))
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))

        # with open('test.csv', 'a', newline='') as csvfile:
        #     # 建立 CSV 檔寫入器
        #     writer = csv.writer(csvfile)
        #     # 寫入一列資料
        #     writer.writerow(record)

        self.ema_update()
    # 保存s，a，r，s_
    def store_transition(self, s1,s2, a, r, s1_, s2_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        # 整理s，s_,方便直接输入网络计算
        s1 = np.array(s1).astype(np.float32).flatten()
        s2 = np.array(s2).astype(np.float32).flatten()
        s1_ = np.array(s1_).astype(np.float32).flatten()
        s2_ = np.array(s2_).astype(np.float32).flatten()
        a = np.array(a)
        r = np.array(r)

        #把s, a, [r], s_横向堆叠
        # TODO
        # 維度不同
        # 13*10, 1*3
        transition = np.hstack((s1, s2, a, r, s1_, s2_))
        # transition = np.hstack((s, a, [r], s_))

        #pointer是记录了曾经有多少数据进来。
        #index是记录当前最新进来的数据位置。
        #所以是一个循环，当MEMORY_CAPACITY满了以后，index就重新在最底开始了
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        #把transition，也就是s, a, [r], s_存进去。
        self.memory[index, :] = transition
        self.pointer += 1

    def save_statistic(self, epoch, reward):
        with open('model/output.csv', 'a', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)
            # 寫入一列資料
            writer.writerow([epoch, reward])
    
    def save_ckpt(self, epoch=0):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists(f'model/epoch{epoch}'):
            os.makedirs(f'model/epoch{epoch}')
        self.actor.save_weights(f'model/epoch{epoch}/ddpg_actor_{epoch}.ckpt')
        self.actor_target.save_weights(f'model/epoch{epoch}/ddpg_actor_target_{epoch}.ckpt')
        self.critic.save_weights(f'model/epoch{epoch}/ddpg_critic_{epoch}.ckpt')
        self.critic_target.save_weights(f'model/epoch{epoch}/ddpg_critic_target_{epoch}.ckpt')

    def load_ckpt(self, epoch=0):
        """
        load trained weights
        :return: None
        """
        self.actor.load_weights(f'model/epoch{epoch}/ddpg_actor_{epoch}.ckpt')
        self.actor_target.load_weights(f'model/epoch{epoch}/ddpg_actor_target_{epoch}.ckpt')
        self.critic.load_weights(f'model/epoch{epoch}/ddpg_critic_{epoch}.ckpt')
        self.critic_target.load_weights(f'model/epoch{epoch}/ddpg_critic_target_{epoch}.ckpt')