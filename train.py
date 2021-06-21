# %%
# import the env 
import fixed_env as fixed_env
import load_trace as load_trace
#import matplotlib.pyplot as plt
import time as tm
import ABR
import os, sys
import numpy as np
import multiprocessing as mp
import DDPG
import Noise

#%%
# TODO
debug = False
testcase = ['game', 'high', debug]
VIDEO_TRACE = testcase[0]
NETWORK_TRACE = testcase[1]
DEBUG = testcase[2]
LOG_FILE_PATH = './log/'
# create result directory
if not os.path.exists(LOG_FILE_PATH):
    os.makedirs(LOG_FILE_PATH)
network_trace_dir = './dataset/network_trace/' + NETWORK_TRACE + '/'
video_trace_prefix = './dataset/video_trace/' + VIDEO_TRACE + '/frame_trace_'

# load the trace
all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(network_trace_dir)

print(f"{VIDEO_TRACE},{NETWORK_TRACE}: Start")

#%%
# System parameter
random_seed = 2
count = 0
FPS = 25
frame_time_len = 0.04

# Train Parameter
EPOCH = 3
MEMORY_CAPACITY = 1000
# %%
# init
BIT_RATE      = [500.0,850.0,1200.0,1850.0] # kpbs
TARGET_BUFFER = [0.5,1.0]   # seconds

# QOE setting
reward_frame = 0
reward_all = 0
SMOOTH_PENALTY= 0.02
REBUF_PENALTY = 1.85
LANTENCY_PENALTY = 0.005
SKIP_PENALTY = 0.5
# past_info setting
past_frame_num  = 10
# params setting
call_time_sum = 0 

# %% 
def getInitState():
    # duration of cycle
    S_time_interval = [0] * past_frame_num
    # frame size
    S_send_data_size = [0] * past_frame_num
    # frame length
    S_frame_time_lens = [0] * past_frame_num
    # network throughputs
    S_throughputs = [0] * past_frame_num
    # rebuf
    S_rebuf = [0] * past_frame_num
    # play time len
    S_play_time_len = [0] * past_frame_num
    # skip
    S_skip_time = [0] * past_frame_num
    # end delay
    S_end_delay = [0] * past_frame_num
    # buffer size
    S_buffer_size = [0] * past_frame_num
    # cdn flag
    S_cdn_flag = [0] * past_frame_num
    # playing buffer
    S_buffer_flag = [0] * past_frame_num

    state1 = [S_time_interval, S_send_data_size, S_frame_time_lens, S_throughputs, S_rebuf, \
            S_play_time_len, S_skip_time, S_end_delay, S_buffer_size, S_cdn_flag, S_buffer_flag, ]
    state2 = [0]*6
    return state1, state2
# %%
# 每次大概跑 7500 次
def Step(net_env, pre_state1, pre_state2, action, last_bit_rate):
    bit_rate,target_buffer,latency_limit = action[0], action[1], action[2]

    while True:
        reward_frame = 0
        
        time,time_interval, send_data_size, chunk_len,\
               rebuf, buffer_size, play_time_len,end_delay,\
                cdn_newest_id, download_id, cdn_has_frame,skip_frame_time_len, decision_flag,\
                buffer_flag, cdn_flag, skip_flag,end_of_video = net_env.get_video_frame(bit_rate,target_buffer, latency_limit)
        
        if time_interval != 0:
            throughput = send_data_size / time_interval
        else:
            throughput = 0
        # S_info is sequential order
        S_time_interval = pre_state1[0]
        S_send_data_size = pre_state1[1]
        S_frame_time_lens = pre_state1[2]
        S_throughputs = pre_state1[3]
        S_rebuf = pre_state1[4]
        S_play_time_len = pre_state1[5]
        S_skip_time = pre_state1[6]
        S_end_delay = pre_state1[7]
        S_buffer_size = pre_state1[8]
        S_cdn_flag = pre_state1[9]
        S_buffer_flag = pre_state1[10]

        S_time_interval.pop(0)
        S_send_data_size.pop(0)
        S_frame_time_lens.pop(0)
        S_throughputs.pop(0)
        S_rebuf.pop(0)
        S_play_time_len.pop(0)
        S_skip_time.pop(0)
        S_end_delay.pop(0)
        S_buffer_size.pop(0)
        S_cdn_flag.pop(0)
        S_buffer_flag.pop(0)
        
        

        S_time_interval.append(time_interval)
        S_send_data_size.append(send_data_size)
        S_frame_time_lens.append(chunk_len)
        S_throughputs.append(throughput)
        S_rebuf.append(rebuf)
        S_play_time_len.append(play_time_len)
        S_skip_time.append(skip_frame_time_len)
        S_end_delay.append(end_delay)
        S_buffer_size.append(buffer_size)
        S_cdn_flag.append(cdn_flag)
        S_buffer_flag.append(buffer_flag)
         
        

        # QOE setting 
        if end_delay <=1.0:
            LANTENCY_PENALTY = 0.005
        else:
            LANTENCY_PENALTY = 0.01
            
        if not cdn_flag:
            reward_frame = frame_time_len * float(BIT_RATE[bit_rate]) / 1000  - REBUF_PENALTY * rebuf - LANTENCY_PENALTY  * end_delay - SKIP_PENALTY * skip_frame_time_len 
        else:
            reward_frame = -(REBUF_PENALTY * rebuf)
        
        if decision_flag or end_of_video:
            reward_frame += -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
            state1 = [S_time_interval, S_send_data_size, S_frame_time_lens, S_throughputs, S_rebuf, \
            S_play_time_len, S_skip_time, S_end_delay, S_buffer_size, S_cdn_flag, S_buffer_flag, ]
            state2 = [np.sum(S_rebuf), np.sum(S_skip_time), end_delay, cdn_flag, buffer_flag, last_bit_rate]
            return state1, state2, reward_frame, end_of_video

# %%
def train(testcase):
    #定义状态空间，动作空间，动作幅度范围
    s_dim_1 = 11
    s_dim_2 = 6
    a_dim = 3
    a_bound = 1
    ddpg = DDPG.DDPG(past_frame_num, a_dim, s_dim_1, s_dim_2, a_bound)
    # defalut setting
    last_bit_rate = 0
    bit_rate = 0
    target_buffer = 0
    latency_limit = 4
    for i in range(EPOCH):
        call_time_sum = 0 
        cnt = 0
        ep_reward = 0
        reward_all_sum = 0
        reward_all = 0
        run_time = 0
        trace_count = 1
        # feature #

        net_env = fixed_env.Environment(all_cooked_time=all_cooked_time,
                                all_cooked_bw=all_cooked_bw,
                                random_seed=random_seed,
                                logfile_path=LOG_FILE_PATH,
                                VIDEO_SIZE_FILE=video_trace_prefix,
                                Debug = DEBUG)
        std_dev = 0.2
        ou_noise = Noise.OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
        # Init State
        state1, state2 = getInitState()
        while True:
            action = ddpg.choose_action(state1, state2, ou_noise)
            action = [int(action[0]), int(action[1]), action[2]]
            [bit_rate,target_buffer,latency_limit] = action[0], action[1], action[2]
            n_state1, n_state2, reward, end_of_video = Step(net_env, state1 , state2, action, last_bit_rate)
            ddpg.store_transition(state1,state2, action, reward, n_state1, n_state2)
            state1 = n_state1
            state2 = n_state2 
            last_bit_rate = bit_rate 
            ep_reward += reward  #记录当前EP的总reward
            # print(')
            cnt += 1

            # print('---cnt---:', cnt)
            # ---------------- Train Process---------------
            if ddpg.pointer > MEMORY_CAPACITY:
                timestamp_start = tm.time()
                ddpg.learn()
                timestamp_end = tm.time()
                call_time_sum += timestamp_end - timestamp_start
                print(
                        '\rEpisode: {}/{}  | Episode Reward: {:.4f} | Count: {} | Trace Count: {} | Running Time: {:.4f}'.format(
                            i, EPOCH, ep_reward,
                            cnt,
                            trace_count, 
                            timestamp_end - timestamp_start
                        ), end=''
                    )
            # ---------------------End--------------------
            
            if end_of_video:
                reward_all_sum += reward_all
                run_time += call_time_sum / cnt
                if trace_count >= len(all_file_names):
                    break
                trace_count += 1
                cnt = 0
                call_time_sum = 0
                last_bit_rate = 0
                reward_all = 0
                bit_rate = 0
                target_buffer = 0
                
            reward_all += reward
        print(f"{VIDEO_TRACE},{NETWORK_TRACE}: Done")
        print([reward_all_sum / trace_count, run_time / trace_count])

# %%
train(testcase)
        
    






# %%
