# import the env 
import fixed_env as fixed_env
import load_trace as load_trace
#import matplotlib.pyplot as plt
import time as tm
import DDPG
import os, sys
import numpy as np
import multiprocessing as mp

LATENCY_THRESHOLD = [0.3, 2.5]

def map_bit_rate(bit_rate):
        if bit_rate >= -1 and bit_rate < -0.5:
            bit_rate = 0
        elif bit_rate >= -0.5 and bit_rate < 0:
            bit_rate = 1
        elif bit_rate >= 0 and bit_rate < 0.5:
            bit_rate = 2
        else:
            bit_rate = 3
        return bit_rate
def map_target_buffer(target_buffer):
    if target_buffer < 0:
        target_buffer = 0
    else:
        target_buffer = 1
    return target_buffer
def map_latency_limit(latency_limit):
    latency_limit = LATENCY_THRESHOLD[0] +((LATENCY_THRESHOLD[1]-LATENCY_THRESHOLD[0])/(1-(-1))) * (latency_limit - (-1))
    return latency_limit

def test(testcase):
    
    # -- Configuration variables --
    # Edit these variables to configure the simulator
    # Change which set of video trace to use: AsianCup_China_Uzbekistan, Fengtimo_2018_11_3, game, room, sports, YYF_2018_08_12
    VIDEO_TRACE = testcase[0]

    # Change which set of network trace to use: 'fixed' 'low' 'medium' 'high'
    NETWORK_TRACE = testcase[1]

    # Turn on and off logging.  Set to 'True' to create log files.
    # Set to 'False' would speed up the simulator.
    DEBUG = testcase[2]

    # Control the subdirectory where log files will be stored.
    LOG_FILE_PATH = './log/'
    
    # create result directory
    if not os.path.exists(LOG_FILE_PATH):
        os.makedirs(LOG_FILE_PATH)

    print(f"{VIDEO_TRACE},{NETWORK_TRACE}: Start")
    # -- End Configuration --
    # You shouldn't need to change the rest of the code here.

    network_trace_dir = './dataset/network_trace/' + NETWORK_TRACE + '/'
    video_trace_prefix = './dataset/video_trace/' + VIDEO_TRACE + '/frame_trace_'

    # load the trace
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(network_trace_dir)
    #random_seed 
    random_seed = 2
    count = 0
    trace_count = 1
    FPS = 25
    frame_time_len = 0.04
    reward_all_sum = 0
    run_time = 0
    #init 
    #setting one:
    #     1,all_cooked_time : timestamp
    #     2,all_cooked_bw   : throughput
    #     3,all_cooked_rtt  : rtt
    #     4,agent_id        : random_seed
    #     5,logfile_path    : logfile_path
    #     6,VIDEO_SIZE_FILE : Video Size File Path
    #     7,Debug Setting   : Debug
    net_env = fixed_env.Environment(all_cooked_time=all_cooked_time,
                                  all_cooked_bw=all_cooked_bw,
                                  random_seed=random_seed,
                                  logfile_path=LOG_FILE_PATH,
                                  VIDEO_SIZE_FILE=video_trace_prefix,
                                  Debug = DEBUG)
    

    BIT_RATE      = [500.0,850.0,1200.0,1850.0] # kpbs
    TARGET_BUFFER = [0.5,1.0]   # seconds
    # ABR setting
    RESEVOIR = 0.5
    CUSHION  = 2

    cnt = 0
    # defalut setting
    last_bit_rate = 0
    bit_rate = 0
    target_buffer = 0
    latency_limit = 4

    # QOE setting
    reward_frame = 0
    reward_all = 0
    SMOOTH_PENALTY= 0.02
    REBUF_PENALTY = 1.85
    LANTENCY_PENALTY = 0.005
    SKIP_PENALTY = 0.5
    # past_info setting
    past_frame_num  = 10
    S_time_interval = [0] * past_frame_num
    S_send_data_size = [0] * past_frame_num
    S_chunk_len = [0] * past_frame_num
    S_rebuf = [0] * past_frame_num
    S_buffer_size = [0] * past_frame_num
    S_frame_time_lens = [0] * past_frame_num
    S_throughputs = [0] * past_frame_num
    S_end_delay = [0] * past_frame_num
    S_chunk_size = [0] * past_frame_num
    S_play_time_len = [0] * past_frame_num
    S_decision_flag = [0] * past_frame_num
    S_buffer_flag = [0] * past_frame_num
    S_cdn_flag = [0] * past_frame_num
    S_skip_time = [0] * past_frame_num
    # params setting
    call_time_sum = 0 

    s_dim_1 = 11
    s_dim_2 = 6
    a_dim = 3
    a_bound = 1
    ddpg = DDPG.DDPG(past_frame_num, a_dim, s_dim_1, s_dim_2, a_bound)
    ddpg.load_ckpt(epoch=2)

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
        S_time_interval.pop(0)
        S_send_data_size.pop(0)
        S_chunk_len.pop(0)
        S_buffer_size.pop(0)
        S_frame_time_lens.pop(0)
        S_throughputs.pop(0)
        S_rebuf.pop(0)
        S_end_delay.pop(0)
        S_play_time_len.pop(0)
        S_decision_flag.pop(0)
        S_buffer_flag.pop(0)
        S_cdn_flag.pop(0)
        S_skip_time.pop(0)

        S_time_interval.append(time_interval)
        S_send_data_size.append(send_data_size)
        S_chunk_len.append(chunk_len)
        S_buffer_size.append(buffer_size)
        S_frame_time_lens.append(chunk_len)
        S_throughputs.append(throughput)
        S_rebuf.append(rebuf)
        S_end_delay.append(end_delay)
        S_play_time_len.append(play_time_len)
        S_decision_flag.append(decision_flag)
        S_buffer_flag.append(buffer_flag)
        S_cdn_flag.append(cdn_flag) 
        S_skip_time.append(skip_frame_time_len)

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
            # reward formate = play_time * BIT_RATE - 4.3 * rebuf - 1.2 * end_delay
            reward_frame += -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
            # last_bit_rate
            last_bit_rate = bit_rate

            cnt += 1
            timestamp_start = tm.time()
            # ----- 整理 state------
            state1 = [S_time_interval, S_send_data_size, S_frame_time_lens, S_throughputs, S_rebuf, \
            S_play_time_len, S_skip_time, S_end_delay, S_buffer_size, S_cdn_flag, S_buffer_flag, ]
            state2 = [np.sum(S_rebuf), np.sum(S_skip_time), end_delay, cdn_flag, buffer_flag, last_bit_rate]
            # ----- END -----------
            action = ddpg.choose_action(state1, state2)
            bit_rate,target_buffer,latency_limit = map_bit_rate(action[0]), \
                                           map_target_buffer(action[1]), \
                                           map_latency_limit(action[2])

            timestamp_end = tm.time()
            call_time_sum += timestamp_end - timestamp_start
            # -------------------- End --------------------------------
            
        if end_of_video:
            # print("network traceID, network_reward, avg_running_time", trace_count, reward_all, call_time_sum/cnt)
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

            S_time_interval = [0] * past_frame_num
            S_send_data_size = [0] * past_frame_num
            S_chunk_len = [0] * past_frame_num
            S_rebuf = [0] * past_frame_num
            S_buffer_size = [0] * past_frame_num
            S_end_delay = [0] * past_frame_num
            S_chunk_size = [0] * past_frame_num
            S_play_time_len = [0] * past_frame_num
            S_decision_flag = [0] * past_frame_num
            S_buffer_flag = [0] * past_frame_num
            S_cdn_flag = [0] * past_frame_num
            
        reward_all += reward_frame
    print(f"{VIDEO_TRACE},{NETWORK_TRACE}: Done")
    return [reward_all_sum / trace_count, run_time / trace_count]

if __name__ == "__main__":
    if(sys.argv[1]=="all"):
        video_traces = [
            'AsianCup_China_Uzbekistan',
            'Fengtimo_2018_11_3', 
            'game', 
            'room', 
            'sports', 
            'YYF_2018_08_12'
        ]
        netwrok_traces = [
            # 'fixed',
            # 'low',
            # 'medium',
            'high'
        ]
    else:
        video_traces = [sys.argv[1]]
        netwrok_traces = [sys.argv[2]]
    debug = False
    testcases = []
    for video_trace in video_traces:
        for netwrok_trace in netwrok_traces:
            testcases.append([video_trace, netwrok_trace, debug])
    N = mp.cpu_count()
    with mp.Pool(processes=N) as p:
        results = p.map(test,testcases)
    print(results)
    print("score: ", np.mean(results ,axis = 0))