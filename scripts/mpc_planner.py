from local_planner.simulation import Simulation
from config import read_config
from map import costmap
import numpy as np

import argparse
import multiprocessing
import copy
import datetime
import pickle
import copy

def worker(mode, sim, start_ind, shared_dict, process_id, end_ind):
    dataset_config = sim.run_closedLoop(mode, start_ind, end_ind)
    shared_dict[process_id] = dataset_config

def start_processes(mode, sim, subseq):
    """
    Start multiple processes for path planning.
    """
    with multiprocessing.Manager() as manager:
        shared_dict = manager.dict()
        processes = []
        
        for iter, start_ind in enumerate(subseq[:-1]):
            process = multiprocessing.Process(target=worker, args=(mode, sim, start_ind, shared_dict, iter, subseq[iter+1]+1))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        copied_dict = copy.deepcopy(shared_dict)
        return copied_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='local_planner')
    parser.add_argument('--map_name', type=str, default='map414diffuser')
    parser.add_argument('--ritp_filename', type=str, default='randSddrx-414test-static-v2-light')
    parser.add_argument('--mode', type=str, default='static')
    parser.add_argument('--parallel_num', type=int, default='12')
    args = parser.parse_args()

    config = read_config.read_config(config_name='config')

    scence_dir = f"{config['map_dir']}/{args.map_name}.csv"
    park_map = costmap.Map(file=scence_dir, discrete_size=config['map_discrete_size'])
    ego_vehicle = costmap.Vehicle()
    file_name = f"dataset/{args.ritp_filename}.hdf5"
    sim = Simulation('static', park_map.case, ego_vehicle, file_name, config, park_map, args.map_name)

    total_len = len(sim.true_indices)
    subsequences = np.array_split(sim.true_indices[:-1], np.ceil(total_len / args.parallel_num))
    for i in range(1, len(subsequences)):
        subsequences[i] = np.insert(subsequences[i], 0, subsequences[i-1][-1])

    if args.parallel_num == 1:
        ########################### for single test ###########################
        choose_ind = 1000
        start_ind = subsequences[choose_ind][0]
        end_ind = subsequences[choose_ind][0+1]+1
        dataset_config = sim.run_closedLoop('dynamic', start_ind, end_ind)
        # dataset_config = sim.run_closedLoop('static', start_ind, end_ind)
        ########################### for single test ###########################
    else:
        ########################### for parallel dataset generation ###########################
        success_todate, fail_todate, save_freq, total_dataset = 0, 0, 10, [] # save_freq: save every n iters
        for sub_ind, subseq in enumerate(subsequences):
            if sub_ind < 1000: # only first part of trajs consider dynamic obstacles, because the rest trajs are too far away from the dynamic obstacles
                ################### start parallel ###################
                subdataset = start_processes(args.mode, sim, subseq)
                ################### start parallel ###################
                total_dataset.append(subdataset)
                succuss_num, fail_num = 0, 0
                for val in subdataset:
                    succuss_num += sum(subdataset[val]["is_success_list"])
                    fail_num += len(subdataset[val]["fail_cnt"])
                success_todate += succuss_num
                fail_todate += fail_num

                if sub_ind % save_freq == 0 or sub_ind == (len(subsequences)-1):
                    ########### get current time ###########
                    now = datetime.datetime.now()
                    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
                    ########### get current time ###########
                    save_dir = f"./dataset/{timestamp}_{args.mode}_success{success_todate}_fail{fail_todate}.pkl"
                    with open(save_dir, 'wb') as file:
                        pickle.dump(copy.deepcopy(total_dataset), file)
                    print(f"save at iter {sub_ind}, total success trajs is {success_todate}")
        ########################### for parallel dataset generation ###########################
    
