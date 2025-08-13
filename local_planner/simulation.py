import numpy as np
import h5py

import time
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from local_planner.env import Env
from local_planner.closed_loop import closedLoop
from shapely.geometry import Polygon
from collision_check.vertexex_check import VertexexCheck
class Simulation:
    def __init__(self, scene_name, case, ego_vehicle, ritp_filename, config, map, map_name, trans_dist_x=1,trans_dist_y=1):
        self.scene_name = scene_name
        self.case = case
        self.ego_vehicle = ego_vehicle
        self.ritp_filename = ritp_filename
        self.config = config
        self.map = map
        self.map_name = map_name

        self.record_traj = None
        self.true_indices = None
        self.GSP = None

        self.trans_dist_x = trans_dist_x
        self.trans_dist_y = trans_dist_y
        
        obstacle_polygons = []
        for _, val in enumerate(self.case.obs):
            obs_x = val[:, 0]
            obs_y = val[:, 1]
            obstacle_polygons.append(Polygon(zip(obs_x, obs_y)))
        self.obs_polys = obstacle_polygons

        self.collision_checker = VertexexCheck(vehicle=self.ego_vehicle, map=self.map, config=self.config)
        self.load_ref_ritp() # get ritp traj from dataset

    def run_closedLoop(self, mode, start_ind, end_ind):
        use_dynamic, param_ind = {
            "static": (False, 0),
            "dynamic": (True, 1)
        }[mode]

        env_config = Env(self.scene_name, self.case)
        fail_cnt = []
        observations_total, obs_pose_total, is_success_list, total_time_list = [], [], [], []

        reference, GSP_list, yaw, forward_dir = self.get_ritp_traj(start_ind, end_ind)
        ########################## instance mpc planner close-loop simulation ##########################
        closed_mpc = closedLoop(env_config, self.ego_vehicle, reference, yaw, GSP_list, forward_dir)
        ########################## instance mpc planner close-loop simulation ##########################
        
        try:
            v_ref, v_min = 1.1, 0.8  # m/s
            iter_cnt, is_success, start_time, fail_str = 0, False, time.time(), ""

            while True:
                horizon_decrease = 0
                iter_cnt += 1

                #################################### mpc planner ####################################
                pose_traj, obs_pose, is_reach_goal, total_time = closed_mpc.mpc_caller(use_dynamic, v_ref, param_ind, horizon_decrease)
                #################################### mpc planner ####################################
                
                # do collision check
                if not use_dynamic: # static
                    collision_ind = self.collision_checker.pointwise_check(pose_traj, self.obs_polys, reference[::2, :])
                    is_collision_free = True if len(collision_ind) == 0 else False
                else: # dynamic scene do not involve static collision check
                    is_collision_free = True

                if is_reach_goal and is_collision_free:
                    is_success = True
                    break

                v_ref -= 0.1
                if v_ref < v_min:
                    if not is_reach_goal:
                        fail_str += "+notreachgoal" 
                    if not is_collision_free:
                        fail_str += "+collision"
                    break
            
            cost_time = time.time() - start_time
            print(f"cost {cost_time} s to finish")
            observations, obstacle_pose, time_list = [], [], []
            for ind, point in enumerate(pose_traj): # first frame of initial state
                x, y, phi, v = point[0], point[1], point[2], point[3]
                rotation = R.from_euler('zyx', [phi, 0, 0], degrees=False)
                quaternion = rotation.as_quat()  # [x, y, z, w] format
                observations.append([x, y, quaternion[-2], quaternion[-1], v])
                time_list.append(total_time[ind])
                if use_dynamic and ind > 0:
                    obstacle_pose.append(obs_pose[ind-1])
        
        except Exception as e:
            print(f"\nException error while collect dynamic collision-free trajs: {e}")
            pose_traj, obs_pose, total_time_list, observations, obstacle_pose, time_list = \
                [], [], [], [], [], []

        is_success_list.append(is_success)
        observations_total.append(observations)
        obs_pose_total.append(obstacle_pose)
        total_time_list.append(time_list)

        if not is_success:
            cost_time = time.time() - start_time
            print(f"fail to generate collision-free traj after trying {cost_time} s")
            fail_cnt.append(str(start_ind)+'_'+fail_str)
            print(fail_cnt)

        dataset_config = {
            "observations_total": observations_total,
            "obs_pose_total": obs_pose_total,
            "is_success_list": is_success_list,
            "total_time_list": total_time_list,
            "fail_cnt": fail_cnt,
            "reference_path": reference,
            "reference_yaw": yaw,
        }

        return dataset_config

    def load_ref_ritp(self):
        # reference
        ####################### load ritp trajs for tracking #######################
        data_dict = {}
        with h5py.File(self.ritp_filename, 'r') as dataset_file:
            for ind in tqdm(self.get_keys(dataset_file), desc="load dataset"):
                try:  # first try loading as an array
                    data_dict[ind] = dataset_file[ind][:]
                except ValueError as e:  # try loading as a scalar
                    data_dict[ind] = dataset_file[ind][()]

        self.record_traj = data_dict['observations_center']
        self.true_indices = [0] + list(np.where(data_dict['terminals'] == True)[0])
        self.GSP = data_dict["GSP"]

    def get_ritp_traj(self, start_ind, end_ind):
        if start_ind > 0:
            start_ind += 1

        reference = self.record_traj[start_ind:end_ind, :].T
        GSP_list = list(np.where(self.GSP[start_ind:end_ind]==True)[0])
        assert GSP_list[0] == False
        GSP_list.pop(0)

        # calculate phi
        piece_quan_z = reference[2, :]
        piece_quan_w = reference[3, :]
        quaternion = np.column_stack((np.zeros_like(piece_quan_z), np.zeros_like(piece_quan_z), piece_quan_z, piece_quan_w))
        with np.errstate(invalid='ignore'):  # Ignore possible singularity warnings
            rotation = R.from_quat(np.array(quaternion, dtype=np.float64))
            yaw = rotation.as_euler('zyx', degrees=False)[:, 0]
        yaw = np.unwrap(yaw)
        xy_ref = reference[:2, :]
        xy_ref[0, :] -= self.trans_dist_x
        xy_ref[1, :] -= self.trans_dist_y
        
        return xy_ref, GSP_list, yaw, reference[4, :]
    
    def get_keys(self, h5file):
        keys = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)

        h5file.visititems(visitor)
        return keys