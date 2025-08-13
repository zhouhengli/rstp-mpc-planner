import numpy as np
import matplotlib.pyplot as plt
import datetime

from casadi import *
from local_planner.draw import plotClass
from local_planner.mpc import MPCPlanner

warnings.filterwarnings("ignore")
class closedLoop:

    def __init__(self, problem_setting, ego_vehicle, center_path, phi, GSP_list, forward_dir):
        self.ego_vehicle = ego_vehicle
        self.dt_alltraj = None
        self.setting = problem_setting # from the demo_setting.py file
        self.draw = plotClass(self.setting, [], [])
        self.center_path = center_path[0:2, :].T
        self.phi_init = phi[0]
        self.phi_end = np.arctan2(np.sin(phi[-1]), np.cos(phi[-1]))
        self.phi_full = phi.T
        self.GSP_list = GSP_list
        self.horizon = len(phi)
        self.GSP_list.append(self.horizon-1)
        self.current_piece = 0
        self.forward_dir = forward_dir
        self.planner_name = 'ritp'

        ##################################
        # Both
        ##################################

        # state & input numbers
        self.nx = 4 # x,y,phi,v
        self.nu = 2 # delta,a
        self.local_solver = MPCPlanner(self.nx, self.nu, ego_vehicle.lw)

        # state & input constraint
        self.xL = [self.setting.xL[0], self.setting.xL[1], -np.inf, -ego_vehicle.max_v]
        self.xU = [self.setting.xU[0], self.setting.xU[1],  np.inf,  ego_vehicle.max_v]
        # delta, a
        self.uL = [-ego_vehicle.max_steering_angle, -ego_vehicle.max_acc]
        self.uU = [  ego_vehicle.max_steering_angle, ego_vehicle.max_acc]

        # state & input
        self.x0 = self.setting.startPose
        self.xF = self.setting.goalPose
        self.u0 = [0, 0]

        # reference
        self.xref = []
        self.uref = []

        # car
        self.ego = [ego_vehicle.lw + ego_vehicle.lf, ego_vehicle.lb/2, ego_vehicle.lr, ego_vehicle.lb/2]
        
        ##################################
        # free time
        ##################################

        # horizon
        self.N_mpc = 20 # ego vehicle prediction horizon
        self.N_obs = 10 # dynamic obstacle prediction horizon

        ##################################
        # param list for local planner
        ##################################
        self.param_list = [
        # [delata   a]    [u change]   [ x   y   phi   v ]  [terminal] [slack] 
            [0.01, 0.5,] +  [0.1]*4 + [10, 10,  10,  None,   1,       None] , # for static scene 
            [0.01, 0.5,] +  [0.1]*4 + [ 5,  5,   5,  None,   5,       None] ,    # for dynamic scene 
        ]

        ##################################
        # dynamic obstacle for local planner
        # obstacle informaiton:
        ## start center & angle,
        ## length, width (moving direction is length)
        ## const_velocity
        ## end center & angle
        ## start_time, end time
        ##################################
        self.dyn_nObs = 0
        self.dyn_vObs = 0
        self.dyn_lObs = []
        self.dyn_obs_info = [] # orignal size of dynamic obstacles
        
        # dynamic scene 1
        # self.dyn_orignal_info = [[3.5, 2.0, np.pi/4*1.2+np.pi, 0.6, 0.4, 0.4, 0, 50]]
        # dynamic scene 2
        self.dyn_orignal_info = [[3.5, 1.2, -np.pi, 0.6, 0.4, 0.4, 0, 80]] 
        

        self.obs_vel = self.dyn_orignal_info[0][5]
        self.obs_phi = self.dyn_orignal_info[0][2]
        # get dynamic obstacle number, each vertex number & location
        self.add_dynamic_obstacle(self.dyn_orignal_info)
        self.senseDis = 1.0
        self.dyn_loc = []

        ##################################
        # MPC Planner related init
        ##################################
        self.ARC_LENGTH_MIN_DIST_TOL = 0.01

    def calcu_vref(self, vref):
        vref_tmp = vref * (self.forward_dir[self.GSP_list[self.current_piece]-1]*2-1) # correct vref according to forward direction
        return vref_tmp

    def mpc_caller(self, use_dynamic, vref, param_ind, horizon_decrease):

        self.N_mpc -= horizon_decrease
        print(f"current horizon is {self.N_mpc}")

        # count time
        sim_iter, u0 = 0, self.u0

        # store lists
        u_opt, mpc_pred_traj, pose_traj, ref_traj = [], [], [], []
        obs_pose, dynObs_list_total, total_time = [], [], [0]

        # timing and goal flag
        Ts = sim_dt = 0.1
        is_reach_goal = True

        start_ind = 0 # simlulation start pose
        if start_ind > 0: 
            current_s, near_idx = self.find_current_arc_length(self.center_path[start_ind, 0:2])
            psi_next = self.ceter_lut_phi(current_s)
            xyphi_list = [self.center_path[start_ind, 0], self.center_path[start_ind, 1], psi_next, 0.0]
        else: # only for test
            xyphi_list = [self.center_path[start_ind, 0], self.center_path[start_ind, 1], self.phi_init, 0.0]

        sim_state = np.array(xyphi_list).T
        center_path = self.center_path[:self.GSP_list[self.current_piece], :]
        phi_path = self.phi_full[:self.GSP_list[self.current_piece]]
        element_arc_lengths_orig, center_lut_x, center_lut_y, center_lut_phi = self.preprocess_track_data(center_path, phi_path) # update reference piece path
        vref = self.calcu_vref(vref)
        pose_tol = 0.15
        yaw_tol = (20/180)*np.pi
        pose_bias, yaw_bias = np.inf, np.inf

        while ( pose_bias >= pose_tol \
                or yaw_bias >= yaw_tol \
                        ): # fit the terminal constraints
            ######################## store states ##########################
            pose_traj.append(sim_state)
            total_time.append(total_time[-1]+sim_dt)
            ######################## store states ##########################

            ######################## update current piece-wise path for tracking ########################
            dist = np.sqrt((sim_state[0] - self.center_path[self.GSP_list[self.current_piece]][0]) ** 2 + (sim_state[1] - self.center_path[self.GSP_list[self.current_piece]][1])** 2)
            if dist < 0.1 and self.current_piece < len(self.GSP_list)-1:
                self.current_piece += 1
                center_path = self.center_path[self.GSP_list[self.current_piece-1]:self.GSP_list[self.current_piece], :]
                phi_path = self.phi_full[self.GSP_list[self.current_piece-1]:self.GSP_list[self.current_piece]]
                element_arc_lengths_orig, center_lut_x, center_lut_y, center_lut_phi = self.preprocess_track_data(center_path, phi_path) # update reference piece path
                vref = self.calcu_vref(vref)
            ######################## update current piece-wise path for tracking ########################
        
            ##################### update reference for local-mpc #####################
            current_s, near_idx = self.find_current_arc_length(sim_state[0:2], center_path, element_arc_lengths_orig)
            # add reduced-horizon mpc
            x_ref, sim_state = self.compute_x_ref(sim_state, current_s, Ts, vref, element_arc_lengths_orig, center_lut_x, center_lut_y, center_lut_phi)
            ##################### update reference for local-mpc #####################

            if use_dynamic:
                ############ update dynamic obstacle pose ############
                if len (obs_pose) > 0:
                    lastobs_pose = obs_pose[-1]
                else:
                    lastobs_pose = None
                dyn_lObs, cx, cy = self.update_obstacle(sim_iter, sim_dt, lastobs_pose)
                self.dyn_loc.append(dyn_lObs)
                obs_pose.append([cx, cy]) # for diffusion dataset generation

                # search for dynamic obstacles
                dyn_exist = self.sensor(sim_state)
                if dyn_exist:
                    # update dynamic obstacles message
                    dynObs_list = self.rebuild_lObs(sim_dt, self.N_obs, obs_pose[-1])
                else:
                    dynObs_list = []
                ############ update dynamic obstacle pose ############
            else:
                dynObs_list = []

            ########################### collision-free mpc ###########################
            # print("############## dynamic exist ##############") if dyn_exist else None
            xOpt, uOpt, feas \
                = self.local_solver.planner_mpc( \
                    Ts, self.param_list[param_ind], self.N_mpc,
                                                sim_state, self.xL, self.xU, self.uL, self.uU,
                                                x_ref, u0, dynObs_list)
            ########################### collision-free mpc ###########################

            if feas == True:
                # print('MPC -- Success in sim_iter = %i, current obs %i' % (sim_iter,len(dynObs_list)))
                # get control input
                u0 = uOpt[:, 0].T
            else:
                # print('MPC -- Failed in sim_iter = %i, current obs %i' % (sim_iter, len(dynObs_list)))
                pass # use last control input

            # update current state
            sim_state = xOpt[:, 1].T

            ########################## store states ##########################
            dynObs_list_total.append(dynObs_list)
            mpc_pred_traj.append(xOpt.T)
            ref_traj.append(x_ref)
            u_opt.append(u0)
            ########################## store states ##########################

            # update time counting
            sim_iter += 1

            if sim_iter == 150:
                is_reach_goal = False
                break
            
            phi_normalized = np.arctan2(np.sin(sim_state[2]), np.cos(sim_state[2]))
            pose_bias = np.sqrt((sim_state[0] - self.setting.goalPose[0]) ** 2 + (sim_state[1] - self.setting.goalPose[1]) ** 2)
            yaw_bias = abs(phi_normalized - self.setting.goalPose[2])
        
        pose_traj.append(sim_state)
        total_time.append(total_time[-1]+sim_dt)

        # ################### vis traj GIF ###################
        self.xref = self.center_path
        sim_title = 'Dynamic Avoidance With RSTP-MPC-Planner'
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = './media/%s_' % timestamp + self.setting.scene_name + '_N%i' % self.N_mpc + '_SensorDis%i' % self.senseDis + '_steps%i' % sim_iter
        self.draw.fullDimension_closedLoop_animate(self, pose_traj, ref_traj, sim_iter - 1, mpc_pred_traj, self.dyn_loc, sim_title, file_name, self.senseDis, True, dynObs_list_total)
        # ################### vis traj GIF ###################

        return pose_traj, obs_pose, is_reach_goal, total_time

    def add_dynamic_obstacle(self, dyn_obs_info):
        """
        Generate the initial vertex of dynamic obstacle & update dynamic obstacle parameter.
        - obstacle represent as rectangle
        - length define at moving direction
        :param dyn_obs_info: [start_center_x, start_center_y, start_theta,
                              length, width,
                              constant_velocity, constant_angular_velocity,
                              end_center_x, end_center_y, end_theta,
                              start_time, end_time]
        :return:
        """

        dyn_lObs = []
        for i in range(np.size(dyn_obs_info, 0)):
            cx = dyn_obs_info[i][0]
            cy = dyn_obs_info[i][1]
            theta = dyn_obs_info[i][2]
            l = dyn_obs_info[i][3]
            w = dyn_obs_info[i][4]

            obs_initial_vertex = self.get_obstacle(cx, cy, theta, l, w)
            # print(obs_initial_vertex)
            dyn_lObs.append(obs_initial_vertex)

        # update dynamic obstacle parameter
        self.dyn_obs_info = dyn_obs_info

    def ode_vehicle(self, state, input, Ts):
        state = state + Ts * self.rhs(state, input)
        return state

    def rhs(self, state, input):
        L = self.ego_vehicle.lw
        lr = self.ego_vehicle.lr
        X = state[0]
        Y = state[1]
        psi = state[2]
        v = state[3]
        delta = input[0]
        a = input[1]
        
        dot = np.array([
            v * np.cos(psi),
            v * np.sin(psi),
            v * np.tan(delta) / L,
            a
        ])

        return dot

    def update_obstacle(self, sim_iter, sim_dt, lastobs_pose):
        dyn_obs_info = []
        dyn_lObs = []
        if sim_iter == self.dyn_orignal_info[0][6]: # start time
            dyn_obs_info.append(self.dyn_orignal_info[0])
            
        elif sim_iter > self.dyn_orignal_info[0][6] and sim_iter <= self.dyn_orignal_info[0][7]:
            i = 0
            self.dyn_orignal_info[i][0] += sim_dt * self.dyn_orignal_info[i][5] * np.cos(
                self.dyn_orignal_info[i][2])
            self.dyn_orignal_info[i][1] += sim_dt * self.dyn_orignal_info[i][5] * np.sin(
                self.dyn_orignal_info[i][2])
            dyn_obs_info.append(self.dyn_orignal_info[0])
        elif sim_iter > self.dyn_orignal_info[0][7]:
            i = 0
            dyn_obs_info.append(self.dyn_orignal_info[i])

        self.obs_vel = dyn_obs_info[-1][5]
        self.obs_phi = dyn_obs_info[-1][2]

        # calc vertex
        cx = dyn_obs_info[-1][0]
        cy = dyn_obs_info[-1][1]
        theta = dyn_obs_info[-1][2]
        l = dyn_obs_info[-1][3]
        w = dyn_obs_info[-1][4]

        obs_initial_vertex = self.get_obstacle(cx, cy, theta, l, w)
        dyn_lObs.append(obs_initial_vertex)

        return dyn_lObs, cx, cy

    def update_obstacle_constraint(self, lObs):
        fulltime_nObs = np.size(lObs, 0)
        fulltime_vObs = np.ones(fulltime_nObs, dtype=int)
        for i in range(fulltime_nObs):
            fulltime_vObs[i] = np.size(lObs[i], 0)

        return fulltime_nObs, fulltime_vObs

    def sensor(self, sim_state):

        cx = sim_state[0]
        cy = sim_state[1]
        theta = sim_state[2]
        l = self.ego[0]
        w = self.ego[1]

        vertex_2 = [cx + l * np.cos(theta) - w * np.sin(theta), cy + l * np.sin(theta) + w * np.cos(theta)]
        vertex_3 = [cx + l * np.cos(theta) + w * np.sin(theta), cy + l * np.sin(theta) - w * np.cos(theta)]
        carFront = [(vertex_2[0] + vertex_3[0]) / 2, (vertex_2[1] + vertex_3[1]) / 2]

        for i in range(np.size(self.dyn_loc[-1], 0)): # latest obstacles
            isExist = 0
            dyn_obs = self.dyn_loc[-1][i]
            # only check front collision
            for j in range(4):
                dis = np.sqrt((carFront[0] - dyn_obs[j][0]) ** 2 + (carFront[1] - dyn_obs[j][1]) ** 2)
                if dis <= self.senseDis:
                    isExist = 1
                    break

            if isExist == 0:
                self.dyn_loc[-1][i].append(0)
            else:
                self.dyn_loc[-1][i].append(1)

        return isExist

    def get_obstacle(self, center_x, center_y, theta, length, width):
        """
        Generate vertex of rectangle obstacle in clockwise
        :param center_x: x value at the center of the obstacle
        :param center_y: y value at the center of the obstacle
        :param theta: heading angle of the obstacle, unit: rad
        :param length: obstacle's length, define at the moving direction
        :param width: obstacle's width
        :return: obs_vertex = obstacle vertex [left_buttom, left_top, right_top, right_buttom]
        """
        cx = center_x
        cy = center_y
        l = length / 2
        w = width / 2

        # vertex: left_buttom, left_top, right_top, right_buttom
        vertex_1 = [cx - l * np.cos(theta) - w * np.sin(theta), cy - l * np.sin(theta) + w * np.cos(theta)]
        vertex_2 = [cx + l * np.cos(theta) - w * np.sin(theta), cy + l * np.sin(theta) + w * np.cos(theta)]
        vertex_3 = [cx + l * np.cos(theta) + w * np.sin(theta), cy + l * np.sin(theta) - w * np.cos(theta)]
        vertex_4 = [cx - l * np.cos(theta) + w * np.sin(theta), cy - l * np.sin(theta) - w * np.cos(theta)]

        # combine obs_vertex
        obs_vertex = [vertex_1, vertex_2, vertex_3, vertex_4, vertex_1]

        return obs_vertex

    def rebuild_lObs(self, sim_dt, N_obs, obs_pose):
        new_Obs = []
        # do forward simulation for obstacle movement prediction
        for k in range(N_obs + 1):
            # self.obs_vel * sim_dt
            lObs_x = obs_pose[0] + sim_dt * self.obs_vel * np.cos(self.obs_phi) * k
            lObs_y = obs_pose[1] + sim_dt * self.obs_vel * np.sin(self.obs_phi) * k
            new_Obs.append([lObs_x, lObs_y])
            # pass
        return new_Obs
    
    def compute_x_ref(self, sim_state, current_s, Ts, vref, element_arc_lengths_orig, center_lut_x, center_lut_y, center_lut_phi):

        # Construct the reference trajectory array
        x_ref = np.zeros((self.nx, self.N_mpc + 1))
        x_ref[:, 0] = sim_state # fisrt state is current state
        v_tmp = vref
        is_reachend = False
        for i in range(1, self.N_mpc+1):
            if not is_reachend:
                next_s = current_s + Ts * abs(vref)
                if next_s > element_arc_lengths_orig[-1]:
                    next_s = element_arc_lengths_orig[-1]
                    is_reachend = True
                    v_tmp = 0
    
            # Compute the state at next arc length
            x_ref[0, i] = center_lut_x(next_s)
            x_ref[1, i] = center_lut_y(next_s)
            x_ref[2, i] = center_lut_phi(next_s)
            
            # Assign the reference speed
            x_ref[3, i] = v_tmp
            # Update the current arc length
            current_s = next_s

        return x_ref, sim_state,

    def compute_x_ref_for_evaluate(self, sim_state, current_s, Ts, vref, element_arc_lengths_orig, center_lut_x, center_lut_y, center_lut_phi, center_lut_vel):

        # Construct the reference trajectory array
        x_ref = np.zeros((self.nx, self.N_mpc + 1))
        x_ref[:, 0] = sim_state
        current_vel = center_lut_vel(current_s)
        for i in range(1, self.N_mpc+1):
            next_s = current_s + Ts * abs(current_vel)
            if next_s > element_arc_lengths_orig[-1]:
                next_s = element_arc_lengths_orig[-1]
                current_vel = 0
            else:
                current_vel = center_lut_vel(next_s)
            # Compute the state at next arc length
            x_ref[0, i] = center_lut_x(next_s)
            x_ref[1, i] = center_lut_y(next_s)
            x_ref[2, i] = center_lut_phi(next_s)
            
            # Assign the reference speed
            x_ref[3, i] = current_vel
            current_s = next_s

        return x_ref, sim_state,


    @staticmethod
    def preprocess_track_data(center_path, phi_full):
        def get_arc_lengths(waypoints):
            d = np.diff(waypoints, axis=0)
            consecutive_diff = np.sqrt(np.sum(np.power(d, 2), axis=1))
            dists_cum = np.cumsum(consecutive_diff)
            dists_cum = np.insert(dists_cum, 0, 0.0)
            return dists_cum
        
        def get_interpolated_path_casadi(label_x, label_y, pts, arc_lengths_arr):
            u = arc_lengths_arr
            V_X = pts[:, 0]
            # print(V_X)
            V_Y = pts[:, 1]
            lut_x = interpolant(label_x, 'bspline', [u], V_X)
            lut_y = interpolant(label_y, 'bspline', [u], V_Y)
            return lut_x, lut_y
        
        def get_interpolated_path_casadi_phi(label_x, pts, arc_lengths_arr):
            u = arc_lengths_arr
            V_X = pts
            lut_x = interpolant(label_x, 'bspline', [u], V_X)
            return lut_x
        # Interpolate center line upto desired resolution
        element_arc_lengths_orig = get_arc_lengths(center_path)

        center_lut_x, center_lut_y = get_interpolated_path_casadi('lut_center_x', 'lut_center_y', center_path, element_arc_lengths_orig)
        center_lut_phi = get_interpolated_path_casadi_phi('lut_center_phi', phi_full, element_arc_lengths_orig)
        return element_arc_lengths_orig, center_lut_x, center_lut_y, center_lut_phi
    
    def find_nearest_index(self, car_pos, center_path):
        distances_array = np.linalg.norm(center_path[:, 0:2] - car_pos, axis=1)
        min_dist_idx = np.argmin(distances_array)
        
        return min_dist_idx, distances_array[min_dist_idx]
    
    def find_current_arc_length(self, car_pos, center_path, element_arc_lengths_orig):
        nearest_index, minimum_dist = self.find_nearest_index(car_pos, center_path)
        # print(minimum_dist)
        if minimum_dist > self.ARC_LENGTH_MIN_DIST_TOL:
            if nearest_index == (center_path.shape[0] - 1):
                current_s = element_arc_lengths_orig[nearest_index]
            else:
                if nearest_index == 0:
                    next_idx = 1
                    prev_idx = center_path.shape[0] - 1
                else:
                    next_idx = nearest_index + 1
                    prev_idx = nearest_index - 1
                dot_product_value = np.dot(car_pos - center_path[nearest_index, :],
                                        center_path[prev_idx, :] - center_path[nearest_index, :])
                if dot_product_value > 0:
                    nearest_index_actual = prev_idx
                else:
                    nearest_index_actual = nearest_index
                    nearest_index = next_idx
                new_dot_value = np.dot(car_pos - center_path[nearest_index_actual, :],
                                    center_path[nearest_index, :] - center_path[nearest_index_actual, :])
                projection = new_dot_value / np.linalg.norm(
                    center_path[nearest_index, :] - center_path[nearest_index_actual, :])
                current_s = element_arc_lengths_orig[nearest_index_actual] + projection
        else:
            current_s = element_arc_lengths_orig[nearest_index]

        if nearest_index == 0:
            current_s = 0.0

        return current_s, nearest_index
