# draw.py
"""
## Changelog
- trajectory metrics
- Author: ZhouhengLi

Plot the result
Created on 2022/12/13
@author: Pin-Yun Hung
"""
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib.animation import FuncAnimation
import os
class plotClass:

    def __init__(self, map_model, path_model, mpc_model):
        self.map = map_model
        self.ref_traj = path_model
        self.mpc = mpc_model

    def static_map_model(self, ax1):
        ax1.set_xlim([self.map.xL[0], self.map.xU[0]])
        ax1.set_ylim([self.map.xL[1], self.map.xU[1]])

        # plot static obstacle
        for i in range(len(self.map.static_lObs)):
            for j in range(0, self.map.static_vObs[i] - 1):
                v1 = self.map.static_lObs[i][j]  # vertex 1
                v2 = self.map.static_lObs[i][j + 1]  # vertex 2
                ax1.plot([v1[0], v2[0]], [v1[1], v2[1]], '-k')
            if self.map.static_vObs[i] >= 4:
                ax1.plot([self.map.static_lObs[i][-1][0], self.map.static_lObs[i][0][0]],
                         [self.map.static_lObs[i][-1][1], self.map.static_lObs[i][0][1]], '-k')

    def plot_map(self, dynObs_exist, dyn_nObs):
        # setting

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_aspect('equal', adjustable='box')

        # plot map with only static obstacles
        self.static_map_model(ax1)

        # animation plotting for dynamic obstacles
        corner_list = np.zeros((dyn_nObs, 4, 2))
        def animate_map(k):
            ax1.clear()
            # plot
            plt.title('simulate time = %i' % k)

            # plot static obstacle
            self.static_map_model(ax1)

            # plot start & end
            plt.plot([self.map.startPose[0], self.map.goalPose[0]], [self.map.startPose[1], self.map.goalPose[1]], 'ob', markersize='3')

            # plot dynamic obstacle
            if (self.map.dyn_lObs == []) == False:
                for i in range(self.map.dyn_nObs):
                    if self.map.dyn_obs_info[i][9] == k:
                        for j in range(4):
                            corner_list[i][j][0] = self.map.dyn_lObs[i][j][0]
                            corner_list[i][j][1] = self.map.dyn_lObs[i][j][1]
                    if self.map.dyn_obs_info[i][10] == k:
                        corner_list[i] = np.zeros((4, 2))

                for i in range(self.map.dyn_nObs):
                    if np.all(corner_list[i] == 0) == False:
                        for j in range(4):
                            corner_list[i][j][0] = \
                                self.map.dyn_lObs[i][j][0] + self.map.dyn_obs_info[i][5] * np.cos(
                                    self.map.dyn_obs_info[i][2]) * (
                                        k - self.map.dyn_obs_info[i][9])
                            corner_list[i][j][1] = \
                                self.map.dyn_lObs[i][j][1] + self.map.dyn_obs_info[i][5] * np.sin(
                                    self.map.dyn_obs_info[i][2]) * (
                                        k - self.map.dyn_obs_info[i][9])

                        plt.plot([corner_list[i][0][0], corner_list[i][1][0], corner_list[i][2][0],
                                  corner_list[i][3][0], corner_list[i][0][0]],
                                 [corner_list[i][0][1], corner_list[i][1][1], corner_list[i][2][1],
                                  corner_list[i][3][1], corner_list[i][0][1]], '-k')

            print("time = %i" %k)

        # plot animation if has dynamic obstacles
        if (dynObs_exist == 1) and (self.map.dyn_nObs > 0):
            ani = FuncAnimation(fig, animate_map, frames=max([row[10] for row in self.map.dyn_obs_info]) + 1, interval=20, repeat=False)

        # plt.show()

    def fullDimension_closedLoop_animate(self, mpc, pose_traj, ref_traj, N, mpc_pred_traj, dyn_loc, sim_title, file_name, senseDis, is_gif, dynObs_list_total):
        ref_x = mpc.xref
        ego = mpc.ego
        # vehicle params
        W_ev = ego[1] + ego[3]
        L_ev = ego[0] + ego[2]
        offset = L_ev / 2 - ego[2]
        save_dir = os.path.dirname(file_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if len(mpc_pred_traj) > 0:
            v_traj = [val[0, 3] for val in mpc_pred_traj] 
            ref_phi_traj = [ref_traj[i][2, 0] for i in range(len(ref_traj))]
            fig, axes = plt.subplots(2, 1, figsize=(8, 6))
            axes[0].plot(v_traj, label="Predicted Velocity", color='b', linestyle='-')
            axes[0].set_title("Velocity Trajectory")
            axes[0].set_ylabel("Velocity (m/s)")
            axes[0].legend()
            axes[0].grid(True)

            axes[1].plot(ref_phi_traj, label="Reference Yaw Angle", color='r', linestyle='-')
            axes[1].set_title("Reference Yaw Angle Trajectory")
            axes[1].set_ylabel("Yaw Angle (rad)")
            axes[1].set_xlabel("Time Step")
            axes[1].legend()
            axes[1].grid(True)

            plt.tight_layout()
            # plt.savefig(file_name, dpi=100)

            fig = plt.figure(figsize=(4, 3))
            ax1 = fig.add_subplot(111)
            self.static_map_model(ax1)
            for k in range(len(pose_traj)):
                pose_phi = pose_traj[k][2]
                Rot = np.array([[np.cos(pose_phi), - np.sin(pose_phi)], [np.sin(pose_phi), np.cos(pose_phi)]])
                x_cur = pose_traj[k][0:2].reshape(2, 1)
                centerCar = x_cur + Rot @ np.array([offset, 0]).reshape(2, 1)
                self.carBox(centerCar, pose_traj[k][2], W_ev / 2, L_ev / 2)
            plt.plot(mpc.xref[:, 0], mpc.xref[:, 1], '-o', color='royalblue', markersize='3')
            # plot running trajectory so far
            plot_traj_x = [val[0] for val in pose_traj]
            plot_traj_y = [val[1] for val in pose_traj]
            plt.plot(plot_traj_x, plot_traj_y, 'o', color='orange', markersize='3') 
            plt.title('Traj')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid(True)
            # plt.savefig(file_name+'_traj', dpi=100)

        if is_gif:
            if len(pose_traj) > 0: 
                # initial state
                x0_s = pose_traj[0]
                Rot0 = np.array([[np.cos(x0_s[2]), - np.sin(x0_s[2])], [np.sin(x0_s[2]), np.cos(x0_s[2])]])
                x0 = x0_s[0:2].reshape(2, 1)
                centerCar0 = x0 + Rot0 @ np.array([offset, 0]).reshape(2, 1)

                # end state
                xF_s = pose_traj[-1]
                RotF = np.array([[np.cos(xF_s[2]), - np.sin(xF_s[2])], [np.sin(xF_s[2]), np.cos(xF_s[2])]])
                xF = xF_s[0:2].reshape(2, 1)
                centerCarF = xF + RotF @ np.array([offset, 0]).reshape(2, 1)

                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax1.set_aspect('equal', adjustable='box')

                # plot map with only static obstacles
                self.static_map_model(ax1)

                def animate_planner(k):
                    if k % 5 == 0: # improve speed
                        ax1.clear()

                        plt.suptitle(sim_title, fontweight="bold") #  y=0.75
                        plt.xlabel(' x (m) ')
                        plt.ylabel(' y (m) ')

                        # plot static obstacle
                        self.static_map_model(ax1)

                        x = ref_traj[k][0, :]
                        y = ref_traj[k][1, :]
                        yaw = ref_traj[k][2, :]
                        for i in range(len(x)):
                            x_start = x[i]
                            y_start = y[i]
                            
                            phi = yaw[i]
                            x_end = x_start + np.cos(phi) * 0.4
                            y_end = y_start + np.sin(phi) * 0.4
                            
                            plt.arrow(x_start, y_start, x_end - x_start, y_end - y_start, head_width=0.1, head_length=0.2, fc='red', ec='red')

                        # plot pred obs traj
                        choose_obs_traj = dynObs_list_total[k]
                        obs_traj_x = [val[0] for val in choose_obs_traj]
                        obs_traj_y = [val[1] for val in choose_obs_traj]
                        ax1.plot(obs_traj_x, obs_traj_y, '-o', color='black', markersize='2')

                        # plot the whole reference trajectory
                        ax1.plot(ref_x[:, 0], ref_x[:, 1], '-o', color='royalblue', markersize='3')

                        # plot running trajectory so far
                        plot_traj_x = [val[0] for val in pose_traj]
                        plot_traj_y = [val[1] for val in pose_traj]
                        ax1.plot(plot_traj_x, plot_traj_y, 'o', color='orange', markersize='3') 
                        
                        # reference traj
                        ax1.plot(ref_traj[k][0, :], ref_traj[k][1, :], '-', color='red', markersize='2') 
                        ax1.plot(ref_traj[k][0, :], ref_traj[k][1, :], 'o', color='red', markersize='2.5') 
                        
                        # planner predict trajs
                        ax1.plot(mpc_pred_traj[k][:, 0], mpc_pred_traj[k][:, 1], '-om', markersize='3')

                        # plot current vehicle pose
                        pose_phi = pose_traj[k][2]
                        Rot = np.array([[np.cos(pose_phi), - np.sin(pose_phi)], [np.sin(pose_phi), np.cos(pose_phi)]])
                        x_cur = pose_traj[k][0:2].reshape(2, 1)
                        centerCar = x_cur + Rot @ np.array([offset, 0]).reshape(2, 1)

                        # plot dynamic obstacle
                        if not dyn_loc == []:
                            for i in range(len(dyn_loc[k])):
                                dyn_obs = dyn_loc[k][i]

                                plt.plot([dyn_obs[0][0], dyn_obs[1][0], dyn_obs[2][0],
                                        dyn_obs[3][0], dyn_obs[0][0]],
                                        [dyn_obs[0][1], dyn_obs[1][1], dyn_obs[2][1],
                                        dyn_obs[3][1], dyn_obs[0][1]], '-k')
                                if dyn_obs[5] == 1:
                                    self.sensorCircle(ax1, centerCar, pose_traj[k][2], W_ev / 2, L_ev / 2, 'r', senseDis)
                                else:
                                    self.sensorCircle(ax1, centerCar, pose_traj[k][2], W_ev / 2, L_ev / 2, 'g', senseDis)
                        else:
                            self.sensorCircle(ax1, centerCar, pose_traj[k][2], W_ev / 2, L_ev / 2, 'g', senseDis)
                        
                        self.carBox(centerCar, pose_traj[k][2], W_ev / 2, L_ev / 2)

                    plt.plot(x0[0], x0[1], "ob")
                    self.carBox(centerCar0, x0_s[2], W_ev / 2, L_ev / 2)
                    plt.plot(xF[0], xF[1], "or")
                    self.carBox_dashed(centerCarF, xF_s[2], W_ev / 2, L_ev / 2)
                
                ani = FuncAnimation(fig, animate_planner, frames=N+1, interval=200, repeat=False)
                save_name = file_name + '_N_%i' % N + '.gif'
                ani.save(save_name, writer='pillow', fps=30)
            else:
                print('invalid for even one solution')

    def sensorCircle(self, ax1, center, theta, w, l, color, senseDis):
        cx = center[0]
        cy = center[1]
        vertex_2 = [cx + l * np.cos(theta) - w * np.sin(theta), cy + l * np.sin(theta) + w * np.cos(theta)]
        vertex_3 = [cx + l * np.cos(theta) + w * np.sin(theta), cy + l * np.sin(theta) - w * np.cos(theta)]

        carFront = [(vertex_2[0] + vertex_3[0])/2 , (vertex_2[1] + vertex_3[1])/2]

        sensor_circle = plt.Circle(carFront, senseDis, color=color, alpha=0.3)
        ax1.add_patch(sensor_circle)

    def carBox(self, x0, psi, w, l):
        car1 = x0[0:2] + np.array([np.cos(psi) * l + np.sin(psi) * w, np.sin(psi) * l - np.cos(psi) * w]).reshape(2,1)
        car2 = x0[0:2] + np.array([np.cos(psi) * l - np.sin(psi) * w, np.sin(psi) * l + np.cos(psi) * w]).reshape(2, 1)
        car3 = x0[0:2] + np.array([-np.cos(psi) * l + np.sin(psi) * w, -np.sin(psi) * l - np.cos(psi) * w]).reshape(2, 1)
        car4 = x0[0:2] + np.array([-np.cos(psi) * l - np.sin(psi) * w, -np.sin(psi) * l + np.cos(psi) * w]).reshape(2, 1)
        plt.plot([car1[0], car2[0], car4[0], car3[0], car1[0]], [car1[1], car2[1], car4[1], car3[1], car1[1]], "k")

    def carBox_dashed(self, x0, psi, w, l):
        car1 = x0[0:2] + np.array([np.cos(psi) * l + np.sin(psi) * w, np.sin(psi) * l - np.cos(psi) * w]).reshape(2,1)
        car2 = x0[0:2] + np.array([np.cos(psi) * l - np.sin(psi) * w, np.sin(psi) * l + np.cos(psi) * w]).reshape(2, 1)
        car3 = x0[0:2] + np.array([-np.cos(psi) * l + np.sin(psi) * w, -np.sin(psi) * l - np.cos(psi) * w]).reshape(2, 1)
        car4 = x0[0:2] + np.array([-np.cos(psi) * l - np.sin(psi) * w, -np.sin(psi) * l + np.cos(psi) * w]).reshape(2, 1)
        plt.plot([car1[0], car2[0], car4[0], car3[0], car1[0]], [car1[1], car2[1], car4[1], car3[1], car1[1]], "--k")







