import casadi
import numpy as np

class MPCPlanner:

    def __init__(self, nx, nu, wheelbase):
        # pass
        self.nx = nx
        self.nu = nu
        self.wheelbase = wheelbase
        self.vertex_num = 4

    def planner_mpc(self, Ts, param, N, x0, xL, xU, uL, uU, xref, u0, dynObs_list):

        # is_slack = False

        # set nx, nu
        nx = self.nx
        nu = self.nu
        L = self.wheelbase

        # optimization model
        opti = casadi.Opti()

        # state variable
        x = opti.variable(nx, N + 1)

        # control input variable
        u = opti.variable(nu, N)

        N_obs = len(dynObs_list)

        # objective fxn
        def obj_rule(x, u, xref):
            cost = 0
            cost += casadi.sumsqr(param[0]*u[0,:]) + casadi.sumsqr(param[1]*u[1,:])

            diff_u1 = casadi.mtimes([1/Ts, u[0, 1:] - u[0, :-1]])
            diff_u2 = casadi.mtimes([1/Ts, u[1, 1:] - u[1, :-1]])
            cost += param[2]*casadi.sumsqr(diff_u1) + param[3]*casadi.sumsqr(diff_u2)

            cost += param[4] * casadi.sumsqr((u[0, 0] - u0[0]) / Ts)
            cost += param[5] * casadi.sumsqr((u[1, 0] - u0[1]) / Ts)

            cost += param[6] * casadi.sumsqr(x[0,:].T - xref[0,:])
            cost += param[7] * casadi.sumsqr(x[1,:].T - xref[1,:])
            cost += param[8] * casadi.sumsqr((x[2,:]).T - (xref[2,:]))

            cost += param[10] * casadi.sumsqr(x[0:2, N] - xref[0:2,N].T)

            if len(dynObs_list) > 0:
                for i in range(N_obs):
                    obs_pose = dynObs_list[i]
                    cost += 10 * 1/np.sqrt((x[0,i]-obs_pose[0])**2 + (x[1,i]-obs_pose[1])**2)

            return cost

        opti.minimize(obj_rule(x, u, xref))

        #############################
        # dynamic fxn
        #############################
        for k in range(N):
            x1_expr = x[0,k] + Ts * (x[3,k] + Ts/2*u[1,k]) * casadi.cos(x[2,k] + Ts/2 * x[3,k] * casadi.tan(u[0,k]) / L)
            x2_expr = x[1,k] + Ts * (x[3,k] + Ts/2*u[1,k]) * casadi.sin(x[2,k] + Ts/2 * x[3,k] * casadi.tan(u[0,k]) / L)
            x3_expr = x[2,k] + Ts * (x[3,k] + Ts/2*u[1,k]) * casadi.tan(u[0,k]) / L
            x4_expr = x[3,k] + Ts * u[1,k]

            opti.subject_to(x[0,k+1] == x1_expr)
            opti.subject_to(x[1,k+1] == x2_expr)
            opti.subject_to(x[2,k+1] == x3_expr)
            opti.subject_to(x[3,k+1] == x4_expr)
        
        #############################
        # state constraint
        #############################
        for i in range(nx - 1):
            opti.subject_to(opti.bounded(xL[i], x[i, :], xU[i]))

        #############################
        # input constraint
        #############################
        for i in range(nu):
            opti.subject_to(opti.bounded(uL[i], u[i, :], uU[i]))

        #############################
        # acceleration constraint
        #############################
        # for k in range(N):
        #     if k == 0:
        #         angle_vel = (u0[0] - u[0, k]) / Ts
        #     else:
        #         angle_vel = (u[0, k - 1] - u[0, k]) / Ts
        #     opti.subject_to(opti.bounded(-delta_vel, angle_vel, delta_vel))

        #############################
        # initial state constraint
        #############################
        opti.subject_to(x[:, 0] == x0)

        #############################
        # terminal state constraint
        #############################
        # opti.subject_to(x[:, N] == xref[:, N])


        # #############################
        # # obstacle avoidance constraint
        # #############################

        opti.set_initial(x, xref)
        # opti.set_initial(u, uref)        

        opti_setting = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 500,
            'print_time': False,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6
            }

        x_Opt = []
        u_Opt = []
        feas = False
        try:
            # print("========= Runing: full dimension =========")
            opti.solver('ipopt', opti_setting)
            sol = opti.solve()
            x_Opt = np.asarray(sol.value(x))
            u_Opt = np.asarray(sol.value(u))
            feas = True
        except:
            x_Opt = np.asarray(opti.debug.value(x))
            u_Opt = np.asarray(opti.debug.value(u))

        return x_Opt, u_Opt, feas

