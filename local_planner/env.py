# demo_setting.py
"""
Setting poblem (map and obstacles) for demo
Created on 2022/12/13
@author: Pin-Yun Hung
"""
import numpy as np

class Env:

    def __init__(self, scene_name, case):

        # scene_name
        self.scene_name = scene_name

        # map info
        self.map_size = [0, 0] # horizon, vertical, = array v * h
        self.xL = [0, 0] # the smallest x, y of map
        self.xU = [0, 0] # the largest x, y of map

        # start & goal position: [x, y, theta]
        self.startPose = [0, 0, 0]
        self.goalPose = [0, 0, 0]

        # obstacle
        self.nObs = 0
        self.vObs = 0
        self.lObs = 0
        self.obs_info = [] # obstacle informaiton:
                           ## start center & angle,
                           ## length, width (moving direction is length)
                           ## const_velocity
                           ## end center & angle
                           ## start_time, end time

        # static obstacle info
        self.static_nObs = 0 # obstacle number
        self.static_vObs = 0 # vertex number of each obstacle
        self.static_lObs = [] # vertex location (x, y)

        # problem initialize
        self.set_problem(scene_name, case)
        

    def set_problem(self, scene_name, case):
        """
        Initialize problem setting
        :param scene_name: the name of demo
        :return:
        """

        ############################
        # Problem Definition
        ############################
        # if scene_name == 'static':
        # map info
        self.xL = [case.xmin, case.xmin]
        self.xU = [case.xmax, case.xmax]
        # self.map_size = [(self.xU[0] - self.xL[0]) + 1, (self.xU[1] - self.xL[1]) + 1]
        self.map_size = [(self.xU[0] - self.xL[0]), (self.xU[1] - self.xL[1])]

        # start & goal
        # self.startPose = [case.x0, case.y0, case.theta0]
        # self.startPose = [2.15574, -0.18846999999999992, 0, 0]
        self.goalPose = [case.xf, case.yf, case.thetaf, 0]

        self.static_lObs = \
                [
                [[1.531153656,2.418774648],[0.696625579,2.312561983],[0.696625579,1.925644421],[1.614606464,2.069790179]],
                [[2.502240872,2.433947885],[1.485633943,2.418774648],],
                [[2.587067634,2.115309892],[1.493220562,2.069790179],],
                [[4.254749833,2.707066165],[2.525000729,2.145656368],],
                [[4.194056882,2.949837969],[2.396028208,2.479467599],],
                [[0.188322115,0.840757921],[-0.555166535,0.635919211],],
                [[0.271774922,0.514533309],[-0.555166535,0.340041075]],
                [[0.173148877,0.522119928],[2.001524026,0.514533309],[2.001524026,0.795238208],[0.15038902,0.817998065]],
                [[0.620759391,3.450554815],[0.620759391,3.025704158],[1.493220562,3.025704158],[1.50080718,3.435381577],[0.620759391,3.450554815]], # obs without boud
                [[-0.729658769,-0.350341243],[3.147103477,-0.494487001]],
                [[-0.555166535,3.496074528],[-0.502060203,-0.384821529],],
                [[0.35522773,4.103004038],[-0.562753154,3.435381577],],
                [[2.555347204,4.125763895],[0.302121398,4.118177276],],
                [[4.300269546,3.936098423],[2.547760585,4.095417419],],
                [[4.07267098,0.059336177],[4.019564648,3.95885828],],
                [[3.12434362,-0.494487001],[4.087844217,0.112442509],],] 

        ############################
        # Problem Setting
        ############################

        # get static obstacle number & each vertex number
        self.static_nObs = np.size(self.static_lObs, 0)

        self.static_vObs = np.ones(self.static_nObs, dtype=int)
        for i in range(self.static_nObs):
            self.static_vObs[i] = np.size(self.static_lObs[i], 0)

        
        



    