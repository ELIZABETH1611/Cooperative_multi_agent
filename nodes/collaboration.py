#!/usr/bin/env python
import numpy as np
import math
from math import pi
import os
import sys
import random
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class Collaboration(object):
    def __init__(self,data_x,data_y,data_x0,data_y0,theta_1,reward_n,reward_0):
        self.data_x =data_x
        self.data_y =data_y
        self.data_x0 =data_x0
        self.data_y0 =data_y0
        self.theta_1 = theta_1
        self.reward_0 = reward_0
        self.reward_n  = reward_n

    @property
    def get_heading_r2_r1(self):
        '''
        Calculate the orientation among all agents
        '''

        print(self.data_y, self.data_y0, self.data_x , self.data_x0)
        r2_r1_angle = math.atan2(self.data_y- self.data_y0, self.data_x - self.data_x0)

        heading_r2_r1 = r2_r1_angle - self.theta_1
        if heading_r2_r1 > pi:
            heading_r2_r1 -= 2 * pi
        elif heading_r2_r1 < -pi:
            heading_r2_r1 += 2 * pi
        self.heading = heading_r2_r1

        return self.heading


    @property
    def get_dt_r2_r1(self):
        '''
        Calculate the distances among all agents
        '''

        distance_r2_r1 = math.hypot(self.data_x - self.data_x0, self.data_x0 - self.data_y0)
        self.distances = distance_r2_r1
        return self.distances
    #
    # @property
    # def detect_agent(self):
    #     calculate_only=self.change_leader
    #         if i==calculate_only:
    #             # print(i, calculate_only)
    #             mask=(self.headings[i]<np.deg2rad(180)) & (self.headings[i]!=np.nan)
    #             mask1=(self.distances[i]<=3.5) & (self.distances[i]!=np.nan)
    #             headings=np.where(mask)[0]
    #             distances=np.where(mask1)[0]
    #             msg=False
    #             send_agent= None
    #             get_agent= None
    #             for j, k in enumerate(distances):
    #                 if k==headings[j]:
    #                     send_agent=i
    #                     get_agent= k
    #                     msg=True
    #                 else:
    #                     msg = False
    #                     pass
    #                 print("agent_"+str(send_agent)+ " sent network to agent_"+str(get_agent))
    #             return msg
    #
    # @property
    # def change_leader(self):
    #     print("change leader")
    #     return self.reward_0 < self.reward_n
