#!/usr/bin/env python
#################################################################################
#Copyright 2022 Elizabeth
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#distributed under the License is distributed on an "AS IS" BASIS,
#See the License for the specific language governing permissions and
#limitations under the License.
#################################################################################

import numpy as np


class Check_room(object):
    """docstring for ."""

    def __init__(self):
        self.room_0_x =[-3,2.3]
        self.room_0_y =[-2.3,3]
        self.room_1_x =[-3,2.3]
        self.room_1_y =[-7.5,-2.3]
        self.room_2_x =[2.3,7.5]
        self.room_2_y =[-7.5,-2.3]
        self.room_3_x =[2.3,7.5]
        self.room_3_y =[-2.3,3]

    def check_room(self,x_robot,y_robot):
        if (self.room_0_x[1]>= x_robot >= self.room_0_x[0]) and\
           (self.room_0_y[1]>= y_robot >= self.room_0_y[0]):
           return 0
        elif (self.room_1_x[1]>= x_robot >= self.room_1_x[0]) and\
              (self.room_1_y[1]>= y_robot >= self.room_1_y[0]):
           return 1
        elif (self.room_2_x[1]>= x_robot >= self.room_2_x[0]) and\
              (self.room_2_y[1]>= y_robot >= self.room_2_y[0]):
           return 2
        elif (self.room_3_x[1]>= x_robot >= self.room_3_x[0]) and\
              (self.room_3_y[1]>= y_robot >= self.room_3_y[0]):
           return 3
        else:
            raise Exception("There is no room, check your data x: "+str(x_robot)+" y: "+str(y_robot))
