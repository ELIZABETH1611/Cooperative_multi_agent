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
import math
from math import pi
import os
import sys
import random
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class Configuration(object):
    """
    Creates a file with the specified number of robots and targets
    """
    def __init__(self,number_robot):
        self.agents = number_robot
        self.path= "/home/mcg/catkin_ws/src/multi_robot/launch/multi_agent.launch"
        # self.path1= "/home/mcg0102/catkin_ws/src/multi_robot/worlds/goal_box/model"
        self.create_file_target()
        self.create_file_robot()

    def create_file_robot(self):
      out_str= "<launch>\n"+\
               '<param name="robot_description"\n'+\
               'command="$(find xacro)/xacro --inorder $(find turtlebot3_description)/urdf/turtlebot3_burger.urdf.xacro" />"\n'
      x_initial= 1
      Y_initial= 3.13
      ra=random.choice([-1,1])
      #range(start, stop, step)
      for i in range(1,self.agents,1):
          out_str += '<group ns="'+"agent"+str(i)+'">\n'+\
          '<param name="tf_prefix" value="'+"agent"+str(i)+'_tf" />\n'+\
          '<include file="$(find multi_robot)/launch/one_agent.launch" >\n' + \
          '<arg name="init_pose" value="-x '+str(ra*x_initial+i*ra*0.5)+' -y '+str(ra*x_initial+i*ra*0.5)+' -z 0 -Y '+str(Y_initial-i*0.5)+'" />\n'+\
          '<arg name="agent_name"  value="'+"agent"+str(i)+'" />\n'+\
          '</include>\n'+\
          '</group>\n\n'
      out_str+='</launch>'
      with open(self.path,"w") as out :
          out.write(out_str)

    def create_file_target(self):
      for i in range(1,self.agents,1):
          os.system("pwd")
          os.system("mkdir /home/mcg/catkin_ws/src/multi_robot/worlds/goal_box_"+"agent"+str(i))
          # print("inside")
          # os.system("/home/mcg0102/catkin_ws/src/multi_robot/worlds/goal_box_"+"agent"+str(i)+ "&")
          out_str= "<?xml version='1.0'?>\n"+\
                   "<sdf version='1.6'>\n"+\
                   "  <model name='goal_box_"+"agent"+str(i)+"'>\n"+\
                   "    <pose frame=''>0 0 0 0 -0 -1.5708</pose>\n"+\
                   "    <link name='goal_box_"+"agent"+str(i)+"'>\n"+\
                   "      <visual name='goal_box_"+"agent"+str(i)+"'>\n"+\
                   "        <pose frame=''>0 0 0.0005 0 -0 0</pose>\n"+\
                   "        <geometry>\n"+\
                   "          <box>\n"+\
                   "            <size>0.5 0.5 0.001</size>\n"+\
                   "          </box>\n"+\
                   "        </geometry>\n"+\
                   "        <material>\n"+\
                   "          <script>\n"+\
                   "            <uri>file://media/materials/scripts/gazebo.material</uri>\n"+\
                   "            <name>Gazebo/Grey</name>\n"+\
                   "          </script>\n"+\
                   "          <ambient>1 0 0 1</ambient>\n"+\
                   "        </material>\n"+\
                   "      </visual>\n"+\
                   "      <pose frame=''>0 0 0 0 -0 0</pose>\n"+\
                   "    </link>\n"+\
                   "    <static>1</static>\n"+\
                   "  </model>\n"
          out_str+='</sdf>'
          with open("/home/mcg0102/catkin_ws/src/multi_robot/worlds/goal_box_"+str("agent"+str(i))+"/model_"+str("agent"+str(i))+".sdf","w") as out :
              out.write(out_str)

          out_str_1= "<?xml version='1.0'?>\n"+\
                   "<model>\n"+\
                   "  <name>goal_box_"+"agent"+str(i)+"</name>\n"+\
                   "  <version>1.0</version>\n"+\
                   "  <sdf version='1.6'>model_"+"agent"+str(i)+".sdf</sdf>\n"+\
                   "  <author>\n"+\
                   "      <name></name>\n"+\
                   "      <email></email>\n"+\
                   "  <author>\n"+\
                   "  <description></description>\n"
          out_str_1+='</model>'
          with open("/home/mcg0102/catkin_ws/src/multi_robot/worlds/goal_box_"+str("agent"+str(i))+"/model_"+str("agent"+str(i))+".config","w") as out_1 :
              out_1.write(out_str_1)
