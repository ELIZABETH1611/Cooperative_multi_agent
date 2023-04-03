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

import numpy as np
import math
from math import pi
import os
import sys
import random

"""
Creates a file with the specified number of robots and targets
"""
import matplotlib
color=[]
for cname, hex in matplotlib.colors.cnames.items():
    color.append(cname)


"""
Creates a file with the specified number of robots
"""
arguments = sys.argv
agents = int(arguments[1][1])
path= "/home/mcg/catkin_ws/src/multi_robot/launch/multi_agent.launch"
path1= "/home/mcg/catkin_ws/src/multi_robot/worlds/goal_box/model"

out_str= "<launch>\n"+\
       '<param name="robot_description"\n'+\
       'command="$(find xacro)/xacro --inorder $(find turtlebot3_description)/urdf/turtlebot3_burger.urdf.xacro" />"\n'
x_initial=int(arguments[2][0])
xp_initial= 0.5
xy_initial= -1.3
Y_initial= 3.1416
for i in range(x_initial,agents+x_initial,1):
  # ra=random.choice([0,1,-1.5,2,4,-2.5])
  if i==4:
      id=0
  else:
      id=1

  # ra=[-0.8,0]
  # ra=[-1,-1]
  ra=[2, 2]
  # ra=[-1,-1.9]
  # ra=[-0.8,0.8]
  rax=ra[id]
  # ra1=[-0.8,-6]

  ra1=[1.8,-1.8]
  # ra1=[1,-1]
  # ra1=[-2,1.6]
  # ra1=[-0.8,1.1]
  ray=ra1[id]
  out_str += '<group ns="'+"agent"+str(i)+'">\n'+\
  '<param name="tf_prefix" value="'+"agent"+str(i)+'_tf" />\n'+\
  '<include file="$(find multi_robot)/launch/one_agent.launch" >\n' + \
  '<arg name="init_pose" value="-x '+str(rax)+' -y '+str(ray)+' -z 0 -Y '+str(Y_initial)+'" />\n'+\
  '<arg name="agent_name"  value="'+"agent"+str(i)+'" />\n'+\
  '</include>\n'+\
  '</group>\n\n'
out_str+='</launch>'
with open(path,"w") as out :
  out.write(out_str)

"""
Creates a file with the specified number of targets, each of them with a different color
"""

for i in range(x_initial,agents+x_initial,1):
    # color=random.choice(["Black","Red","Green","Yellow","Purple","Turquoise","Blue"])
    color=["Black","Yellow","Purple","Green","Turquoise","Blue","Red"]
    if i >= len(color):
        col = random.choice(["Black","Red","Green","Yellow","Purple","Turquoise","Blue"])
    else:
        col = color [i]
    #col=random.choice(["FlatBlack","Black","Red","Green","Yellow","Purple","Turquoise","RedEmissive","GreenEmissive","PurpleEmissive","BlueLaser","BlueEmissive","JointAnchor","Blue","Skull","ExclamationPoint","QuestionMark","SmileyHappy","SmileySad","SmileyDead","SmileyPlain","WoodFloor","CeilingTiled","PaintedWall","PioneerBody","Pioneer2Body","CloudySky","RustySteel","Chrome","BumpyMetal","GrayGrid","Rocky","GrassFloor","Rockwall","RustyBarrel","WoodPallet","Fish","LightWood","WoodTile","Brick","DepthMap","PCBGreen","Turret","EpuckBody","EpuckRing","EpuckPlate","EpuckLogo","EpuckMagenta","EpuckGold"])
    os.system("mkdir /home/mcg/catkin_ws/src/multi_robot/worlds/goal_box_"+"agent"+str(i))
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
           "            <name>Gazebo/"+col+"</name>\n"+\
           "          </script>\n"+\
           "          <ambient>1 0 0 1</ambient>\n"+\
           "        </material>\n"+\
           "      </visual>\n"+\
           "      <pose frame=''>0 0 0 0 -0 0</pose>\n"+\
           "    </link>\n"+\
           "    <static>1</static>\n"+\
           "  </model>\n"
    out_str+='</sdf>'
    with open("/home/mcg/catkin_ws/src/multi_robot/worlds/goal_box_"+str("agent"+str(i))+"/model_"+str("agent"+str(i))+".sdf","w") as out :
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
    with open("/home/mcg/catkin_ws/src/multi_robot/worlds/goal_box_"+str("agent"+str(i))+"/model_"+str("agent"+str(i))+".config","w") as out_1 :
      out_1.write(out_str_1)
