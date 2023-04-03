#!/usr/bin/env python
#################################################################################
#Copyright 2022 Elizabeth.
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
# version descentralizado
from mpi4py import MPI
import numpy as np
import rospy
import random
import math
from std_msgs.msg import Float32MultiArray
import tensorflow as tf
from agents import Agent
from environment import Behaviour
from collaboration import Collaboration
from reinforcenment import ReinforcementNetwork
from pathfinding import pathfinding
from log_class import Logfile
import os
import sys
import time
import os
import keras
import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.tensorflow_backend.set_session(sess)
from numba import cuda
tf.logging.set_verbosity(tf.logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()
status= MPI.Status()
predict_path       = False
state_size         = 29
# state_size         = 54
action_size        = 6
episodes           = 29000
episode_step       = 6000
e                  = 0
global_step        = 0
number_episode     = 1
number_rooms       = 3
number_robot       = int(size)-number_rooms #Number of robots
tt                 = 0.0
if size<=number_rooms:
    raise Exception("You need more cores than rooms!")

if rank==0:

    #Delete old folders of goal_box
    os.system('rm -r /home/mcg/catkin_ws/src/multi_robot/worlds/goal_box_'+"*")
    # os.system('rm -r /home/mcg/catkin_ws/src/multi_robot/save_model/en'+"*")
    #Creates a file with the specified number of robots and targets
    os.system('python many_robots.py'+" "+str([number_robot,number_rooms]))
    os.system("roslaunch multi_robot turtlebot3_multi.launch &")

comm.Barrier()
log= Logfile(rank)
# Process to be followed for each agent
if number_rooms+2>rank>=number_rooms:

    node = 'network'+str(rank)
    rospy.init_node(node)
    # Instances
    agents  = Agent("agent"+str(rank),action_size,state_size,number_episode,rank,number_rooms,number_robot,log)
    agents.call_sub("agent"+str(rank),action_size,state_size,number_episode,rank)
    # The agent sends its ID to the cloud in order to register.
    log.message("Register new user : "+str(rank))
    comm.send(rank, dest=agents.ID, tag=0*(size+1)+agents.ID*10000)

    #Initialization of variables to check if any agent is sending memory data
    init   = True
    init_w = True
    init_G = True
    init_R = True
    da= time.time()
    # Room_member_ID=[rank]
    Room_member_ID=range(number_rooms,number_robot+number_rooms,1)
    # Resetting the position of each agent within the environment
    e=number_episode
    fail=0
    winner=0
    time_use =time.time()
    if rank==5:
        agents.prefer=False
    else:
        agents.prefer=True

    while not rospy.is_shutdown():
        """ Your main process which runs in thread for each chunk"""
        while (e <= episodes):
            log.message("New episode, robot: " +str(rank))
            e+=1
            # Reseting of variables
            agents.reset()
            agents.environment.reset_gazebo()
            # time.sleep(0.5)

            # Resetting the position of the target and obtaining the distance to it
            agents.environment.reset(agents.robot_position_x, agents.robot_position_y)
            log.message("1. Reset position robots : " +str(rank))
            # Gets the initial state  (scan_data + [heading, current_distance,
            # wall_dist,goal_heading_initial])
            # agents.get_initial_status
            agents.last_state= agents.state
            log.message("heading initial before for robots: " +str(rank))
            agents.goal_heading_initial = agents.heading

            for step in range(episode_step):
                agents.step = step
###################################
                #Sending to robots 
                # dist_yaw_a2a = np.array(list(agents.scan_data)+[agents.robot_position_x, agents.robot_position_y, agents.environment.goal_x, agents.environment.goal_y,float(rank),fail,winner,agents.theta])
                dist_yaw_a2a = np.array([agents.robot_position_x, agents.robot_position_y, float(rank),fail,winner])
                for i in Room_member_ID:
                    if i == rank:
                        pass
                    else:
                        log.message("send dist yaw: " +str(rank)+" to "+str(i))
                        comm.isend(dist_yaw_a2a, dest= i, tag=7*(size+1)+7*10000)

                list_agents=np.array(np.zeros(32))
                for i in Room_member_ID:
                    if i == rank:
                        pass
                    else:
                        log.message("send rec yaw: " +str(rank)+" from "+str(i))
                        dist_yaw_a2a_re = comm.irecv(source=i, tag=7*(size+1)+7*10000)
                        list_agents[-1]== i

                # checks if the agent has data at the moment
                # if it doesn't, cancels the msg
                check_data  = True
                status      = False
                ini_time    =time.time()
                while check_data:
                    if not status:
                        status, dist_yaw_a2a_rec= dist_yaw_a2a_re.test()
                        if status:
                            list_agents = dist_yaw_a2a_rec
                            log.message("after receive data state 4")
                            tt=time.time()-ini_time

                    # if (status) or ((time.time() - ini_time)> 0.20):
                    if (status)  or ((time.time() - ini_time)> tt):
                        # print(" heck data, status, tt ",agents.agent_name,status, tt, time.time() - ini_time)
                        check_data = False
                
                visiting_robot_x= list_agents[0]
                visiting_robot_y = list_agents[1]
                rank1 = list_agents[2]
                fail = list_agents[3]
                winner = list_agents[4]
                if not status:
                     dist_yaw_a2a_re.Cancel()
                     log.message("No data 4, zeros")
                     if step>1:
                         visiting_robot_x=visiting_robot_x_back
                         visiting_robot_y=visiting_robot_y_back
                         rank1 =rank1_back
                         fail =fail_back
                         winner=winner_back
                         print("cancel data")

                visiting_robot_x_back= visiting_robot_x
                visiting_robot_y_back = visiting_robot_y
                rank1_back = rank1
                fail_back = fail
                winner_back = winner
              
                agents.get_heading_a2a(visiting_robot_x,visiting_robot_y, rank1, Room_member_ID)

                # If the agent has a Pa (level of knowledge) > normal proces
                # (ability to execute backward action). Each collision ends
                # the episode and the agent is restarted in a random position
                # within the room where it crashedevolve
                log.message("step "+" "+str(step))

                log.message("12.0 before send state new member from "+" "+str(rank+2)+" "+str(rank-2)+" to:  "+str(40*(size+1)+(rank-2)*10000)+" to:  "+str(40*(size+1)+rank*10000))
                comm.send(agents.last_state, dest=rank-2, tag=40*(size+1)+(rank-2)*10000)
                log.message("12.0 before recive ciostate new member from "+str(rank-2)+" to:  "+str(400*(size+1)+(rank-2)*10000))
                # time.sleep(0.1)
                # log.message("after sleep")
                data_from_cloud=comm.recv(source=(rank-2),tag=400*(size+1)+(rank)*10000)
                log.message("rec from "+str(data_from_cloud)+" to:  "+str(rank-2))

                agents.get_action_value()
                agents.action, agents.evolve_rule = data_from_cloud[0], data_from_cloud[1]


                if agents.learning.Pa >agents.learning.normal_process:
                    if agents.process =="collision":
                        break
                    else:
                        pass

                # get accion of neuronal network or call evolve rule (path planning Algorithm)
                log.message(" 2. Get action of evolve rule "+str(agents.evolve_rule)+" "+str(agents.learning.Pa)+" "+str(rank))
                # According to the type of knowledge or processes. The agent will execute:

                if agents.evolve_rule or agents.process =="collision":
                    agents.action= agents.evolve()
                    agents.action_2 = agents.action
                else:
                    agents.action_2 = agents.evolve()

                #############################################
                # Predict the future position and choose another action
                time_a=time.time()-agents.action_time
                if abs(time_a) < 0.25:
                    time.sleep( abs(0.25-(time.time()-agents.action_time)))
                agents.diff_time = time.time()-agents.action_time
                agents.prediction_position(agents.action,agents.previous_action,agents.robot_position_x,agents.robot_position_y)
                time_use=time.time()
                for i in Room_member_ID:
                    if i == rank:
                        pass
                    else:
                        bufe=[agents.new_position,agents.action]
                        comm.isend(bufe, dest=i, tag=10*(size+1)*10000)

                for i in Room_member_ID:
                    if i == rank:
                        pass
                    else:
                        new_position = comm.recv(source=i, tag=10*(size+1)*10000)
                        new_position_r1  = new_position[0]
                        new_position_r2 = agents.new_position

                if agents.learning.Pa>=0.0:
                    agents.who_has_preference()
                    print("prefer", agents.prefer,rank)
                    if agents.prefer==False:
                        dist,crashing= agents.supervisor(new_position_r1,new_position_r2)
                        if crashing:
                            agents.action=agents.choose_new_action(new_position_r1)
                        else:
                            agents.action=agents.action
                else:
                    pass

                agents.perform_action(agents.action)# Execute the action
                log.message(" 3. Execute action "+str(agents.action)+" heading "+str(np.rad2deg(agents.heading))+ " veloci "+str(agents.velocity_ang))
                agents.next_values() # Return next state, reward, done
                fail=agents.done
                winner=agents.environment.get_goalbox
                # next_Sta=agents.State_2_Se
                if agents.done or agents.environment.get_goalbox:
                    agents.number_tries(winner,fail,e)
                coll_win = np.array([fail,winner,agents.reward_agent,agents.reward_penalty])
                for i in Room_member_ID:
                    if i == rank:
                        pass
                    else:
                        comm.isend(coll_win, dest= i, tag=9*(size+1)+9*10000)

                for i in Room_member_ID:
                    if i == rank:
                        pass
                    else:
                        coll_win_re = comm.recv(source=i, tag=9*(size+1)+9*10000)
                # agents.next_state_2 =coll_win_re[2]
                # Fail, Winner,reward_agent_2,reward_penalty_2
                agents.extra_reward(coll_win_re[0],coll_win_re[1],coll_win_re[2],coll_win_re[3])
                if agents.learning.Pa >= 0.00:
                    agents.who_has_preference()
                    if winner:
                        agents.prefer = False
                else:
                    pass
                agents.save_data(rank,step,e)

                if agents.done==True or agents.environment.get_goalbox == True:
                    agents.save_data_win(rank,step,e)
                    log.message(" 4. save data"+" "+str(rank))

                # Agent append data into memory D and Eps
                agents.append_memory()
                log.message(" 5. Append memory"+" "+str(rank))

                # The agent starts to send its memory to the corresponding cloud(ID)
                # to the room it is browsing
                # At the begining all robots have ID=0, which mean they are inside room 0
                if len(agents.learning.memory_D)>20:
                    log.message(" 6.Start to send memory D"+" "+str(rank))
                    if init:
                        init=False
                        data=np.array(agents.learning.memory_D)
                        req=comm.issend(data, dest=agents.ID, tag=2*(size+1)+agents.ID*10000)
                        agents.learning.memory_D.clear()
                    else:
                        if MPI.Request.Test(req):
                            data=np.array(agents.learning.memory_D)
                            MPI.Request.Wait(req)
                            req=comm.issend(data, dest=agents.ID, tag=2*(size+1)+agents.ID*10000)
                            agents.learning.memory_D.clear()

                if len(agents.total_w)>40:
                     log.message(" 6.Start to send memory D"+" "+str(rank))
                     if init_w:
                         init_w=False
                         data_w=np.array(agents.total_w)
                         req_w=comm.issend(data_w, dest=agents.ID, tag=8*(size+1)+agents.ID*10000)
                         agents.total_w.clear()
                     else:
                         if MPI.Request.Test(req):
                             data_w=np.array(agents.total_w)
                             MPI.Request.Wait(req_w)
                             req_w=comm.issend(data_w, dest=agents.ID, tag=8*(size+1)+agents.ID*10000)
                             agents.total_w.clear()

                if len(agents.learning.memory_rules)>3:
                    log.message("8. Before Sending memory rules"+" "+str(rank))
                    if init_R:
                        init_R=False
                        data_R=np.array(agents.learning.memory_rules)
                        req_R=comm.issend(data_R, dest=agents.ID, tag=5*(size+1)+agents.ID*10000)
                        agents.learning.memory_rules.clear()
                        log.message("8. after init Sending memory rules from: "+" "+str(rank)+" to:  "+str(agents.ID))

                    else:
                        if MPI.Request.Test(req_R):
                            data_R=np.array(agents.learning.memory_rules)
                            MPI.Request.Wait(req_R)
                            req_R=comm.issend(data_R, dest=agents.ID, tag=5*(size+1)+agents.ID*10000)
                            agents.learning.memory_rules.clear()
                            log.message("8. after Sending memory rules from: "+" "+str(rank)+" to:  "+str(agents.ID))

                if len(agents.learning.memory_GT)>2:
                    log.message("8. Before Sending memory GT"+" "+str(rank))
                    if init_G:
                        init_G=False
                        data_G=np.array(agents.learning.memory_GT)
                        req_g=comm.issend(data_G, dest=agents.ID, tag=3*(size+1)+agents.ID*10000)
                        agents.learning.memory_GT.clear()
                        log.message("8. after init Sending memory GT from: "+" "+str(rank)+" to:  "+str(agents.ID))

                    else:
                        if MPI.Request.Test(req_g):
                            data_G=np.array(agents.learning.memory_GT)
                            MPI.Request.Wait(req_g)
                            req_g=comm.issend(data_G, dest=agents.ID, tag=3*(size+1)+agents.ID*10000)
                            agents.learning.memory_GT.clear()
                            log.message("8. after Sending memory GT from: "+" "+str(rank)+" to:  "+str(agents.ID))


                # if the agent has data to receive
                if comm.Iprobe(source=agents.ID,tag=6*(size+1)+agents.ID*10000+rank+100):
                    log.message("12.0 before receive network tag:"+str(6*(size+1)+agents.ID*10000+rank+100))
                    agents.learning.Pa=comm.recv(source=agents.ID,tag=6*(size+1)+agents.ID*10000+rank+100)
                    log.message("12.0 before receive Pa from "+str(rank)+" to:  "+str(agents.ID)+"TAG: "+str(6*(size+1)+agents.ID*10000+rank+100))

                # When the agent achieves the target, it calculates its total
                #reward and sends it to the different memories
                agents.work_out_best_state()
                log.message("15 after work out state " +str(rank))

                if agents.done:
                    #check if has to go at the begining of done

                    if not comm.Iprobe(source=agents.ID,tag=4*(size+1)+agents.ID*10000):
                        Done = [True,rank]
                        log.message("before Done from " +str(agents.ID)+" to "+str(rank))
                        comm.isend(Done, dest=agents.ID, tag=4*(size+1)+agents.ID*10000)
                        log.message("after Done from " +str(agents.ID)+" to "+str(rank))

                        # sys.stdout.flush()
                    agents.keep_or_reset()
                    log.message("after keep or reset " +" to "+str(rank))
                    # agents.done =False
                    if agents.finish:
                        agents.finish = False
                        agents.score = 0
                        agents.Score = 0
                        log.message("break becouse finish" +" to "+str(rank))
                        break

                    if agents.evolve_rule:
                        agents.process="collision"

                        agents.score = 0
                        agents.Score = 0
                        agents.done  = False

                        agents.cont+=1
                        # agents.learning.increase_fact()
                        agents.environment.reset(agents.robot_position_x, agents.robot_position_y)
                        log.message("before heading evolve rule" +str(rank))
                        agents.goal_heading_initial = agents.heading
                        log.message("after goal initial " +" to "+str(rank))

                        if agents.cont > 0:
                            agents.cont=0
                            log.message("break because cont " +" to "+str(rank))
                            agents.score = 0
                            agents.Score = 0
                            break
                    else:
                        break
                    log.message("17 after done "+str(rank))

                agents.last_state=agents.state
                agents.previous_action=agents.action

                log.message("after last state "+str(rank))
            if not comm.Iprobe(source=agents.ID,tag=4*(size+1)+agents.ID*10000):
                Done = [True,rank]
                log.message("before finish episode from " +str(agents.ID)+" to "+str(rank))
                comm.isend(Done, dest=agents.ID, tag=4*(size+1)+agents.ID*10000)
                log.message("after finish episode from " +str(agents.ID)+" to "+str(rank))

            if agents.time_out:
                agents.time_out=False
                agents.score = 0
                agents.Score = 0
                break
#################################################################################
# Process to be followed for each cloud
#################################################################################
# One cloud is one room
if rank<=number_rooms-3:
    state_size_cluster=state_size

    with tf.device('/GPU:0'):
        # cluster.load_model = True
        cluster=ReinforcementNetwork(state_size_cluster,action_size,number_episode,load=True)
        step=0
        # Room_member_ID=[]
        Room_member_ID=range(number_rooms,number_robot+number_rooms,1)
        while not rospy.is_shutdown():
            step+=1
            cluster.global_step=step
            train= cluster.start_training()

            if (step%2000==0) and (train==True) :
                # cluster.update_target_cloud()
                cluster.update_target_cloud()
                cluster.update_policy_cloud()
                log.message( "19.1 UPDATE TARGET NETWORK CLOUD "+str(rank))

            if step%2000==0 and (step>0):
                cluster.save_model(rank,step)
                log.message("20 Save model")

            # if  step%10000==0:
            #     cluster.save_lr
            #     log.message("update lr")

            if not train:
                time.sleep(2)
                log.message("21 Sleeping")

            # Comunication section
            # n*(size+1)+rank*10000
            # Agent sended to the cloud its ID in order to register in the cloud
            # also q, target networks are send to the cloud
            #ready
        
            dones=False
            if comm.Iprobe(source=MPI.ANY_SOURCE,tag=4*(size+1)+rank*10000):
                log.message("before receive done" +str(rank))
                Dones=comm.recv(source=MPI.ANY_SOURCE, tag=4*(size+1)+rank*10000)
                log.message(" receive done" +str(rank)+"tag: "+str(4*(size+1)+rank*10000))
                dones=Dones[0]
                ag=Dones[1]
                cluster.increase_fact()
                cluster.update_target_cloud()
                cluster.update_policy_cloud()
                log.message( "19.1 UPDATE TARGET s CLOUD for done " + str(rank))
            log.message("ROM members before unsubscribe cloud : "+str(Room_member_ID))

            log.message("7.0 waiting append data memory D " + str(rank) +"tag: "+str(2*(size+1)+rank*10000))
            if comm.Iprobe(source=MPI.ANY_SOURCE,tag=2*(size+1)+rank*10000):
                log.message("7.0 append data memory D " + str(rank) +"tag: "+str(2*(size+1)+rank*10000))
                data = comm.recv(source=MPI.ANY_SOURCE, tag=2*(size+1)+rank*10000)
                for i in range(len(data)):
                    cluster.append_D(data[i][0], data[i][1],data[i][2], data[i][3], data[i][4],data[i][5], data[i][6], data[i][7], data[i][8])

            if comm.Iprobe(source=MPI.ANY_SOURCE,tag=8*(size+1)+rank*10000):
                log.message("7.0 append data memory D " + str(rank) +"tag: "+str(8*(size+1)+rank*10000))
                data_w = comm.recv(source=MPI.ANY_SOURCE, tag=8*(size+1)+rank*10000)
                cnt_win=0
                cnt_fail=0
                for i in range(len(data_w)):
                    cluster.total_w.append([data_w[i][0], data_w[i][1],data_w[i][2]])
                    cnt_win+=data_w[i][0]
                    cnt_fail+=data_w[i][1]
                cluster.learning_rate=cluster.lr_soft(cnt_win,cnt_fail)
                cluster.save_lr

             # Ask if any agent has sent data from memory_rules
            log.message("7.0 waiting append data memory rules " + str(rank) +"tag: "+str(5*(size+1)+rank*10000))
            if comm.Iprobe(source=MPI.ANY_SOURCE,tag=5*(size+1)+rank*10000):
                log.message("7.0  append data memory rules " + str(rank) +"tag: "+str(5*(size+1)+rank*10000))
                # log.message("9.1 waiting second append data rules" + str(rank))
                data_R = comm.recv(source=MPI.ANY_SOURCE, tag=5*(size+1)+rank*10000)
                for q in range(len(data_R)):
                    cluster.append_Rules(data_R[q][0], data_R[q][1],data_R[q][2], data_R[q][3], data_R[q][4],data_R[q][5], data_R[q][6], data_R[q][7], data_R[q][8])

            # Ask if any agent has sent data from memory_G
            log.message("7.0 waiting append data memory g " + str(rank) +"tag: "+str(3*(size+1)+rank*10000))
            if comm.Iprobe(source=MPI.ANY_SOURCE,tag=3*(size+1)+rank*10000):
                log.message("7.0  append data memory g " + str(rank) +"tag: "+str(3*(size+1)+rank*10000))
                # log.message("9.1 waiting second append data  g" + str(rank))
                data_G = comm.recv(source=MPI.ANY_SOURCE, tag=3*(size+1)+rank*10000)
                for p in range(len(data_G)):
                    cluster.append_GT(data_G[p][0], data_G[p][1],data_G[p][2], data_G[p][3], data_G[p][4],data_G[p][5], data_G[p][6], data_G[p][7], data_G[p][8])


            # Updating the networks of the registered agents in the cloud
            # with the current cloud network
            if ((step%20==0) and (step>0)) and (train==True):
                for g in Room_member_ID:
                        comm.send(cluster.Pa,dest=g,tag=6*(size+1)+rank*10000+g+100)
                        log.message("11.0  for send pa network " + str(rank)+ " to "+str(g))
                        
            if ((step%20==0) and (step>0)) and (train==True):
                for h in [1,2]:
                    log.message("111.0 waiting for send network " + str(h)+ " to "+str(11*(size+1)+rank*10000+h+100))
                    if not comm.Iprobe(source=MPI.ANY_SOURCE,tag=11*(size+1)+rank*10000+h+100):
                        weight_q      = cluster.q_model.get_weights()
                        log.message("1110 waiting for send network " + str(h)+ " to "+str(11*(size+1)+rank*10000+h+100))
                        #Sending q network to each agent
                        comm.send(weight_q,dest=h,tag=11*(size+1)+rank*10000+h+100)
                        log.message("11.0 send q network " + str(rank) + " to "+ str(h))
                        #Sending target network to each agent
                        #Sending members of the same room to each agent
                        comm.send(Room_member_ID,dest=h,tag=31*(size+1)+rank*10000+h+100)
                        log.message("11.0  for send id network " + str(rank)+ " to "+str(h))
                        #Sending Pa members of the same room to each agent
                        comm.send(cluster.Pa,dest=h,tag=61*(size+1)+rank*10000+h+100)
                        log.message("11.0  for send pa network " + str(rank)+ " to "+str(h))
                  

#######################################
#  Cloud agent selection of actions   #
#######################################

if (number_rooms)>rank>0:
    log.message("rooms  " + str(rank) + str(number_rooms))
    state_size_cluster=state_size
    step=0
    with tf.device('/GPU:0'):
        # cluster_back_model =True
        cluster_back=ReinforcementNetwork(state_size_cluster,action_size,number_episode,load=True)
        Room_member_ID=[]
        while not rospy.is_shutdown():
           
            if step%2000==0 and (step>0):
                cluster_back.save_model(rank,step)
                log.message("20 Save model")

            if comm.Iprobe(source=rank+2,tag=40*(size+1)+rank*10000):
                step=step+1
                state=comm.recv(source=rank+2,tag=40*(size+1)+rank*10000 )
                log.message("achive waiting 2"+str(40*(size+1)+rank*10000)+" "+str(rank+2))
                action, evolve_rule = cluster_back.get_action(np.array(state[0]))
                data=[action, evolve_rule]
                log.message("12.0 befope send state new member from "+str(rank)+" to:  "+str(400*(size+1)+(rank+2)*10000))
                comm.send(data, dest=rank+2, tag=400*(size+1)+(rank+2)*10000)
                log.message("action nube , evolve_rule" +str( action)+ " "+str( evolve_rule)+ " "+str(data) )
                log.message("12.0 after  send cloud to "+str(rank+2)+" from:  "+str(rank)+"TAG: "+str(400*(size+1)+(rank+2)*10000))

            if comm.Iprobe(source=0,tag=11*(size+1)+0*10000+rank+100):
                weight_q=comm.recv(source=0,tag=11*(size+1)+0*10000+rank+100)
                cluster_back.q_model.set_weights(weight_q)
                log.message("12.0 after receive q network from "+str(rank)+" to:  "+str(0)+"TAG: "+str(2*(size+1)+0*10000+rank+100))
                Room_member_ID=comm.recv(source=0,tag=31*(size+1)+0*10000+rank+100)
                log.message("ROM members after unsubscribe : "+str(Room_member_ID)+"TAG: "+str(3*(size+1)+0*10000+rank+100))
                cluster_back.Pa=comm.recv(source=0,tag=61*(size+1)+0*10000+rank+100)
                log.message("12.0 before receive Pa from "+str(rank)+" to:  "+str(0)+"TAG: "+str(6*(size+1)+0*10000+rank+100))

            # log.message("waiting 2"+str(40*(size+1)+rank*10000)+" "+str(rank+2))