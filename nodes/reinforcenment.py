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
import rospy
import random
from collections import deque
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import History, TerminateOnNaN, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


class ReinforcementNetwork(object):
    """
    Algorithm DDRLE-GE
    """
    def __init__(self,state_size,action_size,number_episode,load):
        self.dirPath            = os.path.dirname(os.path.realpath(__file__))
        self.dirPath            = self.dirPath.replace('multi_robot/nodes', 'multi_robot/save_model/environment')
        # self.dirPath            =  "/home/mcg/catkin_ws/src/multi_robot/save_model/environment"
        self.state_size         = state_size
        self.state_size_2       = state_size+1+1
        self.action_size        = action_size
        self.load_episode       = 1
        self.discount_factor    = 0.99
        # self.learning_rate      = 0.01 #0.00025
        # self.learning_rate      = 0.001 #0.00025
        self.learning_rate      = 0.00030 #0.00025
        # self.learning_rate      = 0.0006#0.00030 #0.00025
        self.batch_size         = 96#128s
        self.train_start        = 96#128
        self.Pa                 = 0
        self.Pbest              = 0.01
        self.Pbest_max          = 0.95
        self.size_layer_1       = 512#512
        self.size_layer_2       = 512#512
        self.reward_max         = 0
        self.tau                = 0.1#0.3
        self.target_value       = 0
        self.dropout            = 0.2
        self.lim_q_s            = 0.95
        self.lim_q_i            = 0.40#0.186#0.25  0.6
        self.target_update      = 2000
        self.start_or           = 0.186 #0.186
        self.lim_train          = 0.186 #0.25#0.36
        self.memory_D           = deque(maxlen=100000)
        self.memory_GT          = deque(maxlen=100000)
        self.memory_EPS         = deque(maxlen=100000)
        self.memory_rules       = deque(maxlen=100000)
        self.normal_process     = 0.25
        self.increase_factor    = 0.99#.97 env1 for paper
        self.global_step        = 1
        # self.load_epidose       = 0
        self.load_model         = True
        self.loss               = 'mse'
        self.activation_output  = 'linear'
        self.activation_layer   = 'relu'
        self.kernel_initializador = 'lecun_uniform'
        self.possible_actions        ="left"
        # self.eta                  = 0
        self.q_model              = self.q_network()
        self.target_model         = self.target_network()
        self.total_w =[]
        self.total_f =[]
        self.q_model_actor              = self.q_network()
        self.target_model_critic         = self.target_network()
        if self.load_model:
            self.q_model.set_weights(load_model(self.dirPath+str(self.load_episode)+'_q_model'+".h5").get_weights())
            self.target_model.set_weights(load_model(self.dirPath+str(self.load_episode)+'_target_model'+".h5").get_weights())
            self.Pa,self.Pbest= self.load_mode()
            # print(self.Pa, self.Pbest)
        else:
            self.q_model              = self.q_network()
            self.target_model         = self.target_network()

    def load_mode(self):
        with open(self.dirPath+str(self.load_episode)+'.json') as outfile:
            param = json.load(outfile)
            Pa = param.get('Pa')
            Pbest = param.get('Pbest')
        return  Pa, Pbest

    # @property
    # def decayed_lr(self):
    #     sel.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay( initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9)
    #     return self.learning_rate
    # @property
    # def decayed_lr(self):
    #     if (self.global_step % 10000== 0) and (self.global_step>0):
    #         decay_steps=10000
    #         decay_rate=0.9
    #         if self.learning_rate >= 0.0001:
    #             self.learning_rate = self.learning_rate *decay_rate **(self.global_step / decay_steps)
    #     return self.learning_rate

    def lr_soft(self,win,fails):
        if self.learning_rate >= 0.0001 and self.Pa > 0.94:
            self.learning_rate = self.learning_rate*np.exp(-(win/float(win+fails))**6)
            print("New learning",win,fails,self.learning_rate)
        return self.learning_rate

    def q_network(self):
        '''
        In this network we evaluate the action of the q_network and predict the following value of Q(s',a)
        '''
        q_model = Sequential()
        q_model.add(Dense(self.size_layer_1, input_shape=(self.state_size,), activation= self.activation_layer, kernel_initializer = self.kernel_initializador))
        q_model.add(Dense(self.size_layer_2, activation= self.activation_layer, kernel_initializer=self.kernel_initializador))
        q_model.add(Dropout(self.dropout))
        q_model.add(Dense(self.action_size, kernel_initializer=self.kernel_initializador))
        q_model.add(Activation(self.activation_output))
        # q_model.compile(loss=self.loss, optimizer=RMSprop(lr=self.learning_rate, rho=0.9,  epsilon=1e-08, decay=0.0), metrics=['acc'])
        q_model.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate,  beta_1=0.9, beta_2=0.999, epsilon=1e-08), metrics=['acc'])

        return q_model

    def target_network(self):
        '''
        In this network we evaluate the action of the q_network and predict the following value of Q(s',a)
        '''
        target_model = Sequential()
        target_model.add(Dense(self.size_layer_1, input_shape=(self.state_size_2,), activation= self.activation_layer, kernel_initializer = self.kernel_initializador))
        target_model.add(Dense(self.size_layer_2, activation= self.activation_layer, kernel_initializer=self.kernel_initializador))
        target_model.add(Dropout(self.dropout))
        target_model.add(Dense(self.action_size, kernel_initializer=self.kernel_initializador))
        target_model.add(Activation(self.activation_output))
        # target_model.compile(loss=self.loss, optimizer=RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-08, decay=0.0), metrics=['acc'])
        target_model.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate,  beta_1=0.9, beta_2=0.999, epsilon=1e-08), metrics=['acc'])
        # print("N larning ",self.learning_rate)
        return target_model

    def get_Qvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * next_target


    def update_target_cloud(self):
        '''
        Updtate target network in the cloud
        '''
        q_model_theta = self.q_model.get_weights()
        target_model_theta = self.q_model_actor.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_model_theta, target_model_theta):
            target_weight = q_weight*self.tau + target_weight *(1-self.tau)
            target_model_theta[counter] = target_weight
            counter += 1
        rospy.loginfo("UPDATE TARGET NETWORK CLOUD")
        self.q_model_actor.set_weights(target_model_theta)
        # self.q_model.set_weights(target_model_theta)


    def update_policy_cloud(self):
        '''
        Updtate target network in the cloud
        '''
        q_model_theta = self.target_model.get_weights()
        target_model_theta = self.target_model_critic.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_model_theta, target_model_theta):
            target_weight = q_weight*self.tau + target_weight *(1-self.tau)
            target_model_theta[counter] = target_weight
            counter += 1
        rospy.loginfo("UPDATE TARGET NETWORK CLOUD")
        # self.target_model.set_weights(target_model_theta)
        self.target_model_critic.set_weights(target_model_theta)


    def merge_target_cloud(self,q_model_theta,target_model_theta):
        '''
        Merge Q and target network from one robot (area 2) with the cloud (area 1)
        '''
        q_model_theta = q_model_theta
        target_model_theta = self.q_model.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_model_theta, target_model_theta):
            target_weight = target_weight *(1-self.tau) + q_weight*self.tau
            target_model_theta[counter] = target_weight
            counter += 1
        self.q_model.set_weights(target_model_theta)

        q_model_theta_to = target_model_theta
        target_model_theta_to = self.target_model.get_weights()
        counter_t_to = 0
        for q_weight_to, target_weight_to in zip(q_model_theta_to, target_model_theta_to):
            target_weight_to = target_weight_to *(1-self.tau) + q_weight_to*self.tau
            target_model_theta_to[counter_t_to] = target_weight_to
            counter_t_to += 1
        self.target_model.set_weights(target_model_theta_to)
        rospy.loginfo("UPDATE MERGE NETWORKS AGENT - CLOUD")

    def get_Pa(self):
        '''
        Calculates the probability by "Semi-Uniform Distributed Exploration"
        Pbes=0 purely random exploration, Pbest=1 pure exploitation.
        '''
        self.Pa = self.Pbest + ((1.0-self.Pbest)/self.action_size)

        return self.Pa

    def get_action(self,state):
        '''
        Action is determined based on directed knowledge, hybrid knowledge
        or autonomous knowledge
        '''
        n2 = np.random.rand()
        n3 = np.random.rand()

        if self.Pa <= self.lim_q_i:
            self.q_value = np.zeros(self.action_size)
            action = None
            evolve_rule = True

        elif self.lim_q_s>self.Pa >self.lim_q_i:
            if n3 > self.Pa:
                # self.q_value = np.zeros(self.action_size)
                # action = None
                # evolve_rule = True
                self.q_value = self.q_model.predict(state.reshape(1, len(state)))
                q_value = self.q_value[0][:-1]
                # mask=~(np.arange(self.action_size -1)==np.argmax(q_value))
                # q_value=q_value[mask]
                # q_value= q_value-min(q_value)
                # q_value = q_value/sum(q_value)
                # action = np.random.choice(np.arange(self.action_size-1)[mask],p=q_value)
                action = np.random.choice(np.arange(self.action_size-1))
                evolve_rule = False

            else:
                self.q_value = self.q_model.predict(state.reshape(1,len(state)))
                action = np.argmax(self.q_value[0][:5])
                evolve_rule = False

        else:
            if n2 <= self.Pa:
                self.q_value = self.q_model.predict(state.reshape(1,len(state)))
                action = np.argmax(self.q_value[0][:5])
                evolve_rule = False

            else:
                self.q_value = self.q_model.predict(state.reshape(1, len(state)))
                q_value = self.q_value[0][:-1]
                # mask=~(np.arange(self.action_size -1)==np.argmax(q_value))
                # q_value=q_value[mask]
                # q_value= q_value-min(q_value)
                # q_value = q_value/sum(q_value)
                # action = np.random.choice(np.arange(self.action_size-1)[mask],p=q_value)
                action = np.random.choice(np.arange(self.action_size-1))
                evolve_rule = False

        # print(action, evolve_rule)
        return action, evolve_rule

    def append_D(self, state, action, reward, next_state, done, action_2, state_2,next_states_2,Reward ):
        '''
        Memory used to train the model
        '''
        self.memory_D.append((state, action, reward, next_state, done, action_2, state_2,next_states_2,Reward ))

    def append_EPS(self, state,action, reward, next_state, done, action_2, state_2,next_states_2,Reward ):
        '''
        Memory for each episode,
        '''
        self.memory_EPS.append((state,action, reward, next_state, done, action_2, state_2,next_states_2,Reward ))

    def append_GT(self, state,action, reward, next_state, done, action_2, state_2,next_states_2,Reward ):
        '''
        Memory for each episode,
        '''
        self.memory_GT.append((state,action, reward, next_state, done, action_2, state_2,next_states_2,Reward ))

    def append_Rules(self, state,action, reward, next_state, done, action_2, state_2,next_states_2,Reward ):
        '''
        Memory for all path choose by rules,
        '''
        self.memory_rules.append((state,action, reward, next_state, done, action_2, state_2,next_states_2,Reward ))

    def winning_state(self):
        '''
        When the robot reaches the target, the temporary memory is copied into the
        main memory depending on the average reward.
        '''
        all_rewards = map(lambda x: x[2],self.memory_EPS)
        reward_aver = np.mean(all_rewards)
        if reward_aver > self.reward_max:
            self.reward_max = reward_aver
            self.memory_D.extend(self.memory_EPS)
            self.memory_D.extend(self.memory_EPS)
            self.memory_D.extend(self.memory_EPS)
            self.memory_EPS.clear()
            rospy.loginfo("Winning State with reward_max !!!")
        else:
            self.memory_EPS.clear()
            rospy.loginfo("Normal Win !!!")

    def best_state(self):
        '''
        When the robot reaches the goal with the best time, the temporary
        memory is copied into the main memory depending on the best time.
        '''
        self.memory_GT.extend(self.memory_EPS)
        self.memory_D.extend(self.memory_EPS)
        self.memory_D.extend(self.memory_EPS)
        self.memory_D.extend(self.memory_EPS)
        self.memory_EPS.clear()
        rospy.loginfo("Great Time !!!")

    def increase_fact(self):
        '''
        According with Pbest the value of Pa is updated
        '''
        if self.Pbest < self.Pbest_max:
            self.Pbest /= self.increase_factor
        elif self.Pbest > self.Pbest_max:
            self.Pbest = self.Pbest_max
            self.Pa = self.Pbest_max
        else:
            self.Pbest = self.Pbest
        self.get_Pa()

    def start_training(self):
        '''
        Start to train
        '''
        if len(self.memory_D) > (self.train_start):
            self.train_model()
            return True
        else:
            return False

    def save_model(self,rank,e):
        '''
        Save .h5 files with weights and .json with Pa, Pbest
        '''
        self.q_model.save(self.dirPath + str(e)+"_"+str(rank)+'_q_model' +'.h5')
        self.target_model.save(self.dirPath + str(e) +"_"+str(rank)+'_target_model'+'.h5')
        param_keys = ['Pa','Pbest']
        param_values = [self.Pa, self.Pbest]
        param_dictionary = dict(zip(param_keys, param_values))
        with open(self.dirPath +"_"+str(rank)+ str(e) + '.json', 'w') as outfile:
            json.dump(param_dictionary, outfile)

    def experience_replay(self):
        '''
        Based on probability choose random samples or continuous samples with
        the best average rewards
        '''
        batch_save   = []
        batch_rule_save   = []
        max_rew_save = []
        max_rew_save2 = []
        num_2 = random.randrange(0,len(self.memory_D)-int(self.batch_size/8.0))
        if len(self.memory_GT)>2:
            mini_batch1 = deque(np.array(self.memory_D)[num_2:num_2+int(self.batch_size/8.0)-5])
        else:
            mini_batch1 = deque(np.array(self.memory_D)[num_2:num_2+int(self.batch_size/8.0)-3])
        if len(self.memory_rules)>3:
            num_3 = random.randrange(0,len(self.memory_rules)-3)
            mini_batch2 = deque(np.array(self.memory_rules)[num_3:num_3+3])
        else:
            mini_batch2 = deque(np.array(self.memory_D)[num_2:num_2+3])
        batch_save.append(mini_batch1)
        all_rewards = np.array(map(lambda x: x[2],mini_batch1))
        max_reward = np.sum(all_rewards)
        max_rew_save.append(max_reward)
        idx_max_ = np.argmax(max_rew_save)
        batch_rule_save.append(mini_batch2)
        all_rewards2 = np.array(map(lambda x: x[2],mini_batch2))
        max_reward2 = np.sum(all_rewards2)
        max_rew_save2.append(max_reward2)
        idx_max_2 = np.argmax(max_rew_save2)

        if len(self.memory_GT)>2:
            id_gt =random.sample(self.memory_GT, 2)
            id_ra =random.sample(self.memory_D,int(self.batch_size-(self.batch_size/8.0)))
            mini_batch = batch_save[idx_max_]
            mini_batch.extend(batch_rule_save[idx_max_])
            mini_batch.extend(id_gt)
            mini_batch.extend(id_ra)
        else:
            id_ra =random.sample(self.memory_D,int(self.batch_size-(self.batch_size/8.0)))
            mini_batch = batch_save[idx_max_]
            mini_batch.extend(batch_rule_save[idx_max_])
            mini_batch.extend(id_ra)
        return mini_batch


    def train_model(self):
        '''
        Call experience_replay and start training
        '''
        mini_batch =  self.experience_replay()

        X_batch = np.empty((0, self.state_size), dtype=np.float64)
        # state_action =self.state_size+1+24
        state_action =self.state_size+1+1
        X_batch1 = np.empty((0, state_action), dtype=np.float64)

        Y_batch = np.empty((0, self.action_size), dtype=np.float64)
        Y_batch1 = np.empty((0, self.action_size), dtype=np.float64)
        for i in range(self.batch_size):
            states = mini_batch[i][0]
            actions = mini_batch[i][1]
            rewards = mini_batch[i][2] # reward one robot include crash
            next_states =mini_batch[i][3]
            dones = mini_batch[i][4]
            actions_2 = mini_batch[i][5]
            states_2 = mini_batch[i][6]
            next_states_2 = mini_batch[i][7][0:24]
            reward_2 = mini_batch[i][8]
            # print(len(next_states_2))
            q_value = self.q_model.predict(states.reshape(1, len(states)))
            self.q_value = q_value

            next_target = self.q_model_actor.predict(next_states.reshape(1,len(next_states)))
            id_max = np.argmax(next_target)
            next_t = next_target[0][id_max]
            next_q_value = self.get_Qvalue(rewards, next_t, dones) # reward completo

            X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)
            Y_sample = q_value.copy()
            Y_sample[0][actions] = next_q_value
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

            # critic network

            snext=np.append(states,actions)
            snext=np.append(snext,actions_2)
            # print(snext)
            target_value = self.target_model.predict(snext.reshape(1,len(snext)))

            tnext=np.append(next_states,actions)
            tnext=np.append(tnext,actions_2)
            # print(len(tnext),tnext)
            tnext_target = self.target_model_critic.predict(tnext.reshape(1,len(tnext)))

            tid_max = np.argmax(tnext_target)
            tnext_t = tnext_target[0][tid_max]
            self.target_value = tnext_t
            next_target_value = self.get_Qvalue(reward_2, tnext_t, dones) # reward individual

            X_batch1 = np.append(X_batch1, np.array([snext]), axis=0)
            Y_sample1 = target_value.copy()
            Y_sample1[0][actions] = next_target_value
            # Y_sample1[0][actions_2] = next_q_value1
            Y_batch1 = np.append(Y_batch1, np.array([Y_sample1[0]]), axis=0)

            if dones:
                X_batch = np.append(X_batch, np.array([next_states.copy()]), axis=0)
                ns_a = np.append(next_states.copy(),actions)
                ns_a = np.append(ns_a,actions_2)
                X_batch1 = np.append(X_batch1, np.array([ns_a]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[rewards] * self.action_size]), axis=0)
                Y_batch1 = np.append(Y_batch1, np.array([[rewards] * self.action_size]), axis=0)

        reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.2, patience=0, min_lr=0.0001)
        result = self.q_model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0,callbacks=[reduce_lr])
        result2 = self.target_model.fit(X_batch1, Y_batch1, batch_size=self.batch_size, epochs=1, verbose=0,callbacks=[reduce_lr])
        K.set_value(self.q_model.optimizer.lr, self.learning_rate )
        history_dict = result.history
        self.acc_t  = result.history['acc'][0]
        self.loss_t = (result.history['loss'][0])
        self.lr_t =  result.history['lr'][0]

    @property
    def save_lr(self):
        if not os.path.exists(self.dirPath +"loss_history"+'.txt'):
            with open(self.dirPath +"loss_history"+'.txt', 'a') as outfile:
                outfile.write("acc".rjust(15," ")+ "   "+"loss".rjust(20," ")+ "   "+"lr".rjust(10," ") +"\n")
        with open(self.dirPath +"loss_history"+ '.txt', 'a') as outfile:
            outfile.write("{: .3e}".format(self.acc_t)+"   "+"{: .3e}".format(self.loss_t)+ "   "+"{: .3e}".format(self.lr_t) +"   "+"\n")
