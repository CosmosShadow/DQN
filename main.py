# coding: utf-8
from database import *
from emulator import *
import tensorflow as tf
import numpy as np
import time
from ale_python_interface import ALEInterface
import cv2
from scipy import misc
import gc #garbage colloector
import thread
from DQN import *

gc.enable()

params = {
	'visualize' : False,
	'ckpt_file':None,
	'num_epochs': 100,
	'steps_per_epoch': 50000,
	'eval_freq':50000,
	'steps_per_eval':10000,
	'copy_freq' : 10000,
	'disp_freq':10000,
	'save_interval':10000,
	'db_size': 1000000,
	'batch': 32,
	'num_act': 0,
	'input_dims' : [210, 160, 3],
	'input_dims_proc' : [84, 84, 4],
	'eps': 1.0,
	'eps_step':1000000,
	'eps_min' : 0.1,
	'eps_eval' : 0.05,
	'discount': 0.95,
	'lr': 0.0002,
	'rms_decay':0.99,
	'rms_eps':1e-6,
	'train_start':100,
	'img_scale':255.0,
	'gpu_fraction' : 0.5,
	'only_eval' : False
}

def copy_net(from_predix, to_predix):
	cp_ops = []
	for varible in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
		if varible.name.startswith(from_predix):
			target_name = to_predix + varible.name[len(from_predix):len(varible.name)]
			# print varible.name, target_name
			target_varible = [v for v in tf.all_variables() if v.name == target_name][0]
			cp_ops.append(target_varible.assign(varible))
	return cp_ops

class deep_atari:
	def __init__(self,params):
		print 'Initializing Module...'
		self.params = params

		self.gpu_config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.params['gpu_fraction']))
		self.sess = tf.Session(config=self.gpu_config)
		self.DB = database(self.params)
		self.engine = emulator(rom_name='roms/breakout.bin', vis=self.params['visualize'])
		self.params['num_act'] = len(self.engine.legal_actions)
		self.build_net()
		self.training = True
		self.game_cnt = 0

	def build_net(self):
		print 'Building QNet and targetnet...'		
		self.qnet = DQN(self.params, 'qnet')
		self.targetnet = DQN(self.params, 'targetnet')
		self.sess.run(tf.initialize_all_variables())
		self.saver = tf.train.Saver()

		self.cp_ops = copy_net('qnet', 'targetnet')

		self.sess.run(self.cp_ops)
		
		if self.params['ckpt_file'] is not None:
			print 'loading checkpoint : ' + self.params['ckpt_file']
			self.saver.restore(self.sess, self.params['ckpt_file'])
			temp_train_cnt = self.sess.run(self.qnet.global_step)
			print 'Continue from step: ' + str(temp_train_cnt)

	def start(self):
		self.reset_game()
		self.step = 0
		self.reset_statistics()
		self.train_cnt = self.sess.run(self.qnet.global_step)

		self.s = time.time()
		print self.params
		print 'Start training!'
		print 'Collecting replay memory for ' + str(self.params['train_start']) + ' steps'

		while self.step < (self.params['steps_per_epoch'] * self.params['num_epochs'] + self.params['train_start']):
			# 计数
			if self.training and (self.DB.get_size() >= self.params['train_start']):
				self.step += 1
			if self.step%1000 == 0:
				print self.step

			# 存一帧
			if self.state_gray_old is not None and self.training:
				self.DB.insert(self.state_gray_old[26:110,:],self.reward_scaled,self.action_idx,self.terminal)

			# 每隔一定时间，把训练的网络copy到评估网络上
			if self.training and self.params['copy_freq'] > 0 and self.step % self.params['copy_freq'] == 0 and self.DB.get_size() > self.params['train_start']:
				print '&&& Copying Qnet to targetnet\n'
				self.sess.run(self.cp_ops)

			# 训练
			if self.training and self.DB.get_size() > self.params['train_start']:
				bat_s,bat_a,bat_t,bat_n,bat_r = self.DB.get_batches()
				bat_a = self.get_onehot(bat_a)
				# 计算q_t
				if self.params['copy_freq'] > 0 :
					feed_dict={self.targetnet.x: bat_n}
					q_t = self.sess.run(self.targetnet.y,feed_dict=feed_dict)
				else:
					feed_dict={self.qnet.x: bat_n}
					q_t = self.sess.run(self.qnet.y,feed_dict=feed_dict)
				q_t = np.amax(q_t,axis=1)
				# 训练
				feed_dict={self.qnet.x: bat_s, self.qnet.q_t: q_t, self.qnet.actions: bat_a, self.qnet.terminals:bat_t, self.qnet.rewards: bat_r}
				_,self.train_cnt,self.cost = self.sess.run([self.qnet.rmsprop,self.qnet.global_step,self.qnet.cost],feed_dict=feed_dict)

			# 贪心参数epsilon: 前1million从1到0.1，以后固定为0.1
			if self.training:
				self.params['eps'] = max(self.params['eps_min'],1.0 - float(self.train_cnt)/float(self.params['eps_step']))
			else:
				self.params['eps'] = 0.05

			# 备份
			if self.DB.get_size() > self.params['train_start'] and self.step % self.params['save_interval'] == 0 and self.training:
				self.saver.save(self.sess,'ckpt/model_' + str(self.train_cnt))
			
			# 是不是结束游戏了
			if self.terminal :  
				self.reset_game()
				self.reset_statistics()
				continue

			# 环境里运行一下
			self.action_idx, self.action, self.maxQ = self.select_action(self.state_proc)
			self.state, self.reward, self.terminal = self.engine.next(self.action)
			self.reward_scaled = self.reward // max(1,abs(self.reward))

			# 添加统计
			self.add_statistics()

			self.state_gray_old = np.copy(self.state_gray)
			self.state_proc[:,:,0:3] = self.state_proc[:,:,1:4]
			self.state_resized = cv2.resize(self.state,(84,110))
			self.state_gray = cv2.cvtColor(self.state_resized, cv2.COLOR_BGR2GRAY)
			self.state_proc[:,:,3] = self.state_gray[26:110,:]/self.params['img_scale']

	def reset_game(self):
		self.state_proc = np.zeros((84,84,4)); self.action = -1; self.terminal = False; self.reward = 0
		self.state = self.engine.newGame()		
		self.state_resized = cv2.resize(self.state,(84,110))
		self.state_gray = cv2.cvtColor(self.state_resized, cv2.COLOR_BGR2GRAY)
		self.state_gray_old = None
		self.state_proc[:,:,3] = self.state_gray[26:110,:]/self.params['img_scale']

	def add_statistics(self):
		self.epi_step += 1
		self.epi_reward_train += self.reward
		self.epi_Q_train += self.maxQ

	def reset_statistics(self):
		if self.game_cnt > 1 and self.epi_step != 0:
			print 'Game: ' + str(self.game_cnt) + ', R: ' + str(self.epi_reward_train) + ', Q: ' + str(self.epi_Q_train / self.epi_step)
		self.epi_step = 0
		self.epi_reward_train = 0
		self.epi_Q_train = 0
		self.game_cnt += 1

	def select_action(self, st):
		if np.random.rand() > self.params['eps']:
			#greedy with random tie-breaking
			Q_pred = self.sess.run(self.qnet.y, feed_dict = {self.qnet.x: np.reshape(st, (1,84,84,4))})[0] 
			a_winner = np.argwhere(Q_pred == np.amax(Q_pred))
			if len(a_winner) > 1:
				act_idx = a_winner[np.random.randint(0, len(a_winner))][0]
				return act_idx,self.engine.legal_actions[act_idx], np.amax(Q_pred)
			else:
				act_idx = a_winner[0][0]
				return act_idx,self.engine.legal_actions[act_idx], np.amax(Q_pred)
		else:
			#random
			act_idx = np.random.randint(0,len(self.engine.legal_actions))
			Q_pred = self.sess.run(self.qnet.y, feed_dict = {self.qnet.x: np.reshape(st, (1,84,84,4))})[0]
			return act_idx,self.engine.legal_actions[act_idx], Q_pred[act_idx]

	def get_onehot(self,actions):
		actions_onehot = np.zeros((self.params['batch'], self.params['num_act']))
		for i in range(self.params['batch']):
			actions_onehot[i,actions[i]] = 1
		return actions_onehot


if __name__ == "__main__":
	if params['only_eval']:
		params['eval_freq'] = 1
		params['train_start'] = 100

	da = deep_atari(params)
	da.start()
