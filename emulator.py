# coding: utf-8
import numpy as np
import sys
from ale_python_interface import ALEInterface
import cv2
import time

class emulator:
	def __init__(self, rom_name, vis):
		self.ale = ALEInterface()
		self.max_frames_per_episode = self.ale.getInt("max_num_frames_per_episode");
		self.ale.setInt("random_seed",123)
		self.ale.setInt("frame_skip", 4)
		if vis:
			self.ale.setBool('display_screen', True)
		self.ale.loadROM(rom_name)
		self.legal_actions = self.ale.getMinimalActionSet()
		print(self.legal_actions)
		self.screen_width, self.screen_height = self.ale.getScreenDims()
		print("image width/height: " +str(self.screen_width) + "/" + str(self.screen_height))

	def get_image(self):
		numpy_surface = np.zeros(self.screen_height*self.screen_width*3, dtype=np.uint8)
		self.ale.getScreenRGB(numpy_surface)
		image = np.reshape(numpy_surface, (self.screen_height, self.screen_width, 3))
		return image

	def newGame(self):
		self.ale.reset_game()
		return self.get_image()

	def next(self, action_indx):
		reward = self.ale.act(action_indx)	
		nextstate = self.get_image()
		return nextstate, reward, self.ale.game_over()


"""game breakout
0: stop
1: stop
3: move right
4: move left
"""

if __name__ == "__main__":
	import random
	engine = emulator('roms/breakout.bin', True)
	while True:
		# a = int(input(''))
		a = random.choice(engine.legal_actions)
		time.sleep(0.2)
		nextstate, reward, game_over = engine.next(a)
		if game_over:
			engine.newGame()
