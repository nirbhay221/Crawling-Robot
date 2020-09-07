import gym
from Agents import QAgent
import numpy as np
from crawler_env import CrawlingRobotEnv

##Todo: Build a robot that can learn to crawl 
env = CrawlingRobotEnv(render = True)
agent = QAgent(env,gamma=0.9)
current_state =env.reset()
total_rewards = 0
i = 0 
while i <300000:
	i = i+1
	action = agent.choose_action(current_state)
	next_state,reward,done,info = env.step(action)
	agent.learn(current_state,action,reward,next_state)
	current_state = next_state
	total_rewards += reward


	if i%5000 == 0:
		print("Average reward is : ",total_rewards/i)
		if (total_rewards/i)>1.3:
			break