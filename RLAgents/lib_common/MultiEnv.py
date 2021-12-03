import threading
import multiprocessing
import time
import numpy
import gym

class MultiEnvSeq:
	def __init__(self, env_name, wrapper, envs_count):

		try:
			dummy_env 	= gym.make(env_name)
			if wrapper is not None:
				dummy_env 	= wrapper(dummy_env)
		except:
			dummy_env 	= wrapper(env_name)

		self.observation_space 	= dummy_env.observation_space
		self.action_space 		= dummy_env.action_space

		self.envs	= [] 

		for i in range(envs_count):

			try:
				env = gym.make(env_name)
				if wrapper is not None:
					env 	= wrapper(env)
			except:
				env = wrapper(env_name)

			self.envs.append(env)

	def close(self):
		pass

	def reset(self, env_id):
		return self.envs[env_id].reset()

	def step(self, actions):
		obs 	= numpy.zeros((len(self.envs), ) + self.observation_space.shape, dtype=numpy.float32)
		reward 	= numpy.zeros((len(self.envs), ), dtype=numpy.float32)
		done 	= numpy.zeros((len(self.envs), ), dtype=bool)
		info 	= []

		for e in range(len(self.envs)):
			_obs, _reward, _done, _info = self.envs[e].step(actions[e])

			obs[e] 		= _obs
			reward[e] 	= _reward
			done[e] 	= _done
			info.append(_info)
			
		return obs, reward, done, info

	def render(self, env_id):
		self.envs[env_id].render()
	
	def get(self, env_id):
		return self.envs[env_id]



def env_process_main(id, child_conn, env_name, wrapper):

	print("env_process_main = ", id, env_name)

	try:
		env 	= gym.make(env_name)
		if wrapper is not None:
			env 	= wrapper(env)
	except:
		env 	= wrapper(env_name)
	
	while True:
		val = child_conn.recv()

		if val[0] == "step":
			action = val[1]

			_obs, _reward, _done, _info = env.step(action)

			child_conn.send((_obs, _reward, _done, _info))
		
		elif val[0] == "end":
			break

		elif val[0] == "reset":
			_obs 	= env.reset()
			child_conn.send(_obs)

		elif val[0] == "render":
			env.render() 

		elif val[0] == "get":
			child_conn.send(env)
	

class MultiEnvParallel:
	def __init__(self, env_name, wrapper, envs_count):
		try:
			dummy_env 	= gym.make(env_name)
			if wrapper is not None:
				dummy_env 	= wrapper(dummy_env)
		except:
			dummy_env 	= wrapper(env_name)

		self.observation_space 	= dummy_env.observation_space
		self.action_space 		= dummy_env.action_space

		dummy_env.close()

		self.envs_count = envs_count

		self.parent_conn		= []
		self.child_conn 		= []
		self.workers 	= []

		print("MultiEnvParallel")
		print("envs_count      = ", self.envs_count)
		print("\n\n")

		for i in range(self.envs_count):
			parent_conn, child_conn = multiprocessing.Pipe()

			worker = multiprocessing.Process(target=env_process_main, args=(i, child_conn, env_name, wrapper))
			worker.daemon = True
			
			self.parent_conn.append(parent_conn)
			self.child_conn.append(child_conn)
			self.workers.append(worker) 

		for i in range(self.envs_count):
			self.workers[i].start()

	def close(self):
		for i in range(len(self.workers)):
			self.parent_conn[i].send(["end"])
		
		for i in range(len(self.workers)):
			self.workers[i].join()

	def reset(self, env_id):
		self.parent_conn[env_id].send(["reset"])
		return self.parent_conn[env_id].recv() 

	def render(self, env_id):
		self.parent_conn[env_id].send(["render"])


	def step(self, actions):
		for i in range(self.envs_count):
			self.parent_conn[i].send(["step", actions[i]])

		obs 	= numpy.zeros((self.envs_count, ) + self.observation_space.shape, dtype=numpy.float32)
		rewards = numpy.zeros((self.envs_count, ), dtype=numpy.float32)
		dones 	= numpy.zeros((self.envs_count, ), dtype=bool)
		infos 	= []
 
		for i in range(self.envs_count):
			_obs, _reward, _done, _info = self.parent_conn[i].recv()

			obs[i] 		= _obs
			rewards[i] 	= _reward
			dones[i] 	= _done
			infos.append(_info)
			
		return obs, rewards, dones, infos

	def get(self, env_id):
		self.parent_conn[env_id].send(["get"])
		return self.parent_conn[env_id].recv()


'''

def env_process_main(id, envs_count, child_conn, env_name, wrapper):
	envs = []

	#create envs
	for i in range(envs_count):
		try:
			env 	= gym.make(env_name)
			if wrapper is not None:
				env 	= wrapper(env)
		except:
			env 	= wrapper(env_name)

		envs.append(env)

	print("env_process_main = ", id)


	#init buffers
	observations 	= numpy.zeros((envs_count, ) + envs[0].observation_space.shape, dtype=numpy.float32)
	rewards 		= numpy.zeros(envs_count, dtype=numpy.float32)
	dones 			= numpy.zeros(envs_count, dtype=bool)
	infos 			= []

	while True:
		val 	= child_conn.recv()
		infos 	= []
		
		if val[0] == "step":
			actions = val[1]

			for i in range(envs_count):
				obs, reward, done, info = envs[i].step(actions[i])
				observations[i] = obs.copy()
				rewards[i] 		= reward
				dones[i] 		= done
				infos.append(info)

			child_conn.send((observations, rewards, dones, infos))
		
		elif val[0] == "end":
			for i in range(envs_count):
				envs[i].close()
			break

		elif val[0] == "reset":
			env_id 	= val[1]
			obs 	= envs[env_id].reset()
			child_conn.send(obs)

		elif val[0] == "render":
			env_id 	= val[1]
			envs[env_id].render() 

		elif val[0] == "get":
			env_id 	= val[1]
			child_conn.send(envs[env_id])



class MultiEnvParallel:
	def __init__(self, env_name, wrapper, envs_count, threads_count = 8):
		try:
			dummy_env 	= gym.make(env_name)
			if wrapper is not None:
				dummy_env 	= wrapper(dummy_env)
		except:
			dummy_env 	= wrapper(env_name)

		self.observation_space 	= dummy_env.observation_space
		self.action_space 		= dummy_env.action_space

		dummy_env.close()

		self.envs_count 	 = envs_count
		self.threads_count	 = threads_count

		self.envs_per_thread = self.envs_count//self.threads_count

		self.parent_conn	= []
		self.child_conn		= []
		self.workers		= []

		print("MultiEnvParallel")
		print("env_name		   = ", env_name)
		print("envs_count      = ", self.envs_count)
		print("threads_count   = ", self.threads_count)
		print("\n\n")

		#create threads
		for i in range(self.threads_count):
			parent_conn, child_conn = multiprocessing.Pipe()

			worker = multiprocessing.Process(target=env_process_main, args=(i, self.envs_per_thread, child_conn, env_name, wrapper))
			#worker.daemon = True
			
			self.parent_conn.append(parent_conn)
			self.child_conn.append(child_conn)
			self.workers.append(worker) 

		for i in range(self.threads_count):
			self.workers[i].start()

		time.sleep(2)


	def close(self):
		for i in range(len(self.workers)):
			self.parent_conn[i].send(["end"])
		
		for i in range(len(self.workers)):
			self.workers[i].join()

	def reset(self, env_id):
		thread_id, thread_env = self._get_ids(env_id)

		self.parent_conn[thread_id].send(["reset", thread_env])
		return self.parent_conn[thread_id].recv() 

	def render(self, env_id):
		thread_id, thread_env = self._get_ids(env_id)
		self.parent_conn[thread_id].send(["render", thread_env])


	def step(self, actions):
		for i in range(self.threads_count):
			self.parent_conn[i].send(["step", actions[(i+0)*self.envs_per_thread:(i+1)*self.envs_per_thread]])

		observations 	= numpy.zeros((self.threads_count, self.envs_per_thread) + self.observation_space.shape, dtype=numpy.float32)
		rewards 		= numpy.zeros((self.threads_count, self.envs_per_thread), dtype=numpy.float32)
		dones 			= numpy.zeros((self.threads_count, self.envs_per_thread), dtype=bool)
		infos 			= []
 
		for i in range(self.threads_count):
			_obs, _reward, _done, _info = self.parent_conn[i].recv()

			observations[i]	= _obs
			rewards[i] 		= _reward
			dones[i] 		= _done

			for j in range(len(_info)):
				infos.append(_info[j])

		obs 	= numpy.reshape(observations, (self.envs_count, ) + self.observation_space.shape)
		rewards = numpy.reshape(rewards, (self.envs_count, ))
		dones 	= numpy.reshape(dones, (self.envs_count, ))
			
		return obs, rewards, dones, infos

	def get(self, env_id):
		thread_id, thread_env = self._get_ids(env_id)
		self.parent_conn[thread_id].send(["get", thread_env])
		return self.parent_conn[thread_id].recv()

	def _get_ids(self, env_id):
		return env_id//self.envs_per_thread, env_id%self.envs_per_thread 
'''

if __name__ == "__main__":
	from WrapperAtari import *
	envs_count = 128
	#envs = MultiEnvSeq("MsPacmanNoFrameskip-v4", WrapperAtari, envs_count)
	envs = MultiEnvParallel("MsPacmanNoFrameskip-v4", WrapperAtari, envs_count)
 
	for i in range(envs_count):
		envs.reset(i)

	while True:
		actions = numpy.random.randint(9, size=envs_count)
		ts = time.time()
		states, rewards, dones, _ = envs.step(actions)
		te = time.time()

		for i in range(envs_count):
			if dones[i] == True:
				envs.reset(i)

		fps = envs_count*1.0/(te - ts)
		print("fps = ", fps)
