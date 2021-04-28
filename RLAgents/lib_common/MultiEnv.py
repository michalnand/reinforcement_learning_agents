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
				env 	= gym.make(env_name)
				if wrapper is not None:
					env 	= wrapper(env)
			except:
				env 	= wrapper(env_name)

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

def env_process_main(id, inq, outq, env_name, wrapper, count):

	print("env_process_main = ", id, count, env_name)
	envs 	= []

	for _ in range(count):
		
		try:
			env 	= gym.make(env_name)
			if wrapper is not None:
				env 	= wrapper(env)
		except:
			env 	= wrapper(env_name)

		envs.append(env)

		observation_space 	= env.observation_space

	
	while True:
		val = inq.get()
		
		if val[0] == "end":
			break

		elif val[0] == "reset":
			env_id 	= val[1]

			_obs 	= envs[env_id].reset()
			
			outq.put(_obs)

		elif val[0] == "step":
			actions = val[1]

			obs 		= numpy.zeros((count, ) + observation_space.shape, dtype=numpy.float32)
			rewards 	= numpy.zeros((count, ), dtype=numpy.float32)
			dones 		= numpy.zeros((count, ), dtype=bool)
			infos 		= []

			for i in range(count):
				_obs, _reward, _done, _info = envs[i].step(actions[i])

				obs[i] 		= _obs
				rewards[i] 	= _reward
				dones[i]	= _done 
				infos.append(_info)

			outq.put((obs, rewards, dones, infos))

		elif val[0] == "render":
			env_id = val[1]
			envs[env_id].render()

		elif val[0] == "get":
			env_id = val[1]
			outq.put(envs[env_id])
	

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


		self.inq		= []
		self.outq 		= []
		self.workers 	= []

		envs_per_thread			= 8

		self.envs_count			= envs_count
		self.threads_count 		= envs_count//envs_per_thread
		self.envs_per_thread	= envs_per_thread

		print("MultiEnvParallel")
		print("envs_count      = ", self.envs_count)
		print("threads_count   = ", self.threads_count)
		print("envs_per_thread = ", self.envs_per_thread)
		print("\n\n")

		for i in range(self.threads_count):
			inq	 =	multiprocessing.Queue()
			outq =	multiprocessing.Queue()

			worker = multiprocessing.Process(target=env_process_main, args=(i, inq, outq, env_name, wrapper, envs_per_thread))
			
			self.inq.append(inq)
			self.outq.append(outq)
			self.workers.append(worker) 

		for i in range(self.threads_count):
			self.workers[i].start()

	

	def close(self):
		for i in range(len(self.workers)):
			self.inq[i].put(["end"])
		
		for i in range(len(self.workers)):
			self.workers[i].join()

	def reset(self, env_id):
		thread, id = self._position(env_id)

		self.inq[thread].put(["reset", id])

		obs = self.outq[thread].get()
		return obs 

	def render(self, env_id):
		thread, id = self._position(env_id)

		self.inq[thread].put(["render", id])


	def step(self, actions):
		for j in range(self.threads_count):
			_actions = []
			for i in range(self.envs_per_thread):
				_actions.append(actions[j*self.envs_per_thread + i])

			self.inq[j].put(["step", _actions])


		obs 	= numpy.zeros((self.threads_count, self.envs_per_thread) + self.observation_space.shape, dtype=numpy.float32)
		rewards = numpy.zeros((self.threads_count, self.envs_per_thread), dtype=numpy.float32)
		dones 	= numpy.zeros((self.threads_count, self.envs_per_thread), dtype=bool)
		infos 	= None

		for j in range(self.threads_count):
			_obs, _reward, _done, _info = self.outq[j].get()

			obs[j] 		= _obs
			rewards[j] 	= _reward
			dones[j] 	= _done


		obs 		= numpy.reshape(obs, (self.threads_count*self.envs_per_thread, ) + self.observation_space.shape)
		rewards 	= numpy.reshape(rewards, (self.threads_count*self.envs_per_thread, ))
		dones 		= numpy.reshape(dones, (self.threads_count*self.envs_per_thread, ))

		return obs, rewards, dones, infos

	def get(self, env_id):
		thread, id = self._position(env_id)

		self.inq[thread].put(["get", id])

		return self.outq[thread].get()

	def _position(self, env_id):
		return env_id//self.envs_per_thread, env_id%self.envs_per_thread




'''
def env_process_main(id, child_conn, env_name, wrapper):

	print("env_process_main = ", id, env_name)

	try:
		env 	= gym.make(env_name)
		if wrapper is not None:
			env 	= wrapper(env)
	except:
		env 	= wrapper(env_name)


	observation_space 	= env.observation_space

	
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
			inq	 =	multiprocessing.Queue()
			outq =	multiprocessing.Queue()

			parent_conn, child_conn = multiprocessing.Pipe()

			worker = multiprocessing.Process(target=env_process_main, args=(i, child_conn, env_name, wrapper))
			
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
		infos 	= None
 
		for i in range(self.envs_count):
			_obs, _reward, _done, _info = self.parent_conn[i].recv()

			obs[i] 		= _obs
			rewards[i] 	= _reward
			dones[i] 	= _done

		return obs, rewards, dones, infos

	def get(self, env_id):
		self.parent_conn[env_id].send(["get"])
		return self.parent_conn[env_id].recv()
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
