import numpy
import time




class TrainingNew:
    def __init__(self, envs, agent, iterations_count, saving_path, log_period_iterations = 128):
        self.envs           = envs   
        self.envs_count     = len(self.envs)
        self.state_shape    = self.envs.observation_space.shape
        
        self.agent = agent

        self.iterations_count = iterations_count
        self.saving_path      = saving_path
        self.log_period_iterations = log_period_iterations

        #reset envs and fill initial state
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        self.infos  = []
        for e in range(self.envs_count):
            self.states[e], info = self.envs.reset(e)
            self.infos.append(info)

        #loging    
        self.log_file_name   = self.saving_path + "result/result.log"
        self.log_f           = open(self.log_file_name, "w+")

        self.time_prev = time.time()
        self.time_now  = time.time()
        self.time_steps= 0
      

    def step(self): 
        #starting round
        self.agent.round_start()

        score = numpy.zeros(self.envs_count)

        #obtain binary masking of legal actions, returns None if not provided in info
        legal_actions_mask = self._get_legal_actions_mask(self.infos)

        states_new, rewards, dones, infos = self.agent.step(self.states, True, legal_actions_mask)

        self.states = states_new.copy()
        self.infos  = infos.copy()

        #fill master raw reward, if any
        if "raw_reward" in infos[0]:
            for e in range(self.envs_count):
                score[e] = infos[e]["raw_reward"]
        else:
            score = rewards.copy()
 
        #find all envs where episode is done
        dones_idx = numpy.where(dones)[0]

        for e in dones_idx:
            #send to agent the episode is done with env id
            self.agent.episode_done(e)

            #reset env
            self.states[e], self.infos[e] = self.envs.reset(e)

            #shuffle players order for this env
            self._new_players_order(e)

        #all players are done
        self.agent.round_finish()

        return rewards, score, dones, self.infos

    def run(self):        
        #create log file for append
        self.log_f = open(self.log_file_name, "w+")

        rewards_sum = numpy.zeros(self.envs_count)
        score_sum   = numpy.zeros(self.envs_count)

        rewards_episode = numpy.zeros(self.envs_count)
        score_episode   = numpy.zeros(self.envs_count)

        episodes_count  = numpy.zeros(self.envs_count)
        
        for iteration in range(self.iterations_count):
            rewards, score, dones, infos = self.step()

            rewards_sum+= rewards
            score_sum+= score

            info = infos[0].copy()

            if "legal_actions_mask" in info:
                del info["legal_actions_mask"]

            time_remaining = self._estimate_time(iteration)

            dones_idx = numpy.where(dones)[0]

            for e in dones_idx:
                rewards_episode[e] = rewards_sum[e]
                score_episode[e]   = score_sum[e]

                rewards_sum[e] = 0
                score_sum[e]   = 0

                episodes_count[e]+= 1

            if iteration%self.log_period_iterations == 0:

                log_agent = "" 
                if hasattr(self.agent, "get_log"):
                    log_agent = self.agent.get_log() 
 
                log_str = ""
                log_str+= str(iteration) + " " 
                log_str+= str(round(episodes_count.mean(), 5)) + " "
                log_str+= str(round(rewards_episode.mean(), 5)) + " "
                log_str+= str(round(score_episode.mean(), 5)) + " "
                log_str+= str(round(time_remaining, 2)) + " "
                log_str+= log_agent + " "
                log_str+= info + " "

                print(log_str)
                self.log_f.write(log_str + "\n")

            if iteration%1024 == 0:
                self.log_f.flush()

            if iteration%(self.iterations_count//10) == 0:
                print("saving agent on step ", iteration, "\n\n")
                self.agent.save(self.saving_path)

        self.agent.save(self.saving_path)

        self.log_f.flush() 
        self.log_f.close() 

        

    #after each game, new players order is generated
    def _new_players_order(self, env_id):
        self.players_order[env_id] = numpy.random.permutation(self.players_count)

    #extract binary matrix with shape (envs_count, actions_count) from infos,
    #containing ones where action is legal
    #if not provided, returns ones - all actions allowed
    def _get_legal_actions_mask(self, infos): 
        
        legal_actions_mask  = numpy.ones((self.envs_count,  self.envs.action_space.n), dtype=numpy.float32) 

        if type(infos) is list:
            if type(infos[0]) is dict:
                if "legal_actions_mask" in infos[0]:
                    legal_actions_mask  = numpy.zeros((self.envs_count,  self.envs.action_space.n), dtype=numpy.float32) 
                    for e in range(self.envs_count):
                        legal_actions_mask[e] = infos[e]["legal_actions_mask"]
            
        return legal_actions_mask 

    #returns remaining time in hours
    def _estimate_time(self, iteration, time_iterations = 128):
        if self.time_steps%time_iterations == 0:
            self.time_prev = self.time_now
            self.time_now  = time.time()

        self.time_steps+= 1

        dt              = (self.time_now - self.time_prev)/time_iterations
        time_remaining  = ((self.iterations_count - iteration)*dt)/3600.0

        return time_remaining
    