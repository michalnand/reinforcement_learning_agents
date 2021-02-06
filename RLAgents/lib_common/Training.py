import numpy
from .TrainingLog import *
from .MultiEnv import *
import time


class TrainingIterations:
    def __init__(self, env, agent, iterations_count, saving_path, saving_period_iterations = 10000):
        self.env = env
        self.agent = agent

        self.iterations_count = iterations_count
        
     
        self.saving_path = saving_path
        self.saving_period_iterations = saving_period_iterations

    def run(self):
        log = TrainingLog(self.saving_path + "result/result.log", self.saving_period_iterations)
        new_best = False

    
        raw_score_per_episode_best      = 0.0

        if self.iterations_count < 1000000:
            averaging_episodes = 20
        else:
            averaging_episodes = 100

        fps        = 0.0
        score_per_episode_buffer = numpy.zeros(averaging_episodes)

        for iteration in range(self.iterations_count):
            
            time_start      = time.time()
            reward, done    = self.agent.main()
            time_stop       = time.time()

            raw_episodes            = 0 
            raw_score_total         = 0
            raw_score_per_episode   = 0
 
            if isinstance(self.env, list):
                env = self.env[0]
            elif isinstance(self.env, MultiEnvSeq):
                env = self.env.get(0)
            elif isinstance(self.env, MultiEnvParallel):
                env = self.env.get(0)
            else:
                env = self.env

            if hasattr(env, "raw_episodes"):
                raw_episodes = env.raw_episodes
            
            if hasattr(env, "raw_score_total"):
                raw_score_total = env.raw_score_total
        
            if hasattr(env, "raw_score_per_episode"):
                raw_score_per_episode = env.raw_score_per_episode
          

            log_agent = "" 
            if hasattr(self.agent, "get_log"):
                log_agent = self.agent.get_log() 

            log.add(reward, done, raw_episodes, raw_score_total, raw_score_per_episode, log_agent)

            if hasattr(env, "raw_score_per_episode"):
                score_per_episode_buffer[raw_episodes%len(score_per_episode_buffer)] = raw_score_per_episode
                
                if raw_episodes%len(score_per_episode_buffer) == 0:
                    raw_score_per_hundred_episode = score_per_episode_buffer.mean()

                    if raw_score_per_hundred_episode > raw_score_per_episode_best:
                        raw_score_per_episode_best = raw_score_per_hundred_episode
                        new_best = True
            else:
                if log.is_best:
                    new_best = True
             
            if new_best == True:
                new_best = False 
                print("\n\n")
                print("saving new best with score = ", log.episode_score_best, raw_score_per_episode_best)
                self.agent.save(self.saving_path)
                print("\n\n")

            k   = 0.02
            fps = (1.0-k)*fps + k*1.0/(time_stop - time_start)
  
            if iteration%1000 == 0:

                time_remain = (self.iterations_count - iteration)/fps

                print("FPS = ",  round(fps, 3))
                print("ETA = ",  round(time_remain/3600, 2), " hours")
                print("\n\n")

            
        if new_best == True: 
            new_best = False 
            print("\n\n")
            print("saving new best with score = ", log.episode_score_best)
            self.agent.save(self.saving_path)
            print("\n\n")
