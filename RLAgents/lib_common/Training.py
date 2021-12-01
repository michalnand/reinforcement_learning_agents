import numpy
from .MultiEnv import *
import time


class TrainingIterations:
    def __init__(self, env, agent, iterations_count, saving_path, log_period_iterations = 10000):
        self.env = env
        self.agent = agent

        self.iterations_count = iterations_count
        
        self.saving_path            = saving_path
        self.log_period_iterations  = log_period_iterations

    def run(self):

        log_file_name   = self.saving_path + "result/result.log"
        log_f           = open(log_file_name, "w+")
        log_f.close()

 
        averaging_episodes = 50

        new_best = False

        episodes                        = 0
        raw_episodes                    = 0
        
        raw_score_per_episode           = 0.0
        score_per_episode               = 0.0

        score_per_episode_              = 0.0
        raw_score_per_episode_best      = -100000.0


        score_per_episode_buffer = numpy.zeros(averaging_episodes)

        time_now = time.time()
        dt       = 0.0

        filter_k = 0.3

        time_remaining = 0.0
        for iteration in range(self.iterations_count):
            
            reward, done, info    = self.agent.main()

            if iteration%self.log_period_iterations == 0:
                time_prev  = time_now
                time_now   = time.time()

                #compute fps, and remaining time in hours
                dt              = (time_now - time_prev)/self.log_period_iterations
                time_remaining  = (1.0 - filter_k)*time_remaining + filter_k*((self.iterations_count - iteration)*dt)/3600.0

 
            if isinstance(self.env, list):
                env = self.env[0]
            elif isinstance(self.env, MultiEnvSeq):
                env = self.env.get(0)
            elif isinstance(self.env, MultiEnvParallel):
                env = self.env.get(0)
            else:
                env = self.env

            #episode done, update score per episode
            score_per_episode_+= reward
            if done:
                episodes+= 1
                
                score_per_episode = (1.0 - filter_k)*score_per_episode + filter_k*score_per_episode_
                score_per_episode_= 0.0
                
            #get raw episodes count if availible
            if hasattr(env, "raw_episodes"):
                raw_episodes = env.raw_episodes
            else:
                raw_episodes = episodes

            #get raw score per episode if availible
            if hasattr(env, "raw_score_per_episode"):
                raw_score_per_episode = env.raw_score_per_episode
            else:
                raw_score_per_episode = score_per_episode

            #get aditional log if present
            log_agent = "" 
            if hasattr(self.agent, "get_log"):
                log_agent = self.agent.get_log() 

            log_str = ""
            log_str+= str(iteration)                + " "
            log_str+= str(raw_episodes)             + " "
            log_str+= str(episodes)                 + " "
            log_str+= str(raw_score_per_episode)    + " "
            log_str+= str(score_per_episode)        + " "
            log_str+= str(round(time_remaining, 2)) + " "
            log_str+= log_agent + " "
            log_str+= str(info) + " "
            
            if iteration > 0 and iteration%self.log_period_iterations == 0:
                print(log_str)

                log_f = open(log_file_name, "a+")
                log_f.write(log_str + "\n")
                log_f.flush()
                log_f.close() 


            #check if agent is done
            if done:
                #log score per episode
                score_per_episode_buffer[raw_episodes%len(score_per_episode_buffer)] = raw_score_per_episode
                
                #save the best (if any), every 10episodes
                if raw_episodes > 0 and raw_episodes%10 == 0:
                    raw_score_per_hundred_episode = score_per_episode_buffer.mean()

                    if raw_score_per_hundred_episode > raw_score_per_episode_best:
                        raw_score_per_episode_best = raw_score_per_hundred_episode
                        new_best = True
            
                if new_best == True:
                    new_best = False 
                    print("\n\n")
                    print("saving new best with score = ", raw_score_per_episode_best)
                    self.agent.save(self.saving_path)
                    print("\n\n")