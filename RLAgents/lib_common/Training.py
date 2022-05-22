import numpy
from .MultiEnv import *
import time


class TrainingIterations:
    def __init__(self, env, agent, iterations_count, saving_path, log_period_iterations = 10000, averaging_episodes = 50):
        self.env = env
        self.agent = agent

        self.iterations_count = iterations_count
        
        self.saving_path            = saving_path
        self.log_period_iterations  = log_period_iterations
        self.averaging_episodes     = averaging_episodes

    def run(self):

        log_file_name   = self.saving_path + "result/result.log"
        log_f           = open(log_file_name, "w+")
        log_f.close()

        episodes                        = 0
        raw_episodes                    = 0
        
        raw_score_per_episode           = 0.0
        score_per_episode               = 0.0

        score_per_episode_              = 0.0
        raw_score_per_episode_best      = -100000.0


        score_per_episode_buffer = numpy.zeros(self.averaging_episodes)

        time_now = time.time()
        dt       = 0.0

        filter_k = 0.1

        time_remaining = 0.0
        for iteration in range(self.iterations_count):
            
            reward, done, info    = self.agent.main()

            if iteration%self.log_period_iterations == 0:
                time_prev  = time_now
                time_now   = time.time()

                #compute fps, and remaining time in hours
                dt              = (time_now - time_prev)/self.log_period_iterations
                time_remaining  = (1.0 - filter_k)*time_remaining + filter_k*((self.iterations_count - iteration)*dt)/3600.0
            
            if "raw_score" in info:
                raw_score_per_episode   = float(info["raw_score"])
                print(">>>> info ", raw_score_per_episode)
            elif hasattr(self.env, "get_raw_score"):
                res                     = self.env.get_raw_score(0)
                raw_episodes            = res[0] 
                raw_score_per_episode   = res[1]            
            else:
                raw_episodes            = None
                raw_score_per_episode   = None

            #episode done, update score per episode
            score_per_episode_+= reward
            if done:
                episodes+= 1
                
                score_per_episode = (1.0 - filter_k)*score_per_episode + filter_k*score_per_episode_
                score_per_episode_= 0.0
                
            #get raw episodes count if availible
            if raw_episodes is None:
                raw_episodes = episodes
           
            #get raw score per episode if availible
            if raw_score_per_episode is None:
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
                
                #save the best (if any)
                if raw_episodes >= len(score_per_episode_buffer):
                    mean_score = score_per_episode_buffer.mean()

                    if mean_score > raw_score_per_episode_best and raw_score_per_episode >= 0.98*mean_score:
                        raw_score_per_episode_best = mean_score
                
                        print("\n\n")
                        print("saving new best with score = ", raw_score_per_episode_best)
                        self.agent.save(self.saving_path)
                        print("\n\n")