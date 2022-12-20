import numpy
from .MultiEnv import *
import time
import os


'''
class TrainingIterations:
    def __init__(self, env, agent, iterations_count, saving_path, log_period_iterations = 10000, averaging_episodes = 50):
        self.env = env
        self.agent = agent

        self.iterations_count = iterations_count
        
        self.saving_path            = saving_path
        self.log_period_iterations  = log_period_iterations
        self.averaging_episodes     = averaging_episodes

    def run(self):

        print("starting training")

        log_file_name   = self.saving_path + "result/result.log"
        log_f           = open(log_file_name, "w+")
        log_f.close()

        episodes                        = 0
                
        score_per_episode               = 0.0
        score_per_episode_              = 0.0

        mean_score_per_episode_best     = -100000.0


        score_per_episode_buffer = numpy.zeros(self.averaging_episodes)

        time_now = time.time()
        dt       = 0.0

        filter_k = 0.1

        time_remaining = 0.0
        for iteration in range(self.iterations_count):
            
            print("iteration = ", iteration)
            reward, done, info    = self.agent.main()
            print("agent main done")

            if iteration%self.log_period_iterations == 0:
                time_prev  = time_now
                time_now   = time.time()

                #compute fps, and remaining time in hours
                dt              = (time_now - time_prev)/self.log_period_iterations
                time_remaining  = (1.0 - filter_k)*time_remaining + filter_k*((self.iterations_count - iteration)*dt)/3600.0
            
            #episode done, update score per episode
            score_per_episode_+= reward
            if done:                
                score_per_episode = (1.0 - filter_k)*score_per_episode + filter_k*score_per_episode_
                score_per_episode_= 0.0

            
            #get raw episodes if availible
            if "raw_episodes" in info:
                raw_episodes = float(info["raw_episodes"])
            else:
                raw_episodes = episodes

            #get raw score per episode if availible
            if "raw_score" in info:
                raw_score_per_episode = float(info["raw_score"])
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
                score_per_episode_buffer[episodes%len(score_per_episode_buffer)] = raw_score_per_episode
                
                #save the best (if any)
                if episodes >= len(score_per_episode_buffer):
                    mean_score = score_per_episode_buffer.mean()

                    if mean_score > mean_score_per_episode_best:
                        mean_score_per_episode_best = mean_score
                  
                        print("\n\n")
                        print("saving new best with score = ", mean_score_per_episode_best)
                        
                        if not os.path.exists(self.saving_path + "/trained"):
                            os.mkdir(self.saving_path + "/trained")

                        self.agent.save(self.saving_path) 
                        print("\n\n")

                episodes+= 1
'''





class TrainingIterations:
    def __init__(self, env, agent, iterations_count, saving_path, log_period_iterations = 10000, averaging_episodes = 50):
        self.env = env
        self.agent = agent

        self.iterations_count = iterations_count
        
        self.saving_path            = saving_path
        self.log_period_iterations  = log_period_iterations
        self.averaging_episodes     = averaging_episodes


        self.log_file_name   = self.saving_path + "result/result.log"
        self.log_f           = open(self.log_file_name, "w+")
        

        self.episodes                        = 0
                
        self.score_per_episode               = 0.0
        self.score_per_episode_              = 0.0

        self.mean_score_per_episode_best     = -100000.0

        self.score_per_episode_buffer = numpy.zeros(self.averaging_episodes)

        self.time_now = time.time()

        self.filter_k = 0.1

        self.time_remaining = 0.0

        
    def run(self):

        print("starting training")

        for iteration in range(self.iterations_count):
            self.run_step(iteration)

        self.log_f.flush() 
        self.log_f.close() 


    def run_step(self, iteration):
            
        reward, done, info    = self.agent.main()

        if iteration%self.log_period_iterations == 0:
            self.time_prev  = self.time_now
            self.time_now   = time.time()

            #compute fps, and remaining time in hours
            dt              = (self.time_now - self.time_prev)/self.log_period_iterations
            self.time_remaining  = ((self.iterations_count - iteration)*dt)/3600.0
            #self.time_remaining  = (1.0 - self.filter_k)*self.time_remaining + self.filter_k*((self.iterations_count - iteration)*dt)/3600.0
        
        #episode done, update score per episode
        self.score_per_episode_+= reward
        if done:                
            self.score_per_episode = (1.0 - self.filter_k)*self.score_per_episode + self.filter_k*self.score_per_episode_
            self.score_per_episode_= 0.0

        
        #get raw episodes if availible
        if "raw_episodes" in info:
            raw_episodes = float(info["raw_episodes"])
        else:
            raw_episodes = self.episodes

        #get raw score per episode if availible
        if "raw_score" in info:
            raw_score_per_episode = float(info["raw_score"])
        else:
            raw_score_per_episode = self.score_per_episode

        #get aditional log if present
        log_agent = "" 
        if hasattr(self.agent, "get_log"):
            log_agent = self.agent.get_log() 
 
        log_str = ""
        log_str+= str(iteration)                + " "
        log_str+= str(raw_episodes)             + " "
        log_str+= str(self.episodes)                 + " "
        log_str+= str(raw_score_per_episode)    + " "
        log_str+= str(self.score_per_episode)        + " "
        log_str+= str(round(self.time_remaining, 2)) + " "
        log_str+= log_agent + " "
        log_str+= str(info) + " "
        
        if iteration > 0 and iteration%self.log_period_iterations == 0:
            print(log_str)
            self.log_f.write(log_str + "\n")

            if iteration%1000 == 0:
                self.log_f.flush()
            

        #check if agent is done
        if done:
            #log score per episode
            self.score_per_episode_buffer[self.episodes%len(self.score_per_episode_buffer)] = raw_score_per_episode
            
            #save the best (if any)
            if self.episodes >= len(self.score_per_episode_buffer):
                mean_score = self.score_per_episode_buffer.mean()

                if mean_score > self.mean_score_per_episode_best:
                    self.mean_score_per_episode_best = mean_score
                 
                    print("\n\n")
                    print("saving new best with score = ", self.mean_score_per_episode_best)
                    
                    if not os.path.exists(self.saving_path + "/trained"):
                        os.mkdir(self.saving_path + "/trained")

                    self.agent.save(self.saving_path) 
                    print("\n\n")

            self.episodes+= 1


class TrainingIterationsMultiRuns:

    def __init__(self, envs, agents, iterations_count, saving_paths, log_period_iterations = 10000, averaging_episodes = 50):
        print("TrainingIterationsMultiRuns = ", len(agents))
        self.trainings = []

        self.iterations_count = iterations_count

        for i in range(len(agents)):
            training = TrainingIterations(envs[i], agents[i], iterations_count, saving_paths[i], log_period_iterations, averaging_episodes)
            self.trainings.append(training)
            
    def run(self):
        print("starting trainings")
        for iteration in range(self.iterations_count):
            for i in range(len(self.trainings)): 
                self.trainings[i].run_step(iteration)
