import numpy
import torch
from .ExperienceBuffer import *

import cv2


class AgentDQNDuel():
    def __init__(self, env, Model, Config):
        self.env = env

        config = Config.Config()

        self.batch_size         = config.batch_size
        self.exploration        = config.exploration
        self.gamma              = config.gamma
        
        if hasattr(config, "tau"):
            self.soft_update        = True
            self.tau                = config.tau
        elif hasattr(config, "target_update"):
            self.soft_update        = False
            self.target_update      = config.target_update
        else:
            self.soft_update        = False
            self.target_update      = 10000

        if hasattr(config, "priority_buffer"):
            self.priority_buffer        = True
        else:        
            self.priority_buffer        = False

        self.update_frequency   = config.update_frequency        
        self.bellman_steps      = config.bellman_steps
        
       
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.n

        self.experience_replay = ExperienceBuffer(config.experience_replay_size, self.bellman_steps, self.priority_buffer)

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.model_target   = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
            target_param.data.copy_(param.data)


        self.black_state    = env.reset()
        self.white_state    = self.black_state.copy()

        self.play_as        = "black"

        self.iterations     = 0

        self.enable_training()

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False


    def main(self):
        if self.enabled_training:
            self.exploration.process()
            self.epsilon = self.exploration.get()
        else:
            self.epsilon = self.exploration.get_testing()

        active_player = self.env.get_active_player()

        if active_player == "black":
            self.white_state_next, reward, done, black_action = self.step_eval(self.black_state)
        else:
            self.black_state_next, reward, done, white_action = self.step_eval(self.white_state)
            reward = -1.0*reward
 
        if done:
            if active_player == "black":
                self.train_step(self.white_state_next, reward, True, black_action)
            else:
                self.train_step(self.black_state_next, reward, True, white_action)

        else:
            if self.play_as == "black" and active_player == "black":
                self.train_step(self.black_state, reward, False, black_action)

            if self.play_as == "white" and active_player == "white":
                self.train_step(self.white_state, reward, False, white_action)

        if active_player == "black":
            self.white_state = self.white_state_next.copy()
        else:
            self.black_state = self.black_state_next.copy()

        if done:
            #flip players at the game end
            if self.play_as == "black":
                self.play_as = "white"
            else:
                self.play_as = "black"

            self.black_state    = self.env.reset()
            self.white_state    = self.black_state.copy()

        return reward, done


            

    def step_eval(self, state):
        state_t     = torch.from_numpy(state).to(self.model.device).unsqueeze(0).float()
       
        q_values_t  = self.model(state_t)
        
        q_values    = q_values_t.squeeze(0).to("cpu").detach().numpy()

        state, reward, done, _, action = self.env.step_e_greedy(q_values, self.epsilon)

        return state, reward, done, action

    
    def train_step(self, state, reward, done, action):    
        if self.enabled_training:
            self.experience_replay.add(state, action, reward, done)

        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train_model()
            
            if self.soft_update:
                for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
                    target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)
            else:
                if self.iterations%self.target_update == 0:
                    self.model_target.load_state_dict(self.model.state_dict())


        self.iterations+= 1
    
    

    def train_model(self):
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model.device)

        #q values, state now, state next
        q_predicted      = self.model.forward(state_t)
        q_predicted_next = self.model_target.forward(state_next_t)

        #compute target, n-step Q-learning
        q_target         = q_predicted.clone()
        for j in range(self.batch_size):
            gamma_        = self.gamma

            reward_sum = 0.0
            for i in range(self.bellman_steps):
                if done_t[j][i]:
                    gamma_ = 0.0
                reward_sum+= reward_t[j][i]*(gamma_**i)

            action_idx    = action_t[j]
            q_target[j][action_idx]   = reward_sum + (gamma_**self.bellman_steps)*torch.max(q_predicted_next[j])
 
        #train DQN model
        loss_ = ((q_target.detach() - q_predicted)**2)
        loss  = loss_.mean() 

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer.step()

        loss_ = loss_.mean(dim=1).detach().to("cpu").numpy()
        if self.priority_buffer:
            self.experience_replay.set_loss_for_priority(loss_)

    def _sample_action(self, state_t, epsilon):

        batch_size = state_t.shape[0]

        q_values_t          = self.model(state_t).to("cpu")

        #best actions indices
        q_max_indices_t     = torch.argmax(q_values_t, dim = 1)

        #random actions indices
        q_random_indices_t  = torch.randint(self.actions_count, (batch_size,))

        #create mask, which actions will be from q_random_indices_t and which from q_max_indices_t
        select_random_mask_t= torch.tensor((torch.rand(batch_size) < epsilon).clone(), dtype = int)

        #apply mask
        action_idx_t    = select_random_mask_t*q_random_indices_t + (1 - select_random_mask_t)*q_max_indices_t
        action_idx_t    = torch.tensor(action_idx_t, dtype=int)

        #create one hot encoding
        action_one_hot_t = torch.zeros((batch_size, self.actions_count))
        action_one_hot_t[range(batch_size), action_idx_t] = 1.0  
        action_one_hot_t = action_one_hot_t.to(self.model.device)

        #numpy result
        action_idx_np       = action_idx_t.detach().to("cpu").numpy().astype(dtype=int)

        return action_idx_np, action_one_hot_t

    def save(self, save_path):
        self.model.save(save_path)

    def load(self, save_path):
        self.model.load(save_path)
    



