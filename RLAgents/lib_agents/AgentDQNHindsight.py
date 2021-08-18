import numpy
import torch
from .ExperienceBufferGoals import *


class AgentDQNHindsight():
    def __init__(self, env, Model, config):
        self.env = env
 
        self.batch_size         = config.batch_size
        self.exploration        = config.exploration
        self.gamma              = config.gamma

        self.target_update      = config.target_update
        self.update_frequency   = config.update_frequency        
               
        self.state_shape    = self.env.observation_space.shape
        self.goal_shape     = self.env.goal_space.shape
        self.actions_count  = self.env.action_space.n

        self.experience_replay = ExperienceBufferGoals(config.experience_replay_size, self.state_shape, self.goal_shape, self.actions_count)

        self.model          = Model.Model(self.state_shape, self.goal_shape, self.actions_count)
        self.model_target   = Model.Model(self.state_shape, self.goal_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
            target_param.data.copy_(param.data)

        self.state    = env.reset()
        self.iterations     = 0
        self.enable_training()

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
    
    def main(self):
        if self.enabled_training:
            self.exploration.process()
            epsilon = self.exploration.get()
        else:
            epsilon = self.exploration.get_testing()
             
        state_t             = torch.from_numpy(self.state["observation"]).to(self.model.device).unsqueeze(0).float()
        achieved_goal_t     = torch.from_numpy(self.state["achieved_goal"]).to(self.model.device).unsqueeze(0).float()
        desired_goal_t      = torch.from_numpy(self.state["desired_goal"]).to(self.model.device).unsqueeze(0).float()
      
      
        q_values_t  = self.model(state_t, achieved_goal_t, desired_goal_t)
        q_values_np = q_values_t.squeeze(0).detach().to("cpu").numpy()

        action      = self._sample_action(q_values_np, epsilon)

        state_new, reward, done, info = self.env.step(action)
 
        if self.enabled_training:
            self.experience_replay.add(self.state["observation"], self.state["achieved_goal"], self.state["desired_goal"], action, reward, 0.0, done)

        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train_model()

            if self.iterations%self.target_update == 0:
                self.model_target.load_state_dict(self.model.state_dict())

        if done:
            self.state = self.env.reset()
        else:
            self.state = state_new.copy()

      
        self.iterations+= 1

        return reward, done, info
        
    def train_model(self):
        state_t, achieved_goal_t, desired_goal_t, state_next_t, achieved_goal_next_t, desired_goal_next_t, actions_t, rewards_t, _, dones_t = self.experience_replay.sample(self.batch_size, self.model.device)

        #q values, state now, state next
        q_predicted      = self.model.forward(state_t, achieved_goal_t, desired_goal_t)
        q_predicted_next = self.model_target.forward(state_next_t, achieved_goal_next_t, desired_goal_next_t)

        #q-learning equation
        q_target    = q_predicted.clone()

        q_max, _    = torch.max(q_predicted_next, axis=1)
        q_new       = rewards_t + self.gamma*(1.0 - dones_t)*q_max
        q_target[range(self.batch_size), actions_t.type(torch.long)] = q_new

        #train DQN model, MSE loss
        loss  = ((q_target.detach() - q_predicted)**2)
        loss  = loss.mean() 

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer.step()

    def save(self, save_path):
        self.model.save(save_path + "trained/")

    def load(self, load_path):
        self.model.load(load_path + "trained/")
    
    def _sample_action(self, q_values, epsilon):
        if numpy.random.rand() < epsilon:
            action_idx = numpy.random.randint(self.actions_count)
        else:
            action_idx = numpy.argmax(q_values)

        return action_idx
