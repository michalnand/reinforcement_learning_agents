import numpy
import torch
from .ExperienceBuffer import *

import cv2


class AgentDQN():
    def __init__(self, env, Model, Config):
        self.env = env
 
        config = Config.Config()

        self.batch_size         = config.batch_size
        self.exploration        = config.exploration
        self.gamma              = config.gamma

        self.target_update      = config.target_update
        self.update_frequency   = config.update_frequency        
               
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.n

        self.experience_replay = ExperienceBuffer(config.experience_replay_size, self.state_shape, self.actions_count)

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.model_target   = Model.Model(self.state_shape, self.actions_count)
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
    
    def main(self, show_activity = False):
        if self.enabled_training:
            self.exploration.process()
            epsilon = self.exploration.get()
        else:
            epsilon = self.exploration.get_testing()
             
        state_t     = torch.from_numpy(self.state).to(self.model.device).unsqueeze(0).float()
        q_values_t  = self.model(state_t)
        q_values_np = q_values_t.squeeze(0).detach().to("cpu").numpy()

        action      = self._sample_action(q_values_np, epsilon)

        state_new, self.reward, done, self.info = self.env.step(action)
 
        if self.enabled_training:
            self.experience_replay.add(self.state, action, self.reward, done)

        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train_model()

            if self.iterations%self.target_update == 0:
                self.model_target.load_state_dict(self.model.state_dict())

        if done:
            self.state = self.env.reset()
        else:
            self.state = state_new.copy()

        if show_activity:
            self._show_activity(self.state)

        self.iterations+= 1

        return self.reward, done
        
    def train_model(self):
        state_t, state_next_t, action_t, reward_t, done_t, _ = self.experience_replay.sample(self.batch_size, self.model.device)

        #q values, state now, state next
        q_predicted      = self.model.forward(state_t)
        q_predicted_next = self.model_target.forward(state_next_t)

        #compute target, n-step Q-learning
        q_target         = q_predicted.clone()
        for j in range(self.batch_size): 
            action_idx              = action_t[j]
            q_target[j][action_idx] = reward_t[j] + self.gamma*torch.max(q_predicted_next[j])*(1- done_t[j])
 
        #train DQN model
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

    def _show_activity(self, state, alpha = 0.6):
        activity_map    = self.model.get_activity_map(state)
        activity_map    = numpy.stack((activity_map,)*3, axis=-1)*[0, 0, 1]

        state_map    = numpy.stack((state[0],)*3, axis=-1)
        image        = alpha*state_map + (1.0 - alpha)*activity_map

        image        = (image - image.min())/(image.max() - image.min())

        image = cv2.resize(image, (400, 400), interpolation = cv2.INTER_AREA)
        cv2.imshow('state activity', image)
        cv2.waitKey(1)