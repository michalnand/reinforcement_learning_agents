import numpy
import torch
from .ExperienceBuffer import *


class AgentDQNPolicy():
    def __init__(self, env, Model, config):
        self.env = env
 
        self.batch_size         = config.batch_size
        self.gamma              = config.gamma
        self.entropy_beta       = config.entropy_beta

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
    
    def main(self):     
        state_t         = torch.from_numpy(self.state).to(self.model.device).unsqueeze(0).float()
        logits_t, _     = self.model(state_t)

        action          = self._sample_actions(logits_t)[0]

        state_new, reward, done, info = self.env.step(action)
 
        if self.enabled_training:
            self.experience_replay.add(self.state, action, reward, done)

        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train()

            if self.iterations%self.target_update == 0:
                self.model_target.load_state_dict(self.model.state_dict())

        if done:
            self.state = self.env.reset()
        else:
            self.state = state_new.copy()

      
        self.iterations+= 1

        return reward, done, info
        
    def train(self):
        state_t, state_next_t, actions_t, rewards_t, dones_t, _ = self.experience_replay.sample(self.batch_size, self.model.device)

        #q values, state now, state next
        logits, q_predicted      = self.model.forward(state_t)
        _,      q_predicted_next = self.model_target.forward(state_next_t)

        loss_critic = self._loss_critic(q_predicted, q_predicted_next, actions_t, rewards_t, dones_t)
        loss_actor  = self._loss_actor(logits, q_predicted)
        
        loss = loss_critic + loss_actor

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer.step()

    def _loss_critic(self, q_predicted, q_predicted_next, actions, rewards_t, dones_t):
        #q-learning equation
        q_target    = q_predicted.clone()

        q_max, _    = torch.max(q_predicted_next, axis=1)
        q_new       = rewards_t + self.gamma*(1.0 - dones_t)*q_max
        q_target[range(q_predicted.shape[0]), actions.type(torch.long)] = q_new

        #train DQN model, MSE loss
        loss  = ((q_target.detach() - q_predicted)**2)
        loss  = loss.mean() 

        return loss

    def _loss_actor(self, logits, q_predicted, actions):
        advantages = q_predicted - q_predicted.sum(dim=1, keepdim=True)
        advantages = advantages.detach()

        #maximize logits probs
        loss_policy = -logits*advantages
        loss_policy = loss_policy[range(logits.shape[0]), actions]
        print("loss_policy = ", loss_policy.shape)
        loss_policy = loss_policy.mean()

        #entropy regularisation loss
        probs_new       = torch.nn.functional.softmax(logits, dim = 1)
        log_probs_new   = torch.nn.functional.log_softmax(logits, dim = 1)

        loss_entropy    = (probs_new*log_probs_new).sum(dim = 1)
        loss_entropy    = self.entropy_beta*loss_entropy.mean()


        return loss_policy + loss_entropy

    def save(self, save_path):
        self.model.save(save_path + "trained/")

    def load(self, load_path):
        self.model.load(load_path + "trained/")
    
    def _sample_actions(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().to("cpu").numpy()
        return actions
