import numpy
import torch
import time

from torch.distributions import Categorical

from .PolicyBuffer import *

class AgentA2C():
    def __init__(self, envs, Model, Config):
        self.envs = envs

        config = Config.Config()

        self.gamma              = config.gamma
        self.entropy_beta       = config.entropy_beta

        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.actors             = config.actors

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
 
        self.policy_buffer = PolicyBuffer(self.steps, self.state_shape, self.actions_count, self.actors, self.model.device)

        self.states = []
        for e in range(self.actors):
            self.states.append(self.envs.reset(e))

        self.enable_training()
        self.iterations = 0


    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

    def main(self):        
        states_t            = torch.tensor(self.states, dtype=torch.float).detach().to(self.model.device)
 
        logits_t, values_t  = self.model.forward(states_t)

        states_np = states_t.detach().to("cpu").numpy()
        logits_np = logits_t.detach().to("cpu").numpy()
        values_np = values_t.detach().to("cpu").numpy()

        actions = [] 
        for e in range(self.actors):
            actions.append(self._sample_action(logits_t[e]))

        states, rewards, dones, _ = self.envs.step(actions)
        
        for e in range(self.actors):
            if self.enabled_training:
                self.policy_buffer.add(e, states_np[e], logits_np[e], values_np[e], actions[e], rewards[e], dones[e])

                if self.policy_buffer.is_full():
                    self.train()
                    
            if dones[e]:
                self.states[e] = self.envs.reset(e)
            else:
                self.states[e] = states[e].copy()

        self.iterations+= 1
        return rewards[0], dones[0]
    
    def save(self, save_path):
        self.model.save(save_path + "trained/")

    def load(self, save_path):
        self.model.load(save_path + "trained/")
    
    def _sample_action(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits.squeeze(0), dim = 0)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()

        return action_t.item() 
    
    def train(self): 
        self.policy_buffer.compute_returns(self.gamma)

        batch_count = self.steps//self.batch_size

        for batch_idx in range(batch_count):
            states, logits, values, actions, rewards, dones, returns, advantages = self.policy_buffer.sample_batch(self.batch_size, self.model.device)

            loss = self._compute_loss(states, actions, returns, advantages)

            self.optimizer.zero_grad()        
            loss.backward()
            self.optimizer.step() 

        self.policy_buffer.clear()  
    
    def _compute_loss(self, states, actions, returns, advantages):
        logits, values   = self.model.forward(states)

        probs     = torch.nn.functional.softmax(logits, dim = 1)
        log_probs = torch.nn.functional.log_softmax(logits, dim = 1)

        '''
        compute critic loss, as MSE
        L = (T - V(s))^2
        '''
        values      = values.squeeze(1)
        loss_value  = (returns.detach() - values)**2
        loss_value  = loss_value.mean()

        ''' 
        compute actor loss
        L = -log(pi(s, a))*(T - V(s)) = -log(pi(s, a))*A 
        '''
        loss_policy = -log_probs[range(len(log_probs)), actions]*advantages.detach()
        loss_policy = loss_policy.mean()
        
        '''
        compute entropy loss, to avoid greedy strategy
        L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        '''
        loss_entropy = (probs*log_probs).sum(dim = 1)
        loss_entropy = self.entropy_beta*loss_entropy.mean()

        loss = loss_value + loss_policy + loss_entropy

        return loss
