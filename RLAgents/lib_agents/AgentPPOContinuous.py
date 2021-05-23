import numpy
import torch
import time

from torch.distributions import Categorical

from .PolicyBufferContinuous import *

class AgentPPOContinuous():
    def __init__(self, envs, Model, Config):
        self.envs = envs

        config = Config.Config()

        self.gamma              = config.gamma
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip

        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.actors             = config.actors

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.shape[0]

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
 
        self.policy_buffer  = PolicyBufferContinuous(self.steps, self.state_shape, self.actions_count, self.actors, self.model.device)

        self.states = numpy.zeros((self.actors, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.actors):
            self.states[e] = self.envs.reset(e).copy()

        self.enable_training()
        self.iterations = 0


    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

    def main(self):        
        states_t            = torch.tensor(self.states, dtype=torch.float).detach().to(self.model.device)
 
        mu_t, var_t, values_t  = self.model.forward(states_t)

        states_np = states_t.detach().to("cpu").numpy()
        values_np = values_t.detach().to("cpu").numpy()

        mu_np   = mu_t.detach().to("cpu").numpy()
        var_np  = var_t.detach().to("cpu").numpy()

        actions = []
        for e in range(self.actors):
            action = self._sample_action(mu_np[e], var_np[e])
            actions.append(action)

        states, rewards, dones, infos = self.envs.step(actions)
        
        self.states = states.copy()

        for e in range(self.actors):
            if self.enabled_training:
                self.policy_buffer.add(e, states_np[e], values_np[e], actions[e], mu_np[e], var_np[e], rewards[e], dones[e])

                if self.policy_buffer.is_full():
                    self.train()
                    
            if dones[e]:
                self.states[e] = self.envs.reset(e)
         
        self.iterations+= 1
        return rewards[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.model.save(save_path + "trained/")

    def load(self, save_path):
        self.model.load(save_path + "trained/")

    def _sample_action(self, mu, var):
        sigma    = numpy.sqrt(var)

        action   = numpy.random.normal(mu, sigma)
        action   = numpy.clip(action, -1, 1)
        return action
    
    def train(self): 
        self.policy_buffer.compute_returns(self.gamma)

        batch_count = self.steps//self.batch_size
        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                states, values, actions, actions_mu, actions_var, rewards, dones, returns, advantages = self.policy_buffer.sample_batch(self.batch_size, self.model.device)

                loss = self._compute_loss(states, actions, actions_mu, actions_var, returns, advantages)

                self.optimizer.zero_grad()        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step() 

        self.policy_buffer.clear()   
    
    def _compute_loss(self, states, actions, actions_mu, actions_var, returns, advantages):
        mu_new, var_new, values_new = self.model.forward(states)        

        log_probs_old = self._log_prob(actions, actions_mu, actions_var).detach()
        log_probs_new = self._log_prob(actions, mu_new, var_new)

        '''
        dist_old        = torch.distributions.Normal(actions_mu, actions_var)
        log_probs_old   = dist_old.log_prob(actions).detach()

        dist_new        = torch.distributions.Normal(mu_new, var_new)
        log_probs_new   = dist_new.log_prob(actions)
        '''

        '''
        compute critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_new = values_new.squeeze(1)
        loss_value = (returns.detach() - values_new)**2
        loss_value = loss_value.mean()
        
        ''' 
        compute actor loss, surrogate loss
        ''' 
         
        advantages      = advantages.detach()
        advantages_norm = (advantages - torch.mean(advantages))/(torch.std(advantages) + 1e-10)
        advantages_norm = advantages_norm.unsqueeze(1)
        
        ratio       = torch.exp(log_probs_new - log_probs_old)
        p1          = ratio*advantages_norm
        p2          = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages_norm
        loss_policy = -torch.min(p1, p2)  
        loss_policy = loss_policy.mean()

         
        '''
        compute entropy loss, to avoid greedy strategy
        H = ln(sqrt(2*pi*var))
        ''' 
        loss_entropy = -(torch.log(2.0*numpy.pi*var_new) + 1.0)/2.0
        loss_entropy = self.entropy_beta*loss_entropy.mean()
 
        loss = loss_value + loss_policy + loss_entropy
        
        return loss

    def _log_prob(self, action, mu, var):
        p1 = -((action - mu)**2)/(2.0*var + 0.001)
        p2 = -torch.log(torch.sqrt(2.0*numpy.pi*var)) 

        return p1 + p2
