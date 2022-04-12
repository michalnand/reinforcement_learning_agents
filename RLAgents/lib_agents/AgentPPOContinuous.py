import numpy
import torch
import time

from torch.distributions import Categorical

from .PolicyBufferContinuous import *

class AgentPPOContinuous():
    def __init__(self, envs, Model, config):
        self.envs = envs

        self.gamma              = config.gamma
        self.entropy_beta       = config.entropy_beta
        self.kl_coeff           = config.kl_coeff
        self.kl_cutoff          = config.kl_cutoff

        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.shape[0]

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
 
        self.policy_buffer  = PolicyBufferContinuous(self.steps, self.state_shape, self.actions_count, self.envs_count, self.model.device)

        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e] = self.envs.reset(e)

        self.enable_training()
        self.iterations = 0
 

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

    def main(self):        
        states                = torch.tensor(self.states, dtype=torch.float).detach().to(self.model.device)
 
        mu, var, values   = self.model.forward(states)

       
        mu_np   = mu.detach().to("cpu").numpy()
        var_np  = var.detach().to("cpu").numpy()

        actions = numpy.zeros((self.envs_count, self.actions_count))
        for e in range(self.envs_count):
            actions[e] = self._sample_action(mu_np[e], var_np[e])

        states_new, rewards, dones, infos = self.envs.step(actions)
        
        if self.enabled_training: 
            states      = states.detach().to("cpu")
            values      = values.squeeze(1).detach().to("cpu")
            mu          = mu.detach().to("cpu")
            var         = var.detach().to("cpu")

            actions     = torch.from_numpy(actions).to("cpu")
            rewards_    = torch.from_numpy(rewards).to("cpu")
            dones       = torch.from_numpy(dones).to("cpu")
             
            self.policy_buffer.add(states, values, actions, mu, var, rewards_, dones)
            if self.policy_buffer.is_full():
                self.train()

        self.states = states_new.copy()
        for e in range(self.envs_count):
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
        compute critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_new = values_new.squeeze(1)
        loss_value = (returns.detach() - values_new)**2
        loss_value = loss_value.mean()
        
 
        ''' 
        compute actor loss with KL divergence loss to prevent policy collapse
        see https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#ppo

        adaptive KL beta coefficient
        https://github.com/rrmenon10/PPO/blob/7d18619960913d39a5fb0143548abbaeb02f410e/pgrl/algos/ppo_adpkl.py#L136
        ''' 
        advantages  = advantages.detach()
        advantages  = advantages.unsqueeze(1) 
        
        log_ratio   = log_probs_new - log_probs_old
        loss_policy = -torch.exp(log_ratio)*advantages
        loss_policy = loss_policy.mean()

        
        kl_div      = torch.exp(log_probs_old)*(log_probs_old - log_probs_new) 
        loss_kl     = self.kl_coeff*kl_div + 1000*(kl_div > self.kl_cutoff)*( (kl_div-self.kl_cutoff)**2 )

        loss_kl     = loss_kl.mean()


        '''
        loss_kl     = (self.kl_target - kl_div)**2 
        loss_kl     = self.kl_coeff*loss_kl.mean()

        
        kl_div_mean = kl_div.mean().detach().to("cpu").numpy()

 
        if kl_div_mean > (self.kl_target * 1.5):
            self.kl_coeff *= 2.0
        elif kl_div_mean < (self.kl_target / 1.5):
            self.kl_coeff *= 0.5

        self.kl_coeff = numpy.clip(self.kl_coeff, 0.0001, 10)

        #print(">>> ", loss_kl, kl_div_mean, self.kl_coeff)
        '''

        '''
        compute entropy loss, to avoid greedy strategy
        H = ln(sqrt(2*pi*var))
        ''' 
        loss_entropy = -(torch.log(2.0*numpy.pi*var_new) + 1.0)/2.0
        loss_entropy = self.entropy_beta*loss_entropy.mean()
 
        loss = loss_value + loss_policy + loss_kl + loss_entropy
        
        return loss

    def _log_prob(self, action, mu, var):
        p1 = -((action - mu)**2)/(2.0*var + 0.001)
        p2 = -torch.log(torch.sqrt(2.0*numpy.pi*var)) 

        return p1 + p2

