import numpy
import torch
import time

from .ValuesLogger      import *
from .PolicyBuffer      import *

import cv2

 
class AgentMuesli():
    def __init__(self, envs, Model, config):
        self.envs = envs

        self.gamma              = config.gamma
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip

        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
 
        self.policy_buffer = PolicyBuffer(self.steps, self.state_shape, self.actions_count, self.envs_count)
 
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e] = self.envs.reset(e) 

        self.var        = 0.0
        self.beta_var   = config.beta_var #0.99
        self.beta_prod  = 1.0

        self.enable_training()
        self.iterations = 0 

        self.values_logger  = ValuesLogger()
        self.values_logger.add("loss_actor", 0.0)
        self.values_logger.add("loss_critic", 0.0)
        

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

    def main(self):        
        states  = torch.tensor(self.states, dtype=torch.float).detach().to(self.model.device)
    
        logits, values  = self.model.forward(states)
 
        actions = self._sample_actions(logits)
        
        states_new, rewards, dones, infos = self.envs.step(actions)
    
        if self.enabled_training:
            states      = states.detach().to("cpu")
            logits      = logits.detach().to("cpu")
            values      = values.squeeze(1).detach().to("cpu") 
            actions     = torch.from_numpy(actions).to("cpu")
            rewards_t   = torch.from_numpy(rewards).to("cpu")
            dones       = torch.from_numpy(dones).to("cpu")

            self.policy_buffer.add(states, logits, values, actions, rewards_t, dones)

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

    def get_log(self):
        return self.values_logger.get_str()
    
    def _sample_actions(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().to("cpu").numpy()
        return actions
    
    def train(self): 
        self.policy_buffer.compute_returns(self.gamma)

        batch_count = self.steps//self.batch_size
        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                states, _, logits, actions, returns, advantages = self.policy_buffer.sample_batch(self.batch_size, self.model.device)

                loss = self._compute_loss(states, logits, actions, returns, advantages)

                self.optimizer.zero_grad()        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step() 

        self.policy_buffer.clear()   

    def render(self, env_id):
        width   = 256
        height  = 256
        
        ofs = 0
        state_im       = numpy.array([self.states[env_id][2 + 3*ofs], self.states[env_id][1 + 3*ofs], self.states[env_id][0 + 3*ofs]])
        state_im       = numpy.moveaxis(state_im, 0, 2)
        state_im       = cv2.resize(state_im, (width, height), cv2.INTER_CUBIC)

        state_im   = cv2.resize(state_im, (width, height), cv2.INTER_CUBIC)

        cv2.imshow("PPO agent", state_im)
        cv2.waitKey(1)


    '''
    def render(self, env_id):
        width   = 256
        height  = 256

        states_t        = torch.tensor(self.states, dtype=torch.float).detach().to(self.model.device)

        features_t      = self.model.forward_features(states_t)

        features_t      = features_t.reshape((features_t.shape[0], 64, 12, 12))

        attention_t     = (features_t**2).mean(dim=1)

        min = torch.min(attention_t)
        max = torch.max(attention_t)
        
        attention_t     =(attention_t + min)/(max - min)

        state_im       = states_t[env_id][0].detach().to("cpu").numpy()
        attention_im   = attention_t[env_id].detach().to("cpu").numpy()
        
        state_im       = numpy.array([state_im, state_im, state_im])
        state_im       = numpy.moveaxis(state_im, 0, 2)
        state_im       = cv2.resize(state_im, (width, height), cv2.INTER_CUBIC)

        attention_im   = numpy.array([attention_im*0, attention_im*0, attention_im*0.5])
        attention_im   = numpy.moveaxis(attention_im, 0, 2)
        attention_im   = cv2.resize(attention_im, (width, height), cv2.INTER_CUBIC)


        image = state_im + attention_im


        cv2.imshow("PPO agent", image)
        cv2.waitKey(1)
    '''
    
    def _compute_loss(self, states, logits, actions, returns, advantages):
        
        log_probs_old = torch.nn.functional.log_softmax(logits, dim = 1).detach()

        logits_new, values_new   = self.model.forward(states)

        probs_new     = torch.nn.functional.softmax(logits_new,     dim = 1)
        log_probs_new = torch.nn.functional.log_softmax(logits_new, dim = 1)


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
        advantages        = advantages.detach() 

        self.var          = self.beta_var*self.var + (1.0 - self.beta_var)*(advantages**2).mean()
        self.beta_product = self.beta_product*self.beta_var

        var = self.var/(1.0 - self.beta_product) 

        advantages  = advantages/((var**0.5) + 1e-10)

        log_probs_new_  = log_probs_new[range(len(log_probs_new)), actions]
        log_probs_old_  = log_probs_old[range(len(log_probs_old)), actions]
                        
        ratio       = torch.exp(log_probs_new_ - log_probs_old_)
        p1          = ratio*advantages
        p2          = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages
        loss_policy = -torch.min(p1, p2)  
        loss_policy = loss_policy.mean()  
    
        '''
        compute entropy loss, to avoid greedy strategy
        L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        '''
        loss_entropy = (probs_new*log_probs_new).sum(dim = 1)
        loss_entropy = self.entropy_beta*loss_entropy.mean()

        loss = loss_value + loss_policy + loss_entropy

        self.values_logger.add("loss_actor",  loss_policy.detach().to("cpu").numpy())
        self.values_logger.add("loss_critic", loss_value.detach().to("cpu").numpy())

        return loss