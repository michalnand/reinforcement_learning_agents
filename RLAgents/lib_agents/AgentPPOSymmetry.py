import numpy
import torch
import time

from .ValuesLogger      import *
from .PolicyBuffer      import *

import cv2

 
class AgentPPOSymmetry():
    def __init__(self, envs, Model, config):
        self.envs = envs

        self.gamma              = config.gamma
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip

        self.regularisation_coeff   = config.regularisation_coeff
        self.symmetry_loss_coeff    = config.symmetry_loss_coeff

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


        if config.ppo_symmetry_loss == "mse":
            self._ppo_symmetry_loss = self._symmetry_loss_mse
        elif config.ppo_symmetry_loss == "nce": 
            self._ppo_symmetry_loss = self._symmetry_loss_nce
        else:
            self._ppo_symmetry_loss = None

        print("ppo_symmetry_loss        = ", self._ppo_symmetry_loss)

        self.enable_training()
        self.iterations = 0 

        self.values_logger  = ValuesLogger()
        self.values_logger.add("loss_actor", 0.0)
        self.values_logger.add("loss_critic", 0.0)
        self.values_logger.add("loss_symmetry", 0.0)
        self.values_logger.add("symmetry_accuracy", 0.0)
        self.values_logger.add("symmetry_magnitude", 0.0)

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
                states, states_next, logits, actions, returns, advantages = self.policy_buffer.sample_batch(self.batch_size, self.model.device)

                loss_ppo = self._compute_loss_ppo(states, logits, actions, returns, advantages)

                if self._ppo_symmetry_loss is not None:
                    small_batch = states.shape[0]//8

                    states_         = states[0:small_batch]
                    states_next_    = states_next[0:small_batch]
                    actions_        = actions[0:small_batch]

                    loss_symmetry = self._ppo_symmetry_loss(self.model, states_, states_next_, actions_)

                    loss_ppo+= loss_symmetry

                self.optimizer.zero_grad()        
                loss_ppo.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step() 

        self.policy_buffer.clear()   

    def render(self, env_id):
        size            = 256

        state           = self.states[env_id]

        state           = numpy.moveaxis(state, 0, 2)

        state_im        = cv2.resize(state, (size, size))
        state_im        = numpy.clip(state_im, 0.0, 1.0)

        cv2.imshow("PPO agent", state_im)
        cv2.waitKey(1)
    
    def _compute_loss_ppo(self, states, logits, actions, returns, advantages):
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
        advantages       = advantages.detach() 
        #this normalisation has no effect
        #advantages  = (advantages - torch.mean(advantages))/(torch.std(advantages) + 1e-10)

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

    
    def _symmetry_loss_mse(self, model, states, states_next, actions):

        z = model.forward_features(states, states_next)

        #each by each similarity, dot product and sigmoid to obtain probs
        distances   = torch.cdist(z, z)/z.shape[1] 

        #true labels are where are the same actions
        actions_    = actions.unsqueeze(1)
        labels      = (actions_ == actions_.t()).float().detach()

        #similar features for transitions caused by same action
        #conservation of rules - the rules are the same, no matters the state
        required = 1.0 - labels

        loss_symmetry    = (required - distances)**2
        loss_symmetry    = loss_symmetry.mean()   

        #L2 magnitude regularisation (10**-4) 
        magnitude   = (z.norm(dim=1, p=2)).mean()
        loss_mag    = self.regularisation_coeff*magnitude

        loss = self.symmetry_loss_coeff*(loss_symmetry + loss_mag)
 
        self.values_logger.add("loss_symmetry",  loss_symmetry.detach().to("cpu").numpy())

        #compute weighted accuracy
        true_positive  = torch.logical_and(labels > 0.5, distances < 0.5).float().sum()
        true_negative  = torch.logical_and(labels < 0.5, distances > 0.5).float().sum()
        positive       = (labels > 0.5).float().sum() + 10**-12
        negative       = (labels < 0.5).float().sum() + 10**-12 

        w               = 1.0 - 1.0/self.actions_count
        acc             = w*true_positive/positive + (1.0 - w)*true_negative/negative

        acc = acc.detach().to("cpu").numpy() 

        self.values_logger.add("symmetry_accuracy", acc)
        self.values_logger.add("symmetry_magnitude", magnitude.detach().to("cpu").numpy())

        return loss

    def _symmetry_loss_nce(self, model, states, states_next, actions):
        z = model.forward_features(states, states_next)

        #each by each similarity, dot product and sigmoid to obtain probs
        probs   = torch.sigmoid(torch.matmul(z, z.t())/z.shape[1])

        #true labels are where are the same actions
        actions_    = actions.unsqueeze(1)
        labels      = (actions_ == actions_.t()).float().detach()

        #BCE loss
        loss_symmetry    = (labels - probs)**2
        loss_symmetry    = loss_symmetry.mean()   

        #L2 magnitude regularisation (10**-4) 
        magnitude   = (z.norm(dim=1, p=2)).mean()
        loss_mag    = self.regularisation_coeff*magnitude

        loss = self.symmetry_loss_coeff*(loss_symmetry + loss_mag)
        
        self.values_logger.add("loss_symmetry",  loss_symmetry.detach().to("cpu").numpy())

        #compute weighted accuracy
        true_positive  = torch.logical_and(labels > 0.5, probs > 0.5).float().sum()
        true_negative  = torch.logical_and(labels < 0.5, probs < 0.5).float().sum()
        positive       = (labels > 0.5).float().sum() + 10**-12
        negative       = (labels < 0.5).float().sum() + 10**-12 

        w               = 1.0 - 1.0/self.actions_count
        acc             = w*true_positive/positive + (1.0 - w)*true_negative/negative

        acc = acc.detach().to("cpu").numpy() 

        self.values_logger.add("symmetry_accuracy", acc)
        self.values_logger.add("symmetry_magnitude", magnitude.detach().to("cpu").numpy())

        return loss
