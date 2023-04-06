import numpy
import torch
import time

from .ValuesLogger      import *
from .PolicyBuffer      import *

from .Augmentations         import *
from .SelfSupervisedLoss    import *


 
class AgentPPOSA():
    def __init__(self, envs, Model, config):
        self.envs = envs

        self.gamma              = config.gamma
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip
 
        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count

      

        if config.self_supervised_loss == "vicreg":
            self._self_supervised_loss = loss_vicreg
        else:
            self._self_supervised_loss = None

        if config.self_aware_loss == "action_loss":
            self._self_aware_loss = self._action_loss
        elif config.self_aware_loss == "constructor_loss":
             self._self_aware_loss = self._constructor_loss
        else:
            self._self_aware_loss = None

        self.self_supervised_loss_coeff     = config.self_supervised_loss_coeff
        self.self_aware_loss_coeff          = config.self_aware_loss_coeff
         
        self.augmentations                  = config.augmentations
        self.augmentations_probs            = config.augmentations_probs

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
 
        self.policy_buffer = PolicyBuffer(self.steps, self.state_shape, self.actions_count, self.envs_count)

        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e] = self.envs.reset(e) 

        self.enable_training()
        self.iterations = 0 


        print("self_supervised_loss         = ", self._self_supervised_loss)
        print("self_aware_loss              = ", self._self_aware_loss)
        print("augmentations                = ", self.augmentations)
        print("augmentations_probs          = ", self.augmentations_probs)
        print("self_supervised_loss_coeff   = ", self.self_supervised_loss_coeff)
        print("self_aware_loss_coeff        = ", self.self_aware_loss_coeff)

        print("\n\n")

        self.values_logger  = ValuesLogger()

        self.values_logger.add("loss_actor", 0.0)
        self.values_logger.add("loss_critic", 0.0)

        self.values_logger.add("loss_self_supervised", 0.0)
        self.values_logger.add("loss_self_aware", 0.0)

        self.values_logger.add("ss_accuracy", 0.0)
        self.values_logger.add("sa_accuracy", 0.0)

        

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
        small_batch = 16*self.batch_size 

        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                states, _, logits, actions, returns, advantages = self.policy_buffer.sample_batch(self.batch_size, self.model.device)

                #common PPO loss
                loss_ppo = self._loss_ppo(states, logits, actions, returns, advantages)

                #sample smaller batch for self-supervised regularization
                states_a, states_b, states_c, action = self.policy_buffer.sample_states_action_pairs(small_batch, self.model.device)

                #self supervised regularisation   
                if self._self_supervised_loss is not None:
                    loss_self_supervised, _, _, ss_accuracy = self._self_supervised_loss(self.model, states_a, states_a, self._augmentations)
                else:
                    loss_self_supervised = 0.0
                    ss_accuracy = 0.0

                #self aware loss 
                if self._self_aware_loss is not None:
                    loss_self_aware, sa_accuracy = self._self_aware_loss(self.model, states_a, states_b, states_c, action)                 
                else:
                    loss_self_aware     = 0.0
                    sa_accuracy         = 0.0


                self.values_logger.add("loss_self_supervised", loss_self_supervised.detach().to("cpu").numpy())
                self.values_logger.add("loss_self_aware", loss_self_aware.detach().to("cpu").numpy())

                self.values_logger.add("ss_accuracy", ss_accuracy)
                self.values_logger.add("sa_accuracy", sa_accuracy)


                loss = loss_ppo + self.self_supervised_loss_coeff*loss_self_supervised + self.self_aware_loss_coeff*loss_self_aware

                self.optimizer.zero_grad()        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step() 

        self.policy_buffer.clear()   

   

    
    def _loss_ppo(self, states, logits, actions, returns, advantages):
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

        advantages  = (advantages - torch.mean(advantages))/(torch.std(advantages) + 1e-10)

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
    
     #inverse model for action prediction
    def _action_loss(self, model, states_now, states_next, states_random, action):
        action_pred     = model.forward_aux(states_now, states_next)

        action_one_hot  = torch.nn.functional.one_hot(action, self.actions_count).to(states_now.device)

        loss            =  ((action_one_hot - action_pred)**2).mean()

        #compute accuracy
        pred = torch.argmax(action_pred.detach(), dim=1)
        acc = 100.0*(pred == action).float().mean()
        acc = acc.detach().to("cpu").numpy()

        return loss, acc

    #constructor theory loss
    #inverse model for action prediction
    def _constructor_loss(self, model, states_now, states_next, states_random, action):
        batch_size          = states_now.shape[0]

        #0 : state_now,  state_random, two different states
        #1 : state_now,  state_next, two consecutive states
        #2 : state_next, state_now, two inverted consecutive states
        labels                   = torch.randint(0, 3, (batch_size, )).to(states_now.device)
        transition_label_one_hot = torch.nn.functional.one_hot(labels, 3)

        #mix states
        select  = labels.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        sa      = (select == 0)*states_now    + (select == 1)*states_now  + (select == 2)*states_next
        sb      = (select == 0)*states_random + (select == 1)*states_next + (select == 2)*states_now


        #process augmentation
        sa_aug  = self._augmentations(sa)
        sb_aug  = self._augmentations(sb)

        transition_pred = model.forward_aux(sa_aug, sb_aug)

        loss            = ((transition_label_one_hot - transition_pred)**2).mean()
        
        #compute accuracy
        #compute accuracy
        labels_pred = torch.argmax(transition_pred.detach(), dim=1)
        acc = 100.0*(labels == labels_pred).float().mean()
        acc = acc.detach().to("cpu").numpy()

        return loss, acc
 
  


    def _augmentations(self, x): 

        if "aug_inverse" in self.augmentations:
            x = aug_random_apply(x, self.augmentations_probs, aug_inverse)

        if "pixelate" in self.augmentations:
            x = aug_random_apply(x, self.augmentations_probs, aug_pixelate)

        if "random_tiles" in self.augmentations:
            x = aug_random_apply(x, self.augmentations_probs, aug_random_tiles)

        if "noise" in self.augmentations:
            x = aug_random_apply(x, self.augmentations_probs, aug_noise)
        
        return x.detach() 

   
