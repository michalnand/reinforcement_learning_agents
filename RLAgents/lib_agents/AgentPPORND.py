import numpy
import torch

from .ValuesLogger      import * 
from .PolicyBufferIM    import *  
from .RunningStats      import *  
import cv2
      
class AgentPPORND():   
    def __init__(self, envs, ModelPPO, ModelRND, config):
        self.envs = envs  
    
        self.gamma_ext          = config.gamma_ext 
        self.gamma_int          = config.gamma_int
            
        self.ext_adv_coeff      = config.ext_adv_coeff
        self.int_adv_coeff      = config.int_adv_coeff
    
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip 
    
        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count 


        self.normalise_state_mean = config.normalise_state_mean
        self.normalise_state_std  = config.normalise_state_std


        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)
 
        self.model_rnd      = ModelRND.Model(self.state_shape)
        self.optimizer_rnd  = torch.optim.Adam(self.model_rnd.parameters(), lr=config.learning_rate_rnd)
 
        self.policy_buffer = PolicyBufferIM(self.steps, self.state_shape, self.actions_count, self.envs_count)

        for e in range(self.envs_count):
            self.envs.reset(e)
        
        self.states_running_stats       = RunningStats(self.state_shape)

        self._init_running_stats()

        #reset envs and fill initial state
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e] = self.envs.reset(e).copy()

 
        self.enable_training()
        self.iterations                     = 0 

        self.values_logger                  = ValuesLogger()

        self.values_logger.add("loss_rnd", 0.0)
        self.values_logger.add("loss_actor", 0.0)
        self.values_logger.add("loss_critic", 0.0)
        self.values_logger.add("internal_motivation_mean", 0.0)
        self.values_logger.add("internal_motivation_std", 0.0)

    def enable_training(self):
        self.enabled_training = True
 
    def disable_training(self):
        self.enabled_training = False

    def main(self): 
        #state to tensor
        states        = torch.tensor(self.states, dtype=torch.float).detach().to(self.model_ppo.device)

        #compute model output
        logits, values_ext, values_int  = self.model_ppo.forward(states)
        
       
        #collect actions
        actions = self._sample_actions(logits)
         
        #execute action
        states_new, rewards_ext, dones, infos = self.envs.step(actions)
 
        #update long term states mean and variance
        self.states_running_stats.update(states_new)

        #curiosity motivation
        rewards_int    = self._curiosity(states)        
        rewards_int    = torch.clip(rewards_int, 0.0, 1.0)
        
        #put into policy buffer
        if self.enabled_training:
            states          = states.detach().to("cpu")
            logits          = logits.detach().to("cpu")
            values_ext      = values_ext.squeeze(1).detach().to("cpu") 
            values_int      = values_int.squeeze(1).detach().to("cpu")
            actions         = torch.from_numpy(actions).to("cpu")
            rewards_ext_t   = torch.from_numpy(rewards_ext).to("cpu")
            rewards_int_t   = rewards_int.detach().to("cpu")
            dones           = torch.from_numpy(dones).to("cpu")

            self.policy_buffer.add(states, logits, values_ext, values_int, actions, rewards_ext_t, rewards_int_t, dones)

            if self.policy_buffer.is_full():
                self.train()    


        self.states = states_new.copy()
        
        for e in range(self.envs_count): 
            if dones[e]:
                self.states[e] = self.envs.reset(e).copy()

        #collect stats
        self.values_logger.add("internal_motivation_mean", rewards_int.mean().detach().to("cpu").numpy())
        self.values_logger.add("internal_motivation_std" , rewards_int.std().detach().to("cpu").numpy())

       
        self.iterations+= 1
        return rewards_ext[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_rnd.save(save_path + "trained/")

    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
        self.model_rnd.load(load_path + "trained/")

    def get_log(self): 
        return self.values_logger.get_str()

    
    def render(self, env_id):
        size            = 256

        states_t        = torch.tensor(self.states, dtype=torch.float).detach().to(self.model_ppo.device)

        state           = self._norm_state(states_t)[env_id][0].detach().to("cpu").numpy()

        state_im        = cv2.resize(state, (size, size))
        state_im        = numpy.clip(state_im, 0.0, 1.0)

        cv2.imshow("RND agent", state_im)
        cv2.waitKey(1)
    
        



    def _sample_actions(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().to("cpu").numpy()
        return actions
    
    def train(self): 
        self.policy_buffer.compute_returns(self.gamma_ext, self.gamma_int)

        batch_count = self.steps//self.batch_size

        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                states, _, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)

                #train PPO model
                loss_ppo = self._compute_loss_ppo(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int)

                self.optimizer_ppo.zero_grad()        
                loss_ppo.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()

                #train RND model, MSE loss
                loss_rnd = self._compute_loss_rnd(states)

                self.optimizer_rnd.zero_grad() 
                loss_rnd.backward()
                self.optimizer_rnd.step()

              
                self.values_logger.add("loss_rnd", loss_rnd.detach().to("cpu").numpy())
        
        self.policy_buffer.clear() 

    
    def _compute_loss_ppo(self, states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int):
        logits_new, values_ext_new, values_int_new  = self.model_ppo.forward(states)

        #critic loss
        loss_critic = self._compute_critic_loss(values_ext_new, returns_ext, values_int_new, returns_int)

        #actor loss        
        advantages  = self.ext_adv_coeff*advantages_ext + self.int_adv_coeff*advantages_int
        advantages  = advantages.detach() 
        loss_policy, loss_entropy  = self._compute_actor_loss(logits, logits_new, advantages, actions)

        loss_actor = loss_policy + loss_entropy
        
        #total loss
        loss = 0.5*loss_critic + loss_actor

        #store to log
        self.values_logger.add("loss_actor", loss_actor.mean().detach().to("cpu").numpy())
        self.values_logger.add("loss_critic", loss_critic.mean().detach().to("cpu").numpy())
        
        return loss 

    #MSE critic loss
    def _compute_critic_loss(self, values_ext_new, returns_ext, values_int_new, returns_int):
        ''' 
        compute external critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_ext_new  = values_ext_new.squeeze(1)
        loss_ext_value  = (returns_ext.detach() - values_ext_new)**2
        loss_ext_value  = loss_ext_value.mean()

        '''
        compute internal critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_int_new  = values_int_new.squeeze(1)
        loss_int_value  = (returns_int.detach() - values_int_new)**2
        loss_int_value  = loss_int_value.mean()
        
        loss_critic     = loss_ext_value + loss_int_value
        return loss_critic

    #PPO actor loss
    def _compute_actor_loss(self, logits, logits_new, advantages, actions):
        log_probs_old = torch.nn.functional.log_softmax(logits, dim = 1).detach()

        probs_new     = torch.nn.functional.softmax(logits_new, dim = 1)
        log_probs_new = torch.nn.functional.log_softmax(logits_new, dim = 1)

        ''' 
        compute actor loss, surrogate loss
        '''
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

        return loss_policy, loss_entropy


    #MSE loss for RND model
    def _compute_loss_rnd(self, states):
        
        state_norm_t    = self._norm_state(states).detach()
 
        features_predicted_t, features_target_t  = self.model_rnd(state_norm_t)

        loss_rnd = (features_target_t - features_predicted_t)**2

        #random loss regularisation, 25% non zero for 128envs, 100% non zero for 32envs
        prob            = 32.0/self.envs_count
        random_mask     = torch.rand(loss_rnd.shape).to(loss_rnd.device)
        random_mask     = 1.0*(random_mask < prob) 
        loss_rnd        = (loss_rnd*random_mask).sum() / (random_mask.sum() + 0.00000001)

        return loss_rnd
    
    #compute internal motivation
    def _curiosity(self, state_t):
        state_norm_t    = self._norm_state(state_t)

        features_predicted_t, features_target_t  = self.model_rnd(state_norm_t)

        curiosity_t = (features_target_t - features_predicted_t)**2
        curiosity_t = curiosity_t.sum(dim=1)/2.0
     
        return curiosity_t
 

    #normalise mean and std for state
    def _norm_state(self, states):
        states_norm = states

        if self.normalise_state_mean:
            mean = torch.from_numpy(self.states_running_stats.mean).to(states.device).float()
            states_norm = states_norm - mean

        if self.normalise_state_std:
            std  = torch.from_numpy(self.states_running_stats.std).to(states.device).float()            
            states_norm = torch.clamp(states_norm/std, -1.0, 1.0)

        return states_norm 


    #random policy for stats init
    def _init_running_stats(self, steps = 256):
        for _ in range(steps):
            #random action
            actions = numpy.random.randint(0, self.actions_count, (self.envs_count))
            states, _, dones, _ = self.envs.step(actions)

            #update stats
            self.states_running_stats.update(states)

            for e in range(self.envs_count): 
                if dones[e]:
                    self.envs.reset(e)

