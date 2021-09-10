import numpy
import torch
import time

from torch.distributions import Categorical
 
from .PolicyBufferIMDual    import *  
from .GoalsBuffer           import *
from .RunningStats          import * 
    
class AgentPPOEE():   
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

        self.state_shape        = self.envs.observation_space.shape
        self.goal_shape         = (1, ) + self.state_shape[1:]
        self.actions_count      = self.envs.action_space.n
        
        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)
 
        self.model_rnd      = ModelRND.Model(self.state_shape)
        self.optimizer_rnd  = torch.optim.Adam(self.model_rnd.parameters(), lr=config.learning_rate_rnd)
 
        self.policy_buffer  = PolicyBufferIMDual(self.steps, self.state_shape, self.goal_shape, self.actions_count, self.envs_count, self.model_ppo.device, True)


        #TODO, load from config file
        self.goals_buffer = GoalsBuffer(size, add_threshold, downsample, goals_weights, self.state_shape, self.envs_count, self.model_ppo.device)

        #initial state
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e] = self.envs.reset(e).copy()

        #init moving average for RND
        self.states_running_stats       = RunningStats(self.state_shape, self.states)

        #initial all agents into explore mode 
        self.agent_mode                 = torch.zeros((self.envs_count,)).to(self.model_ppo.device)

        self.enable_training()
        self.iterations                 = 0 

        self.log_loss_rnd               = 0.0
        self.log_curiosity              = 0.0
        self.log_advantages             = 0.0
        self.log_curiosity_advatages    = 0.0

    def enable_training(self):
        self.enabled_training = True
 
    def disable_training(self):
        self.enabled_training = False

    def main(self): 
        #state to tensor
        states_t                = torch.tensor(self.states, dtype=torch.float).detach().to(self.model_ppo.device)
        goals_t, goals_reward_ext, goals_reward_int = self.goals_buffer.get(states_t)

        #compute model output
        logits_a_t, logits_b_t, values_ext_a_t, values_int_a_t, values_ext_b_t, values_int_b_t  = self.model_ppo.forward(states_t, goals_t, self.agent_mode)
        
        states_np       = states_t.detach().to("cpu").numpy()
        logits_a_np     = logits_a_t.detach().to("cpu").numpy()
        logits_b_np     = logits_b_t.detach().to("cpu").numpy()

        values_ext_a_np = values_ext_a_t.squeeze(1).detach().to("cpu").numpy()
        values_int_a_np = values_int_a_t.squeeze(1).detach().to("cpu").numpy()

        values_ext_b_np = values_ext_b_t.squeeze(1).detach().to("cpu").numpy()
        values_int_b_np = values_int_b_t.squeeze(1).detach().to("cpu").numpy()

        #collect actions
        actions = self._sample_actions(logits_a_t, logits_b_t, self.agent_mode)
        
        #execute action
        states, rewards, dones, infos = self.envs.step(actions)

        self.states = states.copy()
 
        #update long term states mean and variance
        self.states_running_stats.update(states_np)

        #curiosity motivation
        states_new_t    = torch.tensor(states, dtype=torch.float).detach().to(self.model_ppo.device)
        curiosity_np    = self._curiosity(states_new_t)
        curiosity_np    = numpy.clip(curiosity_np, -1.0, 1.0)
        
        goals_np        = goals_t.detach().to("cpu").numpy()
        mode_np         = self.agent_mode.detach().to("cpu").numpy()
         
        #put into policy buffer
        if self.enabled_training:
            self.policy_buffer.add(states_np, goals_np, mode_np)
            self.policy_buffer.add_a(logits_a_np, values_ext_a_np, values_int_a_np, actions, reward, curiosity_np, dones)
            self.policy_buffer.add_b(logits_b_np, values_ext_b_np, values_int_b_np, actions, goals_reward_ext, goals_reward_int, dones)
      
            if self.policy_buffer.is_full():
                self.train()
        
        for e in range(self.envs_count): 
            if dones[e]:
                self.states[e] = self.envs.reset(e).copy()

                #switch agent with 50% prob to exploit mode, except env 0
                if numpy.random.rand() > 0.5 and e != 0:
                    self.goals_buffer.new_goal(e)
                    self.agent_mode[e] = 1.0
                else:
                    self.goals_buffer.zero_goal(e)
                    self.agent_mode[e] = 0.0

        #collect stats
        k = 0.02
        self.log_curiosity = (1.0 - k)*self.log_curiosity + k*curiosity_np.mean()

        self.iterations+= 1
        return rewards[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_rnd.save(save_path + "trained/")

    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
        self.model_rnd.load(load_path + "trained/")

    def get_log(self): 
        result = "" 
        result+= str(round(self.log_loss_rnd, 7)) + " "
        result+= str(round(self.log_curiosity, 7)) + " "
        result+= str(round(self.log_advantages, 7)) + " "
        result+= str(round(self.log_curiosity_advatages, 7)) + " "
        return result 
    

    '''
    def _sample_action(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 0)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()

        return action_t.item()
    '''

    def _sample_actions_from_logits(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().to("cpu").numpy()
        return actions

    def _sample_actions(self, logits_a, logits_b, mode):
        mode_ = mode.unsqueeze(1)
        logits = (1.0 - mode_)*logits_a + mode_*logits_b
        return self._sample_actions_from_logits(logits)
        
    def train(self): 
        self.policy_buffer.compute_returns(self.gamma_ext, self.gamma_int)

        batch_count = self.steps//self.batch_size

        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                #TODO, loss + training this monster
                states, goals, modes, logits_a, logits_b, actions, returns_ext, returns_int_a, returns_int_b, advantages_ext, advantages_int_a, advantages_int_b = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)

                #train PPO model
                loss = self._compute_loss(states, goals, modes, logits_a, logits_b, actions, returns_ext, returns_int_a, returns_int_b, advantages_ext, advantages_int_a, advantages_int_b)

                self.optimizer_ppo.zero_grad()        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()

                #train RND model, MSE loss
                state_norm_t    = self._norm_state(states).detach()

                features_predicted_t, features_target_t  = self.model_rnd(state_norm_t)

                loss_rnd        = (features_target_t - features_predicted_t)**2
                
                random_mask     = torch.rand(loss_rnd.shape).to(loss_rnd.device)
                random_mask     = 1.0*(random_mask < 0.25)
                loss_rnd        = (loss_rnd*random_mask).sum() / (random_mask.sum() + 0.00000001)

                self.optimizer_rnd.zero_grad() 
                loss_rnd.backward()
                self.optimizer_rnd.step()

                k = 0.02
                self.log_loss_rnd  = (1.0 - k)*self.log_loss_rnd + k*loss_rnd.detach().to("cpu").numpy()

        self.policy_buffer.clear() 

    
    def _compute_loss(self, states, goals, modes, logits_a, logits_b, actions, returns_ext_a, returns_ext_b, returns_int_a, returns_int_b, advantages_ext_a, advantages_ext_b, advantages_int_a, advantages_int_b):
        log_probs_a_old = torch.nn.functional.log_softmax(logits_a, dim = 1).detach()
        log_probs_b_old = torch.nn.functional.log_softmax(logits_b, dim = 1).detach()

        logits_a_new, logits_b_new, values_ext_a_new, values_ext_b_new, values_int_a_new, values_int_b_new  = self.model_ppo.forward(states, goals, modes)

        probs_a_new     = torch.nn.functional.softmax(logits_a_new, dim = 1)
        log_probs_a_new = torch.nn.functional.log_softmax(logits_a_new, dim = 1)

        probs_b_new     = torch.nn.functional.softmax(logits_b_new, dim = 1)
        log_probs_b_new = torch.nn.functional.log_softmax(logits_b_new, dim = 1)

        ''' 
        compute external critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_ext_a_new    = values_ext_a_new.squeeze(1)
        loss_ext_value_a    = (returns_ext_a.detach() - values_ext_a_new)**2
        loss_ext_value_a    = loss_ext_value_a.mean()

        values_ext_b_new    = values_ext_b_new.squeeze(1)
        loss_ext_value_b    = (returns_ext_b.detach() - values_ext_b_new)**2
        loss_ext_value_b    = loss_ext_value_b.mean()

     

        '''
        compute internal critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_int_a_new  = values_int_a_new.squeeze(1)
        loss_int_value_a  = (returns_int_a.detach() - values_int_a_new)**2
        loss_int_value_a  = loss_int_value_a.mean()

        values_int_b_new  = values_int_b_new.squeeze(1)
        loss_int_value_b  = (returns_int_b.detach() - values_int_b_new)**2
        loss_int_value_b  = loss_int_value_b.mean()
        
        
        loss_critic     = loss_ext_value_a + loss_ext_value_b + loss_int_value_a + loss_int_value_b
 
        ''' 
        compute actor loss, surrogate loss
        '''
        loss_actor_a = self._actor_loss(self.ext_adv_coeff*advantages_ext_a + self.int_adv_coeff*advantages_int_a, probs_a_new, log_probs_a_new, log_probs_a_old, actions, 1.0 - modes)
        loss_actor_b = self._actor_loss(self.ext_adv_coeff*advantages_ext_b + self.int_adv_coeff*advantages_int_b, probs_b_new, log_probs_b_new, log_probs_b_old, actions, modes)
        

        loss = 0.5*loss_critic + loss_actor_a + loss_actor_b

        k = 0.02
        self.log_advantages             = (1.0 - k)*self.log_advantages + k*advantages_ext.mean().detach().to("cpu").numpy()
        self.log_curiosity_advatages    = (1.0 - k)*self.log_curiosity_advatages + k*advantages_int.mean().detach().to("cpu").numpy()

        return loss 

    def _actor_loss(self, advantages, probs_new, log_probs_new, log_probs_old, actions, mask):
        ''' 
        compute actor loss, surrogate loss
        '''
        advantages      = advantages.detach() 
        
        log_probs_new_  = log_probs_new[range(len(log_probs_new)), actions]
        log_probs_old_  = log_probs_old[range(len(log_probs_old)), actions]
                        
        ratio       = torch.exp(log_probs_new_ - log_probs_old_)
        p1          = ratio*advantages
        p2          = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages
        loss_policy = -torch.min(p1, p2)  
        loss_policy = loss_policy*mask
        loss_policy = loss_policy.mean()
    
        ''' 
        compute entropy loss, to avoid greedy strategy
        L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        '''
        loss_entropy = (probs_new*log_probs_new).sum(dim = 1)
        loss_entropy = loss_entropy*mask
        loss_entropy = self.entropy_beta*loss_entropy.mean()

        return loss_policy + loss_entropy

    def _curiosity(self, state_t):
        state_norm_t            = self._norm_state(state_t)

        features_predicted_t, features_target_t  = self.model_rnd(state_norm_t)

        curiosity_t    = (features_target_t - features_predicted_t)**2
        
        curiosity_t    = curiosity_t.sum(dim=1)/2.0
        
        return curiosity_t.detach().to("cpu").numpy()

    def _norm_state(self, state_t):
        mean = torch.from_numpy(self.states_running_stats.mean).to(state_t.device).float()
        std  = torch.from_numpy(self.states_running_stats.std).to(state_t.device).float()

        state_norm_t = state_t - mean 
        #state_norm_t = torch.clip((state_t - mean)/std, -4.0, 4.0)

        return state_norm_t
