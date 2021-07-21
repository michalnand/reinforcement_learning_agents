import numpy
import torch
import time

from torch.distributions import Categorical
 
from .PolicyBufferIMDual    import *  
from .GoalsMemory           import *
from .RunningStats          import *
  
   
class AgentPPORNDSkills():   
    def __init__(self, envs, ModelPPO, ModelRND, config):
        self.envs = envs  
   
        self.gamma_ext          = config.gamma_ext
        self.gamma_int          = config.gamma_int
           
        self.ext_adv_coeff      = config.ext_adv_coeff
        self.int_a_adv_coeff    = config.int_a_adv_coeff
        self.int_b_adv_coeff    = config.int_b_adv_coeff
    
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip 
   
        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.actors             = config.actors 

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n
        

        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)

        self.model_rnd      = ModelRND.Model(self.state_shape)
        self.optimizer_rnd  = torch.optim.Adam(self.model_rnd.parameters(), lr=config.learning_rate_rnd)

        self.policy_buffer  = PolicyBufferIMDual(self.steps, self.state_shape, self.actions_count, self.actors, self.model_ppo.device)
 
        self.states = numpy.zeros((self.actors, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.actors):
            self.states[e] = self.envs.reset(e).copy()

        #self.goals_memory = GoalsMemoryNovelty(config.goals_memory_size, downsample = 4, add_threshold= config.goals_memory_threshold, alpha=config.goals_memory_alpha, epsilon = 0.0001, device = self.model_ppo.device)
        self.goals_memory  = GoalsMemoryGraph(config.goals_memory_size, downsample = 8, add_threshold = config.goals_memory_threshold, device = self.model_ppo.device)

        self.steps_t      = torch.zeros((self.actors, )).to(self.model_ppo.device)

        self.states_running_stats       = RunningStats(self.state_shape, self.states)

        self.enable_training()
        self.iterations                 = 0 

        self.log_loss_rnd               = 0.0
        self.log_curiosity              = 0.0
        self.log_skills                 = 0.0
        self.log_advantages             = 0.0
        self.log_curiosity_advatages    = 0.0
        self.log_skills_advatages       = 0.0


    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

    def main(self):
        #state to tensor
        states_t            = torch.tensor(self.states, dtype=torch.float).detach().to(self.model_ppo.device)

        #compute model output
        logits_t, values_ext_t, values_int_a_t, values_int_b_t  = self.model_ppo.forward(states_t)
        
        states_np       = states_t.detach().to("cpu").numpy()
        logits_np       = logits_t.detach().to("cpu").numpy()
        values_ext_np   = values_ext_t.squeeze(1).detach().to("cpu").numpy()
        values_int_a_np = values_int_a_t.squeeze(1).detach().to("cpu").numpy()
        values_int_b_np = values_int_b_t.squeeze(1).detach().to("cpu").numpy()

        #collect actions
        actions = self._sample_actions(logits_t)
        
        #execute action
        states, rewards, dones, infos = self.envs.step(actions)

        #update long term states mean and variance
        self.states_running_stats.update(states_np)

        #curiosity motivation
        states_new_t    = torch.tensor(states, dtype=torch.float).detach().to(self.model_ppo.device)
        curiosity_np    = self._curiosity(states_new_t)
        curiosity_np    = numpy.clip(curiosity_np, -1.0, 1.0)
 
        #skills motivation
        skills_np       = self._skills(states_t)
        skills_np       = numpy.clip(skills_np, -1.0, 1.0)
        
        self.states     = states.copy()

        self.steps_t+= 1

        #put into policy buffer
        if self.enabled_training:
            self.policy_buffer.add(states_np, logits_np, values_ext_np, values_int_a_np, values_int_b_np, actions, rewards, curiosity_np, skills_np, dones)

            if self.policy_buffer.is_full():
                self.train()
        
        for e in range(self.actors): 
            if dones[e]:
                s_new = self.envs.reset(e)
                self.states[e]  = s_new.copy() 
                self.steps_t[e] = 0

        #collect stats
        k = 0.02
        self.log_curiosity  = (1.0 - k)*self.log_curiosity  + k*curiosity_np.mean()
        self.log_skills     = (1.0 - k)*self.log_skills     + k*skills_np.mean()

        self.iterations+= 1
        return rewards[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_rnd.save(save_path + "trained/")
        self.goals_memory.save(save_path + "result/")

    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
        self.model_rnd.load(load_path + "trained/")

    def get_log(self): 
        result = "" 
        result+= str(round(self.log_loss_rnd, 7)) + " "
        result+= str(round(self.log_curiosity, 7)) + " "
        result+= str(round(self.log_skills, 7)) + " "
        result+= str(round(self.log_advantages, 7)) + " "
        result+= str(round(self.log_curiosity_advatages, 7)) + " "
        result+= str(round(self.log_skills_advatages, 7)) + " "
        result+= str(round(self.goals_memory.total_targets, 7)) + " "

        return result 
    

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
                states, logits, actions, returns_ext, returns_int_a, returns_int_b, advantages_ext, advantages_int_a, advantages_int_b = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)

                #train PPO model
                loss = self._compute_loss(states, logits, actions, returns_ext, returns_int_a, returns_int_b, advantages_ext, advantages_int_a, advantages_int_b)

                self.optimizer_ppo.zero_grad()        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()



                #train RND model, MSE loss
                state_norm_t    = self._norm_state(states).detach()

                features_predicted_t, features_target_t  = self.model_rnd(state_norm_t)

                loss_rnd        = (features_target_t - features_predicted_t)**2
                
                random_mask     = torch.rand(loss_rnd.shape).to(loss_rnd.device)
                random_mask     = 1.0*(random_mask < 1.0/self.training_epochs)
                loss_rnd        = (loss_rnd*random_mask).sum() / (random_mask.sum() + 0.00000001)

                self.optimizer_rnd.zero_grad() 
                loss_rnd.backward()
                self.optimizer_rnd.step()

                k = 0.02
                self.log_loss_rnd  = (1.0 - k)*self.log_loss_rnd + k*loss_rnd.detach().to("cpu").numpy()


        self.policy_buffer.clear() 

    
    def _compute_loss(self, states, logits, actions,  returns_ext, returns_int_a, returns_int_b, advantages_ext, advantages_int_a, advantages_int_b):
        log_probs_old = torch.nn.functional.log_softmax(logits, dim = 1).detach()

        logits_new, values_ext_new, values_int_a_new, values_int_b_new  = self.model_ppo.forward(states)

        probs_new     = torch.nn.functional.softmax(logits_new, dim = 1)
        log_probs_new = torch.nn.functional.log_softmax(logits_new, dim = 1)

        ''' 
        compute external critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_ext_new  = values_ext_new.squeeze(1)
        loss_ext_value  = (returns_ext.detach() - values_ext_new)**2
        loss_ext_value  = loss_ext_value.mean()


        '''
        compute internal A critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_int_a_new  = values_int_a_new.squeeze(1)
        loss_int_a_value  = (returns_int_a.detach() - values_int_a_new)**2
        loss_int_a_value  = loss_int_a_value.mean()

        '''
        compute internal B critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_int_b_new  = values_int_b_new.squeeze(1)
        loss_int_b_value  = (returns_int_b.detach() - values_int_b_new)**2
        loss_int_b_value  = loss_int_b_value.mean()
        
        
        loss_critic     = loss_ext_value + loss_int_a_value + loss_int_b_value
 
        ''' 
        compute actor loss, surrogate loss
        '''
        advantages      = self.ext_adv_coeff*advantages_ext + self.int_a_adv_coeff*advantages_int_a + self.int_b_adv_coeff*advantages_int_b
        advantages      = advantages.detach() 
        
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

        loss = 0.5*loss_critic + loss_policy + loss_entropy

        k = 0.02
        self.log_advantages             = (1.0 - k)*self.log_advantages             + k*advantages_ext.mean().detach().to("cpu").numpy()
        self.log_curiosity_advatages    = (1.0 - k)*self.log_curiosity_advatages    + k*advantages_int_a.mean().detach().to("cpu").numpy()
        self.log_skills_advatages       = (1.0 - k)*self.log_skills_advatages      + k*advantages_int_b.mean().detach().to("cpu").numpy()

        return loss 

    def _norm_state(self, state_t):
        mean = torch.from_numpy(self.states_running_stats.mean).to(state_t.device).float()
        state_norm_t = state_t - mean

        return state_norm_t

    def _curiosity(self, state_t):
        state_norm_t            = self._norm_state(state_t)

        features_predicted_t, features_target_t  = self.model_rnd(state_norm_t)

        curiosity_t    = (features_target_t - features_predicted_t)**2
        
        curiosity_t    = curiosity_t.sum(dim=1)/2.0
        
        return curiosity_t.detach().to("cpu").numpy()


    def _skills(self, states_t):
        motivation = self.goals_memory.process(states_t[:, 0])
        return motivation.detach().to("cpu").numpy()
