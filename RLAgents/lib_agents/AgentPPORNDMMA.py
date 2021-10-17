import numpy
import torch

from torch.distributions import Categorical
 
from .PolicyBufferIMDual    import *  
from .RunningStats          import * 

#ppo with rnd multi meta actor - MMA
class AgentPPORNDMMA():    
    def __init__(self, envs, ModelPPO, ModelRND, config):
        self.envs = envs  
     
        self.gamma_ext_a          = config.gamma_ext_a
        self.gamma_int_a          = config.gamma_int_a
        self.gamma_ext_b          = config.gamma_ext_b
        self.gamma_int_b          = config.gamma_int_b
            
        self.ext_adv_coeff      = config.ext_adv_coeff
        self.int_adv_coeff      = config.int_adv_coeff
    
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip 
   
        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count 

        state_shape         = self.envs.observation_space.shape
        self.state_shape    = (state_shape[0] + 1, ) + state_shape[1:]
        self.actions_count  = self.envs.action_space.n

        self.rnd_heads      = config.rnd_heads

        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count, self.rnd_heads)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)
 
        self.model_rnd      = ModelRND.Model(self.state_shape, self.rnd_heads)
        self.optimizer_rnd  = torch.optim.Adam(self.model_rnd.parameters(), lr=config.learning_rate_rnd)
 
        self.policy_buffer  = PolicyBufferIMDual(self.steps, self.state_shape, self.actions_count, self.rnd_heads, self.envs_count, self.model_ppo.device, True)
 

        states = numpy.zeros((self.envs_count, ) + state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            states[e] = self.envs.reset(e).copy()

        self.episode_score_sum      = numpy.zeros(self.envs_count)
        self.states                 = self._make_states(states, self.episode_score_sum)

        self.states_running_stats   = RunningStats(self.state_shape, self.states)
 
        self.enable_training()
        self.iterations                 = 0 

        self.log_loss_rnd               = 0.0
        self.log_internal_motivation    = 0.0
        self.log_loss_actor             = 0.0
        self.log_loss_critic            = 0.0
 
        self.log_heads_usage            = numpy.ones(self.rnd_heads)/self.rnd_heads

    def enable_training(self):
        self.enabled_training = True
 
    def disable_training(self):
        self.enabled_training = False

    def main(self): 
        #state to tensor
        states_t  = torch.tensor(self.states, dtype=torch.float).detach().to(self.model_ppo.device)

        #compute model output
        logits_a_t, logits_b_t, values_ext_a_t, values_int_a_t, values_ext_b_t, values_int_b_t  = self.model_ppo.forward(states_t)
        
        states_np       = states_t.detach().to("cpu").numpy()

        logits_a_np     = logits_a_t.detach().to("cpu").numpy()
        logits_b_np     = logits_b_t.detach().to("cpu").numpy()

        values_ext_a_np   = values_ext_a_t.squeeze(1).detach().to("cpu").numpy()
        values_int_a_np   = values_int_a_t.squeeze(1).detach().to("cpu").numpy()

        values_ext_b_np   = values_ext_b_t.squeeze(1).detach().to("cpu").numpy()
        values_int_b_np   = values_int_b_t.squeeze(1).detach().to("cpu").numpy()

        #collect actions
        actions = self._sample_actions(logits_a_t)

        #select RND head
        rnd_head_ids = self._sample_actions(logits_b_t)
        
        #execute action
        states, rewards_ext, dones, infos = self.envs.step(actions)

        self.episode_score_sum+= rewards_ext
 
        #update long term states mean and variance
        self.states_running_stats.update(states_np)

        #curiosity motivation
        rewards_int = self._curiosity(states_t, rnd_head_ids)
        rewards_int = numpy.clip(rewards_int, -1.0, 1.0)
         
        #put into policy buffer
        if self.enabled_training:
            self.policy_buffer.add(states_np)
            self.policy_buffer.add_a(logits_a_np, values_ext_a_np, values_int_a_np, actions, rewards_ext, rewards_int, dones)
            self.policy_buffer.add_b(logits_b_np, values_ext_b_np, values_int_b_np, rnd_head_ids, rewards_ext, rewards_int, dones)
            
            if self.policy_buffer.is_full():
                self.train()
        
        for e in range(self.envs_count): 
            if dones[e]:
                states[e]                   = self.envs.reset(e)
                self.episode_score_sum[e]   = 0

        self.states = self._make_states(states, self.episode_score_sum)

        #collect stats
        k = 0.02
        self.log_internal_motivation = (1.0 - k)*self.log_internal_motivation + k*rewards_int.mean()

        heads_usage = numpy.zeros(self.rnd_heads)
        for h in range(self.rnd_heads):
            heads_usage[h] = (rnd_head_ids == h).sum()
        heads_usage = heads_usage/heads_usage.sum()

        self.log_heads_usage = (1.0 - k)*self.log_heads_usage + k*heads_usage

        self.iterations+= 1
        return rewards_ext[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_rnd.save(save_path + "trained/")

    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
        self.model_rnd.load(load_path + "trained/")

    def get_log(self): 
        result = "" 
        result+= str(round(self.log_loss_rnd, 7)) + " "
        result+= str(round(self.log_internal_motivation, 7)) + " "
        result+= str(round(self.log_loss_actor, 7)) + " "
        result+= str(round(self.log_loss_critic, 7)) + " "

        for h in range(self.rnd_heads):
            result+= str(round(self.log_heads_usage[h], 3)) + " "

        return result 

    def _sample_actions(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().to("cpu").numpy()
        return actions
    
    def train(self): 
        self.policy_buffer.compute_returns(self.gamma_ext_a, self.gamma_int_a, self.gamma_ext_b, self.gamma_int_b)

        batch_count = self.steps//self.batch_size

        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):

                states, res_a, res_b = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)

                #train PPO model
                loss_ppo = self._compute_loss_ppo(states, res_a, res_b)

                self.optimizer_ppo.zero_grad()        
                loss_ppo.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()


                #train RND model, MSE loss
                _, rnd_head_ids, _, _, _, _ = res_b 
                loss_rnd = self._compute_loss_rnd(states, rnd_head_ids)

                self.optimizer_rnd.zero_grad() 
                loss_rnd.backward()
                self.optimizer_rnd.step()

                k = 0.02
                self.log_loss_rnd  = (1.0 - k)*self.log_loss_rnd + k*loss_rnd.detach().to("cpu").numpy()

        self.policy_buffer.clear() 

    
    def _compute_loss_ppo(self, states, res_a, res_b):

        logits_a, actions_a, returns_ext_a, returns_int_a, advantages_ext_a, advantages_int_a  = res_a
        logits_b, actions_b, returns_ext_b, returns_int_b, advantages_ext_b, advantages_int_b  = res_b

         
        logits_a_new, logits_b_new, values_ext_a_new, values_int_a_new, values_ext_b_new, values_int_b_new  = self.model_ppo.forward(states)
        
        #actor loss A
        advantages_a  = self.ext_adv_coeff*advantages_ext_a + self.int_adv_coeff*advantages_int_a
        advantages_a  = advantages_a.detach() 
        loss_policy_a, loss_entropy_a  = self._compute_actor_loss(logits_a, logits_a_new, advantages_a, actions_a)

        #actor loss B
        advantages_b  = self.ext_adv_coeff*advantages_ext_b + self.int_adv_coeff*advantages_int_b
        advantages_b  = advantages_b.detach() 
        loss_policy_b, loss_entropy_b  = self._compute_actor_loss(logits_b, logits_b_new, advantages_b, actions_b)

        loss_actor = loss_policy_a + loss_entropy_a + loss_policy_b + loss_entropy_b

        #critic loss A
        loss_critic_a = self._compute_critic_loss(values_ext_a_new, returns_ext_a, values_int_a_new, returns_int_a)

        #critic loss B
        loss_critic_b = self._compute_critic_loss(values_ext_b_new, returns_ext_b, values_int_b_new, returns_int_b)

        loss_critic   = loss_critic_a + loss_critic_b

        loss = 0.5*loss_critic + loss_actor


        k = 0.02
        self.log_loss_actor  = (1.0 - k)*self.log_loss_actor    + k*loss_actor.mean().detach().to("cpu").numpy()
        self.log_loss_critic = (1.0 - k)*self.log_loss_critic   + k*loss_critic.mean().detach().to("cpu").numpy()

        return loss
 

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


    def _compute_actor_loss(self, logits_old, logits_new, advantages, actions):
        log_probs_old = torch.nn.functional.log_softmax(logits_old, dim = 1).detach()

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


    def _compute_loss_rnd(self, states, heads_ids):
        #MSE loss for RND model
        state_norm_t    = self._norm_state(states).detach()

        features_predicted_t, features_target_t  = self.model_rnd(state_norm_t, heads_ids)

        loss_rnd        = (features_target_t - features_predicted_t)**2 
        
        #regularisation
        prob            = 32.0/self.envs_count
        random_mask     = torch.rand(loss_rnd.shape).to(loss_rnd.device)
        random_mask     = 1.0*(random_mask < prob)
        loss_rnd        = (loss_rnd*random_mask).sum() / (random_mask.sum() + 0.00000001)

        return loss_rnd
        
    def _curiosity(self, state_t, heads_ids):
        state_norm_t    = self._norm_state(state_t)

        head_ids_t    = torch.from_numpy(heads_ids).to(state_norm_t.device)

        features_predicted_t, features_target_t  = self.model_rnd(state_norm_t, head_ids_t)

        curiosity_t    = (features_target_t - features_predicted_t)**2
        
        curiosity_t    = curiosity_t.sum(dim=1)/2.0
        
        return curiosity_t.detach().to("cpu").numpy()

    def _norm_state(self, state_t):
        mean = torch.from_numpy(self.states_running_stats.mean).to(state_t.device).float()
        std  = torch.from_numpy(self.states_running_stats.std).to(state_t.device).float()
        
        state_norm_t = state_t - mean 

        #state_norm_t = (state_t - mean)/std
        #state_norm_t = torch.clamp(state_norm_t, -4.0, 4.0)

        return state_norm_t

    def _make_states(self, state, score, max_range = 16):
        tmp     = (numpy.floor(score)%max_range)/(1.0*max_range)

        tmp     = numpy.reshape(tmp, (score.shape[0], 1, 1, 1))
        tmp     = numpy.repeat(tmp, state.shape[2], axis=2)
        tmp     = numpy.repeat(tmp, state.shape[3], axis=3)

        result  = numpy.concatenate([state, tmp], axis=1)

        return result

