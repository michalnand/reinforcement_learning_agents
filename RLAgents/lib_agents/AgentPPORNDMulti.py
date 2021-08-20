import numpy
import torch
import time

from torch.distributions import Categorical
 
from .PolicyBufferIM    import *  
from .RunningStats      import * 
    
class AgentPPORNDMulti():   
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
        self.actors             = config.actors 

        self.rnd_count          = config.rnd_count

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)

        self.rnd_models     = []
        self.rnd_optimizers = []
        

        for i in range(self.rnd_count):
            model_rnd      = ModelRND.Model(self.state_shape)
            optimizer_rnd  = torch.optim.Adam(model_rnd.parameters(), lr=config.learning_rate_rnd)

            self.rnd_models.append(model_rnd)
            self.rnd_optimizers.append(optimizer_rnd)

        self.rnd_model_score = numpy.zeros((config.rnd_count, self.actors))
        
 
        self.policy_buffer = PolicyBufferIM(self.steps, self.state_shape, self.actions_count, self.actors, self.model_ppo.device, True)
 
        self.states = numpy.zeros((self.actors, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.actors):
            self.states[e] = self.envs.reset(e).copy()

        features_predicted_t, _     = self.rnd_models[0](torch.from_numpy(self.states).to(self.rnd_models[0].device))  
        self.rnd_features_shape     = features_predicted_t.shape


        self.states_running_stats       = RunningStats(self.state_shape, self.states)
 
        self.enable_training()
        self.iterations                 = 0 

        self.log_loss_rnd               = 0.0
        self.log_curiosity              = 0.0
        self.log_advantages             = 0.0
        self.log_curiosity_advatages    = 0.0
        self.rnd_models_using           = numpy.zeros(config.rnd_count)

    def enable_training(self):
        self.enabled_training = True
 
    def disable_training(self):
        self.enabled_training = False

    def main(self):
        #state to tensor
        states_t            = torch.tensor(self.states, dtype=torch.float).detach().to(self.model_ppo.device)

        #compute model output
        logits_t, values_ext_t, values_int_t  = self.model_ppo.forward(states_t)
        
        states_np       = states_t.detach().to("cpu").numpy()
        logits_np       = logits_t.detach().to("cpu").numpy()
        values_ext_np   = values_ext_t.squeeze(1).detach().to("cpu").numpy()
        values_int_np   = values_int_t.squeeze(1).detach().to("cpu").numpy()

        #collect actions
        actions = self._sample_actions(logits_t)
        
        #execute action
        states, rewards, dones, infos = self.envs.step(actions)

        self.states = states.copy()
 
        #update long term states mean and variance
        self.states_running_stats.update(states_np)

        #curiosity motivation
        states_new_t    = torch.tensor(states, dtype=torch.float).detach().to(self.model_ppo.device)
        curiosity_np, rnd_models_using    = self._curiosity(states_new_t)
        curiosity_np    = numpy.clip(curiosity_np, -1.0, 1.0)

        
         
        #put into policy buffer
        if self.enabled_training:
            self.policy_buffer.add(states_np, logits_np, values_ext_np, values_int_np, actions, rewards, curiosity_np, dones)

            if self.policy_buffer.is_full():
                self.train()
        
        for e in range(self.actors): 
            if dones[e]:
                self.states[e] = self.envs.reset(e).copy()

        #collect stats
        k = 0.02
        self.log_curiosity      = (1.0 - k)*self.log_curiosity + k*curiosity_np.mean()
        self.rnd_models_using   = (1.0 - k)*self.rnd_models_using + k*rnd_models_using

        self.iterations+= 1
        return rewards[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")

        for m in range(self.rnd_count):
            self.model_rnd[m].save(save_path + "trained/" + str(m) + "_")

    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
 
        for m in range(self.rnd_count):
            self.model_rnd.load(load_path + "trained/" + str(m) + "_")

    def get_log(self): 
        result = "" 
        result+= str(round(self.log_loss_rnd, 7)) + " "
        result+= str(round(self.log_curiosity, 7)) + " "
        result+= str(round(self.log_advantages, 7)) + " "
        result+= str(round(self.log_curiosity_advatages, 7)) + " "
        
        rnd_models_using_relative = self.rnd_models_using/(self.rnd_models_using.sum() + 10**-15)
        for m in range(self.rnd_count):
            result+= str(round(rnd_models_using_relative[m], 4)) + " "

        return result  
    

    '''
    def _sample_action(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 0)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()

        return action_t.item()
    '''

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
                loss = self._compute_loss(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int)

                self.optimizer_ppo.zero_grad()        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()

                #train RND models, MSE loss
                for m in range(self.rnd_count):
                    state_norm_t    = self._norm_state(states).detach()

                    features_predicted_t, features_target_t  = self.rnd_models[m](state_norm_t)

                    loss_rnd        = (features_target_t - features_predicted_t)**2
                    
                    random_mask     = torch.rand(loss_rnd.shape).to(loss_rnd.device)
                    random_mask     = 1.0*(random_mask < 0.25)
                    loss_rnd        = (loss_rnd*random_mask).sum() / (random_mask.sum() + 0.00000001)

                    self.rnd_optimizers[m].zero_grad() 
                    loss_rnd.backward()
                    self.rnd_optimizers[m].step()

                    k = 0.02 
                    self.log_loss_rnd  = (1.0 - k)*self.log_loss_rnd + k*loss_rnd.detach().to("cpu").numpy()

        self.policy_buffer.clear() 

    
    def _compute_loss(self, states, logits, actions,  returns_ext, returns_int, advantages_ext, advantages_int):
        log_probs_old = torch.nn.functional.log_softmax(logits, dim = 1).detach()
        logits_new, values_ext_new, values_int_new  = self.model_ppo.forward(states)

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
        compute internal critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_int_new  = values_int_new.squeeze(1)
        loss_int_value  = (returns_int.detach() - values_int_new)**2
        loss_int_value  = loss_int_value.mean()
        
        loss_critic     = loss_ext_value + loss_int_value
 
        ''' 
        compute actor loss, surrogate loss
        '''
        advantages      = self.ext_adv_coeff*advantages_ext + self.int_adv_coeff*advantages_int
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
        self.log_advantages             = (1.0 - k)*self.log_advantages + k*advantages_ext.mean().detach().to("cpu").numpy()
        self.log_curiosity_advatages    = (1.0 - k)*self.log_curiosity_advatages + k*advantages_int.mean().detach().to("cpu").numpy()

        return loss 


    def _norm_state(self, state_t):
        mean = torch.from_numpy(self.states_running_stats.mean).to(state_t.device).float()
        std  = torch.from_numpy(self.states_running_stats.std).to(state_t.device).float()

        state_norm_t = state_t - mean
        #state_norm_t = torch.clip((state_t - mean)/std, -4.0, 4.0)

        return state_norm_t


    def _rnd_get_dif(self, state_t):
        state_norm_t = self._norm_state(state_t)

        features_predicted_all_t    = torch.zeros((self.rnd_count, ) + self.rnd_features_shape, device=self.rnd_models[0].device)
        features_target_all_t       = torch.zeros((self.rnd_count, ) + self.rnd_features_shape, device=self.rnd_models[0].device)
        
        #find curiosity across all RND models 
        for i in range(self.rnd_count):
            features_predicted_t, features_target_t  = self.rnd_models[i](state_norm_t)  
            features_predicted_all_t[i] = features_predicted_t
            features_target_all_t[i]    = features_target_t

        return features_predicted_all_t, features_target_all_t

      
    def _curiosity(self, state_t, k = 0.9):
        features_predicted_all_t, features_target_all_t = self._rnd_get_dif(state_t)

        curiosity_t    = (features_target_all_t - features_predicted_all_t)**2
        curiosity_t    = curiosity_t.sum(dim=2)/2.0

        curiosity      = curiosity_t.detach().to("cpu").numpy()

        self.rnd_model_score    = k*self.rnd_model_score + (1.0 - k)*curiosity
        max_rnd_indices         = numpy.argmax(self.rnd_model_score, axis=0)

        batch_size = state_t.shape[0]

        curiosity_result = numpy.zeros(batch_size)
        for i in range(batch_size):
            idx = max_rnd_indices[i]
            curiosity_result[i] = curiosity[idx][i]

        #count rnd using
        counts = numpy.zeros(self.rnd_count)
        for i in range(self.rnd_count):
            counts[i] = (max_rnd_indices == i).sum()

        return curiosity_result, counts
            
