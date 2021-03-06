from math import log
import numpy
import torch
import time
 
from torch.distributions import Categorical
 
from .PolicyBufferIMDual    import *  
from .RunningStats          import *
    
class AgentPPORNDImagination():   
    def __init__(self, envs, ModelPPO, ModelRND, ModelForward, ModelFeaturesRND, config):
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

        self.rollouts_count     = config.rollouts_count
        self.rollouts_length    = config.rollouts_length

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)
 
        self.model_rnd      = ModelRND.Model(self.state_shape)
        self.optimizer_rnd  = torch.optim.Adam(self.model_rnd.parameters(), lr=config.learning_rate_rnd)

        features_count = self.model_ppo.forward_features(torch.randn((1, ) + self.state_shape).to(self.model_ppo.device)).shape[1]

        self.model_forward          = ModelForward.Model(features_count, self.actions_count)
        self.optimizer_forward      = torch.optim.Adam(self.model_forward.parameters(), lr=config.learning_rate_forward)

        self.model_features_rnd     = ModelFeaturesRND.Model(features_count)
        self.optimizer_features_rnd = torch.optim.Adam(self.model_features_rnd.parameters(), lr=config.learning_rate_rnd)

        self.policy_buffer = PolicyBufferIMDual(self.steps, self.state_shape, self.actions_count, self.actors, self.model_ppo.device)
 
        self.states = numpy.zeros((self.actors, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.actors):
            self.states[e] = self.envs.reset(e).copy()

        self.states_running_stats       = RunningStats(self.state_shape, self.states)
        self.int_reward_running_stats   = RunningStats(( ))
 
        self.enable_training()
        self.iterations             = 0 

        self.log_loss_rnd           = 0.0
        self.log_loss_features_rnd  = 0.0
        self.log_loss_forward       = 0.0
        self.log_curiosity_a        = 0.0
        self.log_curiosity_b        = 0.0
        self.log_advantages         = 0.0
        self.log_int_advatages_a    = 0.0
        self.log_int_advatages_b    = 0.0

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

    def main(self):
        #state to tensor
        states_t            = torch.tensor(self.states, dtype=torch.float).detach().to(self.model_ppo.device)

        #compute model output
        features_t = self.model_ppo.forward_features(states_t)

        logits_t, values_ext_t, values_int_a_t, values_int_b_t  = self.model_ppo.forward_heads(features_t)
        
        states_np           = states_t.detach().to("cpu").numpy()
        logits_np           = logits_t.detach().to("cpu").numpy()
        values_ext_np       = values_ext_t.detach().to("cpu").numpy()
        values_int_a_np     = values_int_a_t.detach().to("cpu").numpy()
        values_int_b_np     = values_int_b_t.detach().to("cpu").numpy()

        #collect actions
        actions = self._sample_actions(logits_t)

        #use imagination, to find most curious actions
        curiosity_b_np, _       = self._imagine_future(features_t)
        curiosity_b_np          = numpy.clip(curiosity_b_np, -1.0, 1.0)

        #execute action
        states, rewards, dones, infos = self.envs.step(actions)

        self.states = states.copy()
 
        #update long term states mean and variance
        self.states_running_stats.update(states_np)

        #curiosity motivation
        states_new_t      = torch.tensor(states, dtype=torch.float).detach().to(self.model_ppo.device)
        curiosity_a_np    = self._curiosity(states_new_t)
        curiosity_a_np    = numpy.clip(curiosity_a_np, -1.0, 1.0)

        
        #put into policy buffer
        for e in range(self.actors):            
            if self.enabled_training:
                self.policy_buffer.add(e, states_np[e], logits_np[e], values_ext_np[e], values_int_a_np[e], values_int_b_np[e], actions[e], rewards[e], curiosity_a_np[e], curiosity_b_np[e], dones[e])

                if self.policy_buffer.is_full():
                    self.train()

            if dones[e]:
                self.states[e] = self.envs.reset(e).copy()

        #collect stats
        k = 0.02
        self.log_curiosity_a = (1.0 - k)*self.log_curiosity_a + k*curiosity_a_np.mean()
        self.log_curiosity_b = (1.0 - k)*self.log_curiosity_b + k*curiosity_b_np.mean()

        self.iterations+= 1
        return rewards[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_rnd.save(save_path + "trained/")
        self.model_forward.save(save_path + "trained/")
        self.model_features_rnd.save(save_path + "trained/")

    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
        self.model_rnd.load(load_path + "trained/")
        self.model_forward.load(load_path + "trained/")
        self.model_features_rnd.load(load_path + "trained/")
 
    def get_log(self): 
        result = "" 
        result+= str(round(self.log_loss_rnd, 7)) + " "
        result+= str(round(self.log_loss_features_rnd, 7)) + " "
        result+= str(round(self.log_loss_forward, 7)) + " "
        result+= str(round(self.log_curiosity_a, 7)) + " "
        result+= str(round(self.log_curiosity_b, 7)) + " "
        result+= str(round(self.log_advantages, 7)) + " "
        result+= str(round(self.log_int_advatages_a, 7)) + " "
        result+= str(round(self.log_int_advatages_b, 7)) + " "
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
                states, states_next, logits, actions, returns_ext, returns_int_a, returns_int_b, advantages_ext, advantages_int_a, advantages_int_b = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)

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

                #predict features
                features_t      = self.model_ppo.forward_features(states).detach()
                features_next_t = self.model_ppo.forward_features(states_next).detach()
                
                #train features RND model, MSE loss
                features_predicted_t, features_target_t  = self.model_features_rnd(features_t)

                loss_features_rnd   = (features_target_t - features_predicted_t)**2
                
                random_mask         = torch.rand(loss_features_rnd.shape).to(loss_features_rnd.device)
                random_mask         = 1.0*(random_mask < 1.0/self.training_epochs)
                loss_features_rnd   = (loss_features_rnd*random_mask).sum() / (random_mask.sum() + 0.00000001)

                self.optimizer_features_rnd.zero_grad() 
                loss_features_rnd.backward()
                self.optimizer_features_rnd.step()

                #train forward model, MSE loss
                action_one_hot_t = self._action_one_hot(actions)

                features_next_predicted_t  = self.model_forward(features_t, action_one_hot_t)

                loss_forward = ((features_next_t - features_next_predicted_t)**2.0).mean()
                self.optimizer_forward.zero_grad() 
                loss_forward.backward()
                self.optimizer_forward.step()

                k = 0.02
                self.log_loss_rnd           = (1.0 - k)*self.log_loss_rnd + k*loss_rnd.detach().to("cpu").numpy()
                self.log_loss_features_rnd  = (1.0 - k)*self.log_loss_features_rnd + k*loss_features_rnd.detach().to("cpu").numpy()
                self.log_loss_forward       = (1.0 - k)*self.log_loss_forward + k*loss_forward.detach().to("cpu").numpy()

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
        compute internal critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_int_a_new  = values_int_a_new.squeeze(1)
        loss_int_a_value  = (returns_int_a.detach() - values_int_a_new)**2
        loss_int_a_value  = loss_int_a_value.mean()

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
        self.log_advantages         = (1.0 - k)*self.log_advantages + k*advantages_ext.mean().detach().to("cpu").numpy()
        self.log_int_advatages_a    = (1.0 - k)*self.log_int_advatages_a + k*advantages_int_a.mean().detach().to("cpu").numpy()
        self.log_int_advatages_b    = (1.0 - k)*self.log_int_advatages_b + k*advantages_int_b.mean().detach().to("cpu").numpy()

        return loss 

    def _action_one_hot(self, action_idx_t):
        batch_size = action_idx_t.shape[0]

        action_one_hot_t = torch.zeros((batch_size, self.actions_count))
        action_one_hot_t[range(batch_size), action_idx_t] = 1.0  
        action_one_hot_t = action_one_hot_t.to(self.model_ppo.device)

        return action_one_hot_t

    def _curiosity(self, state_t):
        state_norm_t            = self._norm_state(state_t)

        features_predicted_t, features_target_t  = self.model_rnd(state_norm_t)

        curiosity_t    = (features_target_t - features_predicted_t)**2
        
        curiosity_t    = curiosity_t.sum(dim=1)/2.0
        
        return curiosity_t.detach().to("cpu").numpy()

    def _norm_state(self, state_t):
        mean = torch.from_numpy(self.states_running_stats.mean).to(state_t.device).float()

        state_norm_t = state_t - mean  

        return state_norm_t

    def _imagine_future(self, features_t):

        features_t              = features_t.detach()
        
        batch_size              = features_t.shape[0]

        features_imagined       = torch.zeros((self.rollouts_length + 1, batch_size*self.rollouts_count, features_t.shape[1])).to(features_t.device)
        actions_imagined        = numpy.zeros((self.rollouts_length, batch_size*self.rollouts_count), dtype=int)
        curiosity_imagined      = torch.zeros((self.rollouts_length + 1, batch_size*self.rollouts_count)).to(features_t.device)
        features_imagined[0]    = torch.repeat_interleave(features_t, self.rollouts_count, dim=0)

        for n in range(self.rollouts_length):
            #use policy to select action
            logits, _, _, _ = self.model_ppo.forward_heads(features_imagined[n])

            #sample actions
            actions = self._sample_actions(logits)

            actions_imagined[n] = actions

            actions_one_hot_t = self._action_one_hot(actions)

            #imagine next state
            features_imagined[n+1]  = self.model_forward(features_imagined[n], actions_one_hot_t)   

            #RND on features space
            features_predicted_t, features_target_t  = self.model_features_rnd(features_imagined[n+1])

            #store curiosity
            curiosity_t    = (features_target_t - features_predicted_t)**2
            curiosity_t    = curiosity_t.sum(dim=1)/2.0

            curiosity_imagined[n+1] = curiosity_t


        features_imagined   = features_imagined[1:]
        curiosity_imagined  = curiosity_imagined[1:]

        features_imagined   = features_imagined.reshape(self.rollouts_length,  batch_size, self.rollouts_count, features_t.shape[1])
        actions_imagined    = actions_imagined.reshape(self.rollouts_length, batch_size, self.rollouts_count)
        curiosity_imagined  = curiosity_imagined.reshape(self.rollouts_length, batch_size, self.rollouts_count)

        features_imagined   = torch.transpose(features_imagined, 0, 1)
        curiosity_imagined  = torch.transpose(curiosity_imagined, 0, 1)

        actions_first        = actions_imagined[0]
        curiosity_evaluation = curiosity_imagined.mean(dim=1)
        curiosity_total      = curiosity_imagined.sum(1).sum(1)/(self.rollouts_length*self.rollouts_count)
        curiosity_total      = curiosity_total.detach().to("cpu").numpy()

        best_action_idx     = torch.argmax(curiosity_evaluation, dim=1).detach().to("cpu").numpy()
        best_actions        = actions_first[range(batch_size), best_action_idx]

        '''
        print("features_imagined        = ", features_imagined.shape)
        print("actions_imagined         = ", actions_imagined.shape)
        print("actions_first            = ", actions_first.shape)
        print("curiosity_imagined       = ", curiosity_imagined.shape)
        print("curiosity_evaluation     = ", curiosity_evaluation.shape)
        print("best_action_idx          = ", best_action_idx.shape)
        print("best_actions             = ", best_actions.shape)
        print("\n\n\n")
        '''

        return curiosity_total, best_actions

       