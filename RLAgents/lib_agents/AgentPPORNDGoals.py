import numpy
import torch 
from .PolicyBufferIMDual    import *  
from .RunningStats          import *  
from .GoalsBuffer           import *

import cv2

class AgentPPORNDGoals():    
    def __init__(self, envs, ModelPPO, ModelRND, config):
        self.envs = envs  
    
        self.gamma_ext          = config.gamma_ext 
        self.gamma_int_a        = config.gamma_int_a
        self.gamma_int_b        = config.gamma_int_b
            
        self.ext_adv_coeff      = config.ext_adv_coeff
        self.int_a_adv_coeff    = config.int_a_adv_coeff
        self.int_b_adv_coeff    = config.int_b_adv_coeff
    
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip 
    
        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count 


        self.normalise_state_std = config.normalise_state_std
        self.normalise_im_std    = config.normalise_im_std

        self.goals_reactivate    = config.goals_reactivate

        state_shape         = self.envs.observation_space.shape
        self.state_shape    = (state_shape[0] + 2, ) + state_shape[1:]

        self.goal_shape     = (1, ) + state_shape[1:]
        self.actions_count  = self.envs.action_space.n

        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)
 
        self.model_rnd      = ModelRND.Model(self.state_shape)
        self.optimizer_rnd  = torch.optim.Adam(self.model_rnd.parameters(), lr=config.learning_rate_rnd)
 
        self.policy_buffer  = PolicyBufferIMDual(self.steps, self.state_shape, self.actions_count, self.envs_count, self.model_ppo.device, True)
        self.goals_buffer   = GoalsBuffer(self.envs_count, config.goals_count, config.goals_add_threshold, config.goals_reach_threshold, config.goals_downsample, state_shape)
        
        self.episode_score_sum          = numpy.zeros(self.envs_count)

        self.log_reached_goals_episode  = numpy.zeros(self.envs_count)
        self.episode_goals_reached      = numpy.zeros(self.envs_count)

        for e in range(self.envs_count):
            self.envs.reset(e)
        
        self.states_running_stats       = RunningStats(self.state_shape)
        self.rewards_int_running_stats  = RunningStats((1, ))

        self._init_running_stats()
        
        #reset envs and fill initial state
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            state = self.envs.reset(e) 
            self.states[e][0:state.shape[0]] = state.copy()
 
 
        self.enable_training()
        self.iterations                     = 0 

        self.log_loss_rnd                   = 0.0
        self.log_loss_actor                 = 0.0
        self.log_loss_critic                = 0.0

        self.log_internal_motivation_a_mean = 0.0
        self.log_internal_motivation_a_std  = 0.0
        
        self.log_internal_motivation_b_mean = 0.0
        self.log_internal_motivation_b_std  = 0.0



    def enable_training(self):
        self.enabled_training = True
 
    def disable_training(self):
        self.enabled_training = False

    def main(self): 
        #state to tensor
        states_t = torch.tensor(self.states, dtype=torch.float).detach().to(self.model_ppo.device)

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
        states, rewards_ext, dones, infos = self.envs.step(actions)

        #acumulate score per episode
        episode_score_sum_old = self.episode_score_sum.copy()
        self.episode_score_sum+= rewards_ext

        #curiosity motivation
        rewards_int_a       = self._curiosity(states_t)
        self.rewards_int_running_stats.update(rewards_int_a)

        #normalise RND internal motivation
        if self.normalise_im_std:
            rewards_int_a   = rewards_int_a/self.rewards_int_running_stats.std
            
        rewards_int_a = numpy.clip(rewards_int_a, 0.0, 1.0)


        #goal motivation - state transfer reached
        rewards_int_b, goals, _ = self.goals_buffer.step(self.states)  
        rewards_int_b = numpy.clip(rewards_int_b, 0.0, 1.0)

        self.episode_goals_reached+= (rewards_int_b > 0.9)

        #update long term states mean and variance
        self.states_running_stats.update(states_np)

        #add score information to state
        #value range from <0, 1>
        #progress period is 32score points
        score_progress = (1.0 - numpy.cos(2.0*numpy.pi*self.episode_score_sum/32.0))/2.0

        #tile score information to state
        score_progress = numpy.expand_dims(score_progress, axis=(1, 2, 3))
        score_progress = numpy.tile(score_progress, (1, 1, self.state_shape[1], self.state_shape[2]))

        #create new state
        self.states = numpy.concatenate([states, goals, score_progress], axis=1)
      
        #put into policy buffer
        if self.enabled_training:
            self.policy_buffer.add(states_np, logits_np, values_ext_np, values_int_a_np, values_int_b_np, actions, rewards_ext, rewards_int_a, rewards_int_b, dones)
            if self.policy_buffer.is_full():
                self.train()

        for e in range(self.envs_count): 
           
            if dones[e]:
                s       = self.envs.reset(e)
                zero    = numpy.zeros(self.goal_shape)

                self.states[e] = numpy.concatenate([s, zero, zero], axis=0)

                self.episode_score_sum[e] = 0.0

                self.goals_buffer.activate_goals(e)

                #log for counting goals reached per episode
                self.log_reached_goals_episode[e]   = self.episode_goals_reached[e]
                self.episode_goals_reached[e]       = 0.0


        score_int = numpy.floor(self.episode_score_sum).astype(int)
        for e in range(self.envs_count):
            #only when score changed
            if episode_score_sum_old[e] != self.episode_score_sum[e]:
                if score_int[e]%self.goals_reactivate == self.goals_reactivate-1:
                    self.goals_buffer.activate_goals(e)            
        

        #collect stats
        k = 0.02
        self.log_internal_motivation_a_mean   = (1.0 - k)*self.log_internal_motivation_a_mean + k*rewards_int_a.mean()
        self.log_internal_motivation_a_std    = (1.0 - k)*self.log_internal_motivation_a_std  + k*rewards_int_a.std()
        self.log_internal_motivation_b_mean   = (1.0 - k)*self.log_internal_motivation_b_mean + k*rewards_int_b.mean()
        self.log_internal_motivation_b_std    = (1.0 - k)*self.log_internal_motivation_b_std  + k*rewards_int_b.std()
        

        self.iterations+= 1
        return rewards_ext[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_rnd.save(save_path + "trained/")
        self.goals_buffer.save(save_path + "trained/")

    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
        self.model_rnd.load(load_path + "trained/")
        self.goals_buffer.load(load_path + "trained/")

    def get_log(self):  
        result = "" 

        result+= str(round(self.log_loss_rnd, 7)) + " "
        result+= str(round(self.log_loss_actor, 7)) + " "
        result+= str(round(self.log_loss_critic, 7)) + " "

        result+= str(round(self.log_internal_motivation_a_mean, 7)) + " "
        result+= str(round(self.log_internal_motivation_a_std, 7)) + " "

        result+= str(round(self.log_internal_motivation_b_mean, 7)) + " "
        result+= str(round(self.log_internal_motivation_b_std, 7)) + " "

        result+= str(round(self.goals_buffer.log_used_goals, 7)) + " "
        result+= str(round(self.log_reached_goals_episode.mean(), 7)) + " "
        
        return result 

    
    def render(self, env_id):
        size    = 256
        state   = self.states[env_id]


        goals, _   = self.goals_buffer.get_goals_for_render()

        goal_height = goals.shape[2]
        goal_width  = goals.shape[3]

        count       = goals.shape[0]
        grid_size   = int(count**0.5) 

        goals_result  = numpy.zeros((grid_size*goals.shape[2], grid_size*goals.shape[3]))


        for y in range(grid_size):
            for x in range(grid_size):
                y_ = y*goal_height
                x_ = x*goal_width
                goals_result[y_:y_+goal_height, x_:x_+goal_width]   = goals[y*grid_size + x][0]

        goals_result   = cv2.resize(goals_result, (size, size))

        result_im   = numpy.concatenate([state[0], state[3], state[4]], axis=1)
        result_im   = cv2.resize(result_im, (3*size, size)) 
        result_im   = numpy.concatenate([result_im, goals_result], axis=1)

        cv2.putText(result_im, "observation", (10 + 0*size, size - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.putText(result_im, "goal",  (10 + 1*size, size - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.putText(result_im, "progress", (10 + 2*size, size - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.putText(result_im, "goals buffer", (10 + 3*size, size - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

        cv2.imshow("RND goals agent", result_im)
        cv2.waitKey(1)
    
        



    def _sample_actions(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().to("cpu").numpy()
        return actions
    
    def train(self): 
        self.policy_buffer.compute_returns(self.gamma_ext, self.gamma_int_a, self.gamma_int_b)

        batch_count = self.steps//self.batch_size

        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                states, _, logits, actions, returns_ext, returns_int_a, returns_int_b, advantages_ext, advantages_int_a, advantages_int_b = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)

                #train PPO model
                loss_ppo = self._compute_loss_ppo(states, logits, actions, returns_ext, returns_int_a, returns_int_b, advantages_ext, advantages_int_a, advantages_int_b)

                self.optimizer_ppo.zero_grad()        
                loss_ppo.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()

                #train RND model, MSE loss
                loss_rnd = self._compute_loss_rnd(states)

                self.optimizer_rnd.zero_grad() 
                loss_rnd.backward()
                self.optimizer_rnd.step()

                k = 0.02
                self.log_loss_rnd  = (1.0 - k)*self.log_loss_rnd + k*loss_rnd.detach().to("cpu").numpy()

        self.policy_buffer.clear() 

    
    def _compute_loss_ppo(self, states, logits, actions, returns_ext, returns_int_a, returns_int_b, advantages_ext, advantages_int_a, advantages_int_b):
        logits_new, values_ext_new, values_int_a_new, values_int_b_new  = self.model_ppo.forward(states)

        #critic loss
        loss_critic = self._compute_critic_loss(values_ext_new, returns_ext, values_int_a_new, returns_int_a, values_int_b_new, returns_int_b)

        #actor loss        
        advantages  = self.ext_adv_coeff*advantages_ext + self.int_a_adv_coeff*advantages_int_a + self.int_b_adv_coeff*advantages_int_b
        advantages  = advantages.detach() 
        loss_policy, loss_entropy  = self._compute_actor_loss(logits, logits_new, advantages, actions)

        loss_actor = loss_policy + loss_entropy
        
        #total loss
        loss = 0.5*loss_critic + loss_actor

        #store to log
        k = 0.02
        self.log_loss_actor     = (1.0 - k)*self.log_loss_actor  + k*loss_actor.mean().detach().to("cpu").numpy()
        self.log_loss_critic    = (1.0 - k)*self.log_loss_critic + k*loss_critic.mean().detach().to("cpu").numpy()

        return loss 

    #MSE critic loss
    def _compute_critic_loss(self, values_ext_new, returns_ext, values_int_a_new, returns_int_a, values_int_b_new, returns_int_b):
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

        loss_rnd        = (features_target_t - features_predicted_t)**2
        
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

        return curiosity_t.detach().to("cpu").numpy()

    #normalise mean and std for state
    def _norm_state(self, state_t):
        mean = torch.from_numpy(self.states_running_stats.mean).to(state_t.device).float()
        std  = torch.from_numpy(self.states_running_stats.std).to(state_t.device).float()
        
        state_norm_t = state_t - mean

        if self.normalise_state_std:
            state_norm_t = torch.clamp(state_norm_t/std, -5.0, 5.0)

        return state_norm_t 

    #random policy for stats init
    def _init_running_stats(self, steps = 256):
        for _ in range(steps):
            #random action
            actions = numpy.random.randint(0, self.actions_count, (self.envs_count))
            states, _, dones, _ = self.envs.step(actions)

            zeros       = numpy.zeros((self.envs_count, ) + self.goal_shape)
            states_     = numpy.concatenate([states, zeros, zeros], axis=1)

            #update stats
            self.states_running_stats.update(states_)

            for e in range(self.envs_count): 
                if dones[e]:
                    self.envs.reset(e)
    
    