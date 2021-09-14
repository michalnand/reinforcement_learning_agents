import numpy
import torch
import time

from torch.distributions import Categorical
 
from .PolicyBufferIMDual    import *  
from .GoalsBuffer           import * 
from .RunningStats          import * 


class GoalAchievedCounter:
    def __init__(self, envs_count, k = 0.1):

        self.goal_mode      = numpy.zeros(envs_count, dtype=bool)
        self.goal_achieved  = numpy.zeros(envs_count, dtype=bool)

        #smoothing factor
        self.k      = k

        #result in [%]
        self.result = 0.0

        #average goal ID
        self.goal_id = 0.0

    def add(self, goal_reward, dones):
        #set flag if goal achieved
        self.goal_achieved = numpy.logical_or(self.goal_achieved, numpy.logical_and(self.goal_mode, goal_reward > numpy.zeros_like(goal_reward)))

        #process results on episode end
        for e in range(len(self.goal_mode)):
            if dones[e]:
                #achieved goals increase score
                if self.goal_achieved[e]:
                    self.result = (1.0 - self.k)*self.result + self.k*100.0

                #not achieved goals decrease score
                elif self.goal_mode[e]:
                    self.result = (1.0 - self.k)*self.result + self.k*0.0
        
    def set_goal_mode(self, env_idx, mode, goal_id = -1):
        self.goal_mode[env_idx]     = mode
        self.goal_achieved[env_idx] = False

        if goal_id != -1:
            self.goal_id = (1.0 - self.k)*self.goal_id + self.k*goal_id

        


    
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


        self.goals_buffer = GoalsBuffer(config.goals_buffer_size, config.goals_add_threshold, config.goals_downsampling, config.goals_weights, self.state_shape, self.envs_count, self.model_ppo.device)

        #initial state
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e] = self.envs.reset(e).copy()

        #init moving average for RND
        self.states_running_stats       = RunningStats(self.state_shape, self.states)

        #initial all agents into explore mode 
        self.agent_mode                 = torch.zeros((self.envs_count, )).to(self.model_ppo.device)

        self.episode_rewards_sum        = numpy.zeros((self.envs_count, ))

        #stats for goal achieved
        self.goal_echieved_stats  = GoalAchievedCounter(self.envs_count)
       

        self.enable_training()
        self.iterations           = 0 

        self.log_loss_rnd         = 0.0

        self.log_ext_a            = 0.0
        self.log_ext_b            = 0.0

        self.log_int_a            = 0.0
        self.log_int_b            = 0.0

        self.log_advantages_ext_a = 0.0
        self.log_advantages_int_a = 0.0
        self.log_advantages_ext_b = 0.0
        self.log_advantages_int_b = 0.0

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
        curiosity_np    = self._curiosity(states_t)
        curiosity_np    = numpy.clip(curiosity_np, -1.0, 1.0)
        
        goals_np        = goals_t.detach().to("cpu").numpy()
        mode_np         = self.agent_mode.detach().to("cpu").numpy()

        self.episode_rewards_sum+= rewards

        self.goals_buffer.add(self.episode_rewards_sum)
        #self.goals_buffer.visualise(states_t[1], goals_t[1], goals_reward_ext[1], goals_reward_int[1])
         
        #put into policy buffer
        if self.enabled_training:
            self.policy_buffer.add(states_np, goals_np, mode_np)
            self.policy_buffer.add_a(logits_a_np, values_ext_a_np, values_int_a_np, actions, rewards, curiosity_np, dones)
            self.policy_buffer.add_b(logits_b_np, values_ext_b_np, values_int_b_np, actions, goals_reward_ext, goals_reward_int, dones)
      
            if self.policy_buffer.is_full():
                self.train()

        #log achieved goal stats
        self.goal_echieved_stats.add(goals_reward_ext, dones)

        for e in range(self.envs_count): 
            #switch agent to explore mode, if goal achieved
            if goals_reward_ext[e] > 0:
                self.agent_mode[e] = 0.0

            #episode done
            if dones[e]:
                self.states[e] = self.envs.reset(e).copy()
                self.episode_rewards_sum[e] = 0.0

                #switch agent with 50% prob to exploit mode, except env 0
                if e != 0 and numpy.random.rand() > 0.5:
                    goal_id = self.goals_buffer.new_goal(e)
                    self.agent_mode[e]      = 1.0
                    self.goal_echieved_stats.set_goal_mode(e, True, goal_id)
                else:
                    self.goals_buffer.zero_goal(e)
                    self.agent_mode[e]      = 0.0
                    self.goal_echieved_stats.set_goal_mode(e, False)

        #collect stats
        k = 0.02
        self.log_ext_a = (1.0 - k)*self.log_ext_a + k*rewards.mean()
        self.log_ext_b = (1.0 - k)*self.log_ext_b + k*goals_reward_ext.mean()

        self.log_int_a = (1.0 - k)*self.log_int_a + k*curiosity_np.mean()
        self.log_int_b = (1.0 - k)*self.log_int_b + k*goals_reward_int.mean()


        self.iterations+= 1
        return rewards[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_rnd.save(save_path + "trained/")
        self.goals_buffer.save(save_path)

    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
        self.model_rnd.load(load_path + "trained/")

    def get_log(self): 
        result = "" 
        result+= str(round(self.log_loss_rnd, 7)) + " "
        
        result+= str(round(self.log_ext_a, 7)) + " "
        result+= str(round(self.log_int_a, 7)) + " "

        result+= str(round(self.log_ext_b, 7)) + " "
        result+= str(round(self.log_int_b, 7)) + " "

        result+= str(round(self.log_advantages_ext_a, 7)) + " "
        result+= str(round(self.log_advantages_int_a, 7)) + " "
        result+= str(round(self.log_advantages_ext_b, 7)) + " "
        result+= str(round(self.log_advantages_int_b, 7)) + " "

        result+= str(round(self.goals_buffer.total_goals, 7)) + " "
        result+= str(round(self.goal_echieved_stats.result, 7)) + " "
        result+= str(round(self.goal_echieved_stats.goal_id, 7)) + " "

        return result 
    

    def _sample_actions_from_logits(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().to("cpu").numpy()
        return actions

    def _sample_actions(self, logits_a, logits_b, mode):
        mode_  = mode.unsqueeze(1)
        logits = (1.0 - mode_)*logits_a + mode_*logits_b
        
        return self._sample_actions_from_logits(logits)
        
    def train(self): 
        self.policy_buffer.compute_returns(self.gamma_ext, self.gamma_int)

        batch_count = self.steps//self.batch_size

        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                states, goals, modes, res_a, res_b = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)

                #train PPO model
                loss = self._compute_loss(states, goals, modes, res_a, res_b)

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

    
    def _compute_loss(self, states, goals, modes, res_a, res_b):
        logits_a, actions_a, returns_ext_a, returns_int_a, advantages_ext_a, advantages_int_a = res_a
        logits_b, actions_b, returns_ext_b, returns_int_b, advantages_ext_b, advantages_int_b = res_b

        log_probs_a_old = torch.nn.functional.log_softmax(logits_a, dim = 1).detach()
        log_probs_b_old = torch.nn.functional.log_softmax(logits_b, dim = 1).detach()

        logits_a_new, logits_b_new, values_ext_a_new, values_int_a_new, values_ext_b_new, values_int_b_new  = self.model_ppo.forward(states, goals, modes)

        probs_a_new     = torch.nn.functional.softmax(logits_a_new, dim = 1)
        log_probs_a_new = torch.nn.functional.log_softmax(logits_a_new, dim = 1)

        probs_b_new     = torch.nn.functional.softmax(logits_b_new, dim = 1)
        log_probs_b_new = torch.nn.functional.log_softmax(logits_b_new, dim = 1)

 
        #compute critic A loss, as MSE
        loss_critic_a = self._critic_loss(returns_ext_a, values_ext_a_new, returns_int_a, values_int_a_new, 1.0 - modes)

        #compute critic B loss, as MSE
        loss_critic_b = self._critic_loss(returns_ext_b, values_ext_b_new, returns_int_b, values_int_b_new, modes)

        #sum to single critic loss
        loss_critic     = loss_critic_a + loss_critic_b
 
        #compute actors loss
        loss_actor_a = self._actor_loss(self.ext_adv_coeff*advantages_ext_a + self.int_adv_coeff*advantages_int_a, probs_a_new, log_probs_a_new, log_probs_a_old, actions_a, 1.0 - modes)
        loss_actor_b = self._actor_loss(self.ext_adv_coeff*advantages_ext_b + self.int_adv_coeff*advantages_int_b, probs_b_new, log_probs_b_new, log_probs_b_old, actions_b, modes)
        
        #sum to single actor loss
        loss_actor     = loss_actor_a + loss_actor_b
 

        loss = 0.5*loss_critic + loss_actor


        k = 0.02
        self.log_advantages_ext_a   = (1.0 - k)*self.log_advantages_ext_a + k*advantages_ext_a.mean().detach().to("cpu").numpy()
        self.log_advantages_int_a   = (1.0 - k)*self.log_advantages_int_a + k*advantages_int_a.mean().detach().to("cpu").numpy()
        self.log_advantages_ext_b   = (1.0 - k)*self.log_advantages_ext_b + k*advantages_ext_b.mean().detach().to("cpu").numpy()
        self.log_advantages_int_b   = (1.0 - k)*self.log_advantages_int_b + k*advantages_int_b.mean().detach().to("cpu").numpy()
        
        return loss 

    def _critic_loss(self, returns_ext, values_ext, returns_int, values_int, mask):
        eps = 0.000001

        #compute external critic loss, as MSE
        loss_ext_value    = ((returns_ext.detach() - values_ext.squeeze(1))**2)*mask
        loss_ext_value    = loss_ext_value.sum()/(mask.sum() + eps)

        #compute internal critic loss, as MSE
        loss_int_value    = ((returns_int.detach() - values_int.squeeze(1))**2)*mask
        loss_int_value    = loss_int_value.sum()/(mask.sum() + eps)

        return loss_ext_value + loss_int_value

    def _actor_loss(self, advantages, probs_new, log_probs_new, log_probs_old, actions, mask):
        eps = 0.000001
        
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
        loss_policy = loss_policy.sum()/(mask.sum() + eps)
    
        ''' 
        compute entropy loss, to avoid greedy strategy
        L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        '''
        loss_entropy = (probs_new*log_probs_new).sum(dim = 1)
        loss_entropy = self.entropy_beta*loss_entropy*mask
        loss_entropy = loss_entropy.sum()/(mask.sum() + eps)

        return loss_policy + loss_entropy

    def _curiosity(self, state_t):
        state_norm_t = self._norm_state(state_t)

        features_predicted_t, features_target_t  = self.model_rnd(state_norm_t)

        curiosity_t    = (features_target_t - features_predicted_t)**2
        
        curiosity_t    = curiosity_t.sum(dim=1)/2.0
        
        return curiosity_t.detach().to("cpu").numpy()

    def _norm_state(self, state_t):
        mean = torch.from_numpy(self.states_running_stats.mean).to(state_t.device).float()

        state_norm_t = state_t - mean 

        return state_norm_t
