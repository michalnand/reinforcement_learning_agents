import numpy
import torch

from .ExperienceBuffer import *
from .RunningStats     import * 


class AgentDQNRND():
    def __init__(self, env, ModelDQN, ModelRND, config):
        self.env = env
 
        self.batch_size         = config.batch_size
        self.exploration        = config.exploration
        self.gamma              = config.gamma

        self.ext_reward_coeff      = config.ext_reward_coeff
        self.int_reward_coeff      = config.int_reward_coeff

        self.target_update      = config.target_update
        self.update_frequency   = config.update_frequency        
               
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.n

        self.experience_replay = ExperienceBuffer(config.experience_replay_size, self.state_shape, self.actions_count, True)

        self.model_dqn          = ModelDQN.Model(self.state_shape, self.actions_count)
        self.model_dqn_target   = ModelDQN.Model(self.state_shape, self.actions_count)
        self.optimizer_dqn  = torch.optim.Adam(self.model_dqn.parameters(), lr= config.learning_rate_dqn)

        for target_param, param in zip(self.model_dqn_target.parameters(), self.model_dqn.parameters()):
            target_param.data.copy_(param.data)

        self.model_rnd      = ModelRND.Model(self.state_shape)
        self.optimizer_rnd  = torch.optim.Adam(self.model_rnd.parameters(), lr= config.learning_rate_rnd)


        self.state    = env.reset()

        #init moving average for RND
        self.states_running_stats  = RunningStats(self.state_shape, numpy.expand_dims(self.state, axis=0))


        self.iterations     = 0

        self.log_loss_dqn = 0.0
        self.log_loss_rnd = 0.0
        self.log_reward_int = 0.0
        self.log_reward_ext = 0.0

        self.enable_training()

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

    def get_log(self): 
        result = "" 
        result+= str(round(self.log_loss_dqn, 7)) + " "
        result+= str(round(self.log_loss_rnd, 7)) + " "

        result+= str(round(self.log_reward_int, 7)) + " "
        result+= str(round(self.log_reward_ext, 7)) + " "

        return result 
    
    
    def main(self):
        if self.enabled_training:
            self.exploration.process()
            epsilon = self.exploration.get()
        else:
            epsilon = self.exploration.get_testing()
             
        state_t     = torch.from_numpy(self.state).to(self.model_dqn.device).unsqueeze(0).float()
        q_values_t  = self.model_dqn(state_t)
        q_values_np = q_values_t.squeeze(0).detach().to("cpu").numpy()

        action      = self._sample_action(q_values_np, epsilon)

        state_new, reward, done, info = self.env.step(action)

        reward_int    = self._curiosity(state_t)
        reward_int    = numpy.clip(reward_int, -1.0, 1.0)[0]
 
        if self.enabled_training:
            self.experience_replay.add(self.state, action, reward, done, reward_int)

        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train()

            if self.iterations%self.target_update == 0:
                self.model_dqn_target.load_state_dict(self.model_dqn.state_dict())

        if done:
            self.state = self.env.reset()
        else:
            self.state = state_new.copy()

      
        self.iterations+= 1

        k = 0.02
        self.log_reward_int = (1.0 - k)*self.log_reward_int + k*reward_int
        self.log_reward_ext = (1.0 - k)*self.log_reward_ext + k*reward


        return reward, done, info
        
    def train(self):
        state_t, state_next_t, actions_t, rewards_ext_t, dones_t, rewards_int_t = self.experience_replay.sample(self.batch_size, self.model_dqn.device)

        #q values, state now, state next
        q_predicted      = self.model_dqn.forward(state_t)
        q_predicted_next = self.model_dqn_target.forward(state_next_t)

        

        #q-learning
        q_target    = q_predicted.clone()

        q_max, _    = torch.max(q_predicted_next, axis=1)
        rewards_t = self.ext_reward_coeff*rewards_ext_t +  self.int_reward_coeff*rewards_int_t
        q_new       = rewards_t + self.gamma*(1.0 - dones_t)*q_max
        q_target[range(self.batch_size), actions_t.type(torch.long)] = q_new

        #train DQN model, MSE loss
        loss_dqn  = ((q_target.detach() - q_predicted)**2)
        loss_dqn  = loss_dqn.mean() 

        self.optimizer_dqn.zero_grad()
        loss_dqn.backward()
        for param in self.model_dqn.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer_dqn.step()



        #train RND model, MSE loss
        state_norm_t    = self._norm_state(state_t).detach()

        features_predicted_t, features_target_t  = self.model_rnd(state_norm_t)

        loss_rnd        = (features_target_t - features_predicted_t)**2

        
        #andom 75% regularisation mask and mask with explore mode only
        random_mask     = torch.rand(loss_rnd.shape).to(loss_rnd.device)
        random_mask     = 1.0*(random_mask < 0.25) #*(1.0 - modes.unsqueeze(1))
        loss_rnd        = (loss_rnd*random_mask).sum() / (random_mask.sum() + 0.00000001)

        self.optimizer_rnd.zero_grad() 
        loss_rnd.backward()
        self.optimizer_rnd.step()

        #add logs
        k = 0.02
        self.log_loss_dqn  = (1.0 - k)*self.log_loss_dqn + k*loss_dqn.detach().to("cpu").numpy()
        self.log_loss_rnd  = (1.0 - k)*self.log_loss_rnd + k*loss_rnd.detach().to("cpu").numpy()


    def save(self, save_path):
        self.model_dqn.save(save_path + "trained/")

    def load(self, load_path):
        self.model_dqn.load(load_path + "trained/")
    
    def _sample_action(self, q_values, epsilon):
        if numpy.random.rand() < epsilon:
            action_idx = numpy.random.randint(self.actions_count)
        else:
            action_idx = numpy.argmax(q_values)

        return action_idx


    
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

