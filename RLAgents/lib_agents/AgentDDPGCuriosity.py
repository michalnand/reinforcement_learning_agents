import numpy
import torch
from .ExperienceBufferContinuous import *


class AgentDDPGCuriosity():
    def __init__(self, env, ModelCritic, ModelActor, ModelForward, ModelForwardTarget, Config):
        self.env = env

        config = Config.Config()

        self.batch_size     = config.batch_size
        self.gamma          = config.gamma
        self.update_frequency = config.update_frequency
        self.tau                =  config.tau
        self.beta           = config.beta

        self.exploration    = config.exploration
    
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.shape[0]

        self.experience_replay = ExperienceBufferContinuous(config.experience_replay_size, self.state_shape, self.actions_count)

        self.model_actor            = ModelActor.Model(self.state_shape, self.actions_count)
        self.model_actor_target     = ModelActor.Model(self.state_shape, self.actions_count)

        self.model_critic           = ModelCritic.Model(self.state_shape, self.actions_count)
        self.model_critic_target    = ModelCritic.Model(self.state_shape, self.actions_count)

        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.model_critic_target.parameters(), self.model_critic.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer_actor    = torch.optim.Adam(self.model_actor.parameters(), lr= config.actor_learning_rate)
        self.optimizer_critic   = torch.optim.Adam(self.model_critic.parameters(), lr= config.critic_learning_rate)


        self.model_forward          = ModelForward.Model(self.state_shape, self.actions_count)
        self.model_forward_target   = ModelForwardTarget.Model(self.state_shape, self.actions_count)
        self.optimizer_forward      = torch.optim.Adam(self.model_forward.parameters(), lr=config.forward_learning_rate)

        self.state              = env.reset()
        self.iterations         = 0

        self.loss_forward             = 0.0
        self.internal_motivation      = 0.0

        self.enable_training()

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
    
    def main(self):
        if self.enabled_training:
            self.exploration.process()
            epsilon = self.exploration.get()
        else:
            epsilon = self.exploration.get_testing()
       
        state_t     = torch.from_numpy(self.state).to(self.model_actor.device).unsqueeze(0).float()
        action_t, action = self._sample_action(state_t, epsilon)
 
        action = action.squeeze()

        state_next, reward, done, self.info = self.env.step(action)

        if self.enabled_training: 
            self.experience_replay.add(self.state, action, reward, done)

        if self.enabled_training and self.iterations > 0.1*self.experience_replay.size:
            if self.iterations%self.update_frequency == 0:
                self.train_model()

        if done:
            self.state = self.env.reset()
        else:
            self.state = state_next.copy()

        self.iterations+= 1

        return reward, done
        
        
    def train_model(self):
        state_t, state_next_t, action_t, reward_t, done_t, _ = self.experience_replay.sample(self.batch_size, self.model_critic.device)

        action_next_t   = self.model_actor_target.forward(state_next_t).detach()
        value_next_t    = self.model_critic_target.forward(state_next_t, action_next_t).detach()

        #curiosity internal motivation
        curiosity_prediction_t      = self._curiosity(state_t, action_t)
        curiosity_t                 = self.beta*torch.tanh(curiosity_prediction_t.detach())

        #train forward model, MSE loss
        loss_forward = curiosity_prediction_t.mean()
        self.optimizer_forward.zero_grad()
        loss_forward.backward()
        self.optimizer_forward.step()

        reward_t = reward_t.unsqueeze(-1)
        done_t   = (1.0 - done_t).unsqueeze(-1)

        #critic loss
        value_target    = reward_t + curiosity_t + self.gamma*done_t*value_next_t
        value_predicted = self.model_critic.forward(state_t, action_t)

        loss_critic     = ((value_target - value_predicted)**2)
        loss_critic     = loss_critic.mean()
     
        #update critic
        self.optimizer_critic.zero_grad()
        loss_critic.backward() 
        self.optimizer_critic.step()

        #actor loss
        loss_actor      = -self.model_critic.forward(state_t, self.model_actor.forward(state_t))
        loss_actor      = loss_actor.mean()

        #update actor
        self.optimizer_actor.zero_grad()       
        loss_actor.backward()
        self.optimizer_actor.step()

        # update target networks 
        for target_param, param in zip(self.model_actor_target.parameters(), self.model_actor.parameters()):
            target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)
       
        for target_param, param in zip(self.model_critic_target.parameters(), self.model_critic.parameters()):
            target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)

        k = 0.02
        self.loss_forward           = (1.0 - k)*self.loss_forward        + k*loss_forward.detach().to("cpu").numpy()
        self.internal_motivation    = (1.0 - k)*self.internal_motivation + k*curiosity_t.mean().detach().to("cpu").numpy()

        #print(">>> ", self.loss_forward, self.internal_motivation)

    def save(self, save_path):
        self.model_critic.save(save_path+"trained/")
        self.model_actor.save(save_path+"trained/")
        self.model_forward.save(save_path+"trained/")
        self.model_forward_target.save(save_path+"trained/")

    def load(self, load_path):
        self.model_critic.load(load_path+"trained/")
        self.model_actor.load(load_path+"trained/")
        self.model_forward.load(load_path+"trained/")
        self.model_forward_target.load(load_path+"trained/")

    
    def get_log(self):
        result = "" 
        result+= str(round(self.loss_forward, 7)) + " "
        result+= str(round(self.internal_motivation, 7)) + " "
        return result
    

    def _sample_action(self, state_t, epsilon):
        action_t    = self.model_actor(state_t)
        action_t    = action_t + epsilon*torch.randn(action_t.shape).to(self.model_actor.device)
        action_t    = action_t.clamp(-1.0, 1.0)

        action_np   = action_t.detach().to("cpu").numpy()

        return action_t, action_np

    def _curiosity(self, state_t, action_t):
        state_next_predicted_t       = self.model_forward(state_t, action_t)
        state_next_predicted_t_t     = self.model_forward_target(state_t, action_t)

        curiosity_t    = (state_next_predicted_t_t.detach() - state_next_predicted_t)**2
        curiosity_t    = curiosity_t.mean(dim=1)

        return curiosity_t