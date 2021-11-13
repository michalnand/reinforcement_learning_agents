import numpy
import torch
from .ExperienceBuffer import *


class AgentDQNPolicy():
    def __init__(self, env, Model, config):
        self.env = env
 
        self.gamma              = config.gamma

        self.update_frequency   = config.update_frequency        
        self.batch_size         = config.batch_size
        
        self.entropy_beta       = config.entropy_beta
        self.tau                = config.tau

                
        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.n

        self.experience_replay = ExperienceBuffer(config.experience_replay_size, self.state_shape, self.actions_count)

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.model_target   = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
            target_param.data.copy_(param.data)


        self.state    = env.reset()

        self.iterations     = 0

        self.log_loss_actor                 = 0.0
        self.log_loss_critic                 = 0.0

        self.enable_training()

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

    def get_log(self):
        result = "" 
        result+= str(round(self.log_loss_actor, 7)) + " "
        result+= str(round(self.log_loss_critic, 7)) + " "
        return result
    
    def main(self):     
        state_t                 = torch.from_numpy(self.state).to(self.model.device).unsqueeze(0).float()
        logits_t, q_values_t    = self.model(state_t)

        action          = self._sample_action(q_values_t.detach().to("cpu").numpy()[0])
        #action          = self._sample_actions(logits_t)[0]
        state_new, reward, done, info = self.env.step(action)
 
        if self.enabled_training:
            self.experience_replay.add(self.state, action, reward, done)

        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train()

            #smooth update target model 
            for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)
       

        if done:
            self.state = self.env.reset()
        else:
            self.state = state_new.copy()

      
        self.iterations+= 1

        if self.iterations%10000 == 0:
            print("logits_t = ", logits_t, "\n\n")

        return reward, done, info
        
    def train(self):
        state_t, state_next_t, actions_t, rewards_t, dones_t, _ = self.experience_replay.sample(self.batch_size, self.model.device)

        #q values, state now, state next
        logits,         q_values      = self.model.forward(state_t)
        logits_next,    q_values_next = self.model_target.forward(state_next_t)

        probs       = torch.nn.functional.softmax(logits, dim = 1)
        log_probs   = torch.nn.functional.log_softmax(logits, dim = 1)

        probs_next  = torch.nn.functional.softmax(logits_next, dim = 1)

        #V(s(n+1)) = Ea∼pi Q(s(n+1), a)
        state_value_next = (probs_next*q_values_next).sum(dim=1)

        q_target    = rewards_t + self.gamma*state_value_next*(1.0 - dones_t)

        delta       = q_target.detach() - q_values[range(self.batch_size), actions_t]


        #critic, MSE loss
        loss_critic = (delta**2).mean()

        #actor, policy gradient loss
        loss_actor  = -delta.detach()*log_probs[range(self.batch_size), actions_t]
        loss_actor  = loss_actor.mean()

        #entropy regularisation loss
        loss_entropy    = (probs*log_probs).sum(dim = 1)
        loss_entropy    = self.entropy_beta*loss_entropy.mean()


        loss = loss_critic # + loss_actor + loss_entropy


        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step() 

        k = 0.02
        self.log_loss_actor  = (1.0 - k)*self.log_loss_actor + k*loss_actor.mean().detach().to("cpu").numpy()
        self.log_loss_critic = (1.0 - k)*self.log_loss_critic + k*loss_critic.mean().detach().to("cpu").numpy()
      

   


    def save(self, save_path):
        self.model.save(save_path + "trained/")

    def load(self, load_path):
        self.model.load(load_path + "trained/")
    

    
    def _sample_actions(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().to("cpu").numpy()
        return actions
    

    def _sample_action(self, q_values, epsilon = 0.1):
        if numpy.random.rand() < epsilon:
            action_idx = numpy.random.randint(self.actions_count)
        else:
            action_idx = numpy.argmax(q_values)

        return action_idx
    