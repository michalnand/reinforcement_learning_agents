import numpy
import torch
from .ExperienceBuffer import *

class AgentDQNEntropy():
    def __init__(self, env, ModelDQN, ModelForward, ModelForwardTarget, ModelAutoencoder, Config):
        self.env    = env
 
        config      = Config.Config()

        self.batch_size         = config.batch_size
        self.exploration        = config.exploration
        self.gamma              = config.gamma
        self.beta1              = config.beta1
        self.beta2              = config.beta2

        
        self.target_update      = config.target_update
        self.update_frequency   = config.update_frequency        
               
        self.state_shape        = self.env.observation_space.shape
        self.actions_count      = self.env.action_space.n

        
        self.experience_replay  = ExperienceBuffer(config.experience_replay_size, self.state_shape, self.actions_count)

        self.model_dqn          = ModelDQN.Model(self.state_shape, self.actions_count)
        self.model_dqn_target   = ModelDQN.Model(self.state_shape, self.actions_count)
        self.optimizer_dqn      = torch.optim.Adam(self.model_dqn.parameters(), lr=config.learning_rate_dqn)

        for target_param, param in zip(self.model_dqn_target.parameters(), self.model_dqn.parameters()):
            target_param.data.copy_(param.data)

        self.model_forward          = ModelForward.Model(self.state_shape, self.actions_count)
        self.model_forward_target   = ModelForwardTarget.Model(self.state_shape, self.actions_count)
        self.optimizer_forward      = torch.optim.Adam(self.model_forward.parameters(), lr=config.learning_rate_forward)
        
        self.model_autoencoder       = ModelAutoencoder.Model(self.state_shape)
        self.optimizer_autoencoder   = torch.optim.Adam(self.model_autoencoder.parameters(), lr= config.learning_rate_autoencoder)

        self.state              = env.reset()

        self.episodic_memory_size   = config.episodic_memory_size 
        self._init_episodic_memory(self.state)

        self.iterations             = 0

        self.loss_forward           = 0.0
        self.curiosity_motivation   = 0.0

        self.loss_autoencoder       = 0.0
        self.entropy_motivation     = 0.0

        self.enable_training()


    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
    
    def main(self, show_activity = False):
        if self.enabled_training:
            self.exploration.process()
            epsilon = self.exploration.get()
        else:
            epsilon = self.exploration.get_testing()
             
        state_t     = torch.from_numpy(self.state).to(self.model_dqn.device).unsqueeze(0).float()
        q_values_t  = self.model_dqn(state_t)
        q_values_np = q_values_t.squeeze(0).detach().to("cpu").numpy()

        action      = self._sample_action(q_values_np, epsilon)

        state_next, self.reward, done, self.info = self.env.step(action)
 
        if self.enabled_training:
            entropy = self._add_episodic_memory(state_t, action)
            self.experience_replay.add(self.state, action, self.reward, done, entropy)

        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train_model()

            if self.iterations%self.target_update == 0:
                self.model_dqn_target.load_state_dict(self.model_dqn.state_dict())

        if done:
            self.state  = self.env.reset()
            self._init_episodic_memory(self.state)
        else:
            self.state = state_next.copy()

        if show_activity:
            self._show_activity(self.state)

        self.iterations+= 1

        return self.reward, done

        
    def train_model(self):
        state_t, state_next_t, action_t, reward_t, done_t, entropy_t = self.experience_replay.sample(self.batch_size, self.model_dqn.device)
 
        #curiosity internal motivation
        action_one_hot_t            = self._action_one_hot(action_t)
        curiosity_prediction_t      = self._curiosity(state_t, action_one_hot_t)
        curiosity_t                 = self.beta1*torch.tanh(curiosity_prediction_t.detach())
       
        #train forward model, MSE loss
        loss_forward = curiosity_prediction_t.mean()
        self.optimizer_forward.zero_grad()
        loss_forward.backward()
        self.optimizer_forward.step()

        #train denoising autoencoder model, MSE loss
        state_noised_t          = state_t + 0.1*torch.randn(state_t.shape).to(state_t.device)
        state_predicted_t, _    = self.model_autoencoder(state_noised_t)
        loss_autoencoder        = (state_t.detach() - state_predicted_t)**2

        loss_autoencoder = loss_autoencoder.mean()
        self.optimizer_autoencoder.zero_grad()
        loss_autoencoder.backward()
        self.optimizer_autoencoder.step()


        #q values, state now, state next
        q_predicted      = self.model_dqn.forward(state_t)
        q_predicted_next = self.model_dqn_target.forward(state_next_t)

        #compute target, n-step Q-learning
        q_target         = q_predicted.clone()
        for j in range(self.batch_size): 
            action_idx              = action_t[j] 
            q_target[j][action_idx] = reward_t[j] + curiosity_t[j] + entropy_t[j] + self.gamma*torch.max(q_predicted_next[j])*(1 - done_t[j])
 
        #train DQN model
        loss_dqn  = (q_target.detach() - q_predicted)**2
        loss_dqn  = loss_dqn.mean() 

        self.optimizer_dqn.zero_grad()
        loss_dqn.backward()
        for param in self.model_dqn.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer_dqn.step()

        k = 0.02
        self.loss_forward           = (1.0 - k)*self.loss_forward           + k*loss_forward.detach().to("cpu").numpy()
        self.curiosity_motivation   = (1.0 - k)*self.curiosity_motivation   + k*curiosity_t.mean().detach().to("cpu").numpy()

        self.loss_autoencoder       = (1.0 - k)*self.loss_autoencoder       + k*loss_autoencoder.detach().to("cpu").numpy()
        self.entropy_motivation     = (1.0 - k)*self.entropy_motivation     + k*entropy_t.mean().detach().to("cpu").numpy()

        # print(self.loss_forward, self.curiosity_motivation, self.loss_autoencoder, self.entropy_motivation)
    
    def save(self, save_path):
        self.model_dqn.save(save_path + "trained/")
        self.model_forward.save(save_path + "trained/")
        self.model_forward_target.save(save_path + "trained/")
        self.model_autoencoder.save(save_path + "trained/")

    def load(self, load_path):
        self.model_dqn.load(load_path + "trained/")
        self.model_forward.load(load_path + "trained/")
        self.model_forward_target.load(load_path + "trained/")
        self.model_autoencoder.load(load_path + "trained/")

    def get_log(self):
        result = "" 
        result+= str(round(self.loss_forward, 7)) + " "
        result+= str(round(self.curiosity_motivation, 7)) + " "
        result+= str(round(self.loss_autoencoder, 7)) + " "
        result+= str(round(self.entropy_motivation, 7)) + " "
        return result

    def _action_one_hot(self, action_idx_t):
        batch_size = action_idx_t.shape[0]

        action_one_hot_t = torch.zeros((batch_size, self.actions_count))
        action_one_hot_t[range(batch_size), action_idx_t] = 1.0  
        action_one_hot_t = action_one_hot_t.to(self.model_dqn.device)

        return action_one_hot_t

    def _sample_action(self, q_values, epsilon):
        if numpy.random.rand() < epsilon:
            action_idx = numpy.random.randint(self.actions_count)
        else:
            action_idx = numpy.argmax(q_values)

        return action_idx

    def _curiosity(self, state_t, action_one_hot_t):
        state_next_predicted_t       = self.model_forward(state_t, action_one_hot_t)
        state_next_predicted_t_t     = self.model_forward_target(state_t, action_one_hot_t)

        curiosity_t    = (state_next_predicted_t_t.detach() - state_next_predicted_t)**2
        curiosity_t    = curiosity_t.mean(dim=1)

        return curiosity_t

    def _init_episodic_memory(self, state):   
        state_t     = torch.from_numpy(state).to(self.model_autoencoder.device).unsqueeze(0).float()
        features_t  = self.model_autoencoder.eval_features(state_t)
        features_np = features_t.squeeze(0).detach().to("cpu").numpy()
        
        features_t  = self.model_autoencoder.eval_features(state_t)
        features_t  = features_t.view(features_t.size(0), -1)

        features_np = features_t.detach().to("cpu").numpy()

        self.episodic_memory_features  = numpy.tile(features_np, (self.episodic_memory_size, 1))
        self.episodic_memory_actions   = numpy.zeros((self.episodic_memory_size, self.actions_count))

        for i in range(self.episodic_memory_size):
            action_idx = numpy.random.randint(self.actions_count)
            self.episodic_memory_actions[i][action_idx] = 1.0
        
        self.motivation_long_term = 0.0

    def _add_episodic_memory(self, state_t, action):
        action_one_hot          = numpy.zeros(self.actions_count)
        action_one_hot[action]  = 1.0

        features_t  = self.model_autoencoder.eval_features(state_t)

        features_t  = features_t.view(features_t.size(0), -1)
        features_np = features_t.detach().to("cpu").numpy()

        #put current features and action into episodic memory, on random place
        idx = numpy.random.randint(self.episodic_memory_size)
        self.episodic_memory_features[idx]  = features_np.copy()
        self.episodic_memory_actions[idx]   = action_one_hot.copy()

        #compute relative entropy
        #the higher features variance s.t. low actions variance results to high relative entropy
        episodic_memory_features_std    = self.episodic_memory_features.std(axis=0).mean()        
        episodic_memory_actions_std     = self.episodic_memory_actions.std(axis=0).mean()

        ratio                          = episodic_memory_features_std/(0.01 + episodic_memory_actions_std)
        motivation                     = self.beta2*numpy.tanh(ratio)

        k = 0.05 
        self.motivation_long_term = (1.0 - k)*self.motivation_long_term + k*motivation

        return motivation - self.motivation_long_term

