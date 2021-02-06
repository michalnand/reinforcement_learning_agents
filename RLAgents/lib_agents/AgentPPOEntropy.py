import numpy
import torch
import time

from torch.distributions import Categorical

from .PolicyBuffer import *

class AgentPPOEntropy():
    def __init__(self, envs, ModelPPO, ModelForward, ModelForwardTarget, ModelAutoencoder, Config):
        self.envs = envs

        config = Config.Config()

        self.gamma              = config.gamma
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip

        if hasattr(config, "critic_loss_proportion"):
            self.critic_loss_proportion = config.critic_loss_proportion
        else:
            self.critic_loss_proportion = 1.0

        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.actors             = config.actors

        self.beta1              = config.beta1
        self.beta2              = config.beta2


        self.state_shape    = self.envs[0].observation_space.shape
        self.actions_count  = self.envs[0].action_space.n

        self.model_ppo          = ModelPPO.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo      = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)
 
        self.model_forward          = ModelForward.Model(self.state_shape, self.actions_count)
        self.optimizer_forward      = torch.optim.Adam(self.model_forward.parameters(), lr=config.learning_rate_forward)

        self.model_forward_target   = ModelForwardTarget.Model(self.state_shape, self.actions_count)

        self.model_autoencoder       = ModelAutoencoder.Model(self.state_shape)
        self.optimizer_autoencoder   = torch.optim.Adam(self.model_autoencoder.parameters(), lr=config.learning_rate_autoencoder)

        self.policy_buffer = PolicyBuffer(self.steps, self.state_shape, self.actions_count, self.actors, self.model_ppo.device)

        self.episodic_memory_size   = config.episodic_memory_size 
        self._init_episodic_memory()

        self.states = []
        for e in range(self.actors):
            state = self.envs[e].reset()
            self.states.append(state)
            self._reset_episodic_memory(e, state)

        self.iterations = 0

        self.loss_forward           = 0.0
        self.curiosity_motivation   = 0.0

        self.loss_autoencoder       = 0.0
        self.entropy_motivation     = 0.0

        self.enable_training()

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

    def main(self):        
        states_t            = torch.tensor(self.states, dtype=torch.float).detach().to(self.model_ppo.device)
 
        logits_t, values_t  = self.model_ppo.forward(states_t)

        states_np = states_t.detach().to("cpu").numpy()
        logits_np = logits_t.detach().to("cpu").numpy()
        values_np = values_t.detach().to("cpu").numpy()

        actions = []
        for e in range(self.actors):
            actions.append(self._sample_action(logits_t[e]))

        action_one_hot_t    = self._action_one_hot(numpy.array(actions))

        curiosity_t         = self._curiosity(states_t, action_one_hot_t)
        curiosity_np        = self.beta1*numpy.tanh(curiosity_t.detach().to("cpu").numpy())
        entropy_np          = self._add_episodic_memory(states_t)

        states, rewards, dones, _ = self.envs.step(actions)

        for e in range(self.actors):            
            if self.enabled_training:
                self.policy_buffer.add(e, states_np[e], logits_np[e], values_np[e], actions[e], rewards[e] + curiosity_np[e] + entropy_np[e], dones[e])

                if self.policy_buffer.is_full():
                    self.train()

            if dones[e]:
                self.states[e] = self.envs[e].reset()
                self._reset_episodic_memory(e, self.states[e])
            else:
                self.states[e] = states[e].copy()

        k = 0.02
        self.curiosity_motivation   = (1.0 - k)*self.curiosity_motivation + k*curiosity_np.mean()
        self.entropy_motivation     = (1.0 - k)*self.entropy_motivation + k*entropy_np.mean()

        self.iterations+= 1
        return rewards[0], dones[0]
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_forward.save(save_path + "trained/")
        self.model_forward_target.save(save_path + "trained/")
        self.model_autoencoder.save(save_path + "trained/")

    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
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
    
    def _sample_action(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 0)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()

        return action_t.item() 
    
    def train(self): 
        print("train")
        self.policy_buffer.compute_returns(self.gamma)

        batch_count = self.steps//self.batch_size

        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                states, logits, values, actions, rewards, dones, returns, advantages = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)

                loss = self._compute_loss(states, logits, actions, returns, advantages)

                self.optimizer_ppo.zero_grad()        
                loss.backward()
                self.optimizer_ppo.step()

                #train forward model, MSE loss
                action_one_hot_t    = self._action_one_hot(actions)
                curiosity_t         = self._curiosity(states, action_one_hot_t)

                loss_forward = curiosity_t.mean()
                self.optimizer_forward.zero_grad()
                loss_forward.backward()
                self.optimizer_forward.step()


                #train denoising autoencoder model, MSE loss
                state_noised_t          = states + 0.1*torch.randn(states.shape).to(states.device)
                state_predicted_t, _    = self.model_autoencoder(state_noised_t)
                loss_autoencoder        = (states.detach() - state_predicted_t)**2

                loss_autoencoder = loss_autoencoder.mean()
                self.optimizer_autoencoder.zero_grad()
                loss_autoencoder.backward()
                self.optimizer_autoencoder.step()

 
                k = 0.02
                self.loss_forward       = (1.0 - k)*self.loss_forward + k*loss_forward.detach().to("cpu").numpy()
                self.loss_autoencoder   = (1.0 - k)*self.loss_autoencoder + k*loss_autoencoder.detach().to("cpu").numpy()

        self.policy_buffer.clear() 

        #print("loss ", self.loss_forward, self.curiosity_motivation, self.loss_autoencoder, self.entropy_motivation)

    
    def _compute_loss(self, states, logits, actions, returns, advantages):
        probs_old     = torch.nn.functional.softmax(logits, dim = 1).detach()
        log_probs_old = torch.nn.functional.log_softmax(logits, dim = 1).detach()

        logits_new, values_new   = self.model_ppo.forward(states)

        probs_new     = torch.nn.functional.softmax(logits_new, dim = 1)
        log_probs_new = torch.nn.functional.log_softmax(logits_new, dim = 1)

        '''
        compute critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_new = values_new.squeeze(1)
        loss_value = (returns.detach() - values_new)**2
        loss_value = self.critic_loss_proportion*loss_value.mean()

        ''' 
        compute actor loss, surrogate loss
        '''
        log_probs_new_  = log_probs_new[range(len(log_probs_new)), actions]
        log_probs_old_  = log_probs_old[range(len(log_probs_old)), actions]
                        
        ratio       = torch.exp(log_probs_new_ - log_probs_old_)
        p1          = ratio*advantages.detach()
        p2          = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages.detach()
        loss_policy = -torch.min(p1, p2)  
        loss_policy = loss_policy.mean()
    
        '''
        compute entropy loss, to avoid greedy strategy
        L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        '''
        loss_entropy = (probs_new*log_probs_new).sum(dim = 1)
        loss_entropy = self.entropy_beta*loss_entropy.mean()

        loss = loss_value + loss_policy + loss_entropy

        return loss

    def _action_one_hot(self, action_idx_t):
        batch_size = action_idx_t.shape[0]

        action_one_hot_t = torch.zeros((batch_size, self.actions_count))
        action_one_hot_t[range(batch_size), action_idx_t] = 1.0  
        action_one_hot_t = action_one_hot_t.to(self.model_ppo.device)

        return action_one_hot_t

    def _curiosity(self, state_t, action_one_hot_t):
        state_next_predicted_t       = self.model_forward(state_t, action_one_hot_t)
        state_next_predicted_t_t     = self.model_forward_target(state_t, action_one_hot_t)

        curiosity_t    = (state_next_predicted_t_t.detach() - state_next_predicted_t)**2
        curiosity_t    = curiosity_t.mean(dim=1)

        return curiosity_t

    
    def _init_episodic_memory(self):   
        states_t        = torch.randn((1, ) + self.state_shape).to(self.model_autoencoder.device)
        features_t      = self.model_autoencoder.eval_features(states_t)
        features_t      = features_t.view(features_t.size(0), -1)
        features_count  = features_t.shape[1]

        self.episodic_memory_features   = numpy.zeros((self.actors, self.episodic_memory_size, features_count))

    def _reset_episodic_memory(self, env_idx, state):
        state_t     = torch.from_numpy(state).to(self.model_autoencoder.device).unsqueeze(0).float()
        features_t  = self.model_autoencoder.eval_features(state_t)
        features_t  = features_t.view(features_t.size(0), -1)

        features_np = features_t.squeeze(0).detach().to("cpu").numpy()

        for i in range(self.episodic_memory_size):
            self.episodic_memory_features[env_idx][i]   = features_np.copy()
            

            
    def _add_episodic_memory(self, state_t):
        features_t  = self.model_autoencoder.eval_features(state_t)

        features_t  = features_t.view(features_t.size(0), -1)
        features_np = features_t.detach().to("cpu").numpy()

        motivation = []
        #put current features and action into episodic memory, on random place
        for e in range(self.actors):
            episodic_memory_std_old    = self.episodic_memory_features[e].std(axis=0).mean() 

            #put current features into episodic memory, on random place
            idx = numpy.random.randint(self.episodic_memory_size)
            self.episodic_memory_features[e][idx]   = features_np[e].copy()

            episodic_memory_std_new    = self.episodic_memory_features.std(axis=0).mean() 

            #compute relative entropy
            dif = episodic_memory_std_new - episodic_memory_std_old
            motivation_ = self.beta2*numpy.tanh(dif)

            motivation.append(motivation_)

        return numpy.array(motivation)

