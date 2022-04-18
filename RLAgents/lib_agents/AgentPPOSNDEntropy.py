import numpy
import torch 

from .ValuesLogger          import *
from .RunningStats          import *  
from .PolicyBufferIMDual    import *  
from .FeaturesBuffer        import *


import sklearn.manifold
import matplotlib.pyplot as plt
import cv2
 
       
class AgentPPOSNDEntropy():   
    def __init__(self, envs, ModelPPO, ModelSNDTarget, ModelSND, ModelEntropy, config):
        self.envs = envs  
      
        self.gamma_ext              = config.gamma_ext 
        self.gamma_int_a            = config.gamma_int_a
        self.gamma_int_b            = config.gamma_int_b
            
        self.ext_adv_coeff          = config.ext_adv_coeff
        self.int_a_adv_coeff        = config.int_a_adv_coeff
        self.int_b_adv_coeff        = config.int_b_adv_coeff

        self.int_a_reward_coeff     = config.int_a_reward_coeff
        self.int_b_reward_coeff     = config.int_b_reward_coeff
    
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip 
    
        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count 


        if config.snd_regularisation_loss == "mse":
            self._snd_regularisation_loss = self._contrastive_loss_mse
        elif config.snd_regularisation_loss == "info_nce":
            self._snd_regularisation_loss = self._contrastive_loss_info_nce
        else: 
            self._snd_regularisation_loss = None

        if config.ppo_regularisation_loss == "mse":
            self._ppo_regularisation_loss = self._contrastive_loss_mse
        elif config.ppo_regularisation_loss == "info_nce":
            self._ppo_regularisation_loss = self._contrastive_loss_info_nce
        else:
            self._ppo_regularisation_loss = None

        if config.entropy_regularisation_loss == "mse":
            self._entropy_regularisation_loss = self._contrastive_loss_mse
        elif config.entropy_regularisation_loss == "info_nce":
            self._entropy_regularisation_loss = self._contrastive_loss_info_nce
        else:
            self._entropy_regularisation_loss = None


        print("snd_regularisation_loss = ",     self._snd_regularisation_loss)
        print("ppo_regularisation_loss = ",     self._ppo_regularisation_loss)
        print("entropy_regularisation_loss = ", self._entropy_regularisation_loss)

        self.normalise_state_mean = config.normalise_state_mean
        self.normalise_state_std  = config.normalise_state_std

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)

        self.model_snd_target      = ModelSNDTarget.Model(self.state_shape)
        self.optimizer_snd_target  = torch.optim.Adam(self.model_snd_target.parameters(), lr=config.learning_rate_snd_target)

        self.model_snd      = ModelSND.Model(self.state_shape)
        self.optimizer_snd  = torch.optim.Adam(self.model_snd.parameters(), lr=config.learning_rate_snd)
 
        self.model_entropy      = ModelEntropy.Model(self.state_shape)
        self.optimizer_entropy  = torch.optim.Adam(self.model_entropy.parameters(), lr=config.learning_rate_entropy)
 
        self.policy_buffer       = PolicyBufferIMDual(self.steps, self.state_shape, self.actions_count, self.envs_count)
        
        self.features_buffer     = FeaturesBuffer(config.features_buffer_size, self.envs_count, (256, ))

        for e in range(self.envs_count):
            self.envs.reset(e)
        
        self.states_running_stats       = RunningStats(self.state_shape)

        if self.envs_count > 1:
            self._init_running_stats()

        #reset envs and fill initial state
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e][0:4] = self.envs.reset(e).copy()

        self.enable_training()
        self.iterations                     = 0 

        self.values_logger                  = ValuesLogger()

        self.values_logger.add("loss_snd", 0.0)
        self.values_logger.add("loss_entropy_regularization", 0.0)
        self.values_logger.add("loss_snd_regularization", 0.0)
        self.values_logger.add("loss_ppo_regularization", 0.0)
        self.values_logger.add("loss_actor", 0.0)
        self.values_logger.add("loss_critic", 0.0)
        self.values_logger.add("internal_motivation_a_mean", 0.0)
        self.values_logger.add("internal_motivation_a_std", 0.0)
        self.values_logger.add("internal_motivation_b_mean", 0.0)
        self.values_logger.add("internal_motivation_b_std", 0.0)
 
        #self.vis_features = []
        #self.vis_labels   = []


    def enable_training(self):
        self.enabled_training = True
 
    def disable_training(self):
        self.enabled_training = False

    def main(self): 
        #state to tensor
        states        = torch.tensor(self.states, dtype=torch.float).to(self.model_ppo.device)

        #compute model output
        logits, values_ext, values_int_a, values_int_b   = self.model_ppo.forward(states)
        
        #collect actions
        actions = self._sample_actions(logits)
         
        #execute action
        states_new, rewards_ext, dones, infos = self.envs.step(actions)

        #update long term stats (mean and variance)
        self.states_running_stats.update(self.states)

        #curiosity motivation
        rewards_int_a   = self._curiosity(states)
        rewards_int_a   = torch.clip(self.int_a_reward_coeff*rewards_int_a, 0.0, 1.0)

        #curiosity motivation
        rewards_int_b   = self._entropy(states)
        rewards_int_b   = torch.clip(self.int_b_reward_coeff*rewards_int_b, 0.0, 1.0)
        

        #put into policy buffer
        if self.enabled_training:
            states          = states.detach().to("cpu")
            logits          = logits.detach().to("cpu")
            values_ext      = values_ext.squeeze(1).detach().to("cpu") 
            values_int_a    = values_int_a.squeeze(1).detach().to("cpu")
            values_int_b    = values_int_b.squeeze(1).detach().to("cpu")
            actions         = torch.from_numpy(actions).to("cpu")
            rewards_ext_t   = torch.from_numpy(rewards_ext).to("cpu")
            rewards_int_a_t = rewards_int_a.detach().to("cpu")
            rewards_int_b_t = rewards_int_b.detach().to("cpu")
            dones           = torch.from_numpy(dones).to("cpu")

            self.policy_buffer.add(states, logits, values_ext, values_int_a, values_int_b, actions, rewards_ext_t, rewards_int_a_t, rewards_int_b_t, dones)

            if self.policy_buffer.is_full():
                self.train()
         
        #update new state
        self.states = states_new.copy()

        #or reset env if done
        for e in range(self.envs_count): 
            if dones[e]:
                self.states[e] = self.envs.reset(e).copy()
                self.features_buffer.reset(e)

        '''
        #states_norm_t   = self._norm_state(states_t)
        #features        = self.model_snd_target(states_norm_t)
        features = self.model_entropy(states.to(self.model_entropy.device))
        features        = features.detach().to("cpu").numpy()
        
        self.vis_features.append(features[0])
        self.vis_labels.append(infos[0]["room_id"])

        if dones[0]:
            print("training t-sne")

            features_embedded = sklearn.manifold.TSNE(n_components=2).fit_transform(self.vis_features)

            print("result shape = ", features_embedded.shape)

            plt.clf()
            plt.scatter(features_embedded[:, 0], features_embedded[:, 1], c=self.vis_labels, cmap=plt.cm.get_cmap("jet", numpy.max(self.vis_labels)))
            plt.colorbar(ticks=range(16))
            plt.tight_layout()
            plt.show()

            self.vis_features   = []
            self.vis_labels     = []
        '''
        
        #collect stats
        self.values_logger.add("internal_motivation_a_mean", rewards_int_a.mean().detach().to("cpu").numpy())
        self.values_logger.add("internal_motivation_a_std" , rewards_int_a.std().detach().to("cpu").numpy())

        self.values_logger.add("internal_motivation_b_mean", rewards_int_b.mean().detach().to("cpu").numpy())
        self.values_logger.add("internal_motivation_b_std" , rewards_int_b.std().detach().to("cpu").numpy())

        self.iterations+= 1
        return rewards_ext[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_snd.save(save_path + "trained/")
        self.model_snd_target.save(save_path + "trained/")
        self.model_entropy.save(save_path + "trained/")

    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
        self.model_snd.load(load_path + "trained/")
        self.model_snd_target.load(load_path + "trained/")
        self.model_entropy.load(load_path + "trained/")
 
    def get_log(self): 
        return self.values_logger.get_str()

    def render(self, env_id):
        size            = 256

        states_t        = torch.tensor(self.states, dtype=torch.float).detach().to(self.model_ppo.device)

        state           = self._norm_state(states_t)[env_id][0].detach().to("cpu").numpy()

        state_im        = cv2.resize(state, (size, size))
        state_im        = numpy.clip(state_im, 0.0, 1.0)

        cv2.imshow("RND agent", state_im)
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
                states, states_next, logits, actions, returns_ext, returns_int_a, returns_int_b, advantages_ext, advantages_int_a, advantages_int_b = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)

                #train PPO model
                loss_ppo = self._compute_loss_ppo(states, logits, actions, returns_ext, returns_int_a, returns_int_b, advantages_ext, advantages_int_a, advantages_int_b)

                self.optimizer_ppo.zero_grad()        
                loss_ppo.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()

                #train snd model, MSE loss
                loss_snd = self._compute_loss_snd(states)

                self.optimizer_snd.zero_grad() 
                loss_snd.backward()
                self.optimizer_snd.step()

                #log results
                self.values_logger.add("loss_snd", loss_snd.detach().to("cpu").numpy())
              
                #smaller batch for regularisation
                states_a, states_b, labels = self.policy_buffer.sample_states(64, 0.5, self.model_ppo.device)

                #contrastive loss for better features space (optional)
                if self._ppo_regularisation_loss is not None:
                    loss = self._ppo_regularisation_loss(self.model_ppo, states_a, states_b, labels, normalise=False, augmentation=True)

                    self.optimizer_ppo.zero_grad()        
                    loss.backward()
                    self.optimizer_ppo.step()

                    self.values_logger.add("loss_ppo_regularization", loss.detach().to("cpu").numpy())

                #train snd target model for regularisation (optional)
                if self._snd_regularisation_loss is not None:                    
                    loss = self._snd_regularisation_loss(self.model_snd_target, states_a, states_b, labels, normalise=True, augmentation=True)                
    
                    self.optimizer_snd_target.zero_grad() 
                    loss.backward()
                    self.optimizer_snd_target.step()

                    self.values_logger.add("loss_snd_regularization", loss.detach().to("cpu").numpy())

                #train entropy target model for regularisation (optional)
                if self._entropy_regularisation_loss is not None:                    
                    loss = self._entropy_regularisation_loss(self.model_entropy, states_a, states_b, labels, normalise=False, augmentation=True)                
    
                    self.optimizer_entropy.zero_grad() 
                    loss.backward()
                    self.optimizer_entropy.step()

                    self.values_logger.add("loss_entropy_regularization", loss.detach().to("cpu").numpy())

               
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
        self.values_logger.add("loss_actor", loss_actor.mean().detach().to("cpu").numpy())
        self.values_logger.add("loss_critic", loss_critic.mean().detach().to("cpu").numpy())

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
        compute internal critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_int_a_new  = values_int_a_new.squeeze(1)
        loss_int_value_a  = (returns_int_a.detach() - values_int_a_new)**2
        loss_int_value_a  = loss_int_value_a.mean()

        '''
        compute internal critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_int_b_new  = values_int_b_new.squeeze(1)
        loss_int_value_b  = (returns_int_b.detach() - values_int_b_new)**2
        loss_int_value_b  = loss_int_value_b.mean()
        
        loss_critic     = loss_ext_value + loss_int_value_a + loss_int_value_b
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

    #MSE loss for snd model
    def _compute_loss_snd(self, states):
        
        state_norm_t    = self._norm_state(states).detach()
 
        features_predicted_t  = self.model_snd(state_norm_t)
        features_target_t     = self.model_snd_target(state_norm_t).detach()

        loss_snd = (features_target_t - features_predicted_t)**2

        #random loss regularisation, 25% non zero for 128envs, 100% non zero for 32envs
        prob            = 32.0/self.envs_count
        random_mask     = torch.rand(loss_snd.shape).to(loss_snd.device)
        random_mask     = 1.0*(random_mask < prob) 
        loss_snd        = (loss_snd*random_mask).sum() / (random_mask.sum() + 0.00000001)

        return loss_snd

    
    def _contrastive_loss_mse(self, model, states_a, states_b, target, normalise, augmentation):
        xa = states_a.clone()
        xb = states_b.clone()

        #normalise states
        if normalise:
            xa = self._norm_state(xa) 
            xb = self._norm_state(xb)

        #states augmentation
        if augmentation:
            xa = self._aug(xa)
            xb = self._aug(xb)
 
        #obtain features from model
        if hasattr(model, "forward_features"):
            za = model.forward_features(xa)  
            zb = model.forward_features(xb) 
        else:
            za = model(xa)  
            zb = model(xb) 

        #predict close distance for similar, far distance for different states
        predicted = ((za - zb)**2).mean(dim=1)

        #MSE loss
        loss_mse = ((target - predicted)**2).mean()

        return loss_mse
    
    def _contrastive_loss_info_nce(self, model, states_a, states_b, target, normalise, augmentation):
        xa = states_a.clone()
        xb = states_a.clone() 

        #normalise states
        if normalise:
            xa = self._norm_state(xa)
            xb = self._norm_state(xb)

        #states augmentation
        if augmentation:
            xa = self._aug(xa)
            xb = self._aug(xb)
 
        #obtain features from model
        if hasattr(model, "forward_features"):
            za = model.forward_features(xa)  
            zb = model.forward_features(xb) 
        else:
            za = model(xa)  
            zb = model(xb)

        logits = torch.matmul(za, zb.t())

        #place target class ID on diagonal
        labels = torch.tensor(range(logits.shape[0])).to(logits.device)

        #info NCE loss, train to strong correlation for similar states
        loss_nce   = torch.nn.functional.cross_entropy(logits, labels)

        #magnitude regularisation
        mag_za = (za**2).mean()
        mag_zb = (zb**2).mean()

        loss_magnitude = 0.1*(mag_za + mag_zb)
    
        loss = loss_nce + loss_magnitude  

        return loss


    #compute internal motivation
    def _curiosity(self, states):
        states_norm            = self._norm_state(states)

        features_predicted_t    = self.model_snd(states_norm)
        features_target_t       = self.model_snd_target(states_norm)
 
        curiosity_t = ((features_target_t - features_predicted_t)**2).mean(dim=1)
        return curiosity_t

    def _entropy(self, states):
        features_t    = self.model_entropy(states)
        self.features_buffer.add(features_t)

        
        return self.features_buffer.compute_entropy()


    #normalise mean and std for state
    def _norm_state(self, states):
        states_norm = states

        if self.normalise_state_mean:
            mean = torch.from_numpy(self.states_running_stats.mean).to(states.device).float()
            states_norm = states_norm - mean

        if self.normalise_state_std:
            std  = torch.from_numpy(self.states_running_stats.std).to(states.device).float()            
            states_norm = torch.clamp(states_norm/std, -1.0, 1.0)

        return states_norm 

    #random policy for stats init
    def _init_running_stats(self, steps = 256):
        states_ = numpy.zeros((self.envs_count, ) + self.state_shape)

        for _ in range(steps):
            #random action
            actions             = numpy.random.randint(0, self.actions_count, (self.envs_count))
            states, _, dones, _ = self.envs.step(actions)

            states_[:, 0:4] = states

            #update stats
            self.states_running_stats.update(states_)

            for e in range(self.envs_count): 
                if dones[e]:
                    self.envs.reset(e)

    def _aug(self, x):
        '''
        x = self._aug_random_apply(x, 0.5, self._aug_mask)
        x = self._aug_random_apply(x, 0.5, self._aug_resize2)
        x = self._aug_noise(x, k = 0.2)
        '''

        #this works perfect
        x = self._aug_random_apply(x, 0.5, self._aug_resize2)
        x = self._aug_random_apply(x, 0.25, self._aug_resize4)
        x = self._aug_random_apply(x, 0.125, self._aug_mask)
        x = self._aug_noise(x, k = 0.2)
        
        return x

    def _aug_random_apply(self, x, p, aug_func):
        shape  = (x.shape[0], ) + (1,)*(len(x.shape)-1)
        apply  = 1.0*(torch.rand(shape, device=x.device) < p)

        return (1 - apply)*x + apply*aug_func(x) 
 

    def _aug_resize(self, x, scale = 2):
        ds      = torch.nn.AvgPool2d(scale, scale).to(x.device)
        us      = torch.nn.Upsample(scale_factor=scale).to(x.device)

        scaled  = us(ds(x))  
        return scaled

    def _aug_resize2(self, x):
        return self._aug_resize(x, 2)

    def _aug_resize4(self, x):
        return self._aug_resize(x, 4)

    def _aug_mask(self, x, p = 0.1):
        mask = 1.0*(torch.rand_like(x) < (1.0 - p))
        return x*mask  

    def _aug_noise(self, x, k = 0.2): 
        pointwise_noise   = k*(2.0*torch.rand(x.shape, device=x.device) - 1.0)
        return x + pointwise_noise


   