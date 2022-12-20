import numpy
import torch 

from .ValuesLogger      import *
from .RunningStats      import *  
from .PolicyBufferIM    import *  

from .PPOLoss               import *
from .SelfSupervisedLoss    import *
from .Augmentations         import *

import sklearn.manifold
import matplotlib.pyplot as plt 
import cv2
 
        
class AgentPPOCND():   
    def __init__(self, envs, ModelPPO, ModelCNDTarget, ModelCND, config):
        self.envs = envs  
       
        self.gamma_ext          = config.gamma_ext 
        self.gamma_int          = config.gamma_int
            
        self.ext_adv_coeff      = config.ext_adv_coeff
        self.int_adv_coeff      = config.int_adv_coeff
        self.int_reward_coeff   = config.int_reward_coeff
        self.cnd_dropout        = config.cnd_dropout
    
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip 
    
        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count 

        self.ppo_regularization_loss_coeff    = config.ppo_regularization_loss_coeff
 

        if config.ppo_regularization_loss == "mse":
            self._ppo_regularization_loss = contrastive_loss_mse
        elif config.ppo_regularization_loss == "nce":
            self._ppo_regularization_loss = contrastive_loss_nce
        elif config.ppo_regularization_loss == "vicreg":
            self._ppo_regularization_loss = contrastive_loss_vicreg
        elif config.ppo_regularization_loss == "vicreg2":
            self._ppo_regularization_loss = contrastive_loss_vicreg2
        else:
            self._ppo_regularization_loss = None 
  
        if config.cnd_regularization_loss == "mse":
            self._cnd_regularization_loss = contrastive_loss_mse
        elif config.cnd_regularization_loss == "nce":
            self._cnd_regularization_loss = contrastive_loss_nce
        elif config.cnd_regularization_loss == "vicreg":
            self._cnd_regularization_loss = contrastive_loss_vicreg
        elif config.cnd_regularization_loss == "vicreg2":
            self._cnd_regularization_loss = contrastive_loss_vicreg2
        else:
            self._cnd_regularization_loss = None

        self.ppo_augmentations = config.ppo_augmentations
        self.ppo_reg_augmentations = config.ppo_reg_augmentations
        
        self.cnd_augmentations = config.cnd_augmentations
        
        print("ppo_regularization_loss  = ", self._ppo_regularization_loss)
        print("cnd_regularization_loss  = ", self._cnd_regularization_loss)
        print("ppo_augmentations        = ", self.ppo_augmentations)
        print("ppo_reg_augmentations    = ", self.ppo_reg_augmentations)
        print("cnd_augmentations        = ", self.cnd_augmentations)

        print("\n\n")

        self.normalise_state_mean = config.normalise_state_mean
        self.normalise_state_std  = config.normalise_state_std

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)

        self.model_cnd_target      = ModelCNDTarget.Model(self.state_shape)
        self.optimizer_cnd_target  = torch.optim.Adam(self.model_cnd_target.parameters(), lr=config.learning_rate_cnd_target)

        self.model_cnd      = ModelCND.Model(self.state_shape)
        self.optimizer_cnd  = torch.optim.Adam(self.model_cnd.parameters(), lr=config.learning_rate_cnd)
 
        self.policy_buffer = PolicyBufferIM(self.steps, self.state_shape, self.actions_count, self.envs_count)
 
        for e in range(self.envs_count):
            self.envs.reset(e)
 
        self.states_running_stats       = RunningStats(self.state_shape)

        if self.envs_count > 1:
            self._init_running_stats()

        #reset envs and fill initial state
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e] = self.envs.reset(e).copy()

        self.enable_training()
        self.iterations = 0 

        self.values_logger = ValuesLogger()

        self.values_logger.add("loss_cnd", 0.0)
        self.values_logger.add("loss_cnd_regularization", 0.0)
        self.values_logger.add("cnd_magnitude", 0.0)

        self.values_logger.add("loss_actor", 0.0)
        self.values_logger.add("loss_critic", 0.0)

        self.values_logger.add("internal_motivation_mean", 0.0)
        self.values_logger.add("internal_motivation_std", 0.0)

        self.values_logger.add("loss_ppo_regularization", 0.0)
        self.values_logger.add("symmetry_accuracy", 0.0)
        self.values_logger.add("symmetry_magnitude", 0.0)

        self.vis_features = []
        self.vis_labels   = []


    def enable_training(self):
        self.enabled_training = True
 
    def disable_training(self):
        self.enabled_training = False
 
    def main(self): 
        #state to tensor
        states = torch.tensor(self.states, dtype=torch.float).to(self.model_ppo.device)
        
        #states augmentations, if any
        #states = self._aug_ppo(states)
        
        #compute model output
        logits, values_ext, values_int  = self.model_ppo.forward(states)
        
        #collect actions 
        actions = self._sample_actions(logits)
        
        #execute action
        states_new, rewards_ext, dones, infos = self.envs.step(actions)

        #update long term stats (mean and variance)
        self.states_running_stats.update(self.states)

        #curiosity motivation
        rewards_int    = self._curiosity(states)
        rewards_int    = torch.clip(self.int_reward_coeff*rewards_int, 0.0, 1.0)
        
        #put into policy buffer
        if self.enabled_training:
            states          = states.detach().to("cpu")
            logits          = logits.detach().to("cpu")
            values_ext      = values_ext.squeeze(1).detach().to("cpu") 
            values_int      = values_int.squeeze(1).detach().to("cpu")
            actions         = torch.from_numpy(actions).to("cpu")
            rewards_ext_t   = torch.from_numpy(rewards_ext).to("cpu")
            rewards_int_t   = rewards_int.detach().to("cpu")
            dones           = torch.from_numpy(dones).to("cpu")

            self.policy_buffer.add(states, logits, values_ext, values_int, actions, rewards_ext_t, rewards_int_t, dones)

            if self.policy_buffer.is_full():
                self.train()

        
        #update new state
        self.states = states_new.copy()

        #or reset env if done
        for e in range(self.envs_count): 
            if dones[e]:
                self.states[e] = self.envs.reset(e).copy()

         
        #self._add_for_plot(states, infos, dones)
        
        #collect stats
        self.values_logger.add("internal_motivation_mean", rewards_int.mean().detach().to("cpu").numpy())
        self.values_logger.add("internal_motivation_std" , rewards_int.std().detach().to("cpu").numpy())

        self.iterations+= 1


        return rewards_ext[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_cnd.save(save_path + "trained/")
        self.model_cnd_target.save(save_path + "trained/")
 
    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
        self.model_cnd.load(load_path + "trained/")
        self.model_cnd_target.load(load_path + "trained/")
 
    def get_log(self): 
        return self.values_logger.get_str()

    def render(self, env_id):
        size            = 256

        states_t        = torch.tensor(self.states, dtype=torch.float).detach().to(self.model_ppo.device)

        #state           = self._norm_state(states_t)[env_id][0].detach().to("cpu").numpy()
        state           = states_t[env_id][0].detach().to("cpu").numpy()

        state_im        = cv2.resize(state, (size, size))
        state_im        = numpy.clip(state_im, 0.0, 1.0)

        cv2.imshow("CND agent", state_im)
        cv2.waitKey(1)

    def _sample_actions(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().to("cpu").numpy()
        return actions
    
    def train(self): 
        self.policy_buffer.compute_returns(self.gamma_ext, self.gamma_int)

        batch_count = self.steps//self.batch_size

        small_batch = 16*self.batch_size 

        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                states, states_next, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)

                #train PPO model
                loss_ppo     = self._compute_loss_ppo(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int)
                
                #train ppo model features
                if self._ppo_regularization_loss is not None:

                    #smaller batch for self-supervised regularization
                    states_a, states_b, labels = self.policy_buffer.sample_states(small_batch, 0.5, self.model_ppo.device)

                    loss_ppo_regularization, magnitude, acc = self._ppo_regularization_loss(self.model_ppo, states_a, states_b, labels, None, self._aug_ppo_reg)                

                    loss_ppo+= self.ppo_regularization_loss_coeff*loss_ppo_regularization

                    self.values_logger.add("loss_ppo_regularization", loss_ppo_regularization.detach().to("cpu").numpy())
                    self.values_logger.add("symmetry_accuracy", acc)
                    self.values_logger.add("symmetry_magnitude", magnitude)

 
                self.optimizer_ppo.zero_grad()        
                loss_ppo.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()

                #train CND model, MSE loss (same as RND)
                loss_cnd = self._compute_loss_cnd(states, self.cnd_dropout)

                self.optimizer_cnd.zero_grad() 
                loss_cnd.backward()
                self.optimizer_cnd.step() 

                #log results
                self.values_logger.add("loss_cnd", loss_cnd.detach().to("cpu").numpy())
                
                #train cnd target model for regularization (optional)
                if self._cnd_regularization_loss is not None:                    
                    #smaller batch for self-supervised regularization
                    states_a, states_b, labels = self.policy_buffer.sample_states(small_batch, 0.5, self.model_ppo.device)

                    loss, magnitude, acc = self._cnd_regularization_loss(self.model_cnd_target, states_a, states_b, labels, self._norm_state, self._aug_cnd)                
    
                    self.optimizer_cnd_target.zero_grad() 
                    loss.backward()
                    self.optimizer_cnd_target.step()

                    self.values_logger.add("loss_cnd_regularization", loss.detach().to("cpu").numpy())
                    self.values_logger.add("cnd_magnitude", magnitude)

                    if self._ppo_regularization_loss is None:
                        self.values_logger.add("symmetry_accuracy", acc)

        self.policy_buffer.clear() 

    
    def _compute_loss_ppo(self, states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int):
        logits_new, values_ext_new, values_int_new  = self.model_ppo.forward(states)

        #critic loss
        loss_critic =  ppo_compute_critic_loss(values_ext_new, returns_ext, values_int_new, returns_int)

        #actor loss        
        advantages  = self.ext_adv_coeff*advantages_ext + self.int_adv_coeff*advantages_int
        advantages  = advantages.detach() 
        loss_policy, loss_entropy  = ppo_compute_actor_loss(logits, logits_new, advantages, actions, self.eps_clip, self.entropy_beta)

        loss_actor = loss_policy + loss_entropy

        #total loss
        loss = 0.5*loss_critic + loss_actor

        #store to log
        self.values_logger.add("loss_actor", loss_actor.mean().detach().to("cpu").numpy())
        self.values_logger.add("loss_critic", loss_critic.mean().detach().to("cpu").numpy())

        return loss 

    
    #MSE loss for cnd model
    def _compute_loss_cnd(self, states, dropout = 0.75):
        
        state_norm_t    = self._norm_state(states).detach()
 
        features_predicted_t  = self.model_cnd(state_norm_t)
        features_target_t     = self.model_cnd_target(state_norm_t).detach()

        loss_cnd = (features_target_t - features_predicted_t)**2
 
        #random loss regularization, 25% non zero for 128envs, 100% non zero for 32envs
        '''
        prob            = 1.0 - dropout
        random_mask     = torch.rand(loss_cnd.shape).to(loss_cnd.device)
        random_mask     = 1.0*(random_mask < prob) 
        loss_cnd        = (loss_cnd*random_mask).sum() / (random_mask.sum() + 0.00000001)
        '''
        random_mask     = (torch.rand_like(loss_cnd) > dropout).float()
        loss_cnd        = (loss_cnd*random_mask).sum() / (random_mask.sum() + 0.00000001)

        return loss_cnd


    #compute internal motivation
    def _curiosity(self, states):
        states_norm            = self._norm_state(states)

        features_predicted_t    = self.model_cnd(states_norm)
        features_target_t       = self.model_cnd_target(states_norm)
 
        curiosity_t = ((features_target_t - features_predicted_t)**2).mean(dim=1)
        
        return curiosity_t
 

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
        for _ in range(steps):
            #random action
            actions             = numpy.random.randint(0, self.actions_count, (self.envs_count))
            states, _, dones, _ = self.envs.step(actions)

            #update stats
            self.states_running_stats.update(states)

            for e in range(self.envs_count): 
                if dones[e]:
                    self.envs.reset(e)


    def _aug(self, x, augmentations): 
        if "conv" in augmentations:
            x = aug_random_apply(x, 0.25, aug_conv)

        if "pixelate" in augmentations:
            x = aug_random_apply(x, 0.25, aug_pixelate)

        if "mask" in augmentations:
            x = aug_random_apply(x, 0.25, aug_mask_tiles)

        if "noise" in augmentations:
            x = aug_noise(x, k = 0.2)
        
        return x.detach() 

    def _aug_ppo(self, x):
        return self._aug(x, self.ppo_augmentations)

    def _aug_ppo_reg(self, x):
        return self._aug(x, self.ppo_reg_augmentations)

    def _aug_cnd(self, x):
        return self._aug(x, self.cnd_augmentations)

    def _add_for_plot(self, states, infos, dones):
        
        states_norm_t   = self._norm_state(states)
        features        = self.model_cnd_target(states_norm_t)
        
        '''
        features        = self.model_ppo.forward_features(states)
        '''

        features        = features.detach().to("cpu").numpy()
        
        self.vis_features.append(features[0])

        if "room_id" in infos[0]:
            self.vis_labels.append(infos[0]["room_id"])
        else:
            self.vis_labels.append(0)

        if dones[0]:
            print("training t-sne")

            max_num = numpy.max(self.vis_labels) 

            features_embedded = sklearn.manifold.TSNE(n_components=2).fit_transform(self.vis_features)

            print("result shape = ", features_embedded.shape)

            plt.clf()
            #plt.scatter(features_embedded[:, 0], features_embedded[:, 1])
            plt.scatter(features_embedded[:, 0], features_embedded[:, 1], c=self.vis_labels, cmap=plt.cm.get_cmap("jet", max_num - 1))
            plt.colorbar(ticks=range(max_num))
            plt.tight_layout()
            plt.show()

            self.vis_features   = []
            self.vis_labels     = []