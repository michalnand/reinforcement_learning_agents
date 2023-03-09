import numpy
import torch 

from .ValuesLogger      import *
from .PolicyBufferIM    import *  

from .PPOLoss               import *
from .SelfSupervisedLoss    import *
from .Augmentations         import *

         
class AgentPPOCNDSA():   
    def __init__(self, envs, ModelPPO, ModelCNDTarget, ModelCND, config):
        self.envs = envs  
        
        self.gamma_ext          = config.gamma_ext 
        self.gamma_int          = config.gamma_int
              
        self.ext_adv_coeff      = config.ext_adv_coeff
        self.int_adv_coeff      = config.int_adv_coeff
        self.int_reward_coeff   = config.int_reward_coeff
        self.aux_loss_coeff     = config.aux_loss_coeff
      
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip 
    
        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count

        if hasattr(config, "beta_var"):
            self.beta_var           = config.beta_var
            self.var                = 0.0
            self.beta_product       = 1.0
        else:
            self.beta_var = None

   
        if config.target_regularization_loss == "vicreg":
            self._target_regularization_loss = loss_vicreg
        else:
            self._target_regularization_loss = None

        if config.target_aux_loss == "action_loss":
            self._target_aux_loss = self._action_loss
        elif config.target_aux_loss == "constructor_loss":
             self._target_aux_loss = self._constructor_loss
        else:
            self._target_aux_loss = None
         
        self.augmentations                  = config.augmentations
        self.augmentations_probs            = config.augmentations_probs
        
        print("target_regularization_loss   = ", self._target_regularization_loss)
        print("target_aux_loss              = ", self._target_aux_loss)
        print("augmentations                = ", self.augmentations)
        print("augmentations_probs          = ", self.augmentations_probs)
        print("int_reward_coeff             = ", self.int_reward_coeff)
        print("aux_loss_coeff               = ", self.aux_loss_coeff)

        print("\n\n")

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)

        self.model_cnd_target      = ModelCNDTarget.Model(self.state_shape, self.actions_count)
        self.optimizer_cnd_target  = torch.optim.Adam(self.model_cnd_target.parameters(), lr=config.learning_rate_cnd_target)

        self.model_cnd      = ModelCND.Model(self.state_shape)
        self.optimizer_cnd  = torch.optim.Adam(self.model_cnd.parameters(), lr=config.learning_rate_cnd)
 
        self.policy_buffer = PolicyBufferIM(self.steps, self.state_shape, self.actions_count, self.envs_count)
 
        for e in range(self.envs_count):
            self.envs.reset(e)

        #reset envs and fill initial state
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e] = self.envs.reset(e).copy()
            
        self.enable_training()
        self.iterations     = 0 

        self.values_logger  = ValuesLogger() 

        self.values_logger.add("internal_motivation_mean",      0.0)
        self.values_logger.add("internal_motivation_std" ,      0.0)

        self.values_logger.add("loss_ppo_actor",                0.0)
        self.values_logger.add("loss_ppo_critic",               0.0)
        self.values_logger.add("loss_distillation",             0.0)
        self.values_logger.add("loss_target_regularization",    0.0)
        self.values_logger.add("loss_target_aux",               0.0)

        self.values_logger.add("target_magnitude",              0.0)
        self.values_logger.add("target_magnitude_std",          0.0)
        self.values_logger.add("target_similarity_accuracy",    0.0)
        self.values_logger.add("target_aux_accuracy",        0.0)

        self.vis_features = []
        self.vis_labels   = []


    def enable_training(self): 
        self.enabled_training = True
 
    def disable_training(self):
        self.enabled_training = False
 
    def main(self): 
        #state to tensor
        states = torch.tensor(self.states, dtype=torch.float).to(self.model_ppo.device)
        
        #compute model output
        logits, values_ext, values_int  = self.model_ppo.forward(states)
        
        #collect actions 
        actions = self._sample_actions(logits)
        
        #execute action
        states_new, rewards_ext, dones, infos = self.envs.step(actions)

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
                states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)

                #train PPO model
                loss_ppo     = self._loss_ppo(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int)
                
                self.optimizer_ppo.zero_grad()        
                loss_ppo.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()

                #train CND model, MSE loss (same as RND)
                loss_distillation = self._loss_distillation(states)

                self.optimizer_cnd.zero_grad() 
                loss_distillation.backward()
                self.optimizer_cnd.step() 

                
                #train cnd target model for regularization
                #sample smaller batch for self-supervised regularization
                states_a, states_b, states_c, action = self.policy_buffer.sample_states_action_pairs(small_batch, self.model_ppo.device)
 
                loss_target_regularization, target_magnitude, target_magnitude_std, target_similarity_accuracy = self._target_regularization_loss(self.model_cnd_target, states_a, states_a, self._augmentations)                

                #optional auxliary loss
                #e.g. inverse model : action prediction from two consectuctive states
                if self._target_aux_loss is not None:
                    loss_target_aux, target_aux_accuracy = self._target_aux_loss(states_a, states_b, states_c, action)                 
                else:
                    loss_target_aux         = torch.zeros((1, ), device=self.model_ppo.device)[0]
                    target_aux_accuracy     = 0.0


                #final loss for target model
                loss_target = loss_target_regularization + self.aux_loss_coeff*loss_target_aux

                self.optimizer_cnd_target.zero_grad() 
                loss_target.backward()
                self.optimizer_cnd_target.step()

                #log results
                self.values_logger.add("loss_distillation",             loss_distillation.detach().to("cpu").numpy())
                
                self.values_logger.add("loss_target_regularization",    loss_target_regularization.detach().to("cpu").numpy())
                self.values_logger.add("loss_target_aux",               loss_target_aux.detach().to("cpu").numpy())

                self.values_logger.add("target_magnitude",              target_magnitude)
                self.values_logger.add("target_magnitude_std",          target_magnitude_std)
                self.values_logger.add("target_similarity_accuracy",    target_similarity_accuracy)
                self.values_logger.add("target_aux_accuracy",        target_aux_accuracy)



        self.policy_buffer.clear() 

    
    def _loss_ppo(self, states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int):
        logits_new, values_ext_new, values_int_new  = self.model_ppo.forward(states)

        #critic loss
        loss_critic =  ppo_compute_critic_loss(values_ext_new, returns_ext, values_int_new, returns_int)

        #actor loss        
        advantages  = self.ext_adv_coeff*advantages_ext + self.int_adv_coeff*advantages_int
        advantages  = advantages.detach() 

        #advantages viariance normalisation (optional), mean is already almost zero
        if self.beta_var is not None:
            self.var          = self.beta_var*self.var + (1.0 - self.beta_var)*(advantages**2).mean()
            self.beta_product = self.beta_product*self.beta_var

            var = self.var/(1.0 - self.beta_product) 

            advantages  = advantages/((var**0.5) + 1e-10)

        loss_policy, loss_entropy  = ppo_compute_actor_loss(logits, logits_new, advantages, actions, self.eps_clip, self.entropy_beta)

        loss_actor = loss_policy + loss_entropy

        #total loss
        loss = 0.5*loss_critic + loss_actor

        #store to log
        self.values_logger.add("loss_ppo_actor", loss_actor.mean().detach().to("cpu").numpy())
        self.values_logger.add("loss_ppo_critic", loss_critic.mean().detach().to("cpu").numpy())

        return loss 

    
    #MSE loss for networks distillation model
    def _loss_distillation(self, states): 
        features_predicted_t  = self.model_cnd(states)
        features_target_t     = self.model_cnd_target(states).detach()

        loss_cnd = (features_target_t - features_predicted_t)**2

        loss_cnd  = loss_cnd.mean() 
        return loss_cnd

    #inverse model for action prediction
    def _action_loss(self, states_now, states_next, states_random, action):
        action_pred     = self.model_cnd_target.forward_aux(states_now, states_next)

        action_one_hot  = torch.nn.functional.one_hot(action, self.actions_count).to(states_now.device)

        loss            =  ((action_one_hot - action_pred)**2).mean()

        #compute accuracy
        pred = torch.argmax(action_pred.detach(), dim=1)
        acc = 100.0*(pred == action).float().mean()
        acc = acc.detach().to("cpu").numpy()

        return loss, acc

    #constructor theory loss
    #inverse model for action prediction
    def _constructor_loss(self, states_now, states_next, states_random, action):
        batch_size          = states_now.shape[0]

        #create labels : half consectuctive, half random
        transition_label    = torch.cat([torch.ones((batch_size//2, 1)), torch.zeros((batch_size//2, 1))], dim=0).to(states_now.device)

        #mix states : consectuctive or random
        states_other        = torch.cat([states_next[0:batch_size//2], states_random[batch_size//2:]], dim=0)
        
        #process augmentation
        states_now_aug   = self._augmentations(states_now)
        states_other_aug = self._augmentations(states_other)

        transition_pred = self.model_cnd_target.forward_aux(states_now_aug, states_other_aug)

        loss            = (transition_label - transition_pred)**2
        loss            = loss.mean() 

        #compute accuracy
        transition_pred_label = (transition_pred > 0.5).float()
        acc = (transition_label == transition_pred_label)
        
        
        print(transition_label.shape, transition_pred.shape)
        print(transition_label[0:10, 0])
        print(transition_pred_label[0:10, 0])
        print(acc[0:10, 0])
        print("\n\n")
        
        

        acc = 100.0*acc.float().mean()
        acc = acc.detach().to("cpu").numpy()

        return loss, acc
 
  

    #compute internal motivation
    def _curiosity(self, states):
        features_predicted_t    = self.model_cnd(states)
        features_target_t       = self.model_cnd_target(states)
 
        curiosity_t = ((features_target_t - features_predicted_t)**2).mean(dim=1)
  
        return curiosity_t
 


    def _augmentations(self, x): 
        if "inverse" in self.augmentations:
            x = aug_random_apply(x, self.augmentations_probs, aug_inverse)

        if "conv" in self.augmentations:
            x = aug_random_apply(x, self.augmentations_probs, aug_conv)

        if "pixelate" in self.augmentations:
            x = aug_random_apply(x, self.augmentations_probs, aug_pixelate)

        if "pixel_dropout" in self.augmentations:
            x = aug_random_apply(x, self.augmentations_probs, aug_pixel_dropout)
 
        if "random_tiles" in self.augmentations:
            x = aug_random_apply(x, self.augmentations_probs, aug_random_tiles)

        if "mask" in self.augmentations:
            x = aug_random_apply(x, self.augmentations_probs, aug_mask_tiles)

        if "noise" in self.augmentations:
            x = aug_random_apply(x, self.augmentations_probs, aug_noise)
        
        return x.detach() 

   