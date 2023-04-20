import numpy
import torch
import time

from .ValuesLogger      import *
from .PolicyBuffer      import *


from .PPOLoss               import *
from .Augmentations         import *
from .SelfSupervisedLoss    import *


 
class AgentPPOSA():
    def __init__(self, envs, ModelPPO, ModelIM, config):
        self.envs = envs

        self.gamma_ext          = config.gamma_ext
        self.gamma_int          = config.gamma_int
        self.int_reward_coeff   = config.int_reward_coeff

        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip
 
        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count

      
        self.state_normalise = config.state_normalise

        if config.self_supervised_loss == "vicreg":
            self._self_supervised_loss = loss_vicreg
        else:
            self._self_supervised_loss = None

        if config.self_aware_loss == "action_loss":
            self._self_aware_loss = self._action_loss
            aux_count             = self.actions_count
        elif config.self_aware_loss == "constructor_loss":
            self._self_aware_loss = self._constructor_loss
            aux_count             = 3
        else:
            self._self_aware_loss = None
            aux_count             = 1

        self.self_supervised_loss_coeff     = config.self_supervised_loss_coeff
        self.self_aware_loss_coeff          = config.self_aware_loss_coeff
        self.im_self_supervised_loss_coeff  = config.im_self_supervised_loss_coeff
        self.im_self_aware_loss_coeff       = config.im_self_aware_loss_coeff
         
        self.augmentations                  = config.augmentations
        self.augmentations_probs            = config.augmentations_probs

        self.im_buffer_size                 = config.im_buffer_size

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        self.model_ppo          = ModelPPO.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo      = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)

        features_count          = 256
        
        self.model_im           = ModelIM.Model(self.state_shape, features_count, aux_count)
        self.optimizer_im       = torch.optim.Adam(self.model_im.parameters(), lr=config.learning_rate_im)
 


        self.policy_buffer      = PolicyBuffer(self.steps, self.state_shape, self.actions_count, self.envs_count)
        self.im_buffer          = numpy.zeros((self.im_buffer_size, self.envs_count, features_count), dtype=numpy.float32)
        self.im_ptr             = 0

        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e] = self.envs.reset(e) 

        self.state_mean  = self.states.mean(axis=0)
        self.state_var   = numpy.ones_like(self.state_mean, dtype=numpy.float32)

        for e in range(self.envs_count):
            self._im_buffer_clear(e)

        self.enable_training()
        self.iterations = 0 


        print("self_supervised_loss                 = ", self._self_supervised_loss)
        print("self_aware_loss                      = ", self._self_aware_loss)
        print("augmentations                        = ", self.augmentations)
        print("augmentations_probs                  = ", self.augmentations_probs)
        print("self_supervised_loss_coeff           = ", self.self_supervised_loss_coeff)
        print("self_aware_loss_coeff                = ", self.self_aware_loss_coeff)
        print("im_self_supervised_loss_coeff_im     = ", self.im_self_supervised_loss_coeff)
        print("im_self_aware_loss_coeff             = ", self.im_self_aware_loss_coeff)
        print("state_normalise                      = ", self.state_normalise)

        print("\n\n")

        self.values_logger  = ValuesLogger()

        self.values_logger.add("loss_actor", 0.0)
        self.values_logger.add("loss_critic", 0.0)

        self.values_logger.add("loss_self_supervised", 0.0)
        self.values_logger.add("loss_self_aware", 0.0)

        self.values_logger.add("im_loss_self_supervised", 0.0)
        self.values_logger.add("im_loss_self_aware", 0.0)

        self.values_logger.add("ss_accuracy", 0.0)
        self.values_logger.add("sa_accuracy", 0.0)

        self.values_logger.add("im_ss_accuracy", 0.0)
        self.values_logger.add("im_sa_accuracy", 0.0)

        

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

    def main(self):        
        #normalise state
        if self.state_normalise:
            states = self._state_normalise(self.states)
        else:
            states = self.states
        
        states  = torch.tensor(states, dtype=torch.float).detach().to(self.model.device)
    
        logits, values_ext, values_int = self.model.forward(states)

        #internal motivation
        rewards_int    = self._internal_motivation(states)
        rewards_int    = torch.clip(self.int_reward_coeff*rewards_int, -1.0, 1.0)
        
 
        actions = self._sample_actions(logits)
        
        states_new, rewards_ext, dones, infos = self.envs.step(actions)

        
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

    
        self.states = states_new.copy()
        for e in range(self.envs_count):
            if dones[e]:
                self.states[e] = self.envs.reset(e)

                self._im_buffer_clear(e)
           
        self.iterations+= 1
        return rewards_ext[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_im.save(save_path + "trained/")

        with open(save_path + "trained/" + "state_mean_var.npy", "wb") as f:
            numpy.save(f, self.state_mean)
            numpy.save(f, self.state_var)


    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
        self.model_im.load(load_path + "trained/")

        with open(load_path + "trained/" + "state_mean_var.npy", "rb") as f:
            self.state_mean = numpy.load(f)
            self.state_var  = numpy.load(f) 
 

    def get_log(self):
        return self.values_logger.get_str()
    
    def _sample_actions(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().to("cpu").numpy()
        return actions
    
    def train(self): 
        self.policy_buffer.compute_returns(self.gamma)

        batch_count = self.steps//self.batch_size
        small_batch = 16*self.batch_size 

        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):

                states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)

                #train PPO model
                loss_ppo     = self._loss_ppo(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int)
                
                
                #sample smaller batch for self-supervised regularization
                states_a, states_b, states_c, action = self.policy_buffer.sample_states_action_pairs(small_batch, self.model_ppo.device)

                #self supervised regularisation   
                if self._self_supervised_loss is not None:
                    loss_self_supervised, _, _, ss_accuracy = self._self_supervised_loss(self.model_ppo, states_a, states_a, self._augmentations)
                else:
                    loss_self_supervised = 0.0
                    ss_accuracy = 0.0

                #self aware loss 
                if self._self_aware_loss is not None:
                    loss_self_aware, sa_accuracy = self._self_aware_loss(self.model_ppo, states_a, states_b, states_c, action)                 
                else:
                    loss_self_aware     = 0.0
                    sa_accuracy         = 0.0


                self.values_logger.add("loss_self_supervised", loss_self_supervised.detach().to("cpu").numpy())
                self.values_logger.add("loss_self_aware", loss_self_aware.detach().to("cpu").numpy())

                self.values_logger.add("ss_accuracy", ss_accuracy)
                self.values_logger.add("sa_accuracy", sa_accuracy)


                loss = loss_ppo + self.self_supervised_loss_coeff*loss_self_supervised + self.self_aware_loss_coeff*loss_self_aware

                self.optimizer_ppo.zero_grad()        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step() 



                #train IM model for regularization

                if self._self_supervised_loss is not None:
                    im_loss_self_supervised, _, _, im_ss_accuracy = self._self_supervised_loss(self.model_im, states_a, states_a, self._augmentations)                
                else:
                    im_loss_self_supervised = torch.zeros((1, ), dtype=torch.float32, device=self.model_ppo.device)
                    im_ss_accuracy          = 0.0
 
                #optional auxliary loss
                #e.g. inverse model : action prediction from two consectuctive states
                if self._self_aware_loss is not None:
                    im_loss_self_aware, im_sa_accuracy = self._self_aware_loss(states_a, states_b, states_c, action)                 
                else:
                    im_loss_self_aware = torch.zeros((1, ), dtype=torch.float32, device=self.model_ppo.device)
                    im_sa_accuracy     = 0.0

 
                #final loss for target model
                loss_im = self.im_self_supervised_loss_coeff*im_loss_self_supervised + self.im_self_aware_loss_coeff*im_loss_self_aware

                self.optimizer_im.zero_grad() 
                loss_im.backward()
                self.optimizer_im.step()


                self.values_logger.add("im_loss_self_supervised", im_loss_self_supervised.detach().to("cpu").numpy())
                self.values_logger.add("loss_self_aware", loss_self_aware.detach().to("cpu").numpy())

                self.values_logger.add("im_ss_accuracy", im_ss_accuracy)
                self.values_logger.add("im_sa_accuracy", im_sa_accuracy)


        self.policy_buffer.clear()   

   

    
    def _loss_ppo(self, states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int):
        logits_new, values_ext_new, values_int_new  = self.model_ppo.forward(states)

        #critic loss
        loss_critic =  ppo_compute_critic_loss(values_ext_new, returns_ext, values_int_new, returns_int)

        #actor loss        
        advantages  = 2.0*advantages_ext + 1.0*advantages_int
        advantages  = advantages.detach() 

        loss_policy, loss_entropy  = ppo_compute_actor_loss(logits, logits_new, advantages, actions, self.eps_clip, self.entropy_beta)

        loss_actor = loss_policy + loss_entropy

        #total loss
        loss = 0.5*loss_critic + loss_actor

        #store to log
        self.values_logger.add("loss_actor",  loss_actor.mean().detach().to("cpu").numpy())
        self.values_logger.add("loss_critic", loss_critic.mean().detach().to("cpu").numpy())

        return loss 

    
     #inverse model for action prediction
    def _action_loss(self, model, states_now, states_next, states_random, action):
        action_pred     = model.forward_aux(states_now, states_next)

        action_one_hot  = torch.nn.functional.one_hot(action, self.actions_count).to(states_now.device)

        loss            =  ((action_one_hot - action_pred)**2).mean()

        #compute accuracy
        pred = torch.argmax(action_pred.detach(), dim=1)
        acc = 100.0*(pred == action).float().mean()
        acc = acc.detach().to("cpu").numpy()

        return loss, acc

    #constructor theory loss
    #inverse model for action prediction
    def _constructor_loss(self, model, states_now, states_next, states_random, action):
        batch_size          = states_now.shape[0]

        #0 : state_now,  state_random, two different states
        #1 : state_now,  state_next, two consecutive states
        #2 : state_next, state_now, two inverted consecutive states
        labels                   = torch.randint(0, 3, (batch_size, )).to(states_now.device)
        transition_label_one_hot = torch.nn.functional.one_hot(labels, 3)

        #mix states
        select  = labels.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        sa      = (select == 0)*states_now    + (select == 1)*states_now  + (select == 2)*states_next
        sb      = (select == 0)*states_random + (select == 1)*states_next + (select == 2)*states_now


        #process augmentation
        sa_aug  = self._augmentations(sa)
        sb_aug  = self._augmentations(sb)

        transition_pred = model.forward_aux(sa_aug, sb_aug)

        loss            = ((transition_label_one_hot - transition_pred)**2).mean()
        
        #compute accuracy
        #compute accuracy
        labels_pred = torch.argmax(transition_pred.detach(), dim=1)
        acc = 100.0*(labels == labels_pred).float().mean()
        acc = acc.detach().to("cpu").numpy()

        return loss, acc
 
  


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

   
    '''
    differential entropy intrinsic motivation
    postivie entropy change - more new states discovered
    negative entropy change - visting repeating old states
    '''
    def _internal_motivation(self, states):
        
        entropy_prev = numpy.std(self.im_buffer, axis=0)

        features = self.model_im.forward(states)
        self.im_buffer[self.im_ptr] = features.detach().to("cpu").numpy()
        self.im_ptr = (self.im_ptr + 1)%self.im_buffer_size

        entropy_now  = numpy.std(self.im_buffer, axis=0)

        d_entropy    = entropy_now - entropy_prev

        d_entropy    = d_entropy.mean(axis=1)

        return d_entropy
    

    def _im_buffer_clear(self, env_id):
        state    = torch.from_numpy(self.states).to(self.model_im.device).unsqueeze(0)
        features = self.model_im.forward(state)
        features = features.squeeze(0).detach().to("cpu").numpy()

        self.im_buffer[:, env_id, :] = features

    def _state_normalise(self, states, alpha = 0.99): 
        #update running stats
        mean = states.mean(axis=0)
        self.state_mean = alpha*self.state_mean + (1.0 - alpha)*mean
 
        var = ((states - mean)**2).mean(axis=0)
        self.state_var  = alpha*self.state_var + (1.0 - alpha)*var 
        
        #normalise mean and variance
        states_norm = (states - self.state_mean)/(numpy.sqrt(self.state_var) + 10**-6)
        states_norm = numpy.clip(states_norm, -4.0, 4.0)
        
        return states_norm