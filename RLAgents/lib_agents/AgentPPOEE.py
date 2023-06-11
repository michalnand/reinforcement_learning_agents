import numpy
import torch 

from .ValuesLogger      import *
from .PolicyBufferIM    import *  

from .PPOLoss               import *
from .SelfSupervisedLoss    import *
from .Augmentations         import *



class AgentPPOEE():   
    def __init__(self, envs, ModelPPO, ModelIM, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.envs = envs  
         
        self.gamma_ext          = config.gamma_ext 
        self.gamma_int          = config.gamma_int
              
        self.ext_adv_coeff      = config.ext_adv_coeff
        self.int_adv_coeff      = config.int_adv_coeff
        self.int_reward_coeff   = config.int_reward_coeff
        
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip 
    
        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count

        self.novelty_buffer_size = config.novelty_buffer_size

 
        #self supervised + AUX loss for PPO model
        if config.ppo_self_supervised_loss == "vicreg":
            self.ppo_self_supervised_loss = loss_vicreg
        else:
            self.ppo_self_supervised_loss = None

        if config.ppo_aux_loss == "action_loss":
            self.ppo_aux_loss = self._action_loss
        elif config.ppo_aux_loss == "constructor_loss":
             self.ppo_aux_loss = self._constructor_loss
        else:
            self.ppo_aux_loss = None

       

        #self supervised + AUX loss for internal motivation model
        if config.im_self_supervised_loss == "vicreg":
            self.im_self_supervised_loss = loss_vicreg
        else:
            self.im_self_supervised_loss = None

        if config.im_aux_loss == "action_loss":
            self.im_aux_loss = self._action_loss
        elif config.im_aux_loss == "constructor_loss":
             self.im_aux_loss = self._constructor_loss
        else:
            self.im_aux_loss = None


        if hasattr(config, "state_normalise"):
            self.state_normalise = config.state_normalise
        else:
            self.state_normalise = False
        
         
        self.augmentations                  = config.augmentations
        self.augmentations_probs            = config.augmentations_probs
        
        print("ppo_self_supervised_loss      = ", self.ppo_self_supervised_loss)
        print("ppo_aux_loss                 = ", self.ppo_aux_loss)

        print("im_self_supervised_loss       = ", self.im_self_supervised_loss)
        print("im_aux_loss                  = ", self.im_aux_loss)

        print("augmentations                = ", self.augmentations)
        print("augmentations_probs          = ", self.augmentations_probs)

        print("int_reward_coeff             = ", self.int_reward_coeff)
        print("state_normalise              = ", self.state_normalise)

        print("\n\n")

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.model_ppo.to(self.device)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)

        self.model_im       = ModelIM.Model(self.state_shape,  self.actions_count)
        self.model_im.to(self.device)
        self.optimizer_im   = torch.optim.Adam(self.model_im.parameters(), lr=config.learning_rate_im)
 
        self.policy_buffer = PolicyBufferIM(self.steps, self.state_shape, self.actions_count, self.envs_count)
 
        for e in range(self.envs_count):
            self.envs.reset(e)


        #internal motivation buffer and its entropy
        self.novelty_buffer     = torch.zeros((self.novelty_buffer_size, 512), requires_grad=False, device="cpu")
        self.novelty_buffer_ptr = 0 
        

        #reset envs and fill initial state
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e]  = self.envs.reset(e)

        
        self.state_mean  = self.states.mean(axis=0)
        self.state_var   = numpy.ones_like(self.state_mean, dtype=numpy.float32)

            
        self.enable_training()  
        self.iterations     = 0 

        self.values_logger  = ValuesLogger() 

        self.values_logger.add("internal_motivation_mean",      0.0)
        self.values_logger.add("internal_motivation_std" ,      0.0)

        self.values_logger.add("loss_ppo_actor",                0.0)
        self.values_logger.add("loss_ppo_critic",               0.0)

        self.values_logger.add("loss_ppo_self_supervised",      0.0)
        self.values_logger.add("loss_ppo_aux",                  0.0)
        self.values_logger.add("acc_ppo_self_supervised",       0.0)
        self.values_logger.add("acc_ppo_aux",                   0.0)

        self.values_logger.add("loss_im_self_supervised",       0.0)
        self.values_logger.add("loss_im_aux",                   0.0)
        self.values_logger.add("acc_im_self_supervised",        0.0)
        self.values_logger.add("acc_im_aux",                    0.0)

        self.values_logger.add("im_magnitude",                  0.0)
        self.values_logger.add("im_magnitude_std",              0.0)


        self.vis_features = []
        self.vis_labels   = []


         

        

    def enable_training(self): 
        self.enabled_training = True
 
    def disable_training(self):
        self.enabled_training = False
 
    def main(self): 
        #normalise state
        if self.state_normalise:
            if self.enabled_training:
                states = self._state_normalise(self.states)
            else:
                states = self._state_normalise(self.states, 1.0)
        else:
            states = self.states
        
        #state to tensor
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        #compute model output
        logits, values_ext, values_int  = self.model_ppo.forward(states)
        
        #collect actions 
        actions = self._sample_actions(logits)
        
        #execute action
        states_new, rewards_ext, dones, infos = self.envs.step(actions)

        #curiosity motivation
        rewards_int = self._internal_motivation(states)
        
        rewards_int   = torch.clip(self.int_reward_coeff*rewards_int, 0.0, 1.0)
        
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
                self.states[e]  = self.envs.reset(e)
               
                 
        #collect stats
        self.values_logger.add("internal_motivation_mean", rewards_int.mean().detach().to("cpu").numpy())
        self.values_logger.add("internal_motivation_std" , rewards_int.std().detach().to("cpu").numpy())

        self.iterations+= 1

        return rewards_ext[0], dones[0], infos[0]
    
    def save(self, save_path):
        torch.save(self.model_ppo.state_dict(), save_path + "trained/model_ppo.pt")
        torch.save(self.model_im.state_dict(), save_path + "trained/model_im.pt")

        with open(save_path + "trained/" + "state_mean_var.npy", "wb") as f:
            numpy.save(f, self.state_mean)
            numpy.save(f, self.state_var)

    def load(self, load_path):
        self.model_ppo.load_state_dict(torch.load(load_path + "trained/model_ppo.pt", map_location = self.device))
        self.model_im.load_state_dict(torch.load(load_path + "trained/model_im.pt", map_location = self.device))
       
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
        self.policy_buffer.compute_returns(self.gamma_ext, self.gamma_int)

        batch_count = self.steps//self.batch_size

        small_batch = 16*self.batch_size 

        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int = self.policy_buffer.sample_batch(self.batch_size, self.device)

                #actor + critic PPO loss
                loss_ppo     = self._loss_ppo(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int)

                #features self supervised loss
                if self.ppo_self_supervised_loss is not None:
                    #sample smaller batch for for self supervised loss
                    states_a, states_b, states_c, action = self.policy_buffer.sample_states_action_pairs(small_batch, self.device)

                    loss_ppo_self_supervised , _, _, acc_ppo_self_supervised = self.ppo_self_supervised_loss(self.model_ppo, states_a, states_a, self._augmentations)                
                else:
                    loss_ppo_self_supervised    = torch.zeros((1, ))[0]
                    acc_ppo_self_supervised     = 0.0


                if self.ppo_aux_loss is not None:
                    #sample smaller batch for semi supervised aux loss
                    states_a, states_b, states_c, action = self.policy_buffer.sample_states_action_pairs(small_batch, self.device)

                    loss_ppo_aux , acc_ppo_aux = self.ppo_aux_loss(self.model_ppo, states_a, states_b, states_c, action)  
                else:
                    loss_ppo_aux    = torch.zeros((1, ))[0]
                    acc_ppo_aux     = 0.0
                
                #final loss
                loss = loss_ppo + loss_ppo_self_supervised + 0.01*loss_ppo_aux

                #PPO model training
                self.optimizer_ppo.zero_grad()        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()


                #internal motivation model training

                #features self supervised loss
                if self.im_self_supervised_loss is not None:
                    #sample smaller batch for for self supervised loss
                    states_a, states_b, states_c, action = self.policy_buffer.sample_states_action_pairs(small_batch, self.device)

                    loss_im_self_supervised , im_features_mag, im_features_std, acc_im_self_supervised = self.im_self_supervised_loss(self.model_im, states_a, states_a, self._augmentations)                
                else:
                    loss_im_self_supervised    = torch.zeros((1, ), device=self.device)[0]
                    acc_im_self_supervised     = 0.0


                if self.im_aux_loss is not None:
                    #sample smaller batch for semi supervised aux loss
                    states_a, states_b, states_c, action = self.policy_buffer.sample_states_action_pairs(small_batch, self.device)

                    loss_im_aux, acc_im_aux = self.im_aux_loss(self.model_im, states_a, states_b, states_c, action)                
                else:
                    loss_im_aux    = torch.zeros((1, ), device=self.device)[0]
                    acc_im_aux     = 0.0
 
 
                #final loss for internal motivation model
                loss_im = loss_im_self_supervised + 0.01*loss_im_aux

                self.optimizer_im.zero_grad() 
                loss_im.backward()
                self.optimizer_im.step()

                #log results
                
                self.values_logger.add("loss_ppo_self_supervised",      loss_ppo_self_supervised.detach().to("cpu").numpy())
                self.values_logger.add("loss_ppo_aux",                  loss_ppo_aux.detach().to("cpu").numpy())
                self.values_logger.add("acc_ppo_self_supervised",       acc_ppo_self_supervised)
                self.values_logger.add("acc_ppo_aux",                   acc_ppo_aux)

                self.values_logger.add("loss_im_self_supervised",       loss_im_self_supervised.detach().to("cpu").numpy())
                self.values_logger.add("loss_im_aux",                   loss_im_aux.detach().to("cpu").numpy())
                self.values_logger.add("acc_im_self_supervised",        acc_im_self_supervised)
                self.values_logger.add("acc_im_aux",                    acc_im_aux)

                self.values_logger.add("im_magnitude",                  im_features_mag)
                self.values_logger.add("im_magnitude_std" ,             im_features_std)

            

        self.policy_buffer.clear() 

    
    def _loss_ppo(self, states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int):
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
        self.values_logger.add("loss_ppo_actor", loss_actor.mean().detach().to("cpu").numpy())
        self.values_logger.add("loss_ppo_critic", loss_critic.mean().detach().to("cpu").numpy())

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
    

    def _internal_motivation(self, states):
        #get features from model
        features  = self.model_im(states)
        features  = features.detach().to("cpu")

        '''
        d_mean = torch.zeros((self.envs_count, ), dtype=torch.float32)
        d_var  = torch.zeros((self.envs_count, ), dtype=torch.float32)
        

        #find statistics
        for i in range(features.shape[0]):
            d              = ((features[i].unsqueeze(0) - self.novelty_buffer)**2).mean(dim=-1)
            d_mean[i]      = torch.mean(d)
            d_var[i]       = torch.var(d)
        '''

        d = torch.cdist(features, self.novelty_buffer)
        d_mean = torch.mean(d, dim=1)
        d_var  = torch.var(d, dim=1)

        #add new features into buffer
        for i in range(features.shape[0]):
            self.novelty_buffer[self.novelty_buffer_ptr] = features[i]
            self.novelty_buffer_ptr = (self.novelty_buffer_ptr + 1)%self.novelty_buffer.shape[0]

        result = d_mean/(d_var + 10**-10) 

        return result.to("cpu")
    