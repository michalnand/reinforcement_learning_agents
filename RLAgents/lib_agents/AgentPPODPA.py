import numpy
import torch 

from .ValuesLogger      import *
from .PolicyBufferIMNew import *  

from .PPOLoss               import *
from .SelfSupervised        import * 
from .Augmentations         import *
  

class AgentPPODPA():   
    def __init__(self, envs, ModelPPO, ModelIM, config):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.envs = envs  
          
        self.gamma_ext          = config.gamma_ext 
        self.gamma_int          = config.gamma_int
              
        self.ext_adv_coeff      = config.ext_adv_coeff
        self.int_adv_coeff      = config.int_adv_coeff
 
        self.reward_int_coeff_a   = config.reward_int_coeff_a
        self.reward_int_coeff_b   = config.reward_int_coeff_b

        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip 
    
        self.steps              = config.steps
        self.batch_size         = config.batch_size  
        self.ss_batch_size      = config.ss_batch_size      

        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count

                
        if config.ppo_self_supervised_loss == "vicreg":
            self._ppo_self_supervised_loss = loss_vicreg
        else:
            self._ppo_self_supervised_loss = None

        if config.target_self_supervised_loss == "vicreg":
            self._target_self_supervised_loss = loss_vicreg
        elif config.target_self_supervised_loss == "vicreg_jepa":
            self._target_self_supervised_loss = loss_vicreg_jepa 
        elif config.target_self_supervised_loss == "vicreg_jepa_cross":
            self._target_self_supervised_loss = loss_vicreg_jepa_cross
        else:
            self._target_self_supervised_loss = None


        self.similar_states_distance = config.similar_states_distance
        
        self.state_normalise      = config.state_normalise
        self.int_reward_normalise = config.int_reward_normalise
        
        self.augmentations                  = config.augmentations
        self.augmentations_probs            = config.augmentations_probs
        
        print("ppo_self_supervised_loss              = ", self._ppo_self_supervised_loss)
        print("target_self_supervised_loss           = ", self._target_self_supervised_loss)
        print("augmentations                         = ", self.augmentations)
        print("augmentations_probs                   = ", self.augmentations_probs)
        print("reward_int_coeff_a                    = ", self.reward_int_coeff_a)
        print("reward_int_coeff_b                    = ", self.reward_int_coeff_b)
        print("similar_states_distance               = ", self.similar_states_distance)
        print("state_normalise                       = ", self.state_normalise)
        print("int_reward_normalise                  = ", self.int_reward_normalise)

        print("\n\n")

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        #main ppo agent
        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.model_ppo.to(self.device)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)

        #IM model
        self.model_im      = ModelIM.Model(self.state_shape)
        self.model_im.to(self.device)
        self.optimizer_im  = torch.optim.Adam(self.model_im.parameters(), lr=config.learning_rate_im)
    
        self.policy_buffer = PolicyBufferIMNew(self.steps, self.state_shape, self.actions_count, self.envs_count)

     
        #optional, for state mean and variance normalisation        
        self.state_mean  = numpy.zeros(self.state_shape, dtype=numpy.float32)

        for e in range(self.envs_count):
            state, _ = self.envs.reset(e)
            self.state_mean+= state.copy()

        self.state_mean/= self.envs_count
        self.state_var = numpy.ones(self.state_shape,  dtype=numpy.float32)

        #optional int reward normalisation
        self.reward_mean = 0.0
        self.reward_var  = 1.0


        self.state_prev  = torch.zeros((self.envs_count, ) + self.state_shape, dtype=torch.float32, device=self.device)

        self.iterations = 0 

        self.values_logger  = ValuesLogger() 

        self.values_logger.add("internal_motivation_mean_a", 0.0)
        self.values_logger.add("internal_motivation_std_a" , 0.0)

        self.values_logger.add("internal_motivation_mean_b", 0.0)
        self.values_logger.add("internal_motivation_std_b" , 0.0)
        
        self.values_logger.add("loss_ppo_actor",  0.0)
        self.values_logger.add("loss_ppo_critic", 0.0)
        
        self.values_logger.add("loss_ppo_self_supervised", 0.0)
        self.values_logger.add("loss_target_self_supervised", 0.0)
        self.values_logger.add("loss_distillation", 0.0)
        self.values_logger.add("loss_prediction", 0.0)

        self.info_logger = {} 

   
 
    def round_start(self): 
        pass

    def round_finish(self): 
        pass

    def episode_done(self, env_idx):
        pass

    def step(self, states, training_enabled, legal_actions_mask):        
        #normalise state
        states_norm = self._state_normalise(states, training_enabled)
        
        #state to tensor
        states_t = torch.tensor(states_norm, dtype=torch.float).to(self.device)

        #compute model output
        logits_t, values_ext_t, values_int_t  = self.model_ppo.forward(states_t)
        
        #collect actions 
        actions = self._sample_actions(logits_t, legal_actions_mask)
        
        #execute action
        states_new, rewards_ext, dones, _, infos = self.envs.step(actions)

        #internal motivation
        rewards_int_a, rewards_int_b, _ = self._internal_motivation(self.state_prev, states_t)

        self.state_prev = states_t.clone()

        #weighting and clipping im
        rewards_int = self.reward_int_coeff_a*rewards_int_a + self.reward_int_coeff_b*rewards_int_b
        rewards_int = rewards_int.detach().cpu()

        if self.int_reward_normalise:
            rewards_int = self._reward_normalise(rewards_int) 
        else:
            rewards_int = torch.clip(rewards_int, 0.0, 1.0)
        
        #put into policy buffer
        if training_enabled:
            states_t        = states_t.detach().to("cpu")
            logits_t        = logits_t.detach().to("cpu")
            values_ext_t    = values_ext_t.squeeze(1).detach().to("cpu") 
            values_int_t    = values_int_t.squeeze(1).detach().to("cpu")
            actions         = torch.from_numpy(actions).to("cpu")
            rewards_ext_t   = torch.from_numpy(rewards_ext).to("cpu")
            rewards_int_t   = rewards_int.detach().to("cpu")
            dones           = torch.from_numpy(dones).to("cpu")
            

            self.policy_buffer.add(states_t, logits_t, values_ext_t, values_int_t, actions, rewards_ext_t, rewards_int_t, dones)

            if self.policy_buffer.is_full():
                self.train()
                

        #collect stats
        self.values_logger.add("internal_motivation_mean_a", rewards_int_a.mean().detach().to("cpu").numpy())
        self.values_logger.add("internal_motivation_std_a" , rewards_int_a.std().detach().to("cpu").numpy())

        self.values_logger.add("internal_motivation_mean_b", rewards_int_b.mean().detach().to("cpu").numpy())
        self.values_logger.add("internal_motivation_std_b" , rewards_int_b.std().detach().to("cpu").numpy())
        
        self.iterations+= 1

        return states_new, rewards_ext, dones, infos
   
    
    def save(self, save_path):
        torch.save(self.model_ppo.state_dict(), save_path + "trained/model_ppo.pt")
        torch.save(self.model_im.state_dict(), save_path + "trained/model_im.pt")
    
        if self.state_normalise:
            with open(save_path + "trained/" + "state_mean_var.npy", "wb") as f:
                numpy.save(f, self.state_mean)
                numpy.save(f, self.state_var)
        
    def load(self, load_path):
        self.model_ppo.load_state_dict(torch.load(load_path + "trained/model_ppo.pt", map_location = self.device))
        self.model_im.load_state_dict(torch.load(load_path + "trained/model_im.pt", map_location = self.device))
        
        if self.state_normalise:
            with open(load_path + "trained/" + "state_mean_var.npy", "rb") as f:
                self.state_mean = numpy.load(f) 
                self.state_var  = numpy.load(f)
    
    def get_log(self): 
        return self.values_logger.get_str() + str(self.info_logger)

    def _sample_actions(self, logits, legal_actions_mask):
        legal_actions_mask_t  = torch.from_numpy(legal_actions_mask).to(self.device).float()

        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)

        #keep only legal actions probs, and renormalise probs
        action_probs_t        = action_probs_t*legal_actions_mask_t
        action_probs_t        = action_probs_t/action_probs_t.sum(dim=-1).unsqueeze(1)

        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().to("cpu").numpy()
        return actions
    
    def train(self): 
        self.policy_buffer.compute_returns(self.gamma_ext, self.gamma_int)
        
        samples_count = self.steps*self.envs_count
        batch_count = samples_count//self.batch_size

        #print("ppo_samples = ", self.training_epochs*batch_count*self.batch_size)

        #PPO training
        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int, _ = self.policy_buffer.sample_batch(self.batch_size, self.device)
                
                #train PPO model
                loss_ppo     = self._loss_ppo(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int)
                
                #train ppo features, self supervised
                if self._ppo_self_supervised_loss is not None:
                    #sample smaller batch for self supervised loss
                    states_now, states_similar = self.policy_buffer.sample_states_pairs(self.ss_batch_size, 0, self.device)

                    loss_ppo_self_supervised, ppo_ssl = self._ppo_self_supervised_loss(self.model_ppo.forward_features, self._augmentations, states_now, states_similar)  
                    self.info_logger["ppo_ssl"] = ppo_ssl
                else:
                    loss_ppo_self_supervised    = torch.zeros((1, ), device=self.device)[0]

                #total PPO loss
                loss = loss_ppo + loss_ppo_self_supervised

                self.optimizer_ppo.zero_grad()        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()

                self.values_logger.add("loss_ppo_self_supervised", loss_ppo_self_supervised.detach().cpu().numpy())
        
        
        #IM model training
        batch_count = (samples_count//self.ss_batch_size)//2

        #print("ssl_samples = ", batch_count*self.ss_batch_size)

        for batch_idx in range(batch_count):
            #sample smaller batch for self supervised loss

            states_now, states_similar = self.policy_buffer.sample_states_pairs(self.ss_batch_size, self.similar_states_distance, self.device)
            loss_target_self_supervised, im_ssl  = self._target_self_supervised_loss(self.model_im.forward_self_supervised, self._augmentations, states_now, states_similar)                

            self.info_logger["im_ssl"] = im_ssl
            
            #train distillation and prediction
            states, states_next = self.policy_buffer.sample_states_next_states(self.batch_size, self.device)
            loss_distillation, loss_prediction, loss_hidden, h = self._loss_im(states, states_next)


            h_mag     = round(((h**2).mean()).detach().cpu().numpy().item(), 6)
            h_mag_std = round(((h**2).std()).detach().cpu().numpy().item(), 6)

            self.info_logger["predictor"] = [h_mag, h_mag_std]
            

            #total loss for im model
            loss_im = loss_target_self_supervised + loss_distillation + loss_prediction + 0.01*loss_hidden
            
            self.optimizer_im.zero_grad() 
            loss_im.backward()
            self.optimizer_im.step()

            #log results
            self.values_logger.add("loss_target_self_supervised", loss_target_self_supervised.detach().cpu().numpy())
            self.values_logger.add("loss_distillation", loss_distillation.detach().cpu().numpy())
            self.values_logger.add("loss_prediction", loss_prediction.detach().cpu().numpy())
            
        self.policy_buffer.clear() 

        


    def _loss_ppo(self, states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int):

        logits_new, values_ext_new, values_int_new  = self.model_ppo.forward(states)

        #critic loss
        loss_critic = ppo_compute_critic_loss(values_ext_new, returns_ext, values_int_new, returns_int)

        #actor loss        
        advantages  = self.ext_adv_coeff*advantages_ext + self.int_adv_coeff*advantages_int
        advantages  = advantages.detach() 

        #advantages normalisation 
        advantages_norm  = (advantages - advantages.mean())/(advantages.std() + 1e-8)

        #PPO main actor loss
        loss_policy, loss_entropy = ppo_compute_actor_loss(logits, logits_new, advantages_norm, actions, self.eps_clip, self.entropy_beta)

        loss_actor = loss_policy + loss_entropy

        #total loss
        loss = 0.5*loss_critic + loss_actor

        #store to log
        self.values_logger.add("loss_ppo_actor", loss_actor.mean().detach().to("cpu").numpy())
        self.values_logger.add("loss_ppo_critic", loss_critic.mean().detach().to("cpu").numpy())

        return loss 

    #compute internal motivations
    def _internal_motivation(self, states_prev, states_now):         
        #distillation novelty detection 
        z_target    = self.model_im.forward_target(states_now).detach()
        z_predictor = self.model_im.forward_predictor(states_now)

        distillation_novelty = ((z_target - z_predictor)**2).mean(dim=1)

        #prediction novelty detection
        z_target_prev = self.model_im.forward_target(states_prev).detach()
        z_pred, h = self.model_im.forward_state_predictor(z_target_prev, z_target)

        prediction_novelty = ((z_target - z_pred)**2).mean(dim=1)

        return distillation_novelty, prediction_novelty, h
 
 
    def _loss_im(self, states_prev, states_now):  
        distillation_novelty, prediction_novelty, h = self._internal_motivation(states_prev, states_now)

        loss_distillation = distillation_novelty.mean()
        loss_prediction   = prediction_novelty.mean()

        #hidden information loss, enforce sparsity, and minimize batch-wise variance
        loss_hidden = torch.abs(h).mean() + (h.std(dim=0)).mean()
  
        return loss_distillation, loss_prediction, loss_hidden, h

   
    def _augmentations(self, x): 
        mask_result = torch.zeros((4, x.shape[0]), device=x.device, dtype=torch.float32)

        if "pixelate" in self.augmentations:
            x, mask = aug_random_apply(x, self.augmentations_probs, aug_pixelate)
            mask_result[0] = mask

        if "random_tiles" in self.augmentations:
            x, mask = aug_random_apply(x, self.augmentations_probs, aug_random_tiles)
            mask_result[1] = mask

        if "noise" in self.augmentations:
            x, mask = aug_random_apply(x, self.augmentations_probs, aug_noise)
            mask_result[2] = mask

        if "inverse" in self.augmentations:
            x, mask = aug_random_apply(x, self.augmentations_probs, aug_inverse)
            mask_result[3] = mask
 
        return x.detach(), mask_result 
    

    def _state_normalise(self, states, training_enabled, alpha = 0.99): 

        if self.state_normalise:
            #update running stats only during training
            if training_enabled:
                mean = states.mean(axis=0)
                self.state_mean = alpha*self.state_mean + (1.0 - alpha)*mean
        
                var = ((states - mean)**2).mean(axis=0)
                self.state_var  = alpha*self.state_var + (1.0 - alpha)*var 
             
            #normalise mean and variance
            states_norm = (states - self.state_mean)/(numpy.sqrt(self.state_var) + 10**-6)
            states_norm = numpy.clip(states_norm, -4.0, 4.0)
        
        else:
            states_norm = states
        
        return states_norm
    

    def _reward_normalise(self, rewards, alpha = 0.99): 
        #update running stats
        mean = rewards.mean() 
        self.reward_mean = alpha*self.reward_mean + (1.0 - alpha)*mean

        var = ((rewards - mean)**2).mean()
        self.reward_var  = alpha*self.reward_var + (1.0 - alpha)*var 
            
        #normalise mean and variance
        rewards_result = rewards/(numpy.sqrt(self.reward_var) + 10**-6)
        rewards_result = numpy.clip(rewards_result, -4.0, 4.0)
      
        return rewards_result
   