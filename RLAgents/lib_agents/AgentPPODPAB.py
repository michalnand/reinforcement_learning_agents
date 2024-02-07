import numpy
import torch 

from .ValuesLogger      import *
from .PolicyBufferIMNew import *  

from .PPOLoss               import *
from .SelfSupervised        import * 
from .Augmentations         import *
  

class AgentPPODPAB():   
    def __init__(self, envs, Model, config):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.envs = envs  
          
        self.gamma_ext          = config.gamma_ext 
        self.gamma_int          = config.gamma_int
              
        self.ext_adv_coeff      = config.ext_adv_coeff
        self.int_adv_coeff      = config.int_adv_coeff
 
        self.reward_int_coeff_a = config.reward_int_coeff_a
        self.reward_int_coeff_b = config.reward_int_coeff_b
        self.hidden_coeff       = config.hidden_coeff

        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip 
    
        self.steps              = config.steps
        self.batch_size         = config.batch_size  
        self.ss_batch_size      = config.ss_batch_size       

        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count


        if config.self_supervised_loss == "vicreg_jepa":
            self._self_supervised_loss = loss_vicreg_jepa
        elif config.self_supervised_loss == "vicreg_jepa_cross":
            self._self_supervised_loss = loss_vicreg_jepa_cross
        else:
            self._self_supervised_loss = None 

        self.similar_states_distance = config.similar_states_distance
        self.state_normalise      = config.state_normalise
        
        self.augmentations                  = config.augmentations
        self.augmentations_probs            = config.augmentations_probs
        
        print("self_supervised_loss                  = ", self._self_supervised_loss)
        print("augmentations                         = ", self.augmentations)
        print("augmentations_probs                   = ", self.augmentations_probs)
        print("reward_int_coeff_a                    = ", self.reward_int_coeff_a)
        print("reward_int_coeff_b                    = ", self.reward_int_coeff_b)
        print("state_normalise                       = ", self.state_normalise)
        print("similar_states_distance               = ", self.similar_states_distance)

        print("\n\n")

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        #create model
        self.model      = Model.Model(self.state_shape, self.actions_count)
        self.model.to(self.device)
        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

      
        self.policy_buffer = PolicyBufferIMNew(self.steps, self.state_shape, self.actions_count, self.envs_count)

        print(self.model)
     
        #optional, for state mean and variance normalisation        
        self.state_mean  = numpy.zeros(self.state_shape, dtype=numpy.float32)

        for e in range(self.envs_count):
            state, _ = self.envs.reset(e)
            self.state_mean+= state.copy()

        self.state_mean/= self.envs_count
        self.state_var = numpy.ones(self.state_shape,  dtype=numpy.float32)

    
        self.state_prev  = torch.zeros((self.envs_count, ) + self.state_shape, dtype=torch.float32, device=self.device)

      
        self.iterations = 0 

        self.values_logger  = ValuesLogger() 

        self.values_logger.add("internal_motivation_mean", 0.0)
        self.values_logger.add("internal_motivation_std" , 0.0)

        self.values_logger.add("loss_ppo_actor",  0.0)
        self.values_logger.add("loss_ppo_critic", 0.0)
        self.values_logger.add("loss_self_supervised", 0.0)
        self.values_logger.add("loss_predictor", 0.0)

 
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
        logits_t, values_ext_t, values_int_t = self.model.forward(states_t)
        
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
        self.values_logger.add("internal_motivation_mean", rewards_int.mean().detach().to("cpu").numpy())
        self.values_logger.add("internal_motivation_std" , rewards_int.std().detach().to("cpu").numpy())

        
        
        self.iterations+= 1

        return states_new, rewards_ext, dones, infos
   
    
    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path + "trained/model.pt")
    
        if self.state_normalise:
            with open(save_path + "trained/" + "state_mean_var.npy", "wb") as f:
                numpy.save(f, self.state_mean)
                numpy.save(f, self.state_var)
        
    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path + "trained/model.pt", map_location = self.device))
        
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
                
                # PPO model loss
                loss_ppo = self._loss_ppo(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int)
                
                # sample batch pair for self supervised loss
                states_now, states_similar = self.policy_buffer.sample_states_pairs(self.ss_batch_size, self.similar_states_distance, self.device)

                # train features, self supervised
                loss_self_supervised, im_ssl = self._self_supervised_loss(self.model.forward_self_supervised, self._augmentations, states_now, states_similar, self.hidden_coeff)                

                # future state prediction loss
                states, states_next = self.policy_buffer.sample_states_next_states(self.batch_size, self.device)
                loss_predictor, loss_hidden, h = self._loss_predictor(states, states_next)
                


                #total loss
                loss = loss_ppo + loss_self_supervised + loss_predictor + self.hidden_coeff*loss_hidden

                self.optimizer.zero_grad()        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                #log results
                self.info_logger["im_ssl"] = im_ssl

                h_mag     = round(((h**2).mean()).detach().cpu().numpy().item(), 6)
                h_mag_std = round(((h**2).std()).detach().cpu().numpy().item(), 6)

                self.info_logger["predictor"] = [h_mag, h_mag_std]
            

                self.values_logger.add("loss_self_supervised", loss_self_supervised.detach().cpu().numpy())
                self.values_logger.add("loss_predictor", loss_predictor.detach().cpu().numpy())

    
        self.policy_buffer.clear() 

        


    def _loss_ppo(self, states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int):

        logits_new, values_ext_new, values_int_new  = self.model.forward(states)

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
        za, zb, pa, pb, ha, hb = self.model.forward_self_supervised(states_now, states_now)

        # current state distillation
        distillation_novelty = ((za - pb)**2).mean(dim=-1) + ((zb - pa)**2).mean(dim=-1)
        distillation_novelty = 0.5*distillation_novelty

        # next state prediction
        z_prev = self.model.model_features_a(states_prev)
        z_now  = self.model.model_features_a(states_now)

        z_now_pred, _ = self.model.model_state_predictor(z_prev, z_now)
        prediction_novelty = ((z_now - z_now_pred)**2).mean(dim=-1)

        return distillation_novelty, prediction_novelty
 
    def _loss_predictor(self, states_prev, states_now):
        # next state prediction
        z_prev = self.model.model_features_a(states_prev)
        z_now  = self.model.model_features_a(states_now)

        z_now_pred, h = self.model.model_state_predictor(z_prev, z_now)

        loss_prediction = ((z_now - z_now_pred)**2).mean()
        loss_hidden = torch.abs(h).mean() + (h.std(dim=0)).mean()

        return loss_prediction, loss_hidden, h


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
    
