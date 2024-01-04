import numpy
import torch 

from .ValuesLogger      import *
from .PolicyBufferIM    import *  
from .TemporalBuffer    import *

from .PPOLoss               import *
from .SelfSupervised        import * 
from .Augmentations         import *
  
#import matplotlib.pyplot as plt

class AgentPPOCSND():   
    def __init__(self, envs, ModelPPO, ModelIM, config):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.envs = envs  
          
        self.gamma_ext          = config.gamma_ext 
        self.gamma_int          = config.gamma_int
              
        self.ext_adv_coeff      = config.ext_adv_coeff
        self.int_adv_coeff      = config.int_adv_coeff
 
        self.reward_int_a_coeff     = config.reward_int_a_coeff
        self.reward_int_b_coeff     = config.reward_int_b_coeff
        self.reward_int_dif_coeff   = config.reward_int_dif_coeff
        


        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip 
    
        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        self.seq_length         = config.seq_length

        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count

        self.rnn_policy         = config.rnn_policy

        im_features_size        = config.im_features_size
        im_rnn_size             = config.im_rnn_size
                
        if config.ppo_self_supervised_loss == "vicreg":
            self._ppo_self_supervised_loss = loss_vicreg
        else:
            self._ppo_self_supervised_loss = None

        if config.spatial_target_self_supervised_loss == "vicreg":
            self._spatial_target_self_supervised_loss = loss_vicreg
        else:
            self._spatial_target_self_supervised_loss = None

        if config.temporal_target_self_supervised_loss == "vicreg_temporal":
            self._temporal_target_self_supervised_loss = loss_vicreg_temporal
        else:
            self._temporal_target_self_supervised_loss = None

        self.similar_states_distance = config.similar_states_distance
        
        if hasattr(config, "state_normalise"):
            self.state_normalise = config.state_normalise
        else:
            self.state_normalise = False

        self.augmentations                  = config.augmentations
        self.augmentations_probs            = config.augmentations_probs
        
        print("ppo_self_supervised_loss              = ", self._ppo_self_supervised_loss)
        print("spatial_target_self_supervised_loss   = ", self._spatial_target_self_supervised_loss)
        print("temporal_target_self_supervised_loss  = ", self._temporal_target_self_supervised_loss)

        print("augmentations                         = ", self.augmentations)
        print("augmentations_probs                   = ", self.augmentations_probs)
        print("reward_int_a_coeff                    = ", self.reward_int_a_coeff)
        print("reward_int_b_coeff                    = ", self.reward_int_b_coeff)
        print("reward_int_dif_coeff                  = ", self.reward_int_dif_coeff)
        print("rnn_policy                            = ", self.rnn_policy)
        print("similar_states_distance               = ", self.similar_states_distance)
        print("state_normalise                       = ", self.state_normalise)

        print("\n\n")

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        #main ppo agent
        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.model_ppo.to(self.device)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)

        #IM model
        self.model_im      = ModelIM.Model(self.state_shape, im_features_size, im_rnn_size)
        self.model_im.to(self.device)
        self.optimizer_im  = torch.optim.Adam(self.model_im.parameters(), lr=config.learning_rate_im)
    
        self.policy_buffer   = PolicyBufferIM(self.steps, self.state_shape, self.actions_count, self.envs_count)
        self.temporal_buffer = TemporalBuffer(self.steps, im_features_size, im_rnn_size, self.envs_count)

        #optional hidden state for rnn policy
        if self.rnn_policy:
            self.hidden_state = torch.zeros((self.envs_count, self.model_ppo.rnn_size), dtype=torch.float32, device=self.device)
        else:
            self.hidden_state = torch.zeros((self.envs_count, 8), dtype=torch.float32, device=self.device)

        #temporal hidden state for im (two, one for target one for predictor)
        self.hidden_im_state = torch.zeros((2, self.envs_count, im_rnn_size), dtype=torch.float32, device=self.device)

        #optional, for state mean and variance normalisation        
        self.state_mean  = numpy.zeros(self.state_shape, dtype=numpy.float32)

        for e in range(self.envs_count):
            state, _ = self.envs.reset(e)
            self.state_mean+= state.copy()

        self.state_mean/= self.envs_count

        self.state_var = numpy.ones(self.state_shape,  dtype=numpy.float32)

        self.rewards_int      = torch.zeros(self.envs_count, dtype=torch.float32)
        self.rewards_int_prev = torch.zeros(self.envs_count, dtype=torch.float32)

        self.iterations = 0 

        self.values_logger  = ValuesLogger() 

        self.values_logger.add("internal_motivation_a_mean", 0.0)
        self.values_logger.add("internal_motivation_a_std" , 0.0)
        self.values_logger.add("internal_motivation_b_mean", 0.0)
        self.values_logger.add("internal_motivation_b_std" , 0.0)
        
        self.values_logger.add("loss_ppo_actor",  0.0)
        self.values_logger.add("loss_ppo_critic", 0.0)
        
        self.values_logger.add("loss_ppo_self_supervised", 0.0)
        self.values_logger.add("loss_spatial_target_self_supervised", 0.0)
        self.values_logger.add("loss_temporal_target_self_supervised", 0.0)
        self.values_logger.add("loss_spatial_distillation", 0.0)
        self.values_logger.add("loss_temporal_distillation", 0.0)


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
        if self.rnn_policy:
            logits_t, values_ext_t, values_int_t, hidden_state_new  = self.model_ppo.forward(states_t, self.hidden_state)
        else:
            logits_t, values_ext_t, values_int_t  = self.model_ppo.forward(states_t)
        
        #collect actions 
        actions = self._sample_actions(logits_t, legal_actions_mask)
        
        #execute action
        states_new, rewards_ext, dones, _, infos = self.envs.step(actions)

        #internal motivation
        rewards_int_a, rewards_int_b, zs_target, hidden_im_state_new  = self._internal_motivation(states_t, self.hidden_im_state)

        #store into buffer for spatial IM
        #self.temporal_buffer.add(zs_target, self.hidden_im_state)

        self.hidden_im_state = hidden_im_state_new.detach().clone()


        #im computing, weighting and clipping
        rewards_int_a  = self.reward_int_a_coeff*rewards_int_a
        rewards_int_b  = self.reward_int_b_coeff*rewards_int_b

        self.rewards_int_prev = self.rewards_int.clone()
        self.rewards_int = (rewards_int_a + rewards_int_b).detach().to("cpu")

        rewards_int = torch.clip(self.rewards_int - self.reward_int_dif_coeff*self.rewards_int_prev, 0.0, 1.0)
        
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
            
            hidden_state    = self.hidden_state.detach().to("cpu")

            self.policy_buffer.add(states_t, logits_t, values_ext_t, values_int_t, actions, rewards_ext_t, rewards_int_t, dones, hidden_state)

            if self.policy_buffer.is_full():
                self.train()

        #udpate rnn hidden state if any
        if self.rnn_policy:
            self.hidden_state = hidden_state_new.detach().clone()

        #reset env if done
        dones_idx = numpy.where(dones)[0]
        for e in dones_idx: 
            self.hidden_state[e]        = 0
            self.hidden_im_state[: e]   = 0

            
        #collect stats
        self.values_logger.add("internal_motivation_a_mean", rewards_int_a.mean().detach().to("cpu").numpy())
        self.values_logger.add("internal_motivation_a_std" , rewards_int_a.std().detach().to("cpu").numpy())
        self.values_logger.add("internal_motivation_b_mean", rewards_int_b.mean().detach().to("cpu").numpy())
        self.values_logger.add("internal_motivation_b_std" , rewards_int_b.std().detach().to("cpu").numpy())
        
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

        batch_count = self.steps//self.batch_size

        small_batch = 16*self.batch_size 

        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int, hidden_state = self.policy_buffer.sample_batch(self.batch_size, self.device)
                
                #train PPO model
                loss_ppo     = self._loss_ppo(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int, hidden_state)
                
                #train ppo features, self supervised
                if self._ppo_self_supervised_loss is not None:
                    #sample smaller batch for self supervised loss
                    states_now, states_similar = self.policy_buffer.sample_states_pairs(small_batch, self.device, 0)

                    loss_ppo_self_supervised    = self._ppo_self_supervised_loss(self.model_ppo.forward_features, self._augmentations, states_now, states_similar)  
                else:
                    loss_ppo_self_supervised    = torch.zeros((1, ), device=self.device)[0]

                #total PPO loss
                loss = loss_ppo + loss_ppo_self_supervised

                self.optimizer_ppo.zero_grad()        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()


              
                #sample smaller batch for self supervised loss, different distances for different models
                states_now, states_similar = self.policy_buffer.sample_states_pairs(small_batch, self.device, self.similar_states_distance)

                #z_seq, h_initial = self.temporal_buffer.sample_batch(small_batch, self.seq_length, self.device)

                #train snd target model, self supervised    
                loss_spatial_target_self_supervised = self._spatial_target_self_supervised_loss(self.model_im.forward_spatial_target, self._augmentations, states_now, states_similar)                

                #loss_temporal_target_self_supervised = self._temporal_target_self_supervised_loss(self.model_im.forward_temporal_target, None, z_seq, z_seq, h_initial)                

                #train snd distillation
                loss_spatial_distillation  = self._loss_spatial_distillation(states)
                #loss_temporal_distillation = self._loss_temporal_distillation(z_seq)


                #loss_im = loss_spatial_target_self_supervised + loss_temporal_target_self_supervised + loss_spatial_distillation + loss_temporal_distillation
                loss_im = loss_spatial_target_self_supervised + loss_spatial_distillation
                
                self.optimizer_im.zero_grad() 
                loss_im.backward()
                self.optimizer_im.step()


               
                #log results
                self.values_logger.add("loss_ppo_self_supervised", loss_ppo_self_supervised.detach().cpu().numpy())
                self.values_logger.add("loss_spatial_target_self_supervised", loss_spatial_target_self_supervised.detach().cpu().numpy())
                #self.values_logger.add("loss_temporal_target_self_supervised", loss_temporal_target_self_supervised.detach().cpu().numpy())
                self.values_logger.add("loss_spatial_distillation", loss_spatial_distillation.detach().cpu().numpy())
                #self.values_logger.add("loss_temporal_distillation", loss_temporal_distillation.detach().cpu().numpy())

        self.policy_buffer.clear() 
        self.temporal_buffer.clear()


    def _loss_ppo(self, states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int, hidden_state):

        if self.rnn_policy:
            logits_new, values_ext_new, values_int_new, _ = self.model_ppo.forward(states, hidden_state)
        else:
            logits_new, values_ext_new, values_int_new  = self.model_ppo.forward(states)


        #critic loss
        loss_critic =  ppo_compute_critic_loss(values_ext_new, returns_ext, values_int_new, returns_int)

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

    
    #MSE loss for spatial distillation
    def _loss_spatial_distillation(self, states):         
        z_target_t       = self.model_im.forward_spatial_target(states)        
        z_predicted_t    = self.model_im.forward_spatial_predictor(states)
        
        loss = ((z_target_t.detach() - z_predicted_t)**2).mean()

        return loss  
    

    #MSE loss for temporal distillation
    def _loss_temporal_distillation(self, z_seq, h):         
        z_target_t, _     = self.model_im.forward_temporal_target(z_seq, h)        
        z_predicted_t, _  = self.model_im.forward_temporal_predictor(z_seq, h)
        
        loss = ((z_target_t.detach() - z_predicted_t)**2).mean()

        return loss  
    

    #compute internal motivations
    def _internal_motivation(self, states, hidden_im_state):        
        #spatial distillation novelty detection, mse error
        zs_target_t    = self.model_im.forward_spatial_target(states).detach()
        zs_predictor_t = self.model_im.forward_spatial_predictor(states).detach()

        novelty_spatial_t = ((zs_target_t - zs_predictor_t)**2).mean(dim=1)
        novelty_spatial_t = novelty_spatial_t.detach().cpu()

        #temporal distillation novelty detection, mse error

        zt_target_t,    ht_new = self.model_im.forward_temporal_target(zs_target_t.unsqueeze(1), hidden_im_state[0])
        zt_predictor_t, hs_new = self.model_im.forward_temporal_predictor(zs_predictor_t.unsqueeze(1), hidden_im_state[1])

        zt_target_t    = zt_target_t.squeeze(1)
        zt_predictor_t = zt_predictor_t.squeeze(1)
        novelty_temporal_t = ((zt_target_t - zt_predictor_t)**2).mean(dim=1)
        novelty_temporal_t = novelty_temporal_t.detach().cpu()

        h_new = torch.concatenate([ht_new.unsqueeze(0), hs_new.unsqueeze(0)], dim=0).detach()
 
        return novelty_spatial_t, novelty_temporal_t, zs_target_t, h_new
 
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
   