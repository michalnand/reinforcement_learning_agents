import numpy
import torch 

from .ValuesLogger      import *
from .PolicyBufferIM    import *  

from .PPOLoss               import *
from .SelfSupervised        import * 
from .Augmentations         import *
 
          
class AgentPPOCE():   
    def __init__(self, envs, ModelPPO, ModelIM, config):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.envs   = envs    

        #generic RL parameters
        self.envs_count         = config.envs_count
          
        self.gamma_ext          = config.gamma_ext 
        self.gamma_int          = config.gamma_int
         
        #reward scaling
        self.ext_adv_coeff      = config.ext_adv_coeff
        self.int_adv_coeff      = config.int_adv_coeff
        self.reward_int_a_coeff = config.reward_int_a_coeff
        self.reward_int_b_coeff = config.reward_int_b_coeff

        self.reward_int_dif_coeff = config.reward_int_dif_coeff

        #ppo params
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip 
    
        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        self.training_epochs    = config.training_epochs
        
        #internal motivation params    
        if config.ppo_self_supervised_loss == "vicreg":
            self._ppo_self_supervised_loss = loss_vicreg
        else:
            self._ppo_self_supervised_loss = None

        if config.target_self_supervised_loss == "vicreg":
            self._target_self_supervised_loss = loss_vicreg
        else:
            self._target_self_supervised_loss = None

        self.augmentations              = config.augmentations
        self.augmentations_probs        = config.augmentations_probs
        
        
        self.similar_states_distance    = config.similar_states_distance
        
        self.contextual_buffer_size     = config.contextual_buffer_size
        self.contextual_buffer_skip     = config.contextual_buffer_skip
        self.contextual_average         = config.contextual_average
        
        
        #speacial params 
        self.rnn_policy                 = config.rnn_policy
        self.state_normalise            = config.state_normalise

        print("ppo_self_supervised_loss     = ", self._ppo_self_supervised_loss)
        print("target_self_supervised_loss  = ", self._target_self_supervised_loss)
        print("augmentations                = ", self.augmentations)
        print("augmentations_probs          = ", self.augmentations_probs)
        print("reward_int_a_coeff           = ", self.reward_int_a_coeff)
        print("reward_int_b_coeff           = ", self.reward_int_b_coeff)
        print("similar_states_distance      = ", self.similar_states_distance)
        print("contextual_buffer_size       = ", self.contextual_buffer_size)
        print("contextual_buffer_skip       = ", self.contextual_buffer_skip)
        print("contextual_average           = ", self.contextual_average)
        print("rnn_policy                   = ", self.rnn_policy)
        print("state_normalise              = ", self.state_normalise)
        
        print("\n\n")

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        #main ppo agent
        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.model_ppo.to(self.device)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)

        #internal motivation model
        self.model_im      = ModelIM.Model(self.state_shape)
        self.model_im.to(self.device)
        self.optimizer_im  = torch.optim.Adam(self.model_im.parameters(), lr=config.learning_rate_im)

      
        self.policy_buffer = PolicyBufferIM(self.steps, self.state_shape, self.actions_count, self.envs_count)

        #reset envs and fill initial state
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e], _  = self.envs.reset(e)

        self.hidden_state = torch.zeros((self.envs_count, 512), dtype=torch.float32, device=self.device)
        
        self.z_context_buffer    = torch.zeros((self.envs_count, self.contextual_buffer_size, 512), dtype=torch.float32)
        self.z_context_ptr  = 0

        #optional, for state mean and variance normalisation        
        self.state_mean  = self.states.mean(axis=0)
        self.state_var   = numpy.ones_like(self.state_mean, dtype=numpy.float32)


        self.rewards_int      = torch.zeros(self.envs_count, dtype=torch.float32)
        self.rewards_int_prev = torch.zeros(self.envs_count, dtype=torch.float32)

        self.enable_training() 
        self.iterations     = 0 

        self.values_logger  = ValuesLogger() 

        self.values_logger.add("internal_motivation_mean",      0.0)
        self.values_logger.add("internal_motivation_std" ,      0.0)
        self.values_logger.add("loss_ppo_actor",                0.0)
        self.values_logger.add("loss_ppo_critic",               0.0)
        self.values_logger.add("loss_ppo_self_supervised",      0.0)
        self.values_logger.add("loss_target",                   0.0)
        self.values_logger.add("loss_distillation",             0.0)
        
        
        self.info_logger = {}

        self.info_logger["ln_mean"]      = 0.0
        self.info_logger["ln_std"]       = 0.0
        self.info_logger["ln_max"]       = 0.0
        self.info_logger["en_mean"]      = 0.0
        self.info_logger["en_std"]       = 0.0
        self.info_logger["en_max"]       = 0.0

        
       
    def enable_training(self): 
        self.enabled_training = True
 
    def disable_training(self):
        self.enabled_training = False
 
    def main(self):         
        #normalise if any
        states = self._state_normalise(self.states)
        
        #state to tensor
        states = torch.tensor(states, dtype=torch.float).to(self.device)

        #compute model output
        if self.rnn_policy:
            logits, values_ext, values_int, hidden_state_new  = self.model_ppo.forward(states, self.hidden_state)
        else:
            logits, values_ext, values_int  = self.model_ppo.forward(states)
        
        #collect actions 
        actions = self._sample_actions(logits)
        
        #execute action
        states_new, rewards_ext, dones, _, infos = self.envs.step(actions)

        #internal motivation
        rewards_a_int, rewards_b_int = self._internal_motivation(states)

        rewards_int  = self.reward_int_a_coeff*rewards_a_int + self.reward_int_b_coeff*rewards_b_int

        self.rewards_int_prev   = self.rewards_int.clone()
        self.rewards_int        = rewards_int

        rewards_int = torch.clip(self.rewards_int - self.reward_int_dif_coeff*self.rewards_int_prev, 0.0, 1.0)
        
     

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

            hidden_state    = self.hidden_state.detach().to("cpu")

            self.policy_buffer.add(states, logits, values_ext, values_int, actions, rewards_ext_t, rewards_int_t, dones, hidden_state)

            if self.policy_buffer.is_full():
                self.train()

        
        #update new state
        self.states_prev = self.states.copy()
        self.states      = states_new.copy()

        if self.rnn_policy:
            self.hidden_state = hidden_state_new.detach().clone()

        #or reset env if done
        for e in range(self.envs_count): 
            if dones[e]:
                self.states[e], _       = self.envs.reset(e)
                self.hidden_state[e]    = torch.zeros(self.hidden_state.shape[1], dtype=torch.float32, device=self.device)

                
                #fill initial context after new episode
                states_t                = torch.from_numpy(self.states[e]).to(self.device).unsqueeze(0)
                
                z_target_t, _                  = self.model_im(states_t)
                self.z_context_buffer[e, :, :] = z_target_t.squeeze(0).detach().cpu()
                
        
        #collect stats
        self.values_logger.add("internal_motivation_mean", rewards_int.mean().detach().to("cpu").numpy())
        self.values_logger.add("internal_motivation_std" , rewards_int.std().detach().to("cpu").numpy())

        '''
        attn    = attn.detach().cpu().numpy()

        c_mean  = attn.max(axis=-1).mean()
        c_std   = attn.max(axis=-1).std() 
        p_mean  = attn.mean(axis=-1).mean()
        p_std   = attn.std(axis=-1).mean()

        self.info_logger["c_mean"]      = round(c_mean, 8)
        self.info_logger["c_std"]       = round(c_std, 8)
        self.info_logger["p_mean"]      = round(p_mean, 8)
        self.info_logger["p_std"]       = round(p_std, 8) 
        '''

        self.iterations+= 1

        return rewards_ext[0], dones[0], infos[0]
    
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
                states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int, hidden_state = self.policy_buffer.sample_batch(self.batch_size, self.device)
                
                #train PPO model
                loss_ppo     = self._loss_ppo(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int, hidden_state)
                
                #train ppo features, self supervised
                if self._ppo_self_supervised_loss is not None:
                    #sample smaller batch for self supervised loss
                    states_now, states_next, states_similar, states_random, actions, relations = self.policy_buffer.sample_states_action_pairs(small_batch, self.device, 0)

                    loss_ppo_self_supervised    = self._ppo_self_supervised_loss(self.model_ppo.forward_features, self._augmentations, states_now, states_next, states_similar, states_random, actions, relations)                
                else:
                    loss_ppo_self_supervised    = torch.zeros((1, ), device=self.device)[0]

                #total PPO loss
                loss = loss_ppo + loss_ppo_self_supervised

                self.optimizer_ppo.zero_grad()        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()

                #sample smaller batch for self supervised loss, different distances for different models
                states_now, states_next, states_similar, states_random, actions, relations = self.policy_buffer.sample_states_action_pairs(small_batch, self.device, self.similar_states_distance)

                #target im model, self supervised regularisation 
                loss_target = self._target_self_supervised_loss(self.model_im.forward, self._augmentations, states_now, states_next, states_similar, states_random, actions, relations)                

                #MSE distilation loss
                loss_distillation = self._loss_distillation(states)


                loss_im = loss_target + loss_distillation

                self.optimizer_im.zero_grad() 
                loss_im.backward()
                self.optimizer_im.step() 
               
                #log results
                self.values_logger.add("loss_ppo_self_supervised", loss_ppo_self_supervised.detach().to("cpu").numpy())
                self.values_logger.add("loss_target",   loss_target.detach().to("cpu").numpy())
                self.values_logger.add("loss_distillation", loss_distillation.detach().to("cpu").numpy())

                
        self.policy_buffer.clear() 

    
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

    
 
    def _internal_motivation(self, states):
        z_target_t, z_predictor_t = self.model_im(states)

        z_target_t      = z_target_t.detach().cpu()
        z_predictor_t   = z_predictor_t.detach().cpu()

        #distillation novelty
        long_novelty_t = ((z_target_t - z_predictor_t)**2).mean(dim=1)

        #d.shape = (envs_count, buffer_size, features_count)
        d = (self.z_context_buffer - z_target_t.unsqueeze(1))**2 

        #d.shape = (envs_count, buffer_size)
        d = d.mean(dim=-1)

        #sort, smallest first
        d = torch.sort(d, dim=1)[0]

        #select N-smallest (closest distance)
        d = d[:, 0:self.contextual_average]

        #average for episodic novelty
        episodic_novelty_t = d.mean(dim=1)

        #store every n-th features only 
        #not necessary to store all frames
        if (self.iterations%self.contextual_buffer_skip) == 0: 
            self.z_context_buffer[:, self.z_context_ptr, :] = z_target_t 
            self.z_context_ptr = (self.z_context_ptr + 1)%self.contextual_buffer_size

        
        self.info_logger["ln_mean"]      = round(float(long_novelty_t.mean().numpy()), 6)
        self.info_logger["ln_std"]       = round(float(long_novelty_t.std().numpy()), 6)
        self.info_logger["ln_max"]       = round(float(long_novelty_t.max().numpy()), 6)
        self.info_logger["en_mean"]      = round(float(episodic_novelty_t.mean().numpy()), 6)
        self.info_logger["en_std"]       = round(float(episodic_novelty_t.std().numpy()), 6)
        self.info_logger["en_max"]       = round(float(episodic_novelty_t.max().numpy()), 6)


        return long_novelty_t, episodic_novelty_t



    #MSE loss for networks distillation model
    def _loss_distillation(self, states): 
        z_target_t, z_predictor_t  = self.model_im(states)
        loss = ((z_target_t.detach() - z_predictor_t)**2).mean()        
        return loss
    
 
    def _augmentations(self, x): 
        if "inverse" in self.augmentations:
            x = aug_random_apply(x, self.augmentations_probs, aug_inverse)

        if "random_filter" in self.augmentations:
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

        if self.state_normalise:
            #update running stats only during training
            if self.enabled_training:
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
   