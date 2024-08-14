import numpy
import torch 

from .ValuesLogger      import *
from .TrajectoryBufferIMNew import *  

from .PPOLoss               import *
from .SelfSupervised        import * 
from .Augmentations         import *
   
    
class AgentPPOEuclid():   
    def __init__(self, envs, Model, config):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.envs = envs        
          
        self.gamma_ext          = config.gamma_ext 
        self.gamma_int          = config.gamma_int
              
        self.ext_adv_coeff      = config.ext_adv_coeff
        self.int_adv_coeff      = config.int_adv_coeff 
        self.reward_int_coeff   = config.reward_int_coeff

        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip 
    
        self.steps              = config.steps
        self.batch_size         = config.batch_size  
        self.ss_batch_size      = config.ss_batch_size  

        self.episodic_buffer_size = config.episodic_buffer_size
        

        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count

        self.state_normalise    = config.state_normalise

        if hasattr(config, "rnn_policy"):
            self.rnn_policy          = config.rnn_policy
            self.rnn_seq_length      = config.rnn_seq_length
        else:
            self.rnn_policy          = False
            self.rnn_seq_length      = -1   
            

        if config.rl_ssl_loss == "vicreg":
            self._rl_ssl_loss = loss_vicreg
        else:
            self._rl_ssl_loss = None    

        if config.im_ssl_loss == "vicreg_distance_categorical":
            self._im_ssl_loss = loss_vicreg_distance_categorical
        elif config.im_ssl_loss == "vicreg_distance":
            self._im_ssl_loss = loss_vicreg_distance
        else:
            self._im_ssl_loss = None

        self.max_distance = config.max_distance


        self.metric_scaling_func            = config.metric_scaling_func

        self.augmentations_rl               = config.augmentations_rl
        self.augmentations_im               = config.augmentations_im
        self.augmentations_probs            = config.augmentations_probs
        

        print("state_normalise        = ", self.state_normalise)
        print("rl_ssl_loss            = ", self._rl_ssl_loss)
        print("im_ssl_loss            = ", self._im_ssl_loss)
        print("max_distance           = ", self.max_distance)
        print("metric_scaling_func    = ", self.metric_scaling_func)
        print("augmentations_rl       = ", self.augmentations_rl)
        print("augmentations_im       = ", self.augmentations_im)
        print("augmentations_probs    = ", self.augmentations_probs)
        print("reward_int_coeff       = ", self.reward_int_coeff)
        print("rnn_policy             = ", self.rnn_policy)
        print("rnn_seq_length         = ", self.rnn_seq_length)
        

        print("\n\n")

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        #create model and optimizer
        self.model      = Model.Model(self.state_shape, self.actions_count, self.max_distance + 1)
        self.model.to(self.device)
        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        print(self.model)
     
        self.trajectory_buffer = TrajectoryBufferIMNew(self.steps, self.state_shape, self.actions_count, self.envs_count)

        if self.rnn_policy:
            self.hidden_state   = torch.zeros((self.envs_count, self.model.rnn_size), dtype=torch.float32, device=self.device)
            self.rnn_seq_length = config.rnn_seq_length
        
        self.episode_steps = torch.zeros((self.envs_count), dtype=int)

        #optional, for state mean and variance normalisation        
        self.state_mean  = numpy.zeros(self.state_shape, dtype=numpy.float32)

        for e in range(self.envs_count):
            state, _ = self.envs.reset(e)
            self.state_mean+= state.copy()

        self.state_mean/= self.envs_count
        self.state_var = numpy.ones(self.state_shape,  dtype=numpy.float32)

        self.episodic_buffer = torch.zeros((self.episodic_buffer_size, self.envs_count, self.model.n_features), dtype=torch.float32, device=self.device)

        self.iterations = 0 

        self.values_logger  = ValuesLogger() 

        self.values_logger.add("internal_motivation_mean", 0.0)
        self.values_logger.add("internal_motivation_std" , 0.0)
        
        self.values_logger.add("loss_ppo_actor",  0.0)
        self.values_logger.add("loss_ppo_critic", 0.0)
        
        self.values_logger.add("loss_im", 0.0)

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
            logits_t, values_ext_t, values_int_t, hidden_state_new  = self.model.forward(states_t, self.hidden_state, False)
        else:
            logits_t, values_ext_t, values_int_t = self.model.forward(states_t)

        #collect actions 
        actions = self._sample_actions(logits_t, legal_actions_mask)
        
        #execute action
        states_new, rewards_ext, dones, _, infos = self.envs.step(actions)

        #internal motivation
        rewards_int, im_info = self._internal_motivation(states_t, dones)
        rewards_int = torch.clip(self.reward_int_coeff*rewards_int, 0.0, 1.0).detach().to("cpu")

        self.info_logger["im"] = im_info

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

            if self.rnn_policy:
                hidden_state  = self.hidden_state.detach().to("cpu")
            else:
                hidden_state  = None

            self.trajectory_buffer.add(states_t, logits_t, values_ext_t, values_int_t, actions, rewards_ext_t, rewards_int_t, dones, hidden_state, self.episode_steps)

            if self.trajectory_buffer.is_full():
                self.train()

        self.episode_steps+= 1

        dones_idx = numpy.where(dones)

        #clear episode steps counter
        for e in dones_idx:
            self.episode_steps[e] = 0
  
        #udpate rnn hidden tate
        if self.rnn_policy:
            self.hidden_state = hidden_state_new.detach().clone()

            dones_idx = numpy.where(dones)

            #clear rnn hidden state if done
            for e in dones_idx:
                self.hidden_state[e] = 0.0

            #hidden space stats
            hidden_mean   = (self.hidden_state**2).mean().detach().cpu().numpy().item()
            hidden_std_a  = self.hidden_state.std(dim=0).mean().detach().cpu().numpy().item()
            hidden_std_b  = self.hidden_state.std(dim=1).mean().detach().cpu().numpy().item()
            self.info_logger["hidden"] = [ round(hidden_mean, 5), round(hidden_std_a, 5), round(hidden_std_b, 5)]
      
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

    def _sample_actions(self, logits, legal_actions_mask = None):

        if legal_actions_mask is not None:
            legal_actions_mask_t = torch.from_numpy(legal_actions_mask, dtype=torch.float32, device=self.device)
        else:
            legal_actions_mask_t = 1

        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)

        #keep only legal actions probs, and renormalise probs
        action_probs_t        = action_probs_t*legal_actions_mask_t
        action_probs_t        = action_probs_t/action_probs_t.sum(dim=-1).unsqueeze(1)

        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().to("cpu").numpy()

        return actions
    
    def train(self): 
        self.trajectory_buffer.compute_returns(self.gamma_ext, self.gamma_int)
        
        samples_count = self.steps*self.envs_count
        batch_count = samples_count//self.batch_size

        #main PPO training loop
        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                #PPO RL loss
                if self.rnn_policy:
                    states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int, hidden_states = self.trajectory_buffer.sample_batch_seq(self.rnn_seq_length, self.batch_size, self.device)                    
                    loss_ppo = self._loss_ppo(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int, hidden_states)
                else:
                    states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int = self.trajectory_buffer.sample_batch(self.batch_size, self.device)
                    loss_ppo = self._loss_ppo(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int, None)

                #PPO self supervised loss
                if self._rl_ssl_loss is not None:
                    sa, sb = self.trajectory_buffer.sample_states_pairs(self.ss_batch_size, 0, False, self.device)
                    loss_ssl, rl_ssl = self._rl_ssl_loss(self.model.forward_rl_ssl, self._augmentations_rl_func, sa, sb)

                    self.info_logger["rl_ssl"] = rl_ssl 
                else:
                    loss_ssl = 0    
                
                
                loss = loss_ppo + loss_ssl

                self.optimizer.zero_grad()            
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                

        batch_count = samples_count//self.ss_batch_size
        
        #main IM training loop
        for batch_idx in range(batch_count):    
            #target SSL regularisation
            states, steps = self.trajectory_buffer.sample_states_steps(self.ss_batch_size, self.device)

            loss_ssl, im_ssl = self._im_ssl_loss(self.model.forward_im_ssl, self._augmentations_im_func, states, steps, self.metric_scaling_func)

            self.info_logger["im_ssl"] = im_ssl
        

            self.optimizer.zero_grad()            
            loss_ssl.backward()     
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

            self.values_logger.add("loss_im", loss_ssl.detach().to("cpu").numpy())

        self.trajectory_buffer.clear() 

        
    def _loss_ppo(self, states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int, hidden_states):

        if hidden_states is None:
            logits_new, values_ext_new, values_int_new    = self.model.forward(states)
        else:
            logits_new, values_ext_new, values_int_new, _ = self.model.forward(states, hidden_states, True)

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

   

    def _internal_motivation(self, states, dones, p_smallest = 0.1):        
        # obtain features
        z  = self.model.forward_im_encoder(states).detach()

        # fill with constant where episode done
        dones_idx = numpy.where(dones)
        for e in dones_idx:
            self.episodic_buffer[:, e, :] = z[e]

        # meassure distances from new z and buffered past z
        # dz.shape     = (episodic_buffer_size, envs_count, n_features)
        # d_pred.shape = (episodic_buffer_size, envs_count, n_classes)
        dz      = self.episodic_buffer - z.unsqueeze(0)
        d_pred  = self.model.forward_distance(dz)

        # this part converts categorical distances into real numbers, ranging 0..1
        d_probs = torch.softmax(d_pred, dim=-1)

        print("d_probs = ", d_probs.shape)

        # create normalised category weights
        w = torch.arange(d_pred.shape[-1]).unsqueeze(0).unsqueeze(1).to(self.device)
        w = w/(d_pred.shape[-1]-1)

        # d_weighted.shape = (episodic_buffer_size, envs_count)
        d_weighted = (d_probs*w).sum(dim=-1)

        print("d_weighted = ", d_weighted.shape)

        # sort along buffer size dim, start with smallest
        d_sorted = torch.sort(d_weighted, 0, descending=False)[0]

        # select 10% smallest distances
        max_count  = int(p_smallest*d_sorted.shape[0])
        d_smallest = d_sorted[0:max_count, :]

        print("d_smallest = ", d_smallest.shape)


        # compute internal motivation
        # average distances along 10% closest
        # bigger distance better exploration
        im = d_smallest.mean(dim=0)

        print("im = ", im.shape)

        print("\n\n")

        info = []

        info.append(round(d_weighted.mean().detach().cpu().numpy().item(), 5))
        info.append(round(d_weighted.std().detach().cpu().numpy().item(), 5))
        info.append(round(d_smallest.mean().detach().cpu().numpy().item(), 5))
        info.append(round(d_smallest.std().detach().cpu().numpy().item(), 5))

        return im, info
 
    def _augmentations_rl_func(self, x): 
        mask_result = torch.zeros((x.shape[0], 4), device=x.device, dtype=torch.float32)

        if "mask" in self.augmentations_rl:
            x, mask = aug_random_apply(x, self.augmentations_probs, aug_mask)
            mask_result[:, 0] = mask

        if "channelmask" in self.augmentations_rl:
            x, mask = aug_random_apply(x, self.augmentations_probs, aug_channel_mask)
            mask_result[:, 1] = mask    
       
        if "mask_advanced" in self.augmentations_rl:
            x, mask = aug_random_apply(x, self.augmentations_probs, aug_mask_advanced)
            mask_result[:, 2] = mask
      
        if "noise" in self.augmentations_rl:
            x, mask = aug_random_apply(x, self.augmentations_probs, aug_noise)
            mask_result[:, 3] = mask
      
        return x.detach(), mask_result 
    

    def _augmentations_im_func(self, x): 
        mask_result = torch.zeros((x.shape[0], 4), device=x.device, dtype=torch.float32)

        if "mask" in self.augmentations_im:
            x, mask = aug_random_apply(x, self.augmentations_probs, aug_mask)
            mask_result[:, 0] = mask

        if "channelmask" in self.augmentations_im:
            x, mask = aug_random_apply(x, self.augmentations_probs, aug_channel_mask)
            mask_result[:, 1] = mask    
       
        if "mask_advanced" in self.augmentations_im:
            x, mask = aug_random_apply(x, self.augmentations_probs, aug_mask_advanced)
            mask_result[:, 2] = mask
      
        if "noise" in self.augmentations_im:
            x, mask = aug_random_apply(x, self.augmentations_probs, aug_noise)
            mask_result[:, 3] = mask
      
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
    
