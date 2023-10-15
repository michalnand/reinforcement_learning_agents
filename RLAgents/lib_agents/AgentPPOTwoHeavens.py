import numpy
import torch 

from .ValuesLogger      import *
from .PolicyBufferIM    import *  

from .PPOLoss               import *
from .SelfSupervised        import * 
from .Augmentations         import *
 
           
class AgentPPOTwoHeavens():   
    def __init__(self, envs, ModelPPO, ModelIM, config):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
        self.envs   = envs    

        #generic RL parameters
        self.envs_count         = config.envs_count
          
        self.gamma_ext          = config.gamma_ext 
        self.gamma_int          = config.gamma_int
         
        #reward scaling
        self.ext_adv_coeff        = config.ext_adv_coeff
        self.int_adv_coeff        = config.int_adv_coeff
        self.reward_int_coeff     = config.reward_int_coeff
        self.predictor_loss_coeff = config.predictor_loss_coeff
        
    
        #ppo params
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip 
    
        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        self.training_epochs    = config.training_epochs
        
       

        self.augmentations              = config.augmentations
        self.augmentations_probs        = config.augmentations_probs
        
        self.similar_states_distance    = config.similar_states_distance

    
        
        #speacial params 
        self.rnn_policy                 = config.rnn_policy
        self.state_normalise            = config.state_normalise

        print("augmentations                = ", self.augmentations)
        print("augmentations_probs          = ", self.augmentations_probs)
        print("reward_int_coeff             = ", self.reward_int_coeff)
        print("predictor_loss_coeff         = ", self.predictor_loss_coeff)
        print("similar_states_distance      = ", self.similar_states_distance)
        print("rnn_policy                   = ", self.rnn_policy)
        print("state_normalise              = ", self.state_normalise)
        
        print("\n\n")

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        print("state_shape                  = ", self.state_shape)
        print("actions_count                = ", self.actions_count)
        print("\n\n")

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

    
        #optional, for state mean and variance normalisation        
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
        self.values_logger.add("loss_target",                   0.0)
        self.values_logger.add("loss_predictor",                0.0)
        self.values_logger.add("entropy_mean",                  0.0)
        self.values_logger.add("entropy_std",                   0.0)


        self.info_logger = {}
        

        
       
    def enable_training(self): 
        self.enabled_training = True
 
    def disable_training(self):
        self.enabled_training = False
 
    def main(self):         
        #normalise if any
        states = self._state_normalise(self.states)

        print("A = ", states.shape)
        
        #state to tensor
        states = torch.tensor(states, dtype=torch.float).to(self.device)

        print("B = ", states.shape)
        #compute model output
        if self.rnn_policy:
            logits, values_ext, values_int, hidden_state_new  = self.model_ppo.forward(states, self.hidden_state)
        else:
            logits, values_ext, values_int  = self.model_ppo.forward(states)
        
        print("C = ", states.shape)
        #collect actions 
        actions = self._sample_actions(logits)
        
        #execute action
        states_new, rewards_ext, dones, _, infos = self.envs.step(actions)

        print("D = ", states.shape)
        
        #internal motivation
        rewards_int = self.reward_int_coeff*self._internal_motivation(states)        
        rewards_int = torch.clip(rewards_int, 0.0, 1.0)
        
        print("E = ", states.shape)
    
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

        print("F = ", states.shape)

        
        #update new state
        self.states      = states_new.copy()

        if self.rnn_policy:
            self.hidden_state = hidden_state_new.detach().clone()

        #or reset env if done
        for e in range(self.envs_count): 
            if dones[e]:
                self.states[e], _       = self.envs.reset(e)
                self.hidden_state[e]    = torch.zeros(self.hidden_state.shape[1], dtype=torch.float32, device=self.device)

              
        #collect stats
        self.values_logger.add("internal_motivation_mean", rewards_int.mean().detach().to("cpu").numpy())
        self.values_logger.add("internal_motivation_std" , rewards_int.std().detach().to("cpu").numpy())


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
                states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int, hidden_state = self.policy_buffer.sample_batch(self.batch_size, self.device)
                
                #train PPO model
                loss_ppo     = self._loss_ppo(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int, hidden_state)
                
                #train ppo features, self supervised
                #sample smaller batch for self supervised loss
                states_now, states_next, states_similar, states_random, actions, relations = self.policy_buffer.sample_states_action_pairs(small_batch, self.device, 0)
                loss_ppo_self_supervised    = loss_vicreg(self.model_ppo.forward_features, self._augmentations, states_now, states_next, states_similar, states_random, actions, relations)                
            
                #total PPO loss 
                loss = loss_ppo + loss_ppo_self_supervised

                self.optimizer_ppo.zero_grad()        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()

                #sample smaller batch for self supervised loss, different distances for different models
                states_now, states_next, states_similar, states_random, actions, relations = self.policy_buffer.sample_states_action_pairs(small_batch, self.device, self.similar_states_distance)

                #train snd target model, self supervised to provide better features   
                loss_target = loss_vicreg(self.model_im.forward_target, self._augmentations, states_now, states_next, states_similar, states_random, actions, relations)                

                #distille target model into predictor model, using advanced loss
                loss_predictor, entropy_mean, entropy_std = self._loss_predictor(self.model_im, states)


                loss_im = loss_target + self.predictor_loss_coeff*loss_predictor

                self.optimizer_im.zero_grad()
                loss_im.backward()
                self.optimizer_im.step()
                
                #log results
                self.values_logger.add("loss_ppo_self_supervised",  loss_ppo_self_supervised.detach().cpu().numpy())
                self.values_logger.add("loss_target",               loss_target.detach().cpu().numpy())
                self.values_logger.add("loss_predictor",            loss_predictor.detach().cpu().numpy())
                self.values_logger.add("entropy_mean",              entropy_mean.detach().cpu().numpy())
                self.values_logger.add("entropy_std",               entropy_std.detach().cpu().numpy())



    
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

    def _internal_motivation(self, x):
        #novelty detection based on prediction error
        features_target_t       = self.model_im.forward_target(x)
        features_predicted_t    = self.model_im.forward_predictor(x)

        novelty_t = ((features_target_t - features_predicted_t)**2).mean(dim=1)
        novelty_t = novelty_t.detach().cpu()

        return novelty_t
    
    '''
    this loss maximize mutual informaiton betveen target and predictor,
    with respect to minimal features covariance, and keeping variance close to 1
    '''
    def _loss_predictor(self, model, x):        
        za = model.forward_target(x).detach()
        zb = model.forward_predictor(x)

        loss = loss_vicreg_direct(za, zb)   

        #compute entropy of mutual information, for statistics only
        #and normalise, maximum is 1
        w       = za@zb.T 
        w       = w.detach()
        p       = torch.softmax(w, dim=1)
        entropy = (-p*torch.log2(p + 10**-8)).sum(dim=1)
        entropy = entropy/numpy.log2(w.shape[0])
        entropy_mean = entropy.mean()
        entropy_std  = entropy.std()

        return loss, entropy_mean, entropy_std

    
 
 
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
   