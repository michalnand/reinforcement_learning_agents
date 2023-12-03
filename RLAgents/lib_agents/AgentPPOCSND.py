import numpy
import torch 

from .ValuesLogger      import *
from .PolicyBufferIM    import *  

from .PPOLoss               import *
from .SelfSupervised        import * 
from .Augmentations         import *
  
import matplotlib.pyplot as plt

class AgentPPOCSND():   
    def __init__(self, envs, ModelPPO, ModelIM, config):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.envs = envs  
            
        self.gamma_ext          = config.gamma_ext 
        self.gamma_int          = config.gamma_int
              
        self.ext_adv_coeff      = config.ext_adv_coeff
        self.int_adv_coeff      = config.int_adv_coeff
 
        self.reward_int_a_coeff   = config.reward_int_a_coeff
        self.reward_int_b_coeff   = config.reward_int_b_coeff

      
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip 
    
        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count

        self.rnn_policy         = config.rnn_policy


        if config.ppo_self_supervised_loss == "vicreg":
            self._ppo_self_supervised_loss = loss_vicreg
        else:
            self._ppo_self_supervised_loss = None

        if config.target_self_supervised_loss == "vicreg":
            self._target_self_supervised_loss = loss_vicreg
        else:
            self._target_self_supervised_loss = None


        self.similar_states_distance = config.similar_states_distance
        

        if hasattr(config, "state_normalise"):
            self.state_normalise = config.state_normalise
        else:
            self.state_normalise = False


        self.augmentations                  = config.augmentations
        self.augmentations_probs            = config.augmentations_probs
        

        print("ppo_self_supervised_loss     = ", self._ppo_self_supervised_loss)
        print("target_self_supervised_loss  = ", self._target_self_supervised_loss)
        print("augmentations                = ", self.augmentations)
        print("augmentations_probs          = ", self.augmentations_probs)
        print("reward_int_a_coeff           = ", self.reward_int_a_coeff)
        print("reward_int_b_coeff           = ", self.reward_int_b_coeff)
        print("rnn_policy                   = ", self.rnn_policy)
        print("similar_states_distance      = ", self.similar_states_distance)
        print("state_normalise              = ", self.state_normalise)

        print("\n\n")


        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n


        #main ppo agent
        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.model_ppo.to(self.device)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)


        #snd predictor model
        self.model_im      = ModelIM.Model(self.state_shape)
        self.model_im.to(self.device)
        self.optimizer_im  = torch.optim.Adam(self.model_im.parameters(), lr=config.learning_rate_im)

      
        self.policy_buffer = PolicyBufferIM(self.steps, self.state_shape, self.actions_count, self.envs_count)


        #reset envs and fill initial state
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e], _  = self.envs.reset(e)

        self.states_prev = self.states.copy()


        if self.rnn_policy:
            self.hidden_state = torch.zeros((self.envs_count, self.model_ppo.rnn_size), dtype=torch.float32, device=self.device)
        else:
            self.hidden_state = torch.zeros((self.envs_count, 128), dtype=torch.float32, device=self.device)


        #optional, for state mean and variance normalisation        
        self.state_mean  = self.states.mean(axis=0)
        self.state_var   = numpy.ones_like(self.state_mean, dtype=numpy.float32)

        self.enable_training() 
        self.iterations     = 0 

        self.steps_log = []
        self.z_log     = []

        self.episode_steps = torch.zeros((self.envs_count, ), dtype=torch.float32)

        self.values_logger  = ValuesLogger() 

         
        self.values_logger.add("internal_motivation_a_mean", 0.0)
        self.values_logger.add("internal_motivation_a_std" , 0.0)
        self.values_logger.add("internal_motivation_b_mean", 0.0)
        self.values_logger.add("internal_motivation_b_std" , 0.0)
        
        self.values_logger.add("loss_ppo_actor",  0.0)
        self.values_logger.add("loss_ppo_critic", 0.0) 


        self.values_logger.add("loss_ppo_self_supervised", 0.0)
        self.values_logger.add("loss_target_self_supervised", 0.0)
        self.values_logger.add("loss_target_causality", 0.0)
        self.values_logger.add("loss_distillation", 0.0)
        self.values_logger.add("accuracy", 0.0)

       
    def enable_training(self): 
        self.enabled_training = True
 
    def disable_training(self):
        self.enabled_training = False
 
    def main(self):         
        
        #normalise if any
        states      = self._state_normalise(self.states)
        states_prev = self._state_normalise(self.states_prev)
        
        #state to tensor
        states      = torch.tensor(states, dtype=torch.float).to(self.device)
        states_prev = torch.tensor(states_prev, dtype=torch.float).to(self.device)
        

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
        #prev motivation
        rewards_int_a, rewards_int_b = self._internal_motivation(states_prev, states)
        rewards_int = torch.clip(self.reward_int_a_coeff*rewards_int_a + self.reward_int_b_coeff*rewards_int_b, 0.0, 1.0)
        
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
            
            episode_steps   = self.episode_steps.detach().to("cpu")
            hidden_state    = self.hidden_state.detach().to("cpu")

            self.policy_buffer.add(states, logits, values_ext, values_int, actions, rewards_ext_t, rewards_int_t, dones, hidden_state, episode_steps=episode_steps)

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
                self.episode_steps[e]   = 0
         
        #self._add_for_plot(states, self.episode_steps)
        
        #collect stats
        self.values_logger.add("internal_motivation_a_mean", rewards_int_a.mean().detach().to("cpu").numpy())
        self.values_logger.add("internal_motivation_a_std" , rewards_int_a.std().detach().to("cpu").numpy())
        self.values_logger.add("internal_motivation_b_mean", rewards_int_b.mean().detach().to("cpu").numpy())
        self.values_logger.add("internal_motivation_b_std" , rewards_int_b.std().detach().to("cpu").numpy())
        
        self.iterations+= 1
        self.episode_steps+= 1

        return rewards_ext[0], dones[0], infos[0]
     
    
    def _add_for_plot(self, states, episode_steps):
        
        if  self.iterations%10 == 0: 
            steps = episode_steps.detach().cpu()
            z = self.model_im.forward_target(states.to(self.device)).detach().cpu()[0]

            self.steps_log.append(steps.clone())
            self.z_log.append(z)
            
            if len(self.z_log)%100 == 0:
                s_log = torch.stack(self.steps_log)
                
                s_dist= torch.cdist(s_log, s_log)
                s_dist = s_dist.flatten()
                s_dist = s_dist.numpy()

                z_log = torch.stack(self.z_log)
                z_dist= (torch.cdist(z_log, z_log)**2)/z.shape[0]
                z_dist = z_dist.flatten()
                s_dist = s_dist.flatten()

                
                plt.scatter(s_dist, z_dist, s = 0.1)
                plt.xlabel("episode steps")
                plt.ylabel("z distance")
                plt.show()
            
        
     
    
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
                if self._ppo_self_supervised_loss is not None:
                    #sample smaller batch for self supervised loss
                    states_now, states_similar = self.policy_buffer.sample_states_pairs(small_batch, self.device, self.similar_states_distance)

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

                #train snd target model, self supervised    
                loss_target_self_supervised = self._target_self_supervised_loss(self.model_im.forward_target, self._augmentations, states_now, states_similar)                


                states_a, states_b, distances = self.policy_buffer.sample_random_states_pairs(small_batch, 4, self.device)

                loss_target_causality = self._causality_loss(self.model_im.forward_causality, states_a, states_b, distances)

                #train multiple SND models, MSE loss
                loss_distillation = self._loss_distillation(states)


                loss_im = loss_target_self_supervised + loss_target_causality + loss_distillation
                self.optimizer_im.zero_grad() 
                loss_im.backward()
                self.optimizer_im.step()


               
                #log results
                self.values_logger.add("loss_ppo_self_supervised", loss_ppo_self_supervised.detach().cpu().numpy())
                self.values_logger.add("loss_target_self_supervised",   loss_target_self_supervised.detach().cpu().numpy())
                self.values_logger.add("loss_target_causality",   loss_target_causality.detach().cpu().numpy())
                self.values_logger.add("loss_distillation", loss_distillation.detach().cpu().numpy())

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

    
    #MSE loss for networks distillation model
    def _loss_distillation(self, states):         
        features_target_t       = self.model_im.forward_target(states)
        features_predicted_t    = self.model_im.forward_predictor(states)

        loss = ((features_target_t.detach() - features_predicted_t)**2).mean()

        return loss 
    
    '''
    def _causality_loss(self, forward_func, states_a, states_b, distances):
        
        distances_target = torch.sgn(distances)*torch.log(1.0 + torch.abs(distances))
        distances_target = distances_target.detach()

        distances_pred   = forward_func(states_a, states_b)

        loss = ((distances_target.detach() - distances_pred)**2).mean()


        acc = (torch.sgn(distances_pred) == torch.sgn(distances_target)).float()
        acc = acc.mean()

        self.values_logger.add("accuracy", acc.detach().cpu().numpy())


        return loss
    '''

    
    def _causality_loss(self, forward_func, states_a, states_b, distances):
        
        labels = (distances > 0.0).float()

        pred   = forward_func(states_a, states_b)
        pred   = torch.sigmoid(pred)

        loss_func = torch.nn.BCELoss()
        loss = loss_func(pred, labels)


        acc = ((pred > 0.5) == (distances > 0.0)).float()
        acc = acc.mean()

        self.values_logger.add("accuracy", acc.detach().cpu().numpy())

        return loss
    
        
   
    #compute internal motivation
    #distillation novelty detection
    def _internal_motivation(self, states_prev, states):        
        features_target_t       = self.model_im.forward_target(states)
        features_predicted_t    = self.model_im.forward_predictor(states)

        novelty_t = ((features_target_t - features_predicted_t)**2).mean(dim=1)
        novelty_t = novelty_t.detach().cpu()

        prob        = self.model_im.forward_causality(states, states_prev).squeeze(1)
        prob        = torch.sigmoid(prob)
        causality_t = torch.clip(2.0*(prob - 0.5), 0.0, 1.0)
        causality_t = causality_t.detach().cpu()

        return novelty_t, causality_t
    
 
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
   