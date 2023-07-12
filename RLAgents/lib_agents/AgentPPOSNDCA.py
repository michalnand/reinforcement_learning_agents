import numpy
import torch 

from .ValuesLogger      import *
from .PolicyBufferIM    import *  

from .PPOLoss               import *
from .SelfSupervised        import *
from .Augmentations         import *
 
         
class AgentPPOSNDCA():   
    def __init__(self, envs, ModelPPO, ModelSND, ModelSNDTarget, config):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.envs = envs  
          
        self.gamma_ext          = config.gamma_ext 
        self.gamma_int          = config.gamma_int
              
        self.ext_adv_coeff      = config.ext_adv_coeff
        self.int_adv_coeff      = config.int_adv_coeff
        self.reward_int_a_coeff = config.reward_int_a_coeff
        self.reward_int_b_coeff = config.reward_int_b_coeff
      
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

        self._target_self_awareness_loss = loss_constructor

        self.similar_states_distances = config.similar_states_distances
        

        if hasattr(config, "state_normalise"):
            self.state_normalise = config.state_normalise
        else:
            self.state_normalise = False


         
        self.augmentations                  = config.augmentations
        self.augmentations_probs            = config.augmentations_probs
        
        print("ppo_self_supervised_loss     = ", self._ppo_self_supervised_loss)
        print("target_self_supervised_loss  = ", self._target_self_supervised_loss)
        print("target_self_awareness_loss   = ", self._target_self_awareness_loss)
        print("augmentations                = ", self.augmentations)
        print("augmentations_probs          = ", self.augmentations_probs)
        print("reward_int_a_coeff           = ", self.reward_int_a_coeff)
        print("reward_int_b_coeff           = ", self.reward_int_b_coeff)
        print("rnn_policy                   = ", self.rnn_policy)
        print("similar_states_distances     = ", self.similar_states_distances)
        print("state_normalise              = ", self.state_normalise)

        print("\n\n")

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        #main ppo agent
        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.model_ppo.to(self.device)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)

        #snd destillery
        self.model_snd      = ModelSND.Model(self.state_shape)
        self.model_snd.to(self.device)
        self.optimizer_snd  = torch.optim.Adam(self.model_snd.parameters(), lr=config.learning_rate_snd)

        #multiple target models
        self.model_snd_targets      = []
        self.optimizer_snd_targets  = []
        for i in range(len(self.similar_states_distances)):
            model_snd_target      = ModelSNDTarget.Model(self.state_shape, self.actions_count)
            model_snd_target.to(self.device)
            optimizer_snd_target  = torch.optim.Adam(model_snd_target.parameters(), lr=config.learning_rate_snd_target)

            self.model_snd_targets.append(model_snd_target)
            self.optimizer_snd_targets.append(optimizer_snd_target)

 
        self.policy_buffer = PolicyBufferIM(self.steps, self.state_shape, self.actions_count, self.envs_count)


        #reset envs and fill initial state
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e], _  = self.envs.reset(e)

        self.states_prev = self.states.copy()

        self.hidden_state = torch.zeros((self.envs_count, 512), dtype=torch.float32, device=self.device)

        #optional, for state mean and variance normalisation        
        self.state_mean  = self.states.mean(axis=0)
        self.state_var   = numpy.ones_like(self.state_mean, dtype=numpy.float32)


        self.enable_training()
        self.iterations     = 0 

        self.values_logger  = ValuesLogger() 

        self.values_logger.add("internal_motivation_a_mean",    0.0)
        self.values_logger.add("internal_motivation_a_std" ,    0.0)
        self.values_logger.add("internal_motivation_b_mean",    0.0)
        self.values_logger.add("internal_motivation_b_std" ,    0.0)
        self.values_logger.add("loss_ppo_actor",                0.0)
        self.values_logger.add("loss_ppo_critic",               0.0)
        self.values_logger.add("loss_ppo_self_supervised",      0.0)
        self.values_logger.add("loss_target",                   0.0)
        self.values_logger.add("loss_distillation",             0.0)
        
        self.info_logger = {}

    def enable_training(self): 
        self.enabled_training = True
 
    def disable_training(self):
        self.enabled_training = False
 
    def main(self):         
        
        #normalise if any
        states = self._state_normalise(self.states)
        
        #state to tensor
        states = torch.tensor(states, dtype=torch.float).to(self.device)

        states_prev = torch.tensor(self.states_prev, dtype=torch.float).to(self.device)
        
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
        rewards_int_a, rewards_int_b  = self._internal_motivation(states_prev, states)

        rewards_int_a  = self.reward_int_a_coeff*rewards_int_a
        rewards_int_b  = self.reward_int_b_coeff*rewards_int_b
 
        #total motivation
        rewards_int      = torch.clip(rewards_int_a + rewards_int_b, 0.0, 1.0)
        
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
               
         
        #self._add_for_plot(states, infos, dones)
        
        #collect stats
        self.values_logger.add("internal_motivation_a_mean", rewards_int_a.mean().detach().to("cpu").numpy())
        self.values_logger.add("internal_motivation_a_std" , rewards_int_a.std().detach().to("cpu").numpy())

        self.values_logger.add("internal_motivation_b_mean", rewards_int_b.mean().detach().to("cpu").numpy())
        self.values_logger.add("internal_motivation_b_std" , rewards_int_b.std().detach().to("cpu").numpy())

        self.iterations+= 1

        
        return rewards_ext[0], dones[0], infos[0]
    
    def save(self, save_path):
        torch.save(self.model_ppo.state_dict(), save_path + "trained/model_ppo.pt")
        torch.save(self.model_snd.state_dict(), save_path + "trained/model_snd.pt")

        for i in range(len(self.model_snd_targets)):
            torch.save(self.model_snd_targets[i].state_dict(), save_path + "trained/model_snd_target_" + str(i) + ".pt")
        
        if self.state_normalise:
            with open(save_path + "trained/" + "state_mean_var.npy", "wb") as f:
                numpy.save(f, self.state_mean)
                numpy.save(f, self.state_var)
        
    def load(self, load_path):
        self.model_ppo.load_state_dict(torch.load(load_path + "trained/model_ppo.pt", map_location = self.device))
        self.model_snd.load_state_dict(torch.load(load_path + "trained/model_snd.pt", map_location = self.device))

        for i in range(len(self.model_snd_targets)):
            self.model_snd_targets[i].load_state_dict(torch.load(load_path + "trained/model_snd_target_" + str(i) + ".pt", map_location = self.device))
        
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

                    loss_ppo_self_supervised    = self._ppo_self_supervised_loss(self.model_ppo, self._augmentations, states_now, states_next, states_similar, states_random, actions, relations)                
                else:
                    loss_ppo_self_supervised    = torch.zeros((1, ), device=self.device)[0]

                #total PPO loss
                loss = loss_ppo + loss_ppo_self_supervised

                self.optimizer_ppo.zero_grad()        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()


                loss_target_all = 0
                accuracy_all    = 0
                for i in range(len(self.model_snd_targets)):
                    #sample smaller batch for self supervised loss, different distances for different models
                    states_now, states_next, states_similar, states_random, actions, relations = self.policy_buffer.sample_states_action_pairs(small_batch, self.device, self.similar_states_distances[i])

                    #train snd target model, self supervised    
                    loss_target_self_supervised = self._target_self_supervised_loss(self.model_snd_targets[i], self._augmentations, states_now, states_next, states_similar, states_random, actions, relations)                

                    loss_target_self_awareness, accuracy  = self._target_self_awareness_loss(self.model_snd_targets[i], self._augmentations, states_now, states_next, states_similar, states_random, actions, relations)                

                    #TODO : do we need loss scaling ?
                    loss_target = loss_target_self_supervised + 0.01*loss_target_self_awareness

                    self.optimizer_snd_targets[i].zero_grad() 
                    loss_target.backward()
                    self.optimizer_snd_targets[i].step()

                    loss_target_all+= loss_target.detach().to("cpu").numpy()
                    accuracy_all+= accuracy

                accuracy_all = accuracy_all/len(self.model_snd_targets)



            
                #train SND model, MSE loss (same as RND)
                loss_distillation = self._loss_distillation(states)

                self.optimizer_snd.zero_grad() 
                loss_distillation.backward()
                self.optimizer_snd.step() 
               
                #log results
                self.values_logger.add("loss_ppo_self_supervised", loss_ppo_self_supervised.detach().to("cpu").numpy())
                self.values_logger.add("loss_target",   loss_target_all)
                self.values_logger.add("loss_distillation", loss_distillation.detach().to("cpu").numpy())

                self.info_logger["accuracy"] = list(numpy.round(accuracy_all, 3))
                
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
        features_target_t = 0
        
        #sumarise target for all models
        for i in range(len(self.model_snd_targets)):
            features_target_t+= self.model_snd_targets[i](states)

        #features_target_t = features_target_t/len(self.model_snd_targets)
        
        features_predicted_t  = self.model_snd(states)
        
        loss = (features_target_t.detach() - features_predicted_t)**2

        loss  = loss.mean() 
        return loss 
   
    #compute internal motivation
    def _internal_motivation(self, states_prev, states):        
        #sumarise for all partial models
        features_target_t = 0
        for i in range(len(self.model_snd_targets)):
            features_target_t+= self.model_snd_targets[i](states)
        
        #distilation novelty motivation
        features_predicted_t = self.model_snd(states)
        novelty_t = ((features_target_t - features_predicted_t)**2).mean(dim=1)

        #sumarise for all partial models
        prediction = 0.0
        for i in range(len(self.model_snd_targets)):
            prediction+= self.model_snd_targets[i].forward_aux(states_prev, states)

        #motivation is given by how accurate model predict causality
        #prediction = torch.softmax(prediction, dim=1)


        prediction = torch.argmax(prediction.detach(), dim=1)
        causality_t = (prediction == 0)

        #input is always case : states_prev, states_now transition (class_id = 1)
        #see loss_constructor implementation : 
        #class (column) 0 : different states
        #class (column) 1 : state_prev, state_now
        #class (column) 2 : state_now, state_prev
        #causality_t = prediction[:, 0]
        
        return novelty_t, causality_t
    
 
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
   