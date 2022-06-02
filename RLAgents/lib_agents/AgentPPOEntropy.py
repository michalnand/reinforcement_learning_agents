import numpy
import torch

from .ValuesLogger      import * 
from .PolicyBufferIM    import *  
from .FeaturesBuffer    import *
import cv2
      
class AgentPPOEntropy():   
    def __init__(self, envs, ModelPPO, ModelFeatures, config):
        self.envs = envs  
    
        self.gamma_ext          = config.gamma_ext 
        self.gamma_int          = config.gamma_int
            
        self.ext_adv_coeff          = config.ext_adv_coeff
        self.int_adv_coeff          = config.int_adv_coeff
        self.int_reward_coeff       = config.int_reward_coeff
        self.regularisation_coeff   = config.regularisation_coeff
    
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip 
    
        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count 



        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)
 
        self.model_features      = ModelFeatures.Model(self.state_shape)
        self.optimizer_features  = torch.optim.Adam(self.model_features.parameters(), lr=config.learning_rate_features)
 
        self.policy_buffer      = PolicyBufferIM(self.steps, self.state_shape, self.actions_count, self.envs_count)

        self.features_buffer    = FeaturesBuffer(config.features_buffer_size, self.envs_count, 256)

        if config.features_loss == "mse":
            self._features_loss = self._contrastive_loss_mse
        elif config.features_loss == "nce": 
            self._features_loss = self._contrastive_loss_nce
        else:
            self._features_loss = None

        for e in range(self.envs_count):
            self.envs.reset(e)
        
        #reset envs and fill initial state
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e] = self.envs.reset(e).copy()


        self.enable_training()
        self.iterations                     = 0 

        self.values_logger                  = ValuesLogger()

        self.values_logger.add("loss_features", 0.0)
        self.values_logger.add("loss_actor", 0.0)
        self.values_logger.add("loss_critic", 0.0)
        self.values_logger.add("internal_motivation_mean", 0.0)
        self.values_logger.add("internal_motivation_std", 0.0)
        self.values_logger.add("features_magnitude", 0.0)

    def enable_training(self):
        self.enabled_training = True
 
    def disable_training(self):
        self.enabled_training = False

    def main(self): 
        #state to tensor
        states        = torch.tensor(self.states, dtype=torch.float).detach().to(self.model_ppo.device)

        #compute model output
        logits, values_ext, values_int  = self.model_ppo.forward(states)
        
        #collect actions
        actions = self._sample_actions(logits) 
         
        #execute action
        states_new, rewards_ext, dones, infos = self.envs.step(actions)

        #curiosity motivation
        rewards_int    = self._entropy(states)        
        rewards_int    = torch.clip(self.int_reward_coeff*rewards_int, 0.0, 1.0)
        
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


        self.states = states_new.copy()
        
        for e in range(self.envs_count): 
            if dones[e]:
                self.states[e] = self.envs.reset(e).copy()
                self.features_buffer.reset(e)

        #collect stats
        self.values_logger.add("internal_motivation_mean", rewards_int.mean().detach().to("cpu").numpy())
        self.values_logger.add("internal_motivation_std" , rewards_int.std().detach().to("cpu").numpy())

       
        self.iterations+= 1
        return rewards_ext[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_features.save(save_path + "trained/")

    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
        self.model_features.load(load_path + "trained/")

    def get_log(self): 
        return self.values_logger.get_str()

    
    def render(self, env_id):
        size            = 256

        states_t        = torch.tensor(self.states, dtype=torch.float).detach().to(self.model_ppo.device)

        state           = self._norm_state(states_t)[env_id][0].detach().to("cpu").numpy()

        state_im        = cv2.resize(state, (size, size))
        state_im        = numpy.clip(state_im, 0.0, 1.0)

        cv2.imshow("Entropy agent", state_im)
        cv2.waitKey(1)
    
        



    def _sample_actions(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().to("cpu").numpy()
        return actions
    
    def train(self): 
        self.policy_buffer.compute_returns(self.gamma_ext, self.gamma_int)

        batch_count = self.steps//self.batch_size

        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                states, _, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)

                #train PPO model
                loss_ppo = self._compute_loss_ppo(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int)

                self.optimizer_ppo.zero_grad()        
                loss_ppo.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()


                #smaller batch for regularisation
                small_batch = 64
                states_a, _, _ = self.policy_buffer.sample_states(small_batch, 0.5, self.model_features.device)

                #train features model, contrastive loss
                if self._features_loss is not None:
                    loss_features = self._features_loss(self.model_features, states_a, True)

                    self.optimizer_features.zero_grad() 
                    loss_features.backward()
                    self.optimizer_features.step()

              
                    self.values_logger.add("loss_features", loss_features.detach().to("cpu").numpy())
        
        self.policy_buffer.clear() 

    
    def _compute_loss_ppo(self, states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int):
        logits_new, values_ext_new, values_int_new  = self.model_ppo.forward(states)

        #critic loss
        loss_critic = self._compute_critic_loss(values_ext_new, returns_ext, values_int_new, returns_int)

        #actor loss        
        advantages  = self.ext_adv_coeff*advantages_ext + self.int_adv_coeff*advantages_int
        advantages  = advantages.detach() 
        loss_policy, loss_entropy  = self._compute_actor_loss(logits, logits_new, advantages, actions)

        loss_actor = loss_policy + loss_entropy
        
        #total loss
        loss = 0.5*loss_critic + loss_actor

        #store to log
        self.values_logger.add("loss_actor", loss_actor.mean().detach().to("cpu").numpy())
        self.values_logger.add("loss_critic", loss_critic.mean().detach().to("cpu").numpy())
        
        return loss 

    #MSE critic loss
    def _compute_critic_loss(self, values_ext_new, returns_ext, values_int_new, returns_int):
        ''' 
        compute external critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_ext_new  = values_ext_new.squeeze(1)
        loss_ext_value  = (returns_ext.detach() - values_ext_new)**2
        loss_ext_value  = loss_ext_value.mean()

        '''
        compute internal critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_int_new  = values_int_new.squeeze(1)
        loss_int_value  = (returns_int.detach() - values_int_new)**2
        loss_int_value  = loss_int_value.mean()
        
        loss_critic     = loss_ext_value + loss_int_value
        return loss_critic

    #PPO actor loss
    def _compute_actor_loss(self, logits, logits_new, advantages, actions):
        log_probs_old = torch.nn.functional.log_softmax(logits, dim = 1).detach()

        probs_new     = torch.nn.functional.softmax(logits_new, dim = 1)
        log_probs_new = torch.nn.functional.log_softmax(logits_new, dim = 1)

        ''' 
        compute actor loss, surrogate loss
        '''
        log_probs_new_  = log_probs_new[range(len(log_probs_new)), actions]
        log_probs_old_  = log_probs_old[range(len(log_probs_old)), actions]
                        
        ratio       = torch.exp(log_probs_new_ - log_probs_old_)
        p1          = ratio*advantages
        p2          = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages
        loss_policy = -torch.min(p1, p2)  
        loss_policy = loss_policy.mean()
    
        ''' 
        compute entropy loss, to avoid greedy strategy
        L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
        '''
        loss_entropy = (probs_new*log_probs_new).sum(dim = 1)
        loss_entropy = self.entropy_beta*loss_entropy.mean()

        return loss_policy, loss_entropy


    def _contrastive_loss_mse(self, model, states_a, augmentation):
        xa = states_a.clone() 
        xb = states_a.clone()

        #states augmentation
        if augmentation:
            xa = self._aug(xa)
            xb = self._aug(xb)
 
        #obtain features from model
        if hasattr(model, "forward_features"):
            za = model.forward_features(xa)  
            zb = model.forward_features(xb) 
        else:
            za = model(xa)  
            zb = model(xb) 


        distances  = torch.cdist(za, zb)/za.shape[1]

        batch_size  = xa.shape[0]


        target = 1.0 - torch.eye(batch_size).to(distances.device)

        #de-biasing weights, ones on diagonal, 1/batch_size else
        diag        = torch.eye(batch_size).to(distances.device)
        w           = diag + (1.0/batch_size)*(1.0 - diag)

        #mse loss
        loss_mse = (target - distances)**2
        loss_mse = (w*loss_mse).mean()

        #magnitude regularisation, keep magnitude in small numbers

        #L2 magnitude regularisation
        magnitude       = (za.norm(dim=1, p=2) + zb.norm(dim=1, p=2)).mean()
        loss_magnitude  = self.regularisation_coeff*magnitude
 
        loss = loss_mse + loss_magnitude

        self.values_logger.add("snd_magnitude", magnitude.detach().to("cpu").numpy())

        return loss


    def _contrastive_loss_nce(self, model, states_a, augmentation):
        xa = states_a.clone() 
        xb = states_a.clone()

        #states augmentation
        if augmentation:
            xa = self._aug(xa)
            xb = self._aug(xb)
 
        #obtain features from model
        if hasattr(model, "forward_features"):
            za = model.forward_features(xa)  
            zb = model.forward_features(xb) 
        else:
            za = model(xa)  
            zb = model(xb) 


        probs  = torch.sigmoid(torch.matmul(za, zb.t())/za.shape[1])

        target = torch.eye(probs.shape[0]).to(probs.device)

        #de-biasing weights, ones on diagonal, 1/batch_size else
        batch_size  = xa.shape[0]
        diag        = torch.eye(batch_size).to(probs.device)
        w           = diag + (1.0/batch_size)*(1.0 - diag)

        #info nce loss
        loss_nce = -(target*torch.log(probs) + (1.0 - target)*torch.log(1.0 - probs))
        loss_nce = (w*loss_nce).mean()

        #magnitude regularisation, keep magnitude in small numbers

        #L2 magnitude regularisation
        magnitude       = (za.norm(dim=1, p=2) + zb.norm(dim=1, p=2)).mean()
        loss_magnitude  = self.regularisation_coeff*magnitude

        loss = loss_nce + loss_magnitude

        self.values_logger.add("snd_magnitude", magnitude.detach().to("cpu").numpy())

        return loss
    
    #compute internal motivation
    def _entropy(self, state_t):
        features_t  = self.model_features(state_t)
        entropy_t   = self.features_buffer.add(features_t)
        
        return entropy_t
 
    def _aug(self, x):
        '''
        x = self._aug_random_apply(x, 0.5, self._aug_mask)
        x = self._aug_random_apply(x, 0.5, self._aug_resize2)
        x = self._aug_noise(x, k = 0.2)
        '''

        #this works perfect
        x = self._aug_random_apply(x, 0.5, self._aug_resize2)
        x = self._aug_random_apply(x, 0.25, self._aug_resize4)
        x = self._aug_random_apply(x, 0.125, self._aug_mask)
        x = self._aug_noise(x, k = 0.2)
        
        return x

    def _aug_random_apply(self, x, p, aug_func):
        shape  = (x.shape[0], ) + (1,)*(len(x.shape)-1)
        apply  = 1.0*(torch.rand(shape, device=x.device) < p)

        return (1 - apply)*x + apply*aug_func(x) 
 

    def _aug_resize(self, x, scale = 2):
        ds      = torch.nn.AvgPool2d(scale, scale).to(x.device)
        us      = torch.nn.Upsample(scale_factor=scale).to(x.device)

        scaled  = us(ds(x))  
        return scaled

    def _aug_resize2(self, x):
        return self._aug_resize(x, 2)

    def _aug_resize4(self, x):
        return self._aug_resize(x, 4)

    def _aug_mask(self, x, p = 0.1):
        mask = 1.0*(torch.rand_like(x) < (1.0 - p))
        return x*mask  

    def _aug_noise(self, x, k = 0.2): 
        pointwise_noise   = k*(2.0*torch.rand(x.shape, device=x.device) - 1.0)
        return x + pointwise_noise


   