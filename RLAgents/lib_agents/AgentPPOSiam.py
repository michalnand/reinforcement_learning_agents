from math import dist
from RLAgents.lib_agents.FeaturesBuffer import FeaturesBuffer
import numpy
import torch 
from .PolicyBufferIM    import *  
from .FeaturesBuffer    import *
      
class AgentPPOSiam():   
    def __init__(self, envs, ModelPPO, ModelSimSiam, config):
        self.envs = envs  
    
        self.gamma_ext          = config.gamma_ext 
        self.gamma_int          = config.gamma_int
            
        self.ext_adv_coeff      = config.ext_adv_coeff
        self.int_adv_coeff      = config.int_adv_coeff
    
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

        self.features_count       = config.features_count
        
        self.model_siam      = ModelSimSiam.Model(self.state_shape, self.features_count)
        self.optimizer_siam  = torch.optim.Adam(self.model_siam.parameters(), lr=config.learning_rate_siam)
 
        self.policy_buffer      = PolicyBufferIM(self.steps, self.state_shape, self.actions_count, self.envs_count, self.model_ppo.device, True)
        self.features_buffer    = FeaturesBuffer(config.buffer_size, (self.features_count, ), self.envs_count, "cpu")


        #reset envs and fill initial state
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e] = self.envs.reset(e).copy()
 
        self.enable_training()
        self.iterations                     = 0 

        self.log_loss_siam                  = 0.0
        self.log_loss_actor                 = 0.0
        self.log_loss_critic                = 0.0

        self.log_internal_motivation_mean   = 0.0
        self.log_internal_motivation_std    = 0.0
        self.log_acc_siam                   = 0.0

    def enable_training(self):
        self.enabled_training = True
 
    def disable_training(self):
        self.enabled_training = False

    def main(self): 
        #state to tensor
        states_t        = torch.tensor(self.states, dtype=torch.float).detach().to(self.model_ppo.device)

        #compute model output
        logits_t, values_ext_t, values_int_t  = self.model_ppo.forward(states_t)
        
        states_np       = states_t.detach().to("cpu").numpy()
        logits_np       = logits_t.detach().to("cpu").numpy()
        values_ext_np   = values_ext_t.squeeze(1).detach().to("cpu").numpy()
        values_int_np   = values_int_t.squeeze(1).detach().to("cpu").numpy()

        #collect actions
        actions = self._sample_actions(logits_t)
         
        #execute action
        states, rewards_ext, dones, infos = self.envs.step(actions)

        self.states = states.copy()

        #outlier motivation
        rewards_int    = self._outlier_motivation(states_t)
            
        rewards_int    = numpy.clip(0.1*rewards_int, 0.0, 1.0)
        
        #put into policy buffer
        if self.enabled_training:
            self.policy_buffer.add(states_np, logits_np, values_ext_np, values_int_np, actions, rewards_ext, rewards_int, dones)

            if self.policy_buffer.is_full():
                self.train()
        
        for e in range(self.envs_count): 
            if dones[e]:
                self.states[e] = self.envs.reset(e).copy()

                state_t = torch.from_numpy(self.states[e]).to(self.model_siam.device).float().unsqueeze(0)
                features = self.model_siam(state_t)
                features = features.squeeze(0).detach().to("cpu") 

                self.features_buffer.reset(e, features) 

        #collect stats
        k = 0.02
        self.log_internal_motivation_mean   = (1.0 - k)*self.log_internal_motivation_mean + k*rewards_int.mean()
        self.log_internal_motivation_std    = (1.0 - k)*self.log_internal_motivation_std  + k*rewards_int.std()

        self.iterations+= 1
        return rewards_ext[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_siam.save(save_path + "trained/")
 
    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
        self.model_siam.load(load_path + "trained/")

    def get_log(self): 
        result = "" 

        result+= str(round(self.log_loss_siam, 7)) + " "
        result+= str(round(self.log_loss_actor, 7)) + " "
        result+= str(round(self.log_loss_critic, 7)) + " "

        result+= str(round(self.log_internal_motivation_mean, 7)) + " "
        result+= str(round(self.log_internal_motivation_std, 7)) + " "
        result+= str(round(self.log_acc_siam, 7)) + " "

        return result     



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

                #train Siam model, contrastive loss
                states_a_t, states_b_t, labels_t = self.policy_buffer.sample_states(64)
                
                loss_siam, acc = self._compute_contrastive_loss(states_a_t, states_b_t, labels_t)                
 
                self.optimizer_siam.zero_grad() 
                loss_siam.backward()
                self.optimizer_siam.step()

                k = 0.02
                self.log_loss_siam  = (1.0 - k)*self.log_loss_siam + k*loss_siam.detach().to("cpu").numpy()
                self.log_acc_siam   = (1.0 - k)*self.log_acc_siam  + k*acc

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
        k = 0.02
        self.log_loss_actor     = (1.0 - k)*self.log_loss_actor  + k*loss_actor.mean().detach().to("cpu").numpy()
        self.log_loss_critic    = (1.0 - k)*self.log_loss_critic + k*loss_critic.mean().detach().to("cpu").numpy()

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

    def _compute_contrastive_loss(self, states_a_t, states_b_t, target_t, confidence = 0.5):
        
        target_t = target_t.to(self.model_siam.device)
        xa = self._aug(states_a_t[:, 0]).unsqueeze(1).detach().to(self.model_siam.device)
        xb = self._aug(states_b_t[:, 0]).unsqueeze(1).detach().to(self.model_siam.device)

        za = self.model_siam(xa)  
        zb = self.model_siam(xb) 

        predicted = ((za - zb)**2).mean(dim=1)

        loss = ((target_t - predicted)**2).mean()

        target      = target_t.detach().to("cpu").numpy()
        predicted   = predicted.detach().to("cpu").numpy()

        true_positive = numpy.sum(1.0*(target > confidence)*(predicted > confidence))
        true_negative = numpy.sum(1.0*(target < (1.0-confidence))*(predicted < (1.0-confidence)))
        acc = 100.0*(true_positive + true_negative)/target.shape[0]

        return loss, acc

      

    #compute internal motivation
    def _outlier_motivation(self, state_t):
        features_t = self.model_siam(state_t).detach().to("cpu")

        mean, std, max, min, b_std = self.features_buffer.compute(features_t)

        self.features_buffer.add(features_t) 

        return b_std.detach().to("cpu").numpy()


    def _aug(self, x, k = 0.1):
        x  = self._aug_random_flip(x, dim=1)
        x  = self._aug_random_flip(x, dim=2)
        x  = self._aug_random_noise(x,  k)
  
        return x

    def _aug_random_flip(self, x, dim = 1):
        apply  = 1.0*(torch.rand((x.shape[0], 1, 1)) > 0.5)

        flipped = torch.flip(x, [dim]) 

 
        return (1 - apply)*x + apply*flipped

        
    def _aug_random_noise(self, x, k): 
        pointwise_noise   = k*(2.0*torch.rand(x.shape) - 1.0)
        return x + pointwise_noise

    def _aug_resize(self, x, scale = 2):
        apply  = 1.0*(torch.rand((x.shape[0], 1, 1)) > 0.5)

        ds      = torch.nn.AvgPool2d(scale, scale).to(x.device)
        us      = torch.nn.Upsample(scale_factor=scale).to(x.device)
        scaled  = us(ds(x.unsqueeze(1))).squeeze(1)

        return (1 - apply)*x + apply*scaled



