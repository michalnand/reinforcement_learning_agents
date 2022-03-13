import numpy
import torch 
from .PolicyBufferIM    import *  
from .FeaturesBuffer    import *
from .RunningStats      import *  

import sklearn.manifold
import matplotlib.pyplot as plt
import cv2
 
       
class AgentPPOContrastive():   
    def __init__(self, envs, ModelPPO, ModelContrastive, config):
        self.envs = envs  
      
        self.gamma_ext          = config.gamma_ext 
        self.gamma_int          = config.gamma_int
            
        self.ext_adv_coeff      = config.ext_adv_coeff
        self.int_adv_coeff      = config.int_adv_coeff
        self.int_reward_coeff   = config.int_reward_coeff
    
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip 
    
        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count 

        self.features_buffer_size = config.features_buffer_size


        self.normalise_state_mean = config.normalise_state_mean
        self.normalise_state_std  = config.normalise_state_std

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)

        features_count              = 256
        self.model_contrastive      = ModelContrastive.Model(self.state_shape, features_count)
        self.optimizer_contrastive  = torch.optim.Adam(self.model_contrastive.parameters(), lr=config.learning_rate_contrastive)

      
        self.policy_buffer   = PolicyBufferIM(self.steps, self.state_shape, self.actions_count, self.envs_count, self.model_ppo.device, True)
        self.features_buffer = FeaturesBuffer(self.features_buffer_size, self.envs_count, (features_count, ))

        for e in range(self.envs_count):
            self.envs.reset(e)
        
        self.states_running_stats       = RunningStats(self.state_shape)

        if self.envs_count > 1:
            self._init_running_stats()

        #reset envs and fill initial state
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e] = self.envs.reset(e).copy()

        self.enable_training()
        self.iterations                     = 0 

        self.log_loss_contrastive           = 0.0
        self.log_loss_actor                 = 0.0
        self.log_loss_critic                = 0.0

        self.log_internal_motivation_mean   = 0.0
        self.log_internal_motivation_std    = 0.0

        #self.vis_features = []
        #self.vis_labels   = []


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
 
        #update long term states mean and variance
        self.states_running_stats.update(states_np)

        #curiosity motivation
        rewards_int    = self._curiosity(states_t)

        rewards_int    = numpy.clip(self.int_reward_coeff*rewards_int, 0.0, 1.0)
        

        #put into policy buffer
        if self.enabled_training:
            self.policy_buffer.add(states_np, logits_np, values_ext_np, values_int_np, actions, rewards_ext, rewards_int, dones)

            if self.policy_buffer.is_full():
                self.train()
        
        for e in range(self.envs_count): 
            if dones[e]:
                self.states[e] = self.envs.reset(e).copy()

        '''
        states_norm_t   = self._norm_state(states_t)
        features        = self.model_snd_target(states_norm_t)
        features        = features.detach().to("cpu").numpy()

        self.vis_features.append(features[0])
        self.vis_labels.append(infos[0]["room_id"])

        if dones[0]:
            print("training t-sne")

            features_embedded = sklearn.manifold.TSNE(n_components=2).fit_transform(self.vis_features)

            print("result shape = ", features_embedded.shape)

            plt.clf()
            plt.scatter(features_embedded[:, 0], features_embedded[:, 1], c=self.vis_labels, cmap=plt.cm.get_cmap("jet", numpy.max(self.vis_labels)))
            plt.colorbar(ticks=range(10))
            plt.tight_layout()
            plt.show()

            self.vis_features   = []
            self.vis_labels     = []
        '''
        

        #collect stats
        k = 0.02
        self.log_internal_motivation_mean   = (1.0 - k)*self.log_internal_motivation_mean + k*rewards_int.mean()
        self.log_internal_motivation_std    = (1.0 - k)*self.log_internal_motivation_std  + k*rewards_int.std()

        self.iterations+= 1
        return rewards_ext[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_contrastive.save(save_path + "trained/")

    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
        self.model_contrastive.load(load_path + "trained/")
 
    def get_log(self): 
        result = "" 

        result+= str(round(self.log_loss_contrastive, 7)) + " "
        result+= str(round(self.log_loss_actor, 7)) + " "
        result+= str(round(self.log_loss_critic, 7)) + " "

        result+= str(round(self.log_internal_motivation_mean, 7)) + " "
        result+= str(round(self.log_internal_motivation_std, 7)) + " "

        return result 

    def render(self, env_id):
        size            = 256

        states_t        = torch.tensor(self.states, dtype=torch.float).detach().to(self.model_ppo.device)

        state           = self._norm_state(states_t)[env_id][0].detach().to("cpu").numpy()

        state_im        = cv2.resize(state, (size, size))
        state_im        = numpy.clip(state_im, 0.0, 1.0)

        cv2.imshow("RND agent", state_im)
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
                states, states_next, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int = self.policy_buffer.sample_batch(self.batch_size, self.model_ppo.device)

                #train PPO model
                loss_ppo = self._compute_loss_ppo(states, logits, actions, returns_ext, returns_int, advantages_ext, advantages_int)

                self.optimizer_ppo.zero_grad()        
                loss_ppo.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ppo.parameters(), max_norm=0.5)
                self.optimizer_ppo.step()


                #train contrastive model, MSE loss

                states, _, _ = self.policy_buffer.sample_states(64)

                loss_contrastive = self._compute_loss_contrastive(states)

                self.optimizer_contrastive.zero_grad() 
                loss_contrastive.backward()
                self.optimizer_contrastive.step()

                #log results
                k = 0.02
                self.log_loss_contrastive  = (1.0 - k)*self.log_loss_contrastive + k*loss_contrastive.detach().to("cpu").numpy()
       
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


    
    def _compute_loss_contrastive(self, state):
        

        state_norm = self._norm_state(state)
        
        #states augmentation
        xa = self._aug(state_norm)
        xb = self._aug(state_norm)
 
        za = self.model_contrastive(xa)   
        zb = self.model_contrastive(xb) 

        #distances, each from each 
        distances = ((za.unsqueeze(1) - zb)**2).mean(dim=2)

        #close states are on diagonal, set 0 on diagonal, 1 else
        n = distances.shape[0]
        labels  = 1.0 - torch.eye(n, device=distances.device)

        #MSE loss
        loss = ((labels - distances)**2)
        loss = loss.mean()

        return loss
    
    '''
    def _compute_loss_contrastive(self, state):
        x = self._norm_state(state)
        z = self.model_contrastive(x)  

        #info NCE loss
        logits      = torch.matmul(z, z.t())
        loss        = torch.nn.functional.cross_entropy(logits, torch.arange(z.shape[0]).to(z.device))

        return loss
    '''

    #compute internal motivation
    def _curiosity(self, state_t):
        state_norm_t    = self._norm_state(state_t)

        features_t      = self.model_contrastive(state_norm_t)
        features_t      = features_t.detach().to("cpu")

        curiosity = self.features_buffer.compute(features_t, 32)

        self.features_buffer.add(features_t)

        return curiosity


    #normalise mean and std for state
    def _norm_state(self, state_t):
        
        state_norm_t = state_t

        if self.normalise_state_mean:
            mean = torch.from_numpy(self.states_running_stats.mean).to(state_t.device).float()
            state_norm_t = state_norm_t - mean

        if self.normalise_state_std:
            std  = torch.from_numpy(self.states_running_stats.std).to(state_t.device).float()            
            state_norm_t = torch.clamp(state_norm_t/std, -1.0, 1.0)

        return state_norm_t 

    #random policy for stats init
    def _init_running_stats(self, steps = 256):
        for _ in range(steps):
            #random action
            actions = numpy.random.randint(0, self.actions_count, (self.envs_count))
            states, _, dones, _ = self.envs.step(actions)

            #update stats
            self.states_running_stats.update(states)

            for e in range(self.envs_count): 
                if dones[e]:
                    self.envs.reset(e)

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


   