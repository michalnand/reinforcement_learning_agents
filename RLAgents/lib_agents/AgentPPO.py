import numpy
import torch 
import time 

from .ValuesLogger      import *
from .TrajectoryBuffer  import *
from .SelfSupervised    import * 
from .Augmentations     import *
from .GrokFast          import *
  
class AgentPPO():
    def __init__(self, envs, Model, config):
        self.envs = envs
 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma              = config.gamma
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip
        self.adv_coeff          = config.adv_coeff
        self.val_coeff          = config.val_coeff 

        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count
 
        self.rnn_policy         = config.rnn_policy 

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        if config.self_supervised_loss == "vicreg":
            self.self_supervised_loss_func = loss_vicreg
            self.augmentations              = config.augmentations
        else:
            self.self_supervised_loss_func  = None
            self.augmentations              = None       

        if hasattr(config, "weight_decay"):
            self.weight_decay = config.weight_decay
        else:
            self.weight_decay = 0
        
        if hasattr(config, "use_grok_fast"):
            self.use_grok_fast = config.use_grok_fast
        else:
            self.use_grok_fast = False

        self.model = Model.Model(self.state_shape, self.actions_count)
        self.model.to(self.device)
        print(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=self.weight_decay)

        if self.use_grok_fast:
            self.grok_fast = GrokFast(self.model)
 
        self.trajctory_buffer = TrajectoryBuffer(self.steps, self.state_shape, self.actions_count, self.envs_count)
 
        
        if self.rnn_policy:
            self.hidden_state = torch.zeros((self.envs_count, ) + self.model.rnn_shape , dtype=torch.float32, device=self.device)
            self.rnn_seq_length = config.rnn_seq_length
        else:
            self.hidden_state   = None
            self.rnn_seq_length = -1


        print("gamma                    = ", self.gamma)
        print("entropy_beta             = ", self.entropy_beta)
        print("learning_rate            = ", config.learning_rate)
        print("weight_decay             = ", self.weight_decay)
        print("use_grok_fast            = ", self.use_grok_fast)
        print("adv_coeff                = ", self.adv_coeff)
        print("val_coeff                = ", self.val_coeff)
        print("batch_size               = ", self.batch_size)
        print("rnn_policy               = ", self.rnn_policy)
        print("rnn_seq_length           = ", self.rnn_seq_length)
        print("self_supervised_loss     = ", self.self_supervised_loss_func)
        print("augmentations            = ", self.augmentations)
        print("\n\n")


        self.states_t  = torch.zeros((self.envs_count, ) + self.state_shape, dtype=torch.float32)
        self.logits_t  = torch.zeros((self.envs_count, self.actions_count), dtype=torch.float32)
        self.values_t  = torch.zeros((self.envs_count, ) , dtype=torch.float32)
        self.actions_t = torch.zeros((self.envs_count, ) , dtype=int)
        self.rewards_t = torch.zeros((self.envs_count, ) , dtype=torch.float32)
        self.dones_t   = torch.zeros((self.envs_count, ) , dtype=torch.float32)

       
        self.iterations = 0   

        self.values_logger  = ValuesLogger()
        self.values_logger.add("loss_actor", 0.0)
        self.values_logger.add("loss_critic", 0.0)
        self.values_logger.add("loss_ssl", 0.0)

        self.info_logger = {} 

        
    def round_start(self): 
        pass

    def round_finish(self): 
        pass
        
    def episode_done(self, env_idx):
        pass

    def step(self, states, training_enabled, legal_actions_mask):        
        states_t  = torch.tensor(states, dtype=torch.float).detach().to(self.device)

        if hasattr(self.model, "set_seq_idx"):
            self.model.set_seq_idx(self.iterations)
        
        if self.rnn_policy: 
            logits_t, values_t, hidden_state_new = self.model.forward(states_t, self.hidden_state)
        else:
            logits_t, values_t  = self.model.forward(states_t)

        actions = self._sample_actions(logits_t, legal_actions_mask)

        states_new, rewards, dones, _, infos = self.envs.step(actions)

        #put into policy buffer
        if training_enabled:
            states_t        = states_t.detach().to("cpu")
            logits_t        = logits_t.detach().to("cpu")
            values_t        = values_t.squeeze(1).detach().to("cpu") 
            actions         = torch.from_numpy(actions).to("cpu")
            rewards_t       = torch.from_numpy(rewards).to("cpu")
            dones           = torch.from_numpy(dones).to("cpu")

            if self.rnn_policy:
                hidden_state  = self.hidden_state.detach().to("cpu")
            else:
                hidden_state  = None

            self.trajctory_buffer.add(states_t, logits_t, values_t, actions, rewards_t, dones, hidden_state)

            if self.trajctory_buffer.is_full():
                self.trajctory_buffer.compute_returns(self.gamma)
                self.train()
                self.trajctory_buffer.clear()  

        #udpate rnn hidden tate
        if self.rnn_policy:
            self.hidden_state = hidden_state_new.detach().clone()

            dones_idx = numpy.where(dones)[0]

            #clear rnn hidden state if done
            for e in dones_idx:
                self.hidden_state[e] = 0.0
    
            #hidden space stats 
            hidden_mean = (self.hidden_state**2).mean().detach().cpu().numpy().item()
            hidden_std  = self.hidden_state.std().detach().cpu().numpy().item()
            self.info_logger["hidden"] = [ round(hidden_mean, 5), round(hidden_std, 5)]

      
        self.iterations+= 1
        return states_new, rewards, dones, infos
    
    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path + "trained/model.pt")

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path + "trained/model.pt", map_location = self.device))


    def get_log(self):
        return self.values_logger.get_str() + str(self.info_logger)
    
    def _sample_actions(self, logits, legal_actions_mask):
        if legal_actions_mask is not None:
            legal_actions_mask_t  = torch.from_numpy(legal_actions_mask).to(self.device).float()
        else:
            legal_actions_mask_t = 1.0

        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)
        
        #keep only legal actions probs, and renormalise probs
        action_probs_t        = action_probs_t*legal_actions_mask_t
        action_probs_t        = action_probs_t/action_probs_t.sum(dim=-1).unsqueeze(1)

        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().to("cpu").numpy()
        
        return actions
    
    def train(self): 
        samples_count = self.steps*self.envs_count
        batch_count = samples_count//self.batch_size

        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):

                if self.rnn_policy:
                    states, logits, actions, returns, advantages, hidden_states = self.trajctory_buffer.sample_batch_seq(self.rnn_seq_length, self.batch_size, self.device)                    
                    loss_ppo = self.loss_rnn_ppo(states, logits, actions, returns, advantages, hidden_states)
                else:
                    states, logits, actions, returns, advantages = self.trajctory_buffer.sample_batch(self.batch_size, self.device)
                    loss_ppo = self.loss_ppo(states, logits, actions, returns, advantages)

                if self.self_supervised_loss_func is not None:
                    states_a, states_b = self.trajctory_buffer.sample_states_pairs(self.batch_size//self.training_epochs, 0, self.device)
                    loss_self_supervised, ssl_info = self.self_supervised_loss_func(self.model.forward_rl_ssl, self._augmentations, states_a, states_b)
                    
                    self.info_logger["ppo_ssl"] = ssl_info
                    self.values_logger.add("loss_ssl", loss_self_supervised.detach().cpu().numpy())
                else:
                    loss_self_supervised = 0    

                loss = loss_ppo + loss_self_supervised

                self.optimizer.zero_grad()        
                loss.backward()
                
                if self.use_grok_fast:  
                    self.grok_fast.step()

                if self.use_grok_fast:
                    self.info_logger["params_stats"] = params_stats(self.model)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step() 

         

    
    def loss_ppo(self, states, logits, actions, returns, advantages):
        logits_new, values_new  = self.model.forward(states)

        return self._loss_ppo(logits, actions, returns, advantages, logits_new, values_new)
    

    def loss_rnn_ppo(self, states, logits, actions, returns, advantages, hidden_states):
        seq_length = states.shape[0]

        logits_new, values_new, _  = self.model.forward(states, hidden_states)

        loss = 0.0
        for n in range(seq_length):
            loss+= self._loss_ppo(logits[n], actions[n], returns[n], advantages[n], logits_new[n], values_new[n])

        loss = loss/seq_length
        return loss
       

    def _loss_ppo(self, logits, actions, returns, advantages, logits_new, values_new):
        log_probs_old = torch.nn.functional.log_softmax(logits, dim = 1).detach()

        probs_new     = torch.nn.functional.softmax(logits_new,     dim = 1)
        log_probs_new = torch.nn.functional.log_softmax(logits_new, dim = 1)

        '''
        compute critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_new = values_new.squeeze(1)
        loss_value = (returns.detach() - values_new)**2
        loss_value = loss_value.mean()

        ''' 
        compute actor loss, surrogate loss
        '''
        advantages       = self.adv_coeff*advantages.detach() 
        #this normalisation has no effect
        advantages  = (advantages - torch.mean(advantages))/(torch.std(advantages) + 1e-10)

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

        loss = self.val_coeff*loss_value + loss_policy + loss_entropy

        self.values_logger.add("loss_actor",  loss_policy.detach().to("cpu").numpy())
        self.values_logger.add("loss_critic", loss_value.detach().to("cpu").numpy())

        return loss

    def _augmentations(self, x):    
        mask_result = torch.zeros((3, x.shape[0]), device=x.device, dtype=torch.float32)

        if "noise" in self.augmentations:
            x, mask = aug_random_apply(x, 0.5, aug_noise)
            mask_result[0] = mask
        
        if "mask" in self.augmentations:
            x, mask = aug_random_apply(x, 0.5, aug_mask)
            mask_result[1] = mask
        
        if "conv" in self.augmentations:
            x, mask = aug_random_apply(x, 0.5, aug_conv)
            mask_result[2] = mask

        return x.detach(), mask_result 
    
    