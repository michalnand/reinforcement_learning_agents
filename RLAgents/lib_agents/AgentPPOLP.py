import numpy
import torch
import time

from .ValuesLogger          import *
from .TrajectoryBufferLP    import *
from .SelfSupervised        import * 
from .Augmentations         import *

  
class AgentPPOLP(): 
    def __init__(self, envs, Model, config):
        self.envs = envs
 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.envs_count         = config.envs_count

        self.gamma              = config.gamma

        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip

        self.steps              = config.steps
        self.batch_size         = config.batch_size
        self.ssl_batch_size     = config.ssl_batch_size
        self.prompt_batch_size  = config.prompt_batch_size
        
        self.training_epochs    = config.training_epochs
        
        self.learning_rate      = config.learning_rate

        self.prompt_size        = config.prompt_size
        self.self_prompting     = config.self_prompting

        self.augmentations              = config.augmentations
        self.augmentations_probs        = config.augmentations_probs 

        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        
        

        self.model = Model.Model(self.state_shape, self.actions_count, self.prompt_size, 1)
        self.model.to(self.device)

        print(self.model)                    
        
        self.optimizer_ssl = torch.optim.Adam(self.model.model_ssl.parameters(), lr=config.learning_rate)
        self.optimizer_rl  = torch.optim.Adam(self.model.model_rl.parameters(), lr=config.learning_rate)

        #initial prompt value
        self.prompts_t     = torch.zeros((self.envs_count, self.prompt_size), dtype=torch.float32, device=self.device)
        self.task_id       = torch.zeros((self.envs_count, ), dtype=int, device=self.device)

        self.trajectory_buffer = TrajectoryBufferLP(self.steps, self.state_shape, self.prompt_size, self.actions_count, self.envs_count)
 
        print("gamma                    = ", self.gamma)
        print("entropy_beta             = ", self.entropy_beta)
        print("learning_rate            = ", self.learning_rate)
        print("augmentations            = ", self.augmentations)
        print("augmentations_probs      = ", self.augmentations_probs)
        print("prompt_size              = ", self.prompt_size)
        print("self_prompting           = ", self.self_prompting)
        print("\n\n")
      

        self.iterations = 0   

        self.values_logger  = ValuesLogger()
        self.values_logger.add("loss_actor", 0.0)
        self.values_logger.add("loss_actor_continuous", 0.0)
        self.values_logger.add("loss_critic", 0.0)
        self.values_logger.add("loss_ssl_features", 0.0)
        self.values_logger.add("loss_ssl_prompts",  0.0)

        self.info_logger = {}
        
        
    def round_start(self): 
        pass

    def round_finish(self): 
        pass
        
    def episode_done(self, env_idx):
        pass


    def step(self, states, training_enabled, legal_actions_mask):        
        #state to tensor
        states_t = torch.tensor(states, dtype=torch.float).to(self.device)

        logits_t, values_t, prompt_mean_t, prompt_var_t  = self.model.forward(states_t, self.prompts_t, self.task_id)
        
        actions = self._sample_actions(logits_t)

        states_new, rewards, dones, _, infos = self.envs.step(actions)

        #sample new prompt
        if self.self_prompting:
            self.prompts_t= self._sample_prompt(prompt_mean_t, prompt_var_t)
        
        #put into policy buffer
        if training_enabled:
            states_t        = states_t.detach().to("cpu")
            prompts_t       = self.prompts_t.detach().to("cpu")
            task_id         = self.task_id.detach().to("cpu")

            logits_t        = logits_t.detach().to("cpu")
            values_t        = values_t.squeeze(1).detach().to("cpu") 
            prompt_mean_t   = prompt_mean_t.detach().to("cpu")
            prompt_var_t    = prompt_var_t.detach().to("cpu")

            actions         = torch.from_numpy(actions).to("cpu")
            
            rewards_t       = torch.from_numpy(rewards).to("cpu")
            dones           = torch.from_numpy(dones).to("cpu")


            self.trajectory_buffer.add(states_t, prompts_t, task_id, logits_t, values_t, prompt_mean_t, prompt_var_t, actions, rewards_t, dones)

            if self.trajectory_buffer.is_full():
                self.train()

        dones_idx = numpy.where(dones)

        #clear prompt if new episode
        for e in dones_idx:
            self.prompts_t[e] = 0.0
            
        self.iterations+= 1
        return states_new, rewards, dones, infos
    
    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path + "trained/model.pt")

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path + "trained/model.pt", map_location = self.device))


    def get_log(self):
        return self.values_logger.get_str() + str(self.info_logger)
    
    def _sample_actions(self, logits):
        action_probs_t        = torch.nn.functional.softmax(logits, dim = 1)
        
        action_distribution_t = torch.distributions.Categorical(action_probs_t)
        
        action_t              = action_distribution_t.sample()
        actions               = action_t.detach().to("cpu").numpy()

        return actions
    
    def _sample_prompt(self, mu, var):
        sigma    = torch.sqrt(var)
        result   = torch.normal(mu, sigma)

        result   = torch.clip(result, -1.0, 1.0)

        return result.detach()
    
    def train(self): 
        self.trajectory_buffer.compute_returns(self.gamma)
        
        samples_count   = self.steps*self.envs_count
        batch_count     = samples_count//self.batch_size

        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                states, prompts, task_id, logits, prompt_mean, prompt_var, actions, returns, advantages = self.trajectory_buffer.sample_batch(self.batch_size, self.device)

                loss_ppo = self._loss_ppo(states, prompts, task_id, logits, prompt_mean, prompt_var, actions, returns, advantages)

                self.optimizer_rl.zero_grad()        
                loss_ppo.backward()
                torch.nn.utils.clip_grad_norm_(self.model.model_rl.parameters(), max_norm=0.5)
                self.optimizer_rl.step() 

        
        prompt_repeats        = self.ssl_batch_size//self.prompt_batch_size
        

        batch_count = (samples_count//self.ssl_batch_size)
        for batch_idx in range(batch_count):
            #common self supervised loss, for distinguish states
            states, prompts, task_id = self.trajectory_buffer.sample_states(self.ssl_batch_size, self.device)
            

            states_a = self._augmentations(states)
            states_b = self._augmentations(states)

            za = self.model.forward_ssl(states_a, prompts, task_id)
            zb = self.model.forward_ssl(states_b, prompts, task_id)

            loss_ssl_features, ssl_features = loss_vicreg_direct(za, zb) 
            self.info_logger["ssl_features"] = ssl_features

            #sample monotonic states, and prompts
            #then add little noise to prompts, to force model distinguish small prompt changes
            states, prompts, task_id = self.trajectory_buffer.sample_states(self.prompt_batch_size, self.device)
            states  = states.repeat(prompt_repeats)
            prompts = prompts.repeat(prompt_repeats)
            task_id = task_id.repeat(prompt_repeats)


            prompts_aug = self._augmentations_prompts(prompts)

            za = self.model.forward_ssl(states, prompts_aug, task_id)
            zb = self.model.forward_ssl(states, prompts_aug, task_id)

            loss_ssl_prompts, ssl_prompts = loss_vicreg_direct(za, zb) 
            self.info_logger["ssl_prompts"] = ssl_prompts




            loss_ssl = loss_ssl_features + loss_ssl_prompts

            self.optimizer_ssl.zero_grad()        
            loss_ssl.backward()
            torch.nn.utils.clip_grad_norm_(self.model.model_ssl.parameters(), max_norm=0.5)
            self.optimizer_ssl.step() 

            self.values_logger.add("loss_ssl_features", loss_ssl_features.detach().cpu().numpy())
            self.values_logger.add("loss_ssl_prompts",  loss_ssl_prompts.detach().cpu().numpy())


        self.trajectory_buffer.clear()   

    
    def _loss_ppo(self, states, prompts, task_id, logits, prompt_mean, prompt_var, actions, returns, advantages):
        log_probs_old = torch.nn.functional.log_softmax(logits, dim = 1).detach()

        logits_new, values_new, prompt_mean_new_t, prompt_var_new_t = self.model.forward(states, prompts, task_id)


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
        advantages       = advantages.detach() 
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


        '''
        compute continuous actor loss
        '''
        log_probs_old_ = self._log_prob(prompts, prompt_mean, prompt_var).detach()
        log_probs_new_ = self._log_prob(prompts, prompt_mean_new_t, prompt_var_new_t)

        ratio       = torch.exp(log_probs_new_ - log_probs_old_)
        p1          = ratio*advantages
        p2          = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages
        loss_policy_cont = -torch.min(p1, p2)  
        loss_policy_cont = loss_policy_cont.mean() 

        '''
        compute entropy loss, to avoid greedy strategy
        H = ln(sqrt(2*pi*var))
        ''' 
        loss_entropy_cont = -(torch.log(2.0*numpy.pi*prompt_var_new_t) + 1.0)/2.0
        loss_entropy_cont = self.entropy_beta*loss_entropy_cont.mean()


        loss = loss_value + loss_policy + loss_entropy

        if self.self_prompting:
            loss = loss + loss_policy_cont + loss_entropy_cont

        self.values_logger.add("loss_actor",  loss_policy.detach().to("cpu").numpy())
        self.values_logger.add("loss_actor_continuous",  loss_policy_cont.detach().to("cpu").numpy())
        self.values_logger.add("loss_critic", loss_value.detach().to("cpu").numpy())

        return loss
    
    def _log_prob(self, action, mu, var):
        p1 = -((action - mu)**2)/(2.0*var + 10**-8)
        p2 = -torch.log(torch.sqrt(2.0*numpy.pi*var)) 
        return p1 + p2
    
    
    
    def _augmentations(self, x): 
        if "mask" in self.augmentations:
            x, _ = aug_random_apply(x, self.augmentations_probs, aug_mask)

        if "noise" in self.augmentations:
            x, _ = aug_random_apply(x, self.augmentations_probs, aug_noise)

        return x.detach() 
    

    def _augmentations_prompts(self, x): 
        noise_levels = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
        noise_levels = numpy.random.choice(noise_levels, size=(x.shape[0], 1))

        noise = noise_levels*numpy.random.randn(x.shape[0], x.shape[1])

        x = x + torch.from_numpy(noise).to(x.device)

        return x.detach() 
    
    