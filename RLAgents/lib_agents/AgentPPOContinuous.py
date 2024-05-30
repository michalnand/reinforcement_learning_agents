import numpy
import torch

from .ValuesLogger              import *
from .TrajectoryBufferContinuous    import *

class AgentPPOContinuous():
    def __init__(self, envs, Model, config):
        self.envs = envs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma              = config.gamma
        self.entropy_beta       = config.entropy_beta
        self.eps_clip           = config.eps_clip
        self.adv_coeff          = config.adv_coeff
        self.val_coeff          = config.val_coeff
        self.var_coeff          = config.var_coeff
        

        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count
        self.rnn_policy         = config.rnn_policy


        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.shape[0]

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.model.to(self.device)


        

        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
 
        self.trajectory_buffer  = TrajectoryBufferContinuous(self.steps, self.state_shape, self.actions_count, self.envs_count, self.device)

        if self.rnn_policy:
            self.hidden_state = torch.zeros((self.envs_count, self.model.n_hidden_state), dtype=torch.float32, device=self.device)
        else:
            self.hidden_state = None

        self.iterations = 0

        self.values_logger                  = ValuesLogger()
     
        self.values_logger.add("loss_policy", 0.0)
        self.values_logger.add("loss_value", 0.0)
        self.values_logger.add("loss_entropy", 0.0)
        self.values_logger.add("variance", 0.0)

        print(self.model)

        print("gamma                    = ", self.gamma)
        print("entropy_beta             = ", self.entropy_beta)
        print("learning_rate            = ", config.learning_rate)
        print("adv_coeff                = ", self.adv_coeff)
        print("val_coeff                = ", self.val_coeff)
        print("var_coeff                = ", self.var_coeff)
        print("steps                    = ", self.steps)
        print("batch_size               = ", self.batch_size)
        print("rnn_policy               = ", self.rnn_policy)
        print("\n\n")
     

    def get_log(self): 
        return self.values_logger.get_str()

    def round_start(self): 
        pass

    def round_finish(self): 
        pass
        
    def episode_done(self, env_idx):
        pass

    def step(self, states, training_enabled, legal_actions_mask):        
    
        states_t  = torch.tensor(states, dtype=torch.float).detach().to(self.device)
 
        if self.rnn_policy: 
            mu, var, values, hidden_state_new = self.model.forward(states_t, self.hidden_state)
        else:
            mu, var, values = self.model.forward(states_t)


        mu_np   = mu.detach().to("cpu").numpy()
        var_np  = self.var_coeff*var.detach().to("cpu").numpy()

        '''
        actions = numpy.zeros((self.envs_count, self.actions_count))
        for e in range(self.envs_count):
            actions[e] = self._sample_action(mu_np[e], self.var_coeff*var_np[e])
        '''

        actions = self._sample_action(mu_np, var_np)

        states_new, rewards, dones, _, infos = self.envs.step(actions)
        
        if training_enabled: 
            states      = states_t.detach().to("cpu")
            values      = values.squeeze(1).detach().to("cpu")
            mu          = mu.detach().to("cpu")
            var         = var.detach().to("cpu")

            actions     = torch.from_numpy(actions).to("cpu")
            rewards_    = torch.from_numpy(rewards).to("cpu")
            dones       = torch.from_numpy(dones).to("cpu")

            if self.rnn_policy:
                hidden_state = self.hidden_state.detach().to("cpu")
            else:
                hidden_state = None
             
            self.trajectory_buffer.add(states, values, actions, mu, var, rewards_, dones, hidden_state)

            if self.trajectory_buffer.is_full():
                self.train()

        #udpate rnn hiddens state
        if self.rnn_policy:
            self.hidden_state = hidden_state_new.detach().clone()

            #clear rnn hidden state if done
            dones_idx = numpy.where(dones)
            for e in dones_idx:
                self.hidden_state[e] = torch.zeros(self.hidden_state.shape[1], dtype=torch.float32, device=self.device)
        

        self.iterations+= 1
        return states_new, rewards, dones, infos
    
    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path + "trained/model.pt")

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path + "trained/model.pt", map_location = self.device))


    def _sample_action(self, mu, var):
        sigma    = numpy.sqrt(var)
        action   = numpy.random.normal(mu, sigma)
        action   = numpy.clip(action, -1, 1)
        return action
    
    def train(self): 
        self.trajectory_buffer.compute_returns(self.gamma)

        samples_count = self.steps*self.envs_count
        batch_count = samples_count//self.batch_size

        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                states, values, actions, actions_mu, actions_var, rewards, dones, returns, advantages, hidden_state = self.trajectory_buffer.sample_batch(self.batch_size, self.device)

                loss = self._ppo_loss(states, actions, actions_mu, actions_var, returns, advantages, hidden_state)

                self.optimizer.zero_grad()        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step() 

        self.trajectory_buffer.clear()   
    
    def _ppo_loss(self, states, actions, actions_mu, actions_var, returns, advantages, hidden_state):

        if self.rnn_policy:
            mu_new, var_new, values_new, _ = self.model.forward(states, hidden_state)        
        else:
            mu_new, var_new, values_new = self.model.forward(states)        

        log_probs_old = self._log_prob(actions, actions_mu, actions_var).detach()
        log_probs_new = self._log_prob(actions, mu_new, var_new)

        '''
        compute critic loss, as MSE
        L = (T - V(s))^2
        '''
        values_new = values_new.squeeze(1)
        loss_value = (returns.detach() - values_new)**2
        loss_value = loss_value.mean()
        
 
        '''
        clipped loss
        compute actor loss with KL divergence loss to prevent policy collapse
        see https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#ppo
        with adaptive kl_coeff coefficient
        https://github.com/rrmenon10/PPO/blob/7d18619960913d39a5fb0143548abbaeb02f410e/pgrl/algos/ppo_adpkl.py#L136
        '''
        advantages  = advantages.unsqueeze(1).detach()
        advantages  = (advantages - torch.mean(advantages))/(torch.std(advantages) + 1e-10)

        ratio       = torch.exp(log_probs_new - log_probs_old)
        p1          = ratio*advantages
        p2          = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantages
        loss_policy = -torch.min(p1, p2)  
        loss_policy = loss_policy.mean() 

      

        '''
        compute entropy loss, to avoid greedy strategy
        H = ln(sqrt(2*pi*var))
        ''' 
        loss_entropy = -(torch.log(2.0*numpy.pi*var_new) + 1.0)/2.0
        loss_entropy = self.entropy_beta*loss_entropy.mean()
 
        loss = loss_value + loss_policy + loss_entropy 

        self.values_logger.add("loss_policy",   loss_policy.detach().to("cpu").numpy())
        self.values_logger.add("loss_value",    loss_value.detach().to("cpu").numpy())
        self.values_logger.add("loss_entropy",  loss_entropy.detach().to("cpu").numpy())
        self.values_logger.add("variance",      var_new.mean().detach().to("cpu").numpy())

        
        return loss

    def _log_prob(self, action, mu, var):
        p1 = -((action - mu)**2)/(2.0*var + 0.00000001)
        p2 = -torch.log(torch.sqrt(2.0*numpy.pi*var)) 

        return p1 + p2

