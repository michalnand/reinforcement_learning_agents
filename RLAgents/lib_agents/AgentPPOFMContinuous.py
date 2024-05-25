import numpy
import torch

from .ValuesLogger              import *
from .TrajectoryBufferContinuous    import *

class AgentPPOFMContinuous():
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

        self.prediction_rollout = config.prediction_rollout
        self.training_rollout   = config.training_rollout
        
        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count


        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.shape[0]

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.model.to(self.device)

        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
 
        self.trajectory_buffer     = TrajectoryBufferContinuous(self.steps, self.state_shape, self.actions_count, self.envs_count, self.device)
        self.trajectory_buffer_im  = TrajectoryBufferContinuous(self.steps, self.state_shape, self.actions_count, self.envs_count, self.device)

      
        self.iterations = 0

        self.values_logger                  = ValuesLogger()
     
        self.values_logger.add("loss_policy",   0.0)
        self.values_logger.add("loss_value",    0.0)
        self.values_logger.add("loss_entropy",  0.0)
        self.values_logger.add("variance",      0.0)
        self.values_logger.add("fm_mse_start",  0.0)
        self.values_logger.add("fm_mse_mean",   0.0)
        self.values_logger.add("fm_mse_end",    0.0)
        self.values_logger.add("fm_q",          0.0)
        self.values_logger.add("loss_policy_im",0.0)
        

        print(self.model)

        print("gamma                    = ", self.gamma)
        print("entropy_beta             = ", self.entropy_beta)
        print("learning_rate            = ", config.learning_rate)
        print("adv_coeff                = ", self.adv_coeff)
        print("val_coeff                = ", self.val_coeff)
        print("steps                    = ", self.steps)
        print("batch_size               = ", self.batch_size)
        print("prediction_rollout       = ", self.prediction_rollout)
        print("training_rollout         = ", self.training_rollout)
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
 
        mu, var, values   = self.model.forward(states_t)

       
        mu_np   = mu.detach().to("cpu").numpy()
        var_np  = var.detach().to("cpu").numpy()

        '''
        actions = numpy.zeros((self.envs_count, self.actions_count))
        for e in range(self.envs_count):
            actions[e] = self._sample_action(mu_np[e], var_np[e])
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
             
            self.trajectory_buffer.add(states, values, actions, mu, var, rewards_, dones)


            #inner imagination loop
            if self.iterations%self.training_rollout == 0:
                for n in range(self.training_rollout):
                    #actor action selection
                    mu, var, values   = self.model.forward(states_t)

                    mu_np   = mu.detach().to("cpu").numpy()
                    var_np  = var.detach().to("cpu").numpy()
                    actions = self._sample_action(mu_np, var_np)

                    states  = states_t.detach().to("cpu")
                    values  = values.squeeze(1).detach().to("cpu")
                    mu      = mu.detach().to("cpu")
                    var     = var.detach().to("cpu")

                    actions     = torch.from_numpy(actions).to("cpu").float()
                    rewards_    = torch.zeros((self.envs_count, ), device="cpu")
                    dones       = torch.zeros((self.envs_count, ), device="cpu")

                    #add into inner loop buffer
                    self.trajectory_buffer_im.add(states, values, actions, mu, var, rewards_, dones)

                    #next states prediction with forward model
                    states_t = self.model.forward_fm(states_t, actions.to(self.device))


            if self.trajectory_buffer.is_full():
                self.train()

           
            

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
        self.trajectory_buffer_im.compute_returns(self.gamma)

        samples_count = self.steps*self.envs_count
        batch_count = samples_count//self.batch_size

        # PPO agent training
        for e in range(self.training_epochs):
            for batch_idx in range(batch_count):
                # ppo model training
                states, _, actions, actions_mu, actions_var, _, _, returns, advantages = self.trajectory_buffer.sample_batch(self.batch_size, self.device)
                ppo_loss = self._ppo_loss(states, actions, actions_mu, actions_var, returns, advantages)

                # im ppo policy loss
                states, _, actions, actions_mu, actions_var, _, _, returns, advantages = self.trajectory_buffer_im.sample_batch(self.batch_size, self.device)
                im_loss = self._im_loss(states, actions, actions_mu, actions_var, advantages)
                                        
                # forward model training
                states_seq, actions_seq = self.trajectory_buffer.sample_trajectory(self.batch_size//self.training_epochs, self.prediction_rollout + 1, self.device)
                fm_loss, fm_q = self._fm_loss(states_seq, actions_seq)

                loss = ppo_loss + im_loss + fm_q*fm_loss

                self.optimizer.zero_grad()        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step() 


        self.trajectory_buffer.clear()   
        self.trajectory_buffer_im.clear()
    
    def _ppo_loss(self, states, actions, actions_mu, actions_var, returns, advantages):
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


    def _im_loss(self, states, actions, actions_mu, actions_var, advantages):
        mu_new, var_new, _ = self.model.forward(states)        

        log_probs_old = self._log_prob(actions, actions_mu, actions_var).detach()
        log_probs_new = self._log_prob(actions, mu_new, var_new)

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

        self.values_logger.add("loss_policy_im", loss_policy.mean().detach().to("cpu").numpy())

        return loss_policy

     

    def _log_prob(self, action, mu, var):
        p1 = -((action - mu)**2)/(2.0*var + 0.00000001)
        p2 = -torch.log(torch.sqrt(2.0*numpy.pi*var)) 

        return p1 + p2
    

    def _fm_loss(self, states_seq, actions_seq):
        
        seq_length = states_seq.shape[0] - 1

        #set initial state only
        states_pred = states_seq[0].detach().clone()

        #rollout predictions in whole sequence
        loss_seq = []
        for n in range(seq_length):
            states_pred = self.model.forward_fm(states_pred, actions_seq[n])

            loss_step = ((states_seq[n+1] - states_pred)**2).mean()

            loss_seq.append(loss_step)

        loss_seq = torch.stack(loss_seq, dim=0)
        loss = loss_seq.mean()

        fm_q = 2.0*loss_seq[0]/(loss_seq[0] + loss_seq[-1])
        
    
        #log results, loss on begining, mean and end
        self.values_logger.add("fm_mse_start",  loss_seq[0].detach().to("cpu").numpy())
        self.values_logger.add("fm_mse_mean",   loss.detach().to("cpu").numpy())
        self.values_logger.add("fm_mse_end",    loss_seq[-1].detach().to("cpu").numpy())
        self.values_logger.add("fm_q",          fm_q.detach().to("cpu").numpy())

        return loss, fm_q.detach()