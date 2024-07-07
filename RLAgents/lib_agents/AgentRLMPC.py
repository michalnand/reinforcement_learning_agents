import numpy
import torch
import time

from .ValuesLogger      import *
from .TrajectoryBuffer  import *
from .SelfSupervised    import * 
from .Augmentations     import *
  
class AgentRLMPC():
    def __init__(self, envs, Model, config):
        self.envs = envs
 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.envs_count         = config.envs_count

        self.gamma              = config.gamma
        self.temperature        = config.temperature
        self.rollout_length     = config.rollout_length
        self.value_pred_coeff   = config.value_pred_coeff

        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
 
        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        if config.self_supervised_loss == "vicreg":
            self.self_supervised_loss_func = loss_vicreg
            self.augmentations              = config.augmentations
        else:
            self.self_supervised_loss_func  = None
            self.augmentations              = None


        self.model = Model.Model(self.state_shape, self.actions_count)
        self.model.to(self.device)
        print(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
 
        self.trajctory_buffer = TrajectoryBuffer(self.steps, self.state_shape, self.actions_count, self.envs_count)
 


        print("gamma                    = ", self.gamma)
        print("temperature              = ", self.temperature)
        print("rollout_length           = ", self.rollout_length)
        print("value_pred_coeff         = ", self.value_pred_coeff)
        print("learning_rate            = ", config.learning_rate)

        print("steps                    = ", self.steps)
        print("batch_size               = ", self.batch_size)
        
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
        self.values_logger.add("loss_value",      0.0)
        self.values_logger.add("loss_value_pred", 0.0)
        self.values_logger.add("loss_mpc",        0.0)
        self.values_logger.add("loss_ssl",        0.0)

        self.info_logger = {} 

        
    def round_start(self): 
        pass

    def round_finish(self): 
        pass
        
    def episode_done(self, env_idx):
        pass

    def step(self, states, training_enabled, legal_actions_mask):        
        states_t  = torch.tensor(states, dtype=torch.float).detach().to(self.device)

        batch_size = states_t.shape[0]
        
        logits_t = torch.zeros((batch_size, self.actions_count), device=self.device, dtype=torch.float32)
        z        = self.model.forward_features(states_t)

        # find current state evaluation
        values_t = self.model.model_critic(z)

        # find all actions evaluation using next state prediction
        for a in range(self.actions_count):
            a_one_hot       = torch.zeros((batch_size, self.actions_count), device=self.device, dtype=torch.float32)
            a_one_hot[:, a] = 1.0   

            z_next     = self.model.forward_mpc(z, a_one_hot)
            value_next = self.model.forward_critic(z_next)

            logits_t[:, a] = value_next.squeeze(1)

        # use action evaluation to sample actual action
        actions = self._sample_actions(logits_t)

        states_new, rewards, dones, _, infos = self.envs.step(actions)

        #put into policy buffer
        if training_enabled:
            states_t        = states_t.detach().to("cpu")
            logits_t        = logits_t.detach().to("cpu")
            values_t        = values_t.squeeze(1).detach().to("cpu") 
            actions         = torch.from_numpy(actions).to("cpu")
            rewards_t       = torch.from_numpy(rewards).to("cpu")
            dones           = torch.from_numpy(dones).to("cpu")


            self.trajctory_buffer.add(states_t, logits_t, values_t, actions, rewards_t, dones)

            if self.trajctory_buffer.is_full():
                self.trajctory_buffer.compute_returns(self.gamma)
                self.train()
                self.trajctory_buffer.clear()  

      
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
    
    def train(self): 
        samples_count   = self.steps*self.envs_count
        batch_count     = samples_count//self.batch_size

        for batch_idx in range(batch_count):

            states, logits, actions, returns, advantages = self.trajctory_buffer.sample_batch_trajectory(self.rollout_length, self.batch_size, self.device)                    

            z  = self.model.forward_features(states[0])

            # critic MSE loss
            values_pred = self.model.model_critic(z).squeeze(1)
            loss_value  = ((returns[0] - values_pred)**2).mean()
            

            self.values_logger.add("loss_value", loss_value.detach().to("cpu").numpy())

            
            # MPC unrolled loss
            loss_mpc = 0.0

            loss_value_pred = 0.0

            loss_mpc_trajectory = []
            for n in range(self.rollout_length-1):
                # actions one hot encoding
                action = torch.zeros((self.batch_size, self.actions_count), dtype=torch.float32, device=self.device)
                action[range(self.batch_size), actions[n, range(self.batch_size)] ] = 1.0

                # predict next z-space state using current z, and current action
                z = self.model.forward_mpc(z, action)

                # future z, target
                z_target = self.model.forward_features(states[n+1]).detach()

                # MSE loss for state prediction
                loss_mpc_tmp = ((z_target - z)**2).mean()
                loss_mpc+= loss_mpc_tmp  

                # MSE loss for state evaluation
                values_pred          = self.model.model_critic(z).squeeze(1)
                loss_value_pred_tmp  = ((returns[n+1] - values_pred)**2).mean()

                loss_value_pred+= loss_value_pred_tmp

                loss_mpc_trajectory.append(round(loss_mpc.detach().cpu().numpy().item(), 6))

            loss_mpc        = loss_mpc/(self.rollout_length-1)
            loss_value_pred = loss_value_pred/(self.rollout_length-1)

            self.info_logger["loss_mpc"] = loss_mpc_trajectory            
            self.values_logger.add("loss_mpc", loss_mpc.detach().to("cpu").numpy())
            self.values_logger.add("loss_value_pred", loss_value_pred.detach().to("cpu").numpy())
            

            # self supervised regularisation
            states_a, states_b = self.trajctory_buffer.sample_states_pairs(self.batch_size, 0, self.device)
            loss_self_supervised, ssl_info = self.self_supervised_loss_func(self.model.forward_ssl, self._augmentations, states_a, states_b)
                
            self.info_logger["loss_ssl"] = ssl_info
            self.values_logger.add("loss_ssl", loss_self_supervised.detach().cpu().numpy())

                                   
            loss = loss_value + loss_mpc + self.value_pred_coeff*loss_value_pred + loss_self_supervised

            self.optimizer.zero_grad()        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step() 


    def _augmentations(self, x): 
        mask_result = torch.zeros((2, x.shape[0]), device=x.device, dtype=torch.float32)

        if "mask" in self.augmentations:
            x, mask = aug_random_apply(x, 0.5, aug_mask)
            mask_result[0] = mask

        if "noise" in self.augmentations:
            x, mask = aug_random_apply(x, 0.5, aug_noise)
            mask_result[1] = mask

        return x.detach(), mask_result 
    
    