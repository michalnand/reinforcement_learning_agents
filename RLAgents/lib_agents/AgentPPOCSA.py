import numpy
import torch 
from .PolicyBufferIM    import *  
from .RunningStats      import *  

import sklearn.manifold
import matplotlib.pyplot as plt
 
       
class AgentPPOCSA():   
    def __init__(self, envs, ModelPPO, ModelADM, config):
        self.envs = envs  
      
        self.gamma_ext          = config.gamma_ext 
        self.gamma_int          = config.gamma_int
            
        self.ext_adv_coeff      = config.ext_adv_coeff
        self.int_adv_coeff      = config.int_adv_coeff
        self.int_reward_coeff   = config.int_reward_coeff
    
        self.entropy_beta       = config.entropy_beta
        self.entropy_adm_beta   = config.entropy_adm_beta
        self.eps_clip           = config.eps_clip 
    
        self.steps              = config.steps
        self.batch_size         = config.batch_size        
        
        self.training_epochs    = config.training_epochs
        self.envs_count         = config.envs_count 



        self.state_shape    = self.envs.observation_space.shape
        self.actions_count  = self.envs.action_space.n

        self.model_ppo      = ModelPPO.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate_ppo)

        self.model_adm      = ModelADM.Model(self.state_shape, self.actions_count)
        self.optimizer_adm  = torch.optim.Adam(self.model_adm.parameters(), lr=config.learning_rate_adm)


        self.policy_buffer = PolicyBufferIM(self.steps, self.state_shape, self.actions_count, self.envs_count, self.model_ppo.device, True)

        for e in range(self.envs_count):
            self.envs.reset(e)


        self.rooms_max_count    = 64
        self.features_size      = 12
        self.room_downsample    = 1
        self.rooms_count        = 0

        self.rooms = torch.zeros((self.rooms_max_count, (self.state_shape[1]//self.room_downsample)*(self.state_shape[1]//self.room_downsample)))

        self.visitation_count   = numpy.zeros((self.rooms_max_count, self.features_size*self.features_size))
        

        #reset envs and fill initial state
        self.states = numpy.zeros((self.envs_count, ) + self.state_shape, dtype=numpy.float32)
        for e in range(self.envs_count):
            self.states[e] = self.envs.reset(e).copy()

        self.enable_training()
        self.iterations                     = 0 

        self.log_loss_adm                   = 0.0
        self.log_loss_actor                 = 0.0
        self.log_loss_critic                = 0.0

        self.log_internal_motivation_mean   = 0.0
        self.log_internal_motivation_std    = 0.0
        self.log_acc_adm                    = 0.0



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

        room_ids = self._get_room_ids(states_t)
 
        #internal motivation
        rewards_int    = self._internal_motivation(room_ids, states_t)

        rewards_int    = numpy.clip(self.int_reward_coeff*rewards_int, 0.0, 1.0)
         
        #put into policy buffer
        if self.enabled_training:
            self.policy_buffer.add(states_np, logits_np, values_ext_np, values_int_np, actions, rewards_ext, rewards_int, dones)

            if self.policy_buffer.is_full():
                self.train()
        
        for e in range(self.envs_count): 
            if dones[e]:
                self.states[e] = self.envs.reset(e).copy()


        

        #collect stats
        k = 0.02
        self.log_internal_motivation_mean   = (1.0 - k)*self.log_internal_motivation_mean + k*rewards_int.mean()
        self.log_internal_motivation_std    = (1.0 - k)*self.log_internal_motivation_std  + k*rewards_int.std()

        self.iterations+= 1
        return rewards_ext[0], dones[0], infos[0]
    
    def save(self, save_path):
        self.model_ppo.save(save_path + "trained/")
        self.model_adm.save(save_path + "trained/")
        numpy.save(save_path + "trained/" + "visitation_count.npy", self.visitation_count)


    def load(self, load_path):
        self.model_ppo.load(load_path + "trained/")
        self.model_adm.load(load_path + "trained/")
        self.visitation_count = numpy.load(load_path + "trained/" + "visitation_count.npy")

 
    def get_log(self):  
        result = "" 

        result+= str(round(self.log_loss_adm, 7)) + " "
        result+= str(round(self.log_loss_actor, 7)) + " "
        result+= str(round(self.log_loss_critic, 7)) + " "

        result+= str(round(self.log_internal_motivation_mean, 7)) + " "
        result+= str(round(self.log_internal_motivation_std, 7)) + " "

        result+= str(round(self.log_acc_adm, 7)) + " "

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

                #train adm model, MSE loss
                loss_adm, acc_adm = self._compute_loss_adm(states, states_next, actions)

                self.optimizer_adm.zero_grad() 
                loss_adm.backward()
                self.optimizer_adm.step()

                #log results 
                k = 0.02
                self.log_loss_adm  = (1.0 - k)*self.log_loss_adm + k*loss_adm.detach().to("cpu").numpy()
                self.log_acc_adm   = (1.0 - k)*self.log_acc_adm + k*acc_adm.detach().to("cpu").numpy()

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


    #MSE loss for adm model
    def _compute_loss_adm(self, states, states_next, actions):

        #actions one hot encodings
        actions_one_hot = torch.zeros((states.shape[0], self.actions_count))
        actions_one_hot[range(states.shape[0]), actions] = 1.0
        actions_one_hot = actions_one_hot.to(actions.device)


        actions_predicted, attention = self.model_adm(states, states_next)

        attention   = attention.reshape((attention.shape[0], attention.shape[1], attention.shape[2]*attention.shape[3]))

        #action prediction loss, MSE or logsoftmax ?
        loss_prediction = ((actions_one_hot - actions_predicted)**2)
        loss_prediction = loss_prediction.mean()

        #attention entropy regularisation, -H(attn)
        loss_entropy    = self.entropy_adm_beta*((torch.log(attention)*attention).sum(dim=2)).mean()

        loss_adm        = loss_prediction + loss_entropy

        a_max = torch.argmax(actions_predicted, dim=1)

        #prediction accuracy 
        accuracy = 100.0*(a_max == actions).sum()/actions.shape[0]
       
        return loss_adm, accuracy

    #return closest room ID, add new if distance too big
    def _get_room_ids(self, states):

        if self.room_downsample > 1:
            states_ds = torch.nn.functional.avg_pool2d(states[:,0].to("cpu"), self.room_downsample, self.room_downsample)
        else:
            states_ds = states[:,0].to("cpu")

        states_ds = states_ds.reshape((states_ds.shape[0], states_ds.shape[1]*states_ds.shape[2]))

    
        distances   = torch.cdist(states_ds, self.rooms)

        distances_min, distances_amin  = torch.min(distances, dim=1)

        distances_amin = distances_amin.detach().to("cpu").numpy()

        for i in range(distances_min.shape[0]):
            if distances_min[i] > 10.0 and self.rooms_count < self.rooms.shape[0]:
                self.rooms[self.rooms_count] = states_ds[i].clone()
                self.rooms_count+= 1
                break

        return distances_amin



    #compute internal motivation
    def _internal_motivation(self, room_ids, state_t):
        attention   = self.model_adm.forward_attention(state_t)

        attention   = attention.reshape((attention.shape[0], attention.shape[1], attention.shape[2]*attention.shape[3]))
        position    = torch.argmax(attention, dim=2).squeeze(1)
        position    = position.detach().to("cpu").numpy()

        motivation  = numpy.zeros(attention.shape[0])

        for i in range(position.shape[0]): 
            room_idx = room_ids[i]
            pos_idx  = position[i]

            self.visitation_count[room_idx][pos_idx]+= 1

            motivation[i] = 1.0/self.visitation_count[room_idx][pos_idx]

        return motivation