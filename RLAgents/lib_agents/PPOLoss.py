import numpy
import torch


#MSE critic loss
def ppo_compute_critic_loss(values_ext_new, returns_ext, values_int_new, returns_int):
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
def ppo_compute_actor_loss(logits, logits_new, advantages, actions, eps_clip, entropy_beta):
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
    p2          = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)*advantages
    loss_policy = -torch.min(p1, p2)  
    loss_policy = loss_policy.mean()

    ''' 
    compute entropy loss, to avoid greedy strategy
    L = beta*H(pi(s)) = beta*pi(s)*log(pi(s))
    '''
    loss_entropy = (probs_new*log_probs_new).sum(dim = 1)
    loss_entropy = entropy_beta*loss_entropy.mean()

    return loss_policy, loss_entropy

