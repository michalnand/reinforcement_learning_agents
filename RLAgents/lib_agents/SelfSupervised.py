import torch
import numpy

def _off_diagonal(x):
    mask = 1.0 - torch.eye(x.shape[0], device=x.device)
    return x*mask

def loss_vicreg_direct(za, zb):
    eps = 0.0001 

    # invariance loss
    sim_loss = ((za - zb)**2).mean()

    # variance loss
    std_za = torch.sqrt(za.var(dim=0) + eps)
    std_zb = torch.sqrt(zb.var(dim=0) + eps) 
    
    std_loss = torch.mean(torch.relu(1.0 - std_za)) 
    std_loss+= torch.mean(torch.relu(1.0 - std_zb))
   
    # covariance loss
    za_norm = za - za.mean(dim=0)
    zb_norm = zb - zb.mean(dim=0)
    cov_za = (za_norm.T @ za_norm) / (za.shape[0] - 1.0)
    cov_zb = (zb_norm.T @ zb_norm) / (zb.shape[0] - 1.0)
    
    cov_loss = _off_diagonal(cov_za).pow_(2).sum()/za.shape[1] 
    cov_loss+= _off_diagonal(cov_zb).pow_(2).sum()/zb.shape[1]

    # total vicreg loss
    loss = 1.0*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss

    return loss


def loss_vicreg_spatial(za, zb):

    #bootstrap random spatial tile, one from each batch item
    batch_size = za.shape[0]    

    idx_y = torch.randint(0, za.shape[2], (batch_size, ))
    idx_x = torch.randint(0, za.shape[3], (batch_size, ))
 
    za_tmp = za[range(batch_size), :, idx_y, idx_x]
    zb_tmp = zb[range(batch_size), :, idx_y, idx_x]

    return loss_vicreg_direct(za_tmp, zb_tmp)

def loss_vicreg(model_forward_func, augmentations, states_now, states_next, states_similar, states_random, actions, relations):
    xa = states_now.clone()
    xb = states_similar.clone()

    # states augmentation
    if augmentations is not None:
        xa = augmentations(xa) 
        xb = augmentations(xb)
 
    # obtain features from model
    za = model_forward_func(xa)  
    zb = model_forward_func(xb) 

    return loss_vicreg_direct(za, zb)

    
def loss_vicreg_spatial(model_forward_func, augmentations, states_now, states_next, states_similar, states_random, actions, relations):
    xa = states_now.clone()
    xb = states_similar.clone()

    # states augmentation
    if augmentations is not None:
        xa = augmentations(xa) 
        xb = augmentations(xb)
 
    # obtain features from model
    # both, global, and spatial features
    # global  shape : Nx512
    # spatial shape : Nx64x12x12
    za_g, za_s = model_forward_func(xa)  
    zb_g, zb_s = model_forward_func(xb) 

    #bootstrap, from NxCx12x12 spatial features, takes only one for every item in batch
    #there is no possible to fit into memory complet each-by-each vicreg loss
    idx_y = torch.randint(0, za_s.shape[2], (za_s.shape[0], ))
    idx_x = torch.randint(0, za_s.shape[3], (za_s.shape[0], ))

    za_s = za_s[range(za_s.shape[0]), :, idx_y, idx_x]
    zb_s = zb_s[range(za_s.shape[0]), :, idx_y, idx_x]
    
    loss_global  = loss_vicreg_direct(za_g, zb_g) 
    loss_spatial = loss_vicreg_direct(za_s, zb_s)

    return loss_global + loss_spatial





#constructor theory loss
#for phylosophical stuff read https://www.constructortheory.org
#train model for causality prediction, a.k.a. if state transition is posible
def loss_constructor(model_forward_func, augmentations, states_now, states_next, states_similar, states_random, actions, relations):
    batch_size  = states_now.shape[0]

    #classes target IDs
    #0 : state_now,  state_random, two different states
    #1 : state_now,  state_next, two consecutive states
    #2 : state_next, state_now, two inverted consecutive states
    labels  = torch.randint(0, 3, (batch_size, ), dtype=torch.long).to(states_now.device)
 
    #mix states coresponding to three cases
    select  = labels.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    xa      = (select == 0)*states_now    + (select == 1)*states_now  + (select == 2)*states_next
    xb      = (select == 0)*states_random + (select == 1)*states_next + (select == 2)*states_now

    #states augmentation
    if augmentations is not None:
        xa = augmentations(xa) 
        xb = augmentations(xb)

    transition_pred = model_forward_func(xa, xb)

    #classification loss
    loss_func       = torch.nn.CrossEntropyLoss()
    loss            = loss_func(transition_pred, labels)
    
    
    #compute accuracy, per class
    labels_pred = torch.argmax(transition_pred.detach(), dim=1)

    acc = numpy.zeros((3, ))

    for class_id in range(acc.shape[0]):
        total_count = (labels == class_id).sum()
        hits_count  = ((labels == class_id)*(labels_pred == class_id)).sum()

        acc[class_id] = 100.0*hits_count/(total_count + 10**-6)

    return loss, acc