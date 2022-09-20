import torch

def accuraccy(za, zb):
    #info NCE loss, CE with target classes on diagonal
    similarity  = torch.cdist(za, zb)
    target      = torch.arange(za.shape[0]).to(za.device)
     
    #compute accuraccy in [%]
    hits = torch.argmin(similarity, dim=1) == target
    hits = torch.sum(hits.float()) 

    acc  = 100.0*hits/similarity.shape[0]

    return acc.detach().to("cpu").numpy()

def contrastive_loss_mse(model, states_a, states_b, target, regularisation_coeff = 0.0, normalise = None, augmentation = None):
    xa = states_a.clone()
    xb = states_b.clone()

    #normalise states
    if normalise is not None:
        xa = normalise(xa) 
        xb = normalise(xb)

    #states augmentation
    if augmentation is not None:
        xa = augmentation(xa) 
        xb = augmentation(xb)

    #obtain features from model
    if hasattr(model, "forward_features"):
        za = model.forward_features(xa)  
        zb = model.forward_features(xb) 
    else:
        za = model(xa)  
        zb = model(xb) 

    #predict close distance for similar, far distance for different states 
    predicted = ((za - zb)**2).mean(dim=1)

    #MSE loss
    loss_mse = ((target - predicted)**2).mean()

    #magnitude regularisation, keep magnitude in small range (optional)

    #L2 magnitude regularisation
    magnitude       = (za**2).mean() + (zb**2).mean() 
    loss_magnitude  = regularisation_coeff*magnitude

    loss = loss_mse + loss_magnitude

    #compute accuraccy in [%]
    hits = torch.logical_and(target > 0.5, predicted > 0.5)
    hits = torch.sum(hits.float()) 

    acc  = 100.0*hits/predicted.shape[0]


    return loss, magnitude.detach().to("cpu").numpy(), acc.detach().to("cpu").numpy()


def contrastive_loss_mse_equivariance( model, states_a, states_b, target, regularisation_coeff = 0.0, normalise = None, augmentation = None):
    xa = states_a.clone()
    xb = states_b.clone()

    #normalise states
    if normalise is not None:
        xa = normalise(xa) 
        xb = normalise(xb)

    #states augmentation
    if augmentation is not None:
        xa = augmentation(xa) 
        xb = augmentation(xb)

    #obtain features from model
    if hasattr(model, "forward_features"):
        za, pa = model.forward_features(xa)  
        zb, pb = model.forward_features(xb) 
    else:
        za, pa = model(xa)  
        zb, pb = model(xb) 

    #predict close distance for similar, far distance for different states 
    predicted = ((za - pb)**2).mean(dim=1)
 
    #MSE loss
    loss_mse = ((target - predicted)**2).mean()

    #magnitude regularisation, keep magnitude in small range (optional)

    #L2 magnitude regularisation
    magnitude       = (za**2).mean() + (zb**2).mean() 
    loss_magnitude  = regularisation_coeff*magnitude

    loss = loss_mse + loss_magnitude

    #compute accuraccy in [%]
    hits = torch.logical_and(target > 0.5, predicted > 0.5)
    hits = torch.sum(hits.float()) 

    acc  = 100.0*hits/predicted.shape[0] 


    return loss, magnitude.detach().to("cpu").numpy(), acc.detach().to("cpu").numpy()


def contrastive_loss_mse_all( model, states_a, states_b, target, regularisation_coeff = 0.0, normalise = None, augmentation = None):
    xa = states_a.clone()
    xb = states_a.clone()

    #normalise states
    if normalise is not None:
        xa = normalise(xa) 
        xb = normalise(xb)

    #states augmentation
    if augmentation is not None:
        xa = augmentation(xa) 
        xb = augmentation(xb)

    #obtain features from model
    if hasattr(model, "forward_features"):
        za = model.forward_features(xa)  
        zb = model.forward_features(xb) 
    else:
        za = model(xa)  
        zb = model(xb) 

    #predict distances, each by each
    distances = (torch.cdist(za, zb)**2)/za.shape[1] 

    #zeros on diagonal -> close distances, ones elsewhere
    target_   = (1.0 - torch.eye(za.shape[0])).to(za.device)
    
    #MSE loss
    loss_mse = ((target_ - distances)**2).mean()

    #magnitude regularisation, keep magnitude in small range (optional)

    #L2 magnitude regularisation
    magnitude       = (za**2).mean() + (zb**2).mean() 
    loss_magnitude  = regularisation_coeff*magnitude

    loss = loss_mse + loss_magnitude

    #compute accuraccy in [%], smallest distance should be on diagonal 
    hits = torch.argmin(distances, dim=1) == torch.arange(za.shape[0]).to(za.device)
    hits = torch.sum(hits.float()) 

    acc  = 100.0*hits/distances.shape[0]

    return loss, magnitude.detach().to("cpu").numpy(), acc.detach().to("cpu").numpy()




def contrastive_loss_mse_equivariance_all( model, states_a, states_b, target, regularisation_coeff = 0.0, normalise = None, augmentation = None):
    xa = states_a.clone()
    xb = states_a.clone()

    #normalise states
    if normalise is not None:
        xa = normalise(xa) 
        xb = normalise(xb)

    #states augmentation
    if augmentation is not None:
        xa = augmentation(xa) 
        xb = augmentation(xb)

    #obtain features from model
    if hasattr(model, "forward_features"):
        za, pa = model.forward_features(xa)  
        zb, pb = model.forward_features(xb) 
    else:
        za, pa = model(xa)  
        zb, pb = model(xb) 

    #predict distances, each by each
    distances = (torch.cdist(za, pb)**2)/za.shape[1] 

    #zeros on diagonal -> close distances, ones elsewhere
    target_   = (1.0 - torch.eye(za.shape[0])).to(za.device)
    
    #MSE loss
    loss_mse = ((target_ - distances)**2).mean()

    #magnitude regularisation, keep magnitude in small range (optional)

    #L2 magnitude regularisation
    magnitude       = (za**2).mean() + (zb**2).mean() 
    loss_magnitude  = regularisation_coeff*magnitude

    loss = loss_mse + loss_magnitude

    #compute accuraccy in [%], smallest distance should be on diagonal 
    hits = torch.argmin(distances, dim=1) == torch.arange(za.shape[0]).to(za.device)
    hits = torch.sum(hits.float()) 

    acc  = 100.0*hits/distances.shape[0]

    return loss, magnitude.detach().to("cpu").numpy(), acc.detach().to("cpu").numpy()

 

def contrastive_loss_nce( model, states_a, states_b, target, regularisation_coeff = 0.0, normalise = None, augmentation = None):
    xa = states_a.clone()
    xb = states_a.clone()

    #normalise states
    if normalise is not None:
        xa = normalise(xa) 
        xb = normalise(xb)

    #states augmentation
    if augmentation is not None:
        xa = augmentation(xa) 
        xb = augmentation(xb)

    #obtain features from model
    if hasattr(model, "forward_features"):
        za = model.forward_features(xa)  
        zb = model.forward_features(xb) 
    else:
        za = model(xa)  
        zb = model(xb) 

    #normalise
    eps = 10**-12

    za_norm = (za - za.mean(dim = 0))/(za.std(dim = 0) + eps)
    zb_norm = (zb - zb.mean(dim = 0))/(zb.std(dim = 0) + eps)
    
    #info NCE loss, CE with target classes on diagonal
    similarity      = torch.matmul(za_norm, zb_norm.T)/za_norm.shape[1]
    lf              = torch.nn.CrossEntropyLoss()
    target          = torch.arange(za_norm.shape[0]).to(za_norm.device)
    loss_info_max   = lf(similarity, target)
    
    #magnitude regularisation, keep magnitude in small range (optional)

    #L2 magnitude regularisation
    magnitude       = (za**2).mean() + (zb**2).mean()
    loss_magnitude  = regularisation_coeff*magnitude

    loss = loss_info_max + loss_magnitude

    #compute accuraccy in [%]
    hits = torch.argmax(similarity, dim=1) == target
    hits = torch.sum(hits.float()) 

    acc  = 100.0*hits/similarity.shape[0]

    return loss, magnitude.detach().to("cpu").numpy(), acc.detach().to("cpu").numpy()

#barlow twins supervised
def contrastive_loss_barlow( model, states_a, states_b, target, regularisation_coeff = 0.0, normalise = None, augmentation = None):
    xa = states_a.clone()
    xb = states_a.clone()

    #normalise states
    if normalise is not None:
        xa = normalise(xa) 
        xb = normalise(xb)

    #states augmentation
    if augmentation is not None:
        xa = augmentation(xa) 
        xb = augmentation(xb)
 
    #obtain features from model
    if hasattr(model, "forward_features"):
        za = model.forward_features(xa)  
        zb = model.forward_features(xb) 
    else:
        za = model(xa)  
        zb = model(xb) 


    eps = 10**-12

    za_norm = (za - za.mean(dim = 0, keepdim=True))/(za.std(dim = 0, keepdim=True) + eps)
    zb_norm = (zb - zb.mean(dim = 0, keepdim=True))/(zb.std(dim = 0, keepdim=True) + eps) 
 
    c = torch.mm(za_norm.T, zb_norm)/za.shape[0]
 
    diag        = torch.eye(c.shape[0]).to(c.device)
    off_diag    = 1.0 - diag  

    loss_invariance = (diag*(1.0 - c)**2).mean()
    loss_redundance = (off_diag*(c**2)).mean()/(c.shape[0] - 1)
 

    #magnitude regularisation, keep magnitude in small range (optional)

    #L2 magnitude regularisation
    magnitude       = (za**2).mean() + (zb**2).mean()
    loss_magnitude  = regularisation_coeff*magnitude

    loss = loss_invariance + loss_redundance + loss_magnitude

    acc = accuraccy(za, zb)

    return loss, magnitude.detach().to("cpu").numpy(), acc







'''
this is symmetry loss : 
same action in different state have same features

f(s_i, a) = const, for all i

imput :
@param model    : model to train, should contain forward_features method
@param states_a : batch of states
@param states_b : not used, can be None
@param actions  : batch discrete actions (ints) corresponding with states_a
@param regularisation_coeff : features L2 regularisation rate
@normalise      : ture/false, if states are normalised using running stats
@augmentation   : ture/false, if states are augmented before obtaining features
'''
def symmetry_loss_mse( model, states_a, states_b, actions, regularisation_coeff = 0.0, normalise = None, augmentation = None):
    xa = states_a.clone()
    xb = states_a.clone()

    #normalise states
    if normalise is not None:
        xa = normalise(xa) 
        xb = normalise(xb)

    #states augmentation
    if augmentation is not None:
        xa = augmentation(xa) 
        xb = augmentation(xb)

    #obtain features from model
    if hasattr(model, "forward_features"):
        za = model.forward_features(xa)  
        zb = model.forward_features(xb) 
    else:
        za = model(xa)  
        zb = model(xb) 

    #normalise
    eps = 10**-12

    za_norm = (za - za.mean(dim = 0))/(za.std(dim = 0) + eps)
    zb_norm = (zb - zb.mean(dim = 0))/(zb.std(dim = 0) + eps)
    
    #cosine similarity
    similarity  = torch.matmul(za_norm, zb_norm.T)/za_norm.shape[1]

    #targets matrix, ones where are the same actions
    target    = (actions.unsqueeze(0) == actions.unsqueeze(1)).float()

    loss_similarity = ((target - similarity)**2).mean()

    #L2 magnitude regularisation
    magnitude       = (za**2).mean() + (zb**2).mean()
    loss_magnitude  = regularisation_coeff*magnitude

    loss = loss_similarity + loss_magnitude

    #track only true positive (true negative is almost 100% accuraccy when lot of actions)
    acc = torch.logical_and(target > 0.5, similarity > 0.5).float().mean()
    acc  = 100.0*acc

    return loss, magnitude.detach().to("cpu").numpy(), acc.detach().to("cpu").numpy()
