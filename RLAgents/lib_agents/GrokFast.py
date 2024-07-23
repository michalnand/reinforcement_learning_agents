

def params_stats(model):

    params_mag = 0.0
    params_std = 0.0
    grads_mag  = 0.0
    grads_std  = 0.0
    
    count = 0
    for name, param in self.model.named_parameters():
        gradient = param.grad

        params_mag+= (param**2).mean().detach().cpu().numpy()
        params_std+= param.std().detach().cpu().numpy()
        grads_mag+= (gradient**2).mean().detach().cpu().numpy()
        grads_std+= gradient.std().detach().cpu().numpy()

        count+= 1

    params_mag = params_mag/count
    params_std = params_std/count
    grads_mag  = grads_mag/count
    grads_std  = grads_std/count

    params_mag = params_mag.detach().cpu().numpy()
    params_std = params_std.detach().cpu().numpy()
    grads_mag  = grads_mag.detach().cpu().numpy()
    grads_std  = grads_std.detach().cpu().numpy()

    return params_mag, params_std, grads_mag, grads_std
    

class GrokFast:
    def __init__(self, model, alpha = 0.98, lam = 4.0):

        self.model = model
        self.grads = {}

        self.alpha = alpha
        self.lam   = lam


    def step(self):
        for name, param in self.model.named_parameters():
            gradient = param.grad.clone()

            # add new grads, only small value for smooth start
            if name not in self.grads:
                self.grads[name] = 0.01*gradient

            # low pass filter
            self.grads[name] = self.alpha*self.grads[name] + (1.0 - self.alpha)*gradient

            # amplify grads with low pass filtered grads
            param.grad = gradient + self.lam*self.grads[name]

    