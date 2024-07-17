
class GrokFast:
    def __init__(self, model, alpha = 0.98, lam = 4.0):

        self.model = model
        self.grads = {}

        self.alpha = alpha
        self.lam   = lam


    def step(self):
        for name,param in self.model.named_parameters():
            gradient = param.grad.clone()

            # add new grads, only small value for smooth start
            if name not in self.grads:
                self.grads[name] = 0.01*gradient

            # low pass filter
            self.grads[name] = self.alpha*self.grads[name] + (1.0 - self.alpha)*gradient

            # amplify grads with low pass filtered grads
            param.grad = gradient + self.lam*self.grads[name]


    