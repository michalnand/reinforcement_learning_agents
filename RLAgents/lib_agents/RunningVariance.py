import torch
 

class RunningVariance:

    def __init__(self, batch_size, features_count):

        self.mean         = torch.zeros((batch_size, features_count), dtype=torch.float32)
        self.pwr_sum_mean = torch.zeros((batch_size, features_count), dtype=torch.float32)

        self.counts       = torch.zeros((batch_size, features_count), dtype=torch.float32)

    def reset(self, id):
        self.counts[id]         = 0
        self.mean[id]           = 0
        self.pwr_sum_mean[id]   = 0

    def update(self, x): 
         
        self.counts+= 1 

        self.mean+= (x - self.mean) / self.counts

        self.pwr_sum_mean+= (x**2 - self.pwr_sum_mean) / self.counts

        self.var = self.pwr_sum_mean - self.mean**2

        return self.mean, self.var 
    


if __name__ == "__main__":

    batch_size      = 5
    features_count  = 512

    steps           = 4000

    rv = RunningVariance(batch_size, features_count)

    all_stats = torch.zeros((steps, batch_size, features_count))


    #var     = 5.0*torch.rand((batch_size, features_count))
    #means   = 2.0*torch.rand((batch_size, features_count)) - 1.0

    for i in range(steps):

        x = 3.0*torch.randn((batch_size, features_count)) + 31

        all_stats[i] = x

        mean_test, var_test  = rv.update(x)

        if i > 2 and i%100 == 0:
            mean_ref    = torch.mean(all_stats[0:i,:,:], dim=0).mean(dim=1)
            var_ref     = torch.var(all_stats[0:i,:,:], dim=0).mean(dim=1)
            
            mean_test   = mean_test.mean(dim=1)
            var_test    = var_test.mean(dim=1)
            
            print(mean_ref, var_ref)
            print(mean_test, var_test)

            print("\n\n\n")

    