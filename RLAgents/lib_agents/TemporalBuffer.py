import torch
 
class TemporalBuffer:

    def __init__(self, buffer_size, s_features_count, h_features_count, envs_count):
        self.buffer_size        = buffer_size
        self.s_features_count   = s_features_count
        self.h_features_count   = h_features_count
        self.envs_count         = envs_count
      
        self.clear()     
 
    def add(self, state, hidden_state):
        self.states[self.ptr]       = state.detach().cpu().clone() 
        self.hidden_states[self.ptr]= hidden_state.detach().cpu().clone()


    def is_full(self):
        if self.ptr >= self.buffer_size:
            return True

        return False 
 
    def clear(self):
        self.states         = torch.zeros((self.buffer_size, self.envs_count, self.s_features_count), dtype=torch.float32)
        self.hidden_states  = torch.zeros((self.buffer_size, 2, self.envs_count, self.h_features_count), dtype=torch.float32)

        self.ptr = 0  
 

    def sample_batch(self, batch_size, seq_length, device = "cpu"):
        z_seq       = torch.randn((batch_size, seq_length, self.s_features_count))
        h_initial   = torch.randn((2, batch_size, self.s_features_count))

        idx_seq = torch.randint(0, self.buffer_size - seq_length, size=(batch_size, ))
        idx_env = torch.randint(0, self.envs_count, size=(batch_size, ))

        print(h_initial.shape, self.hidden_states[idx_seq, :, idx_env, :].shape)

        #n = idx.view(-1, 1) + torch.arange(seq_length)

        #idx = torch.randint(0, self.buffer_size, size=envs_count)
        #z_seq  = self.states[:, idx+0:idx+seq_length, :]

        
        return z_seq.to(device), h_initial.to(device)
   