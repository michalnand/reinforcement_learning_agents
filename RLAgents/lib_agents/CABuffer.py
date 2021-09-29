import numpy
import torch

class CABuffer():
    def __init__(self, size, shape, add_threshold = 0.5, downsample = 8, device = "cpu"):

        self.add_threshold  = add_threshold
        shape_flt           = 1*(shape[1]//downsample)*(shape[2]//downsample)
        shape_downsampled   = 1*(shape[1]//downsample)*(shape[2]//downsample)
        self.states_b       = torch.zeros((size, shape_flt), dtype=torch.float32).to(device)
        self.visited_b      = numpy.zeros((size, shape_downsampled), dtype=numpy.float32)

        self.current_idx    = 0
 
        self.layer_downsample = torch.nn.AvgPool2d((downsample, downsample), (downsample, downsample))
        self.layer_downsample.to(device)

        self.layer_flatten = torch.nn.Flatten()
        self.layer_flatten.to(device)


    def add(self, states_t, attentions_t):
        states_flt = self._downsmaple(states_t[:,0].unsqueeze(1))

        attentions_flt = attentions_t.reshape((attentions_t.shape[0], attentions_t.shape[2]*attentions_t.shape[3]))

        distances     = torch.cdist(self.states_b, states_flt)

        min_distances, min_indices = torch.min(distances, dim=0)

        position_indices = torch.max(attentions_flt, dim=1)[1] 

        #add new if necessary
        for b in range(states_t.shape[0]):
            if self.current_idx < self.states_b.shape[0] and min_distances[b] > self.add_threshold:
                self.states_b[self.current_idx] = states_flt[b].clone()                
                self.current_idx+= 1

                #first run - add only first
                if self.current_idx == 1:
                    break

        for b in range(states_t.shape[0]):
            self.visited_b[min_indices[b], position_indices[b]]+= 1

        '''
        b = 0
        print("rooms_count = ", self.current_idx)
        print("room_id = ", min_indices[b])
        print("visited_map = ")
        v = self.visited_b[min_indices[b]].reshape((attentions_t.shape[2], attentions_t.shape[3]))
        print(v)
        print("\n\n\n\n")
        '''
        
        result = 1.0/(self.visited_b[min_indices, position_indices] + 0.001)
        result = numpy.clip(result, 0.0, 1.0)

        return result

    def _downsmaple(self, states_t):
        y = self.layer_downsample(states_t)
        y = self.layer_flatten(y)
        return y 


if __name__ == "__main__":
    size                = 32
    envs_count          = 507
    state_shape         = (4, 96, 96)
    attention_shape     = (1, 12, 12)

    states_t       = torch.randn((envs_count, ) + state_shape)
    attentions_t   = torch.randn((envs_count, ) + attention_shape)

    buffer = CABuffer(size, state_shape)


    buffer.add(states_t, attentions_t)


