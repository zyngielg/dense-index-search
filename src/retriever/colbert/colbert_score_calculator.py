import torch
from itertools import accumulate

from torch.cuda import device_count
from utils.general_utils import torch_percentile


class ColBERTScoreCalculator():
    issue_counter = 0
    device_counter = 0
    num_gpu = 5
    def __init__(self, doclens, embeddings_tensor, device) -> None:
        self.device = device
        self.maxsim_dtype = torch.float32

        self.dim = embeddings_tensor.size(-1)
        self.doclens = doclens
        self.doclens_pfxsum = [0] + list(accumulate(doclens))
        # useful for debugging the views, maybe will come in handy
        # print(f"Num doclens = {len(self.doclens)}")
        # print(f"Sum of doclens = {sum(self.doclens)}")
        # print(f"Num pfxsum = {len(self.doclens_pfxsum)}")
        # print(f"Sum of pfxsum = {sum(self.doclens_pfxsum)}")
        # print(f"Max of pfxsum = {max(self.doclens_pfxsum)}")
        # print(f"last value of doclens = {self.doclens[-1]}")
        # print(f"last value of prefix sum = {self.doclens_pfxsum[-1]}")
        self.doclens = torch.tensor(self.doclens)
        self.doclens_pfxsum = torch.tensor(self.doclens_pfxsum)
        
        self.strides = self.__create_strides()
        self.embeddings_tensor = embeddings_tensor
        self.views = self.__create_views(embeddings_tensor)
        self.buffers = self.__create_buffers(embeddings_tensor.dtype, {
            'cpu', 
            'cuda:0'})
    
    def calculate_scores(self, Q, pids, mode=2):
        assert len(pids) > 0
        assert Q.size(0) in [1, len(pids)]

        Q = Q.contiguous().to(dtype=self.maxsim_dtype)
        
        VIEWS_DEVICE = self.views[0].device
        DEVICE = self.device #f'cuda:{mode}'
        D_buffers = self.buffers[str(VIEWS_DEVICE)]

        raw_pids = pids if type(pids) is list else pids.tolist()
        pids = torch.tensor(pids) if type(pids) is list else pids

        doclens, offsets = self.doclens[pids], self.doclens_pfxsum[pids]      
        output_scores = []
        for i in range(Q.size(0)):
            # inner_device_idx = i % 4
            inner_device_idx = self.device_counter % self.num_gpu
            self.device_counter += 1
            inner_DEVICE = torch.device(f"cuda:{inner_device_idx}")

            group_doclens, group_offsets = doclens[i], offsets[i]
            group_Q = Q[i]
            group_Q = group_Q.to(inner_DEVICE)
            group_offsets = group_offsets.to(VIEWS_DEVICE)
            group_offsets_uniq, group_offsets_expand = torch.unique_consecutive(group_offsets, return_inverse=True)

            D_size = group_offsets_uniq.size(0)
            D = torch.index_select(self.views[i], 0, group_offsets_uniq, out=D_buffers[i][:D_size])
            D = D.to(inner_DEVICE)
            D = D[group_offsets_expand.to(inner_DEVICE)].to(dtype=self.maxsim_dtype)

            
            scores = (D @ group_Q)
            scores = scores.max(1).values.sum(-1)

            # output_pids.append(group_pids)
            output_scores.append(scores.to('cuda:2'))

        output_scores = torch.stack(output_scores)#.tolist()
        # output_pids = torch.cat(output_pids)[output_permutation].tolist()

        # assert len(raw_pids) == len(output_pids)
        # assert len(raw_pids) == len(output_scores)
        # assert raw_pids == output_pids

        return output_scores

    def __create_strides(self):
        # [25, 50, 75]
        percentiles = [25, 50, 75]
        strides = [torch_percentile(self.doclens, p) for p in percentiles]
        strides.append(self.doclens.max().item())
        strides = sorted(list(set(strides)))

        return strides

    def __create_views(self, tensor):
        views = []

        for stride in self.strides:
            outdim = tensor.size(0) - stride + 1
            view = torch.as_strided(tensor, (outdim, stride, self.dim), (self.dim, self.dim, 1))
            views.append(view)

        return views
   
    def __create_buffers(self, dtype, devices):
        buffers = {}
        max_bsize = 1 << 14


        for device in devices:
            buffers[device] = [torch.zeros(max_bsize, stride, self.dim, dtype=dtype,
                                           device=device, pin_memory=(device == 'cpu'))
                               for stride in self.strides]

        return buffers


        
