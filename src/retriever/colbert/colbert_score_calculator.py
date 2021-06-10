import torch
from itertools import accumulate
from utils.general_utils import torch_percentile

class ColBERTScoreCalculator():
    issue_counter = 0

    def __init__(self, doclens, embeddings_tensor, device) -> None:
        self.device = device
        self.maxsim_dtype = torch.float32

        self.dim = embeddings_tensor.size(-1)
        self.doclens = doclens
        self.doclens_pfxsum = [0] + list(accumulate(doclens))
        
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

        Q = Q.contiguous().to(self.device).to(dtype=self.maxsim_dtype)
        
        VIEWS_DEVICE = self.views[0].device
        DEVICE = self.device #f'cuda:{mode}'
        D_buffers = self.buffers[str(VIEWS_DEVICE)]

        raw_pids = pids if type(pids) is list else pids.tolist()
        pids = torch.tensor(pids) if type(pids) is list else pids

        doclens, offsets = self.doclens[pids], self.doclens_pfxsum[pids]      

        x = doclens.unsqueeze(1)
        xx = torch.tensor(self.strides)
        xxx = xx.unsqueeze(0) + 1e-6
        y = (x > xx)
        yy = y.sum(-1)

        assignments = (doclens.unsqueeze(1) > torch.tensor(self.strides).unsqueeze(0) + 1e-6).sum(-1)

        one_to_n = torch.arange(len(raw_pids))
        output_pids, output_scores, output_permutation = [], [], []

        for group_idx, stride in enumerate(self.strides):
            inner_device_idx = group_idx % 2 + 1
            inner_DEVICE = torch.device(f"cuda:{inner_device_idx}")
            locator = (assignments == group_idx)

            if locator.sum() < 1e-5:
                continue

            group_pids, group_doclens, group_offsets = pids[locator], doclens[locator], offsets[locator]
            group_Q = Q if Q.size(0) == 1 else Q[locator]
            group_Q = group_Q.to(inner_DEVICE)
            group_offsets = group_offsets.to(VIEWS_DEVICE)
            group_offsets_uniq, group_offsets_expand = torch.unique_consecutive(group_offsets, return_inverse=True)

            D_size = group_offsets_uniq.size(0)
            
            # print(f"Shape of views num {group_idx}: {self.views[group_idx].shape}")
            # print(f"Shape of group_offsets_uniq: {group_offsets_uniq.shape}")
            # print(f"Max value in group_offsets_uniq: {max(group_offsets_uniq)}")
            # print(f"Shape of D_buffers num {group_idx}: {D_buffers[group_idx].shape}")
            # print(f"D_size={D_size}")
            try:
                D = torch.index_select(self.views[group_idx], 0, group_offsets_uniq, out=D_buffers[group_idx][:D_size])
                # D = torch.index_select(self.views[group_idx], 0, group_offsets_uniq)
                # print(f"Shape of views num {group_idx}: {self.views[group_idx].shape}")
                # print(f"Shape of group_offsets_uniq: {group_offsets_uniq.shape}")
                # print(f"Max value in group_offsets_uniq: {max(group_offsets_uniq)}")
                # print(f"Shape of D_buffers num {group_idx}: {D_buffers[group_idx].shape}")
                # print(f"D_size={D_size}")
            except Exception as inst:
                # # print(type(inst))    # the exception instance
                # # print(inst.args)     # arguments stored in .args
                # # print(inst)
                self.issue_counter += 1
                # # TODO: very often the max is the same number all the time
                # print(f"Shape of views num {group_idx}: {self.views[group_idx].shape}")
                # print(f"Shape of group_offsets_uniq: {group_offsets_uniq.shape}")
                # print(f"Max value in group_offsets_uniq: {max(group_offsets_uniq)}")
                # print(f"Shape of D_buffers num {group_idx}: {D_buffers[group_idx].shape}")
                # print(f"D_size={D_size}")
                zeros = torch.zeros_like(group_offsets_uniq)
                group_offsets_uniq = torch.where(group_offsets_uniq > self.views[group_idx].shape[0], zeros, group_offsets_uniq)
                # D_size = group_offsets_uniq.size(0)
                D = torch.index_select(self.views[group_idx], 0, group_offsets_uniq, out=D_buffers[group_idx][:D_size])
                # D = torch.index_select(self.views[group_idx], 0, group_offsets_uniq)
            D = D.to(inner_DEVICE)
            D = D[group_offsets_expand.to(inner_DEVICE)].to(dtype=self.maxsim_dtype)

            mask = torch.arange(stride, device=inner_DEVICE) + 1
            mask = mask.unsqueeze(0) <= group_doclens.to(inner_DEVICE).unsqueeze(-1)
            
            scores = (D @ group_Q) * mask.unsqueeze(-1)
            scores = scores.max(1).values.sum(-1).cpu()

            output_pids.append(group_pids)
            output_scores.append(scores)
            output_permutation.append(one_to_n[locator])

        output_permutation = torch.cat(output_permutation).sort().indices
        output_pids = torch.cat(output_pids)[output_permutation].tolist()
        output_scores = torch.cat(output_scores)[output_permutation].tolist()

        assert len(raw_pids) == len(output_pids)
        assert len(raw_pids) == len(output_scores)
        assert raw_pids == output_pids

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


        
