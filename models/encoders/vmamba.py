import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
# import cv2
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# try:
#     from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
# except:
#     pass

try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
except:
    pass

if True:
    import selective_scan_cuda_core as selective_scan_cuda
    class SelectiveScan(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
            assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
            assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
            ctx.delta_softplus = delta_softplus
            ctx.nrows = nrows

            # all in float
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if B.dim() == 3:
                B = B.unsqueeze(dim=1)
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = C.unsqueeze(dim=1)
                ctx.squeeze_C = True

            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
            
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, dout, *args):
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)

    class CrossScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            B, C, H, W = x.shape
            ctx.shape = (B, C, H, W)
            xs = x.new_empty((B, 4, C, H * W))
            xs[:, 0] = x.flatten(2, 3)
            xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            return xs
        
        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, k, d, l)
            B, C, H, W = ctx.shape
            L = H * W
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
            return y.view(B, -1, H, W)
    
    class CrossMerge(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, H, W = ys.shape
            ctx.shape = (H, W)
            ys = ys.view(B, K, D, -1)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
            return y
        
        @staticmethod
        def backward(ctx, x: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            H, W = ctx.shape
            B, C, L = x.shape
            xs = x.new_empty((B, 4, C, L))
            xs[:, 0] = x
            xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            xs = xs.view(B, 4, C, H, W)
            return xs, None, None
        
    class CrossScan_multimodal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_rgb: torch.Tensor, x_e: torch.Tensor):
            # B, C, H, W -> B, 2, C, 2 * H * W
            B, C, H, W = x_rgb.shape
            ctx.shape = (B, C, H, W)
            xs_fuse = x_rgb.new_empty((B, 2, C, 2 * H * W))
            xs_fuse[:, 0] = torch.concat([x_rgb.flatten(2, 3), x_e.flatten(2, 3)], dim=2)
            xs_fuse[:, 1] = torch.flip(xs_fuse[:, 0], dims=[-1])
            return xs_fuse

        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            B, C, H, W = ctx.shape
            L = 2 * H * W
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1]) 
            return ys[:, :, 0:H*W].view(B, -1, H, W), ys[:, :, H*W:2*H*W].view(B, -1, H, W) 
         
    class CrossMerge_multimodal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, L = ys.shape
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1]) 
            return ys[:, :, 0:L//2], ys[:, :, L//2:L]
        
        @staticmethod
        def backward(ctx, x1: torch.Tensor, x2: torch.Tensor):
            B, C, L = x1.shape
            xs = x1.new_empty((B, 2, C, 2*L))
            xs[:, 0] = torch.cat([x1, x2], dim=2)
            xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
            xs = xs.view(B, 2, C, 2*L)
            return xs, None, None
    

    def extract_non_zero_values(tensor):
        B, C, H, W = tensor.shape
        flat_tensor = tensor.view(B, C, -1)
        mask = flat_tensor != 0
        non_zero_values = flat_tensor[mask]

        non_zero_values = non_zero_values.view(B, C, -1)

        non_zero_positions = flat_tensor.nonzero(as_tuple=True)[2].view(B, C, -1)

        return non_zero_values, non_zero_positions

    def restore_tensor(original_shape, output_tensor, mask):
        B, C, H, W = original_shape
        flat_restored_tensor = torch.zeros((B, C, H * W), device=output_tensor.device)
        flat_restored_tensor.scatter_(dim=-1, index=mask, src=output_tensor)
        restored_tensor = flat_restored_tensor.view(B, C, H, W)

        return restored_tensor

    def traverse_saliency_tensor(tensor):
        # result = []
        tensor = tensor.squeeze()
        result_index = []
        current_row = 0
        direction = 'left_right'
        while current_row < tensor.size(0):
            non_zero_indices = torch.nonzero(tensor[current_row], as_tuple=False).squeeze()
            # print(non_zero_indices)
            if non_zero_indices.ndim == 0:
                non_zero_indices = non_zero_indices.unsqueeze(0)
            # non_zero_values = tensor[current_row][non_zero_indices]
            if direction == 'right_left':
                # non_zero_values = torch.flip(non_zero_values,dims=[-1])
                non_zero_indices = torch.flip(non_zero_indices,dims=[-1])
            if len(non_zero_indices) > 0:
                # result.extend(non_zero_values.tolist())
                # print(non_zero_indices)
                non_zero_index = non_zero_indices + current_row * tensor.size(0)
                result_index.extend(non_zero_index.tolist())
                # print(result_index)
                if current_row < tensor.size(0) - 1:
                    last_index = non_zero_indices[-1].item()
                    next_non_zero_indices = torch.nonzero(tensor[current_row + 1], as_tuple=False).squeeze()
                    if next_non_zero_indices.ndim == 0:
                        next_non_zero_indices = next_non_zero_indices.unsqueeze(0)

                    if len(next_non_zero_indices) > 0:
                        left_index = next_non_zero_indices[0].item()
                        right_index = next_non_zero_indices[-1].item()
                        left_dist = abs(last_index - left_index)
                        right_dist = abs(last_index - right_index)
                        if left_dist <= right_dist:
                            direction = 'left_right'
                        else:
                            direction = 'right_left'

            current_row += 1
        return torch.tensor(result_index, device=tensor.device)

    def restore_saliency_tensor(original_shape, non_zero_values, non_zero_indexs):
        B, C, H, W = original_shape
        flat_restored_tensor = torch.zeros((B, C, H * W), device= non_zero_values.device)
        flat_restored_tensor.scatter_(dim=-1, index=non_zero_indexs, src=non_zero_values)
        restored_tensor = flat_restored_tensor.view(B, C, H, W)
        return restored_tensor

    def CrossScan_saliency(x_rgb, gt):
        # B, C, H, W -> B, 2, C, H * W
        B, C, H, W = x_rgb.shape

        xs = x_rgb.new_empty((B, 4, C, H * W))
        index_s_list = []
        mask_ns_list = []
        for b in range(B):
            non_zero_indexs = traverse_saliency_tensor(gt[b:b+1,:])
            non_zero_indexs_expanded = non_zero_indexs.unsqueeze(0)
            non_zero_indexs_expanded = non_zero_indexs_expanded.expand(C, -1).unsqueeze(0)
            s = torch.gather(x_rgb[b:b+1,:].view(1,C,H*W), 2, non_zero_indexs_expanded)
            # s, mask_s = extract_non_zero_values(x_rgb[b:b+1,:] * gt[b:b+1,:])
            ns, mask_ns = extract_non_zero_values(x_rgb[b:b+1,:] * (1-gt[b:b+1,:]))
            scan_1 = torch.cat((s,ns), -1)
            scan_1_flip = torch.cat((s.flip(dims=[-1]), ns.flip(dims=[-1])), -1)
            scan_2 = torch.cat((ns,s), -1)
            scan_2_flip = torch.cat((ns.flip(dims=[-1]), s.flip(dims=[-1])), -1)
            # print(scan_1.shape)
            xs[b:b+1, 0] = scan_1
            xs[b:b+1, 1] = scan_1_flip
            xs[b:b+1, 2] = scan_2
            xs[b:b+1, 3] = scan_2_flip
            index_s_list.append(non_zero_indexs_expanded)
            mask_ns_list.append(mask_ns)
        return xs, index_s_list, mask_ns_list
         
    def CrossMerge_saliency(ys, index_s, mask_ns, gt):
        B, K, C, H, W = ys.shape
        
        L = H * W

        ys = ys.view(B, K, C, -1)
        out_ys = ys.new_empty(B, C, H, W)
        for b in range(B):
            len_s = int(gt.view(B, -1, L)[b:b+1,:].sum(dim=-1).item())
            len_ns = int((1-gt).view(B, -1, L)[b:b+1,:].sum(dim=-1).item())

            y1_1 = restore_saliency_tensor([1,C,H,W], ys[b:b+1, 0, :, :len_s], index_s[b])
            y1_2 = restore_tensor([1,C,H,W], ys[b:b+1, 0, :, len_s:], mask_ns[b])
            y1 = y1_1 + y1_2

            y2_1 = restore_saliency_tensor([1,C,H,W], ys[b:b+1, 1, :, :len_s].flip(dims=[-1]), index_s[b])
            y2_2 = restore_tensor([1,C,H,W], ys[b:b+1, 1, :, len_s:].flip(dims=[-1]), mask_ns[b])
            y2 = y2_1 + y2_2

            y3_1 = restore_tensor([1,C,H,W], ys[b:b+1, 2, :, :len_ns], mask_ns[b])
            y3_2 = restore_saliency_tensor([1,C,H,W], ys[b:b+1, 2, :, len_ns:], index_s[b])
            y3 = y3_1 + y3_2

            y4_1 = restore_tensor([1,C,H,W], ys[b:b+1, 3, :, :len_ns].flip(dims=[-1]), mask_ns[b])
            y4_2 = restore_saliency_tensor([1,C,H,W], ys[b:b+1, 3, :, len_ns:].flip(dims=[-1]), index_s[b])
            y4 = y4_1 + y4_2
            
            out_ys[b] = y1 + y2 + y3 + y4 # B, d, H * W
        return out_ys.view(B,C,L)
    
    def CrossScan_modality(x_rgb, x_e):
        B, C, H, W = x_rgb.shape
        L = H * W
        xs = x_rgb.new_empty((B, 4, C, L*2))
        x_rgb = x_rgb.view(B, C, L)
        x_e = x_e.view(B, C, L)

        Hg2, Wg2 = math.ceil(H / 2), math.ceil(W / 2)
        x_rgb_local = x_rgb.view(B, C, H, W).view(B, C, Hg2, 2, Wg2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, Hg2 * Wg2, -1)
        x_e_local = x_e.view(B, C, H, W).view(B, C, Hg2, 2, Wg2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, Hg2 * Wg2, -1)
        scan_1 = torch.cat([x_rgb_local, x_e_local], dim=-1).view(B, C, L * 2)
        scan_1_flip = torch.cat([x_e_local, x_rgb_local], dim=-1).view(B, C, L * 2)
        scan_2 = torch.cat([x_rgb, x_e], dim=-1).view(B, C, L*2)
        scan_2_flip = torch.cat([x_e, x_rgb], dim=-1).view(B, C, L*2)

        xs[:, 0] = scan_1
        xs[:, 1] = scan_1_flip
        xs[:, 2] = scan_2
        xs[:, 3] = scan_2_flip

        return xs
    
    def CrossMerge_modality(ys, H, W):
        B, K, C, L = ys.shape
        Hg2, Wg2 = math.ceil(H / 2), math.ceil(W / 2)

        y1 = ys[:,0].view(B, C, L//8, 8)
        y_rgb1, y_e1 = y1[:,:,:,:4].view(B, C, Hg2, Wg2, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, L//2), y1[:,:,:,4:].view(B, C, Hg2, Wg2, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, L//2)
        
        y2 = ys[:,1].view(B, C, L//8, 8)
        y_rgb2, y_e2 = y2[:,:,:,4:].view(B, C, Hg2, Wg2, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, L//2), y2[:,:,:,:4].view(B, C, Hg2, Wg2, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, L//2)

        y3 = ys[:,2].view(B, C, L)
        y_rgb3, y_e3 = y3[:,:,:L//2], y3[:,:,L//2:]

        y4 = ys[:,3].view(B, C, L)
        y_rgb4, y_e4 = y4[:,:,L//2:], y4[:,:,:L//2]

        y_rgb = y_rgb1 + y_rgb2 + y_rgb3 + y_rgb4
        y_e = y_e1 + y_e2 + y_e3 + y_e4
        
        return y_rgb, y_e
    
    
    def CrossScan_aligned(x_rgb, x_e):
        B, C, H, W = x_rgb.shape
        L = H * W
        xs = x_rgb.new_empty((B, 4, C, 5*L))

        x_rgb_hw = x_rgb.view(B, C, L)
        Hg2, Wg2 = math.ceil(2*H / 2), math.ceil(2*W / 2)
        x_e_hw = x_e.view(B, C, Hg2, 2, Wg2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, L, 4)

        x_rgb_wh = x_rgb.permute(0, 1, 3, 2).reshape(B, C, L)
        x_e_wh = x_e.permute(0, 1, 3, 2).view(B, C, Wg2, 2, Hg2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, L, 4)
        
        scan_1 = torch.cat([x_rgb_hw.unsqueeze(2).transpose(-1, -2), x_e_hw], dim=-1).contiguous().view(B, C, L * 5)
        scan_2 = torch.cat([x_e_hw, x_rgb_hw.unsqueeze(2).transpose(-1, -2)], dim=-1).contiguous().view(B, C, L * 5)
        scan_3 = torch.cat([x_rgb_wh.unsqueeze(2).transpose(-1, -2), x_e_wh], dim=-1).contiguous().view(B, C, L * 5)
        scan_4 = torch.cat([x_e_wh, x_rgb_wh.unsqueeze(2).transpose(-1, -2)], dim=-1).contiguous().view(B, C, L * 5)

        xs[:, 0] = scan_1
        xs[:, 1] = scan_2
        xs[:, 2] = scan_3
        xs[:, 3] = scan_4

        return xs
    
    def CrossMerge_aligned(ys, ori_shape):
        B, K, C, L = ys.shape
        H, W = ori_shape
        y1 = ys[:,0].view(B, C, L//5, 5)
        y_rgb1 = y1[:,:,:,0]

        y2 = ys[:,1].view(B, C, L//5, 5)
        y_rgb2 = y2[:,:,:,4]

        y3 = ys[:,2].view(B, C, L//5, 5)
        y_rgb3 = y3[:,:,:,0].view(B, C, W, H).permute(0, 1, 3, 2).reshape(B, C, H*W)

        y4 = ys[:,3].view(B, C, L//5, 5)
        y_rgb4 = y4[:,:,:,4].view(B, C, W, H).permute(0, 1, 3, 2).reshape(B, C, H*W)

        y = y_rgb1 + y_rgb2 + y_rgb3 + y_rgb4
        return y
    
    def CrossScan_aligned1(x_rgb, x_e):
        B, C, H, W = x_rgb.shape
        L = H * W
        xs = x_rgb.new_empty((B, 2, C, 5*L))

        x_rgb_hw = x_rgb.view(B, C, L)
        Hg2, Wg2 = math.ceil(2*H / 2), math.ceil(2*W / 2)
        x_e_hw = x_e.view(B, C, Hg2, 2, Wg2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, L, 4)

        x_rgb_wh = x_rgb.permute(0, 1, 3, 2).reshape(B, C, L)
        x_e_wh = x_e.permute(0, 1, 3, 2).view(B, C, Wg2, 2, Hg2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, L, 4)
        
        scan_1 = torch.cat([x_e_hw, x_rgb_hw.unsqueeze(2).transpose(-1, -2)], dim=-1).contiguous().view(B, C, L * 5)
        scan_2 = torch.cat([x_e_wh, x_rgb_wh.unsqueeze(2).transpose(-1, -2)], dim=-1).contiguous().view(B, C, L * 5)

        xs[:, 0] = scan_1
        xs[:, 1] = scan_2
        return xs
    
    def CrossMerge_aligned1(ys, ori_shape):
        B, K, C, L = ys.shape
        H, W = ori_shape

        y1 = ys[:,0].view(B, C, L//5, 5)
        y_rgb1 = y1[:,:,:,4]

        y2 = ys[:,1].view(B, C, L//5, 5)
        y_rgb2 = y2[:,:,:,4].view(B, C, W, H).permute(0, 1, 3, 2).reshape(B, C, H*W)

        y = y_rgb1 + y_rgb2
        return y
    
    def CrossScan_aligned2(x_rgb, x_e):
        B, C, H, W = x_rgb.shape
        L = H * W
        xs = x_rgb.new_empty((B, 4, C, 2*L))
        x_rgb_hw = x_rgb.view(B, C, L)
        x_e_hw = x_e.view(B, C, L)
        x_rgb_wh = x_rgb.permute(0, 1, 3, 2).reshape(B, C, L)
        x_e_wh = x_e.permute(0, 1, 3, 2).reshape(B, C, L)

        scan_1 = torch.cat([x_rgb_hw.unsqueeze(2).transpose(-1, -2), x_e_hw.unsqueeze(2).transpose(-1, -2)], dim=-1).contiguous().view(B, C, L * 2)
        scan_2 = torch.cat([x_e_hw.unsqueeze(2).transpose(-1, -2), x_rgb_hw.unsqueeze(2).transpose(-1, -2)], dim=-1).contiguous().view(B, C, L * 2)
        scan_3 = torch.cat([x_rgb_wh.unsqueeze(2).transpose(-1, -2), x_e_wh.unsqueeze(2).transpose(-1, -2)], dim=-1).contiguous().view(B, C, L * 2)
        scan_4 = torch.cat([x_e_wh.unsqueeze(2).transpose(-1, -2), x_rgb_wh.unsqueeze(2).transpose(-1, -2)], dim=-1).contiguous().view(B, C, L * 2)
        xs[:, 0] = scan_1
        xs[:, 1] = scan_2
        xs[:, 2] = scan_3
        xs[:, 3] = scan_4

        return xs
    
    def CrossMerge_aligned2(ys, ori_shape):
        B, K, C, L = ys.shape
        H, W = ori_shape
        y1 = ys[:,0].view(B, C, L//2, 2)
        y_rgb1 = y1[:,:,:,0]

        y2 = ys[:,1].view(B, C, L//2, 2)
        y_rgb2 = y2[:,:,:,1]

        y3 = ys[:,0].view(B, C, L//2, 2)
        y_rgb3 = y3[:,:,:,0].view(B, C, W, H).permute(0, 1, 3, 2).reshape(B, C, L//2)

        y4 = ys[:,1].view(B, C, L//2, 2)
        y_rgb4 = y4[:,:,:,1].view(B, C, W, H).permute(0, 1, 3, 2).reshape(B, C, L//2)

        y = y_rgb1 + y_rgb2 + y_rgb3 + y_rgb4
        return y

    class CrossScan_local(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_rgb: torch.Tensor):
            # B, C, H, W -> B, 2, C, H * W
            B, C, H, W = x_rgb.shape
            ctx.shape = (B, C, H, W)
            x = x_rgb.view(B, C, H, W)
            Hg2, Wg2 = math.ceil(H / 2), math.ceil(W / 2)
            Hg7, Wg7 = math.ceil(H / 7), math.ceil(W / 7)
            h = x.permute(0, 1, 2, 3).reshape(B, C, -1)
            w = x.permute(0, 1, 3, 2).reshape(B, C, -1)
            x2 = x.view(B, C, Hg2, 2, Wg2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)
            x7 = x.view(B, C, Hg7, 7, Wg7, 7).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)

            xs = x_rgb.new_empty((B, 4, C, H * W))
            xs[:, 0] = h
            xs[:, 1] = w
            xs[:, 2] = x2
            xs[:, 3] = x7
            return xs

        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, 2, d, l)
            B, C, H, W = ctx.shape
            L = H * W
            Hg2, Wg2 = math.ceil(H / 2), math.ceil(W / 2)
            Hg7, Wg7 = math.ceil(H / 7), math.ceil(W / 7)
            ys = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L) \
            + ys[:, 2].view(B, C, Hg2, Wg2, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, -1, L) \
            + ys[:, 3].view(B, C, Hg7, Wg7, 7, 7).permute(0, 1, 2, 4, 3, 5).reshape(B, -1, L)  # B, d, H * W
            # get B, d, H*W
            return ys.view(B, -1, H, W)
         
    class CrossMerge_local(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, C, H, W = ys.shape
            ctx.shape = (H, W)
            L = H * W
            Hg2, Wg2 = math.ceil(H / 2), math.ceil(W / 2)
            Hg7, Wg7 = math.ceil(H / 7), math.ceil(W / 7)
            ys = ys.view(B, K, C, -1)
            ys = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L) \
            + ys[:, 2].view(B, C, Hg2, Wg2, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, -1, L) \
            + ys[:, 3].view(B, C, Hg7, Wg7, 7, 7).permute(0, 1, 2, 4, 3, 5).reshape(B, -1, L)  # B, d, H * W
            return ys
        
        @staticmethod
        def backward(ctx, x: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            H, W = ctx.shape
            B, C, L = x.shape
            xs = x.new_empty((B, 4, C, L))
            Hg2, Wg2 = math.ceil(H / 2), math.ceil(W / 2)
            Hg7, Wg7 = math.ceil(H / 7), math.ceil(W / 7)
            h = x
            w = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
            x2 = x.view(B, C, Hg2, 2, Wg2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)
            x7 = x.view(B, C, Hg7, 7, Wg7, 7).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)

            xs[:, 0] = h
            xs[:, 1] = w
            xs[:, 2] = x2
            xs[:, 3] = x7
            xs = xs.view(B, 4, C, H ,W)
            return xs, None, None
        
    class CrossScan_local_flip(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_rgb: torch.Tensor):
            # B, C, H, W -> B, 2, C, H * W
            B, C, H, W = x_rgb.shape
            ctx.shape = (B, C, H, W)
            x = x_rgb.view(B, C, H, W)
            Hg2, Wg2 = math.ceil(H / 2), math.ceil(W / 2)
            Hg7, Wg7 = math.ceil(H / 7), math.ceil(W / 7)
            h = x.permute(0, 1, 2, 3).reshape(B, C, -1)
            w = x.permute(0, 1, 3, 2).reshape(B, C, -1)
            x2 = x.view(B, C, Hg2, 2, Wg2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)
            x7 = x.view(B, C, Hg7, 7, Wg7, 7).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)

            xs = x_rgb.new_empty((B, 4, C, H * W))
            xs[:, 0] = torch.flip(h, dims=[-1])
            xs[:, 1] = torch.flip(w, dims=[-1])
            xs[:, 2] = torch.flip(x2, dims=[-1])
            xs[:, 3] = torch.flip(x7, dims=[-1])
            return xs

        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, 2, d, l)
            B, C, H, W = ctx.shape
            L = H * W
            Hg2, Wg2 = math.ceil(H / 2), math.ceil(W / 2)
            Hg7, Wg7 = math.ceil(H / 7), math.ceil(W / 7)
            ys = ys[:, 0].flip(dims=[-1]) + ys[:, 1].flip(dims=[-1]).view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L) \
            + ys[:, 2].flip(dims=[-1]).view(B, C, Hg2, Wg2, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, -1, L) \
            + ys[:, 3].flip(dims=[-1]).view(B, C, Hg7, Wg7, 7, 7).permute(0, 1, 2, 4, 3, 5).reshape(B, -1, L)   # B, d, H * W
            # get B, d, H*W
            return ys.view(B, -1, H, W)
         
    class CrossMerge_local_flip(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, C, H, W = ys.shape
            ctx.shape = (H, W)
            L = H * W
            Hg2, Wg2 = math.ceil(H / 2), math.ceil(W / 2)
            Hg7, Wg7 = math.ceil(H / 7), math.ceil(W / 7)
            ys = ys.view(B, K, C, -1)
            ys = ys[:, 0].flip(dims=[-1]) + ys[:, 1].flip(dims=[-1]).view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L) \
            + ys[:, 2].flip(dims=[-1]).view(B, C, Hg2, Wg2, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, -1, L) \
            + ys[:, 3].flip(dims=[-1]).view(B, C, Hg7, Wg7, 7, 7).permute(0, 1, 2, 4, 3, 5).reshape(B, -1, L)   # B, d, H * W
            return ys
        
        @staticmethod
        def backward(ctx, x: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            H, W = ctx.shape
            B, C, L = x.shape
            xs = x.new_empty((B, 4, C, L))
            Hg2, Wg2 = math.ceil(H / 2), math.ceil(W / 2)
            Hg7, Wg7 = math.ceil(H / 7), math.ceil(W / 7)
            h = x
            w = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
            x2 = x.view(B, C, Hg2, 2, Wg2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)
            x7 = x.view(B, C, Hg7, 7, Wg7, 7).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)

            xs[:, 0] = torch.flip(h, dims=[-1])
            xs[:, 1] = torch.flip(w, dims=[-1])
            xs[:, 2] = torch.flip(x2, dims=[-1])
            xs[:, 3] = torch.flip(x7, dims=[-1])
            xs = xs.view(B, 4, C, H ,W)
            return xs, None, None


    def cross_selective_scan(
        x: torch.Tensor=None, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        xs = CrossScan.apply(x)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, H, W)
        
        y = CrossMerge.apply(ys)

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)
        
        return y
    
    
    def selective_scan_1d(
        x: torch.Tensor=None, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        A_logs = A_logs[: A_logs.shape[0] // 4]
        Ds = Ds[: Ds.shape[0] // 4]
        B, D, H, W = x.shape
        D, N = A_logs.shape
        # get 1st of dt_projs_weight
        x_proj_weight = x_proj_weight[0].unsqueeze(0)
        x_proj_bias = x_proj_bias[0].unsqueeze(0) if x_proj_bias is not None else None
        dt_projs_weight = dt_projs_weight[0].unsqueeze(0)
        dt_projs_bias = dt_projs_bias[0].unsqueeze(0) if dt_projs_bias is not None else None
        K, D, R = dt_projs_weight.shape # K=1
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # xs = CrossScan.apply(x)
        xs = x.view(B, -1, L).unsqueeze(dim=1)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, L)
        
        # y = CrossMerge.apply(ys)

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = ys[:, 0].transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = ys[:, 0].transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)
        
        return y
    
    
    def cross_selective_scan_multimodal_k1(
        x_rgb: torch.Tensor=None, 
        x_e: torch.Tensor=None,
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        B, D, H, W = x_rgb.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # x_fuse = CrossScan_multimodal.apply(x_rgb, x_e) # B, C, H, W -> B, 1, C, 2 * H * W
        B, C, H, W = x_rgb.shape
        x_fuse = x_rgb.new_empty((B, 1, C, 2 * H * W))
        x_fuse[:, 0] = torch.concat([x_rgb.flatten(2, 3), x_e.flatten(2, 3)], dim=2)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, 2*H*W)
        
        # y = CrossMerge_multimodal.apply(ys)
        y = ys[:, 0, :, 0:L//2] + ys[:, 0, :, L//2:L]

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x_rgb.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x_rgb.dtype)
        
        return y

    def cross_selective_scan_multimodal_k2(
        x_rgb: torch.Tensor=None, 
        x_e: torch.Tensor=None,
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm1: torch.nn.Module=None,
        out_norm2: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        B, D, H, W = x_rgb.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        x_fuse = CrossScan_multimodal.apply(x_rgb, x_e) # B, C, H, W -> B, 2, C, 2 * H * W
       
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, 2*H*W)
        
        y_rgb, y_e = CrossMerge_multimodal.apply(ys)

        y_rgb = y_rgb.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_e = y_e.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_rgb = out_norm1(y_rgb).to(x_rgb.dtype)
        y_e = out_norm2(y_e).to(x_e.dtype)
        
        return y_rgb, y_e
    
    def cross_selective_scan_local_k2(
        x: torch.Tensor=None, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm1: torch.nn.Module=None,
        out_norm2: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
        flip = False
    ):
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1
        if flip:
            x_fuse = CrossScan_local_flip.apply(x) # B, C, H, W -> B, 4, C, H * W
        else:
            x_fuse = CrossScan_local.apply(x) # B, C, H, W -> B, 4, C, H * W
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, H, W)
        if flip:
            y = CrossMerge_local_flip.apply(ys)
        else:
            y = CrossMerge_local.apply(ys)
        y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = out_norm1(y).to(x.dtype)
        
        return y
    
    def cross_selective_scan_saliency_k2(
        x: torch.Tensor=None, 
        gt: torch.Tensor=None, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm1: torch.nn.Module=None,
        out_norm2: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
        flip = False
    ):
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # x_fuse = CrossScan_local.apply(x) # B, C, H, W -> B, 4, C, H * W
        x_fuse, index_s_list, mask_ns = CrossScan_saliency(x, gt)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, H, W)

        y = CrossMerge_saliency(ys, index_s_list, mask_ns, gt)
        y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = out_norm1(y).to(x.dtype)
        
        return y

    def cross_selective_scan_dualmodal_k2(
        x_rgb: torch.Tensor=None, 
        x_e: torch.Tensor=None,
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm1: torch.nn.Module=None,
        out_norm2: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        B, D, H, W = x_rgb.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        x_fuse = CrossScan_modality(x_rgb, x_e) # B, C, H, W -> B, 2, C, 2 * H * W
       
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, 2*H*W)
        
        y_rgb, y_e = CrossMerge_modality(ys, H, W)

        y_rgb = y_rgb.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_e = y_e.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_rgb = out_norm1(y_rgb).to(x_rgb.dtype)
        y_e = out_norm2(y_e).to(x_e.dtype)
        
        return y_rgb, y_e
    
    def cross_selective_scan_aligned_k2(
        x_rgb: torch.Tensor=None, 
        x_e: torch.Tensor=None,
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm1: torch.nn.Module=None,
        softmax_version=False,
        nrows = -1,
        delta_softplus = True,
    ):
        B, D, H, W = x_rgb.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 5 * H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        x_fuse = CrossScan_aligned1(x_rgb, x_e) # B, C, H, W -> B, 2, C, 2 * H * W

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, 5*H*W)
        
        y = CrossMerge_aligned1(ys,[H,W])

        y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = out_norm1(y).to(x_rgb.dtype)
        
        return y




# fvcore flops =======================================

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops


def print_jit_input_names(inputs):
    # tensor.11, dt.1, A.1, B.1, C.1, D.1, z.1, None
    try: 
        print("input params: ", end=" ", flush=True)
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)

    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    assert inputs[0].debugName().startswith("xs") # (B, D, L)
    assert inputs[1].debugName().startswith("dts") # (B, D, L)
    assert inputs[2].debugName().startswith("As") # (D, N)
    assert inputs[3].debugName().startswith("Bs") # (D, N)
    assert inputs[4].debugName().startswith("Cs") # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = len(inputs) > 5 and inputs[5].debugName().startswith("z")
    else:
        with_z = len(inputs) > 6 and inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    # flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops

# =====================================================

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


DEV = False
class SS2D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2,
        dt_rank="auto",
        # dwconv ===============
        # d_conv=-1, # < 2 means no conv 
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        softmax_version=False,
        # ======================
        **kwargs,
    ):
        if DEV:
            d_conv = -1
            
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        # x proj; dt proj ============================
        self.K = 4 if not (self.forward_core == self.forward_corev1_share_ssm) else 1
        # VMamaba set K=4, while original SSM set K=1
        if self.forward_core == self.forward_corev0:
            self.K = 1
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.K2 = self.K if not (self.forward_core == self.forward_corev1_share_a) else 1
        # VMamaba set K=4, while original SSM set K=1
        if self.forward_core == self.forward_corev0:
            self.K2 = 1
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        selective_scan = selective_scan_fn_v1
        B, C, H, W = x.shape
        L = H * W
        K = 1
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)
        xs = x.view(B, -1, L).unsqueeze(dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float() # (b, k, d_state, l)
        Cs = Cs.float() # (b, k, d_state, l)
        As = -torch.exp(self.A_logs.float()) # (k * d, d_state)
        Ds = self.Ds.float() # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, #z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            # return_last_state=False,
        ).view(B, K, -1, L)
        # assert out_y.dtype == torch.float

        # inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = torch.transpose(out_y[:, 0], dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        return y
    
    # def forward_corev0_seq(self, x: torch.Tensor):
    #     selective_scan = selective_scan_fn

    #     B, C, H, W = x.shape
    #     L = H * W
    #     K = 4

    #     x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
    #     xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

    #     x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
    #     # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
    #     dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
    #     dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

    #     xs = xs.float() # (b, k, d, l)
    #     dts = dts.contiguous().float() # (b, k, d, l)
    #     Bs = Bs.float() # (b, k, d_state, l)
    #     Cs = Cs.float() # (b, k, d_state, l)
        
    #     As = -torch.exp(self.A_logs.float()).view(K, -1, self.d_state)  # (k, d, d_state)
    #     Ds = self.Ds.float().view(K, -1) # (k, d)
    #     dt_projs_bias = self.dt_projs_bias.float().view(K, -1) # (k, d)

    #     # assert len(xs.shape) == 4 and len(dts.shape) == 4 and len(Bs.shape) == 4 and len(Cs.shape) == 4
    #     # assert len(As.shape) == 3 and len(Ds.shape) == 2 and len(dt_projs_bias.shape) == 2

    #     out_y = []
    #     for i in range(4):
    #         yi = selective_scan(
    #             xs[:, i], dts[:, i], 
    #             As[i], Bs[:, i], Cs[:, i], Ds[i],
    #             delta_bias=dt_projs_bias[i],
    #             delta_softplus=True,
    #         ).view(B, -1, L)
    #         out_y.append(yi)
    #     out_y = torch.stack(out_y, dim=1)
    #     assert out_y.dtype == torch.float

    #     inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
    #     wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
    #     invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
    #     y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
    #     y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
    #     y = self.out_norm(y)

    #     return y

    # def forward_corev1(self, x: torch.Tensor, float32=True):
    #     # float32 should be true in training!!!! otherwise, the output of selective_scan would be inf...
    #     selective_scan = selective_scan_fn_v1

    #     B, C, H, W = x.shape
    #     L = H * W

    #     xs = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=1)
    #     xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1) # (b, k, d, l)

    #     x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
    #     # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
    #     dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
    #     dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

    #     xs = xs.view(B, -1, L) # (b, k * d, l)
    #     dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
    #     As = -torch.exp(self.A_logs.to(torch.float))  # (k * d, d_state)
    #     Ds = self.Ds.to(torch.float) # (k * d)
    #     dt_projs_bias = self.dt_projs_bias.to(torch.float).view(-1) # (k * d)
        
    #     if float32:
    #         ys: torch.Tensor = selective_scan(
    #             xs.to(torch.float), 
    #             dts.to(torch.float), 
    #             As, 
    #             Bs.to(torch.float), 
    #             Cs.to(torch.float), 
    #             Ds,
    #             delta_bias=dt_projs_bias,
    #             delta_softplus=True,
    #         ).view(B, 4, -1, L)
    #         ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
    #         y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
    #     else:
    #         out_y: torch.Tensor = selective_scan(
    #             xs, dts, 
    #             As, Bs, Cs, Ds,
    #             delta_bias=dt_projs_bias,
    #             delta_softplus=True,
    #         ).view(B, 4, -1, L)
    #         # assert out_y.dtype == torch.float16

    #         inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
    #         wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
    #         invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
    #         y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()
            
    #     if self.softmax_version:
    #         y = torch.softmax(y, dim=-1).to(x.dtype)
    #         y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
    #     else:
    #         y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
    #         y = self.out_norm(y).to(x.dtype)

    #         # if torch.isinf(y).any() or torch.isnan(y).any():
    #         #     for item in [y, xs, dts, As, Bs, Cs, Ds]:
    #         #         print(torch.isinf(item).any(), torch.isnan(item).any(), item.max(), item.min())
    #         #     import time; time.sleep(10000)
        
    #     return y

    def forward_corev1_share_ssm(self, x: torch.Tensor):
        selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W

        def cross_scan_2d(x):
            # (B, C, H, W) => (B, K, C, H * W) with K = len([HW, WH, FHW, FWH])
            x_hwwh = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=1)
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)
            return xs
        
        x_dbl = torch.einsum("b d l, c d -> b c l", x.view(B, -1, L), self.x_proj_weight[0])
        # x_dbl = x_dbl + self.x_proj_bias.view(1, -1, 1)
        dt, BC = torch.split(x_dbl, [self.dt_rank, 2 * self.d_state], dim=1)
        dt = torch.einsum("b r l, d r -> b d l", dt, self.dt_projs_weight[0])
        x_dt_BC = torch.cat([x, dt.view(B, -1, H, W), BC.view(B, -1, H, W)], dim=1) # (b, -1, h, w)

        x_dt_BCs = cross_scan_2d(x_dt_BC) # (b, k, d, l)
        xs, dts, Bs, Cs = torch.split(x_dt_BCs, [self.d_inner, self.d_inner, self.d_state, self.d_state], dim=2)

        xs = xs.contiguous().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        As = -torch.exp(self.A_logs.float()).repeat(4, 1) # (k * d, d_state)
        Ds = self.Ds.repeat(4) # (k * d)
        dt_projs_bias = self.dt_projs_bias.view(-1).repeat(4) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, 4, -1, L)
        # assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()
        
        if self.softmax_version:
            y = torch.softmax(y, dim=-1).to(x.dtype)
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = self.out_norm(y).to(x.dtype)
        
        return y

    def forward_corev1_share_a(self, x: torch.Tensor):
        selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W

        def cross_scan_2d(x, dim=1):
            # (B, C, H, W) => (B, K, C, H * W) with K = len([HW, WH, FHW, FWH])
            x_hwwh = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=dim)
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=dim) # (b, k, d, l)
            return xs
        
        K = 4
        xs = cross_scan_2d(x, dim=1) # (b, d, k, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        dts = dts + self.dt_projs_bias.to(xs.dtype).view(1, K, -1, 1)

        xs = xs.transpose(dim0=1, dim1=2).contiguous().view(B, -1, K * L)
        dts = dts.transpose(dim0=1, dim1=2).contiguous().view(B, -1, K * L)
        As = -torch.exp(self.A_logs.float()) # (D, N)
        Ds = self.Ds.view(-1) # (D)
        Bs = Bs.transpose(dim0=1, dim1=2).contiguous().view(B, 1, -1, K * L)
        Cs = Cs.transpose(dim0=1, dim1=2).contiguous().view(B, 1, -1, K * L)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

        out_y = selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=None,
            delta_softplus=True,
        ).view(B, -1, 4, L)
        # assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, :, 2:4], dims=[-1]).view(B, -1, 2, L)
        wh_y = torch.transpose(out_y[:, :, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, :, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, :, 0].float() + inv_y[:, :, 0].float() + wh_y.float() + invwh_y.float()
        
        if self.softmax_version:
            y = torch.softmax(y, dim=-1).to(x.dtype)
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = self.out_norm(y).to(x.dtype)
        
        return y

    def forward_corev2(self, x: torch.Tensor, nrows=-1):
        return cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None), self.softmax_version, 
            nrows=nrows,
        )
        
    def forward_core_1d(self, x: torch.Tensor):
        return selective_scan_1d(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None), self.softmax_version,
        )
    
    # forward_core = forward_core_share_ssm
    # forward_core = forward_core_share_a
    # forward_core = forward_corev1
    forward_core = forward_corev2 # vmamba
    # forward_core = forward_corev0 # ori mamba
    # forward_core = forward_core_1d # ori mamba

    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x)) # (b, d, h, w)
            y = self.forward_core(x)
            if self.softmax_version:
                y = y * z
            else:
                y = y * F.silu(z)
        else:
            if self.softmax_version:
                x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
                x = F.silu(x)
            else:
                xz = F.silu(xz)
                x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
            x = x.permute(0, 3, 1, 2).contiguous()
            y = self.forward_core(x)
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint: bool = False,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim, 
            dropout=attn_drop_rate, 
            d_state=d_state, 
            ssm_ratio=ssm_ratio, 
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)
        
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=False)

    def _forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class VSSM(nn.Module):
    def __init__(
            self, 
            patch_size=4, 
            in_chans=3, 
            num_classes=1000, 
            depths=[2, 2, 9, 2], 
            dims=[96, 192, 384, 768], 
            # =========================
            d_state=16, 
            dt_rank="auto", 
            ssm_ratio=2.0, 
            attn_drop_rate=0., 
            shared_ssm=False,
            softmax_version=False,
            # =========================
            drop_rate=0., 
            drop_path_rate=0.1, 
            mlp_ratio=4.0,
            patch_norm=True, 
            norm_layer=nn.LayerNorm,
            downsample_version: str = "v2",
            use_checkpoint=False,  
            **kwargs,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, self.embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            Permute(0, 2, 3, 1),
            (norm_layer(self.embed_dim) if patch_norm else nn.Identity()), 
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):

            if downsample_version == "v2":
                downsample = self._make_downsample(
                    self.dims[i_layer], 
                    self.dims[i_layer + 1], 
                    norm_layer=norm_layer,
                ) if (i_layer < self.num_layers - 1) else nn.Identity()
            else:
                downsample = PatchMerging2D(
                    self.dims[i_layer], 
                    self.dims[i_layer + 1], 
                    norm_layer=norm_layer,
                ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                depth = depths[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                d_state=d_state,
                dt_rank=dt_rank,
                ssm_ratio=ssm_ratio,
                attn_drop_rate=attn_drop_rate,
                shared_ssm=shared_ssm,
                softmax_version=softmax_version,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=Permute(0, 3, 1, 2),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96, 
        depth=2,
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        # ===========================
        d_state=16,
        dt_rank="auto",
        ssm_ratio=2.0,
        attn_drop_rate=0.0, 
        shared_ssm=False,
        softmax_version=False,
        # ===========================
        mlp_ratio=4.0,
        drop_rate=0.0,
        **kwargs,
    ):
        assert depth == len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                dt_rank=dt_rank,
                ssm_ratio=ssm_ratio,
                shared_ssm=shared_ssm,
                softmax_version=softmax_version,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                act_layer=nn.GELU,
                drop=drop_rate,
                **kwargs,
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

    def flops(self, shape=(3, 224, 224)):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            "prim::PythonOp.CrossScan": None,
            "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")
        # print('after:', state_dict.keys())

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


# compatible with openmmlab
class Backbone_VSSM_tiny(VSSM):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, 
                 depths=[2, 2, 27, 2], dims=[16, 32, 64, 128], 
                 d_state=16, ssm_ratio=2.0, attn_drop_rate=0., 
                 drop_rate=0., drop_path_rate=0.08, mlp_ratio=4.0,
                 patch_norm=True, norm_layer=nn.LayerNorm,
                 downsample_version: str = "v2",
                 use_checkpoint=False,
                 out_indices=(0, 1, 2, 3), pretrained=None, 
                 **kwargs,
        ):
        super().__init__(patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, 
                         depths=depths, dims=dims, 
                         d_state=d_state, ssm_ratio=ssm_ratio, attn_drop_rate=attn_drop_rate, 
                         drop_rate=drop_rate, drop_path_rate=drop_path_rate, mlp_ratio=mlp_ratio,
                         patch_norm=patch_norm, norm_layer=norm_layer,
                         downsample_version=downsample_version,
                         use_checkpoint=use_checkpoint,
                         **kwargs)
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            # print(f"ckpt keys: {_ckpt['model'].keys()}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print('incompatible:', incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x
        
        return outs


def _get_horizontal_snake_path(H, W):
    coords = []
    for r in range(H):
        if r % 2 == 0:
            for c in range(W):
                coords.append((r, c))
        else:
            for c in range(W - 1, -1, -1):
                coords.append((r, c))
    return coords


def _get_vertical_snake_path(H, W):
    coords = []
    for c in range(W):
        if c % 2 == 0:
            for r in range(H):
                coords.append((r, c))
        else:
            for r in range(H - 1, -1, -1):
                coords.append((r, c))
    return coords


def _get_main_diagonal_snake_path(H, W):
    coords = []
    for d in range(-(W - 1), H):
        row_list = []
        for r in range(max(0, d), min(H, W + d)):
            c = r - d
            if 0 <= c < W:
                row_list.append((r, c))
        if (d + (W - 1)) % 2 == 1:
            row_list = row_list[::-1]
        coords.extend(row_list)
    return coords


def _get_anti_diagonal_snake_path(H, W):
    coords = []
    for d in range(0, H + W - 1):
        row_list = []
        for r in range(max(0, d - W + 1), min(H, d + 1)):
            c = d - r
            if 0 <= c < W:
                row_list.append((r, c))
        if d % 2 == 1:
            row_list = row_list[::-1]
        coords.extend(row_list)
    return coords


def get_scan_path_indices(H, W, direction_mode, device):
    path_generators = {
        0: _get_horizontal_snake_path, 1: _get_vertical_snake_path,
        2: _get_anti_diagonal_snake_path, 3: _get_main_diagonal_snake_path
    }
    path_func = path_generators[direction_mode]
    coord_path = path_func(H, W)
    indices_1d = torch.tensor([r * W + c for r, c in coord_path], device=device, dtype=torch.long)
    return indices_1d

def reorder_by_path_and_saliency(x, guide_map, path_indices):
    B, C, H, W = x.shape
    L = H * W
    saliency_priority = (guide_map < 0.5).float() * 1e9 
    saliency_priority_flat = saliency_priority.view(B, 1, L)
    path_priority = torch.empty_like(path_indices)
    path_priority[path_indices] = torch.arange(L, device=x.device, dtype=torch.long)
    path_priority = path_priority.view(1, 1, L).expand(B, -1, -1) 
    sort_key = saliency_priority_flat + path_priority
    _, sorted_indices = torch.sort(sort_key, dim=-1) 
    _, restore_indices = torch.sort(sorted_indices, dim=-1)
    sorted_indices_expanded = sorted_indices.expand(-1, C, -1)
    reordered_x = torch.gather(x.view(B, C, L), -1, sorted_indices_expanded)
    return reordered_x, restore_indices



class HierarchicalSE_SSM(nn.Module):
    
    def __init__(
    self,
    d_model=96,
    d_state=16,
    ssm_ratio=2.0,
    dt_rank="auto",
    d_conv=3,
    conv_bias=True,
    dropout=0.0,
    bias=False,
    dt_min=0.001,
    dt_max=0.1,
    dt_init="random",
    dt_scale=1.0,
    dt_init_floor=1e-4,
    flip=True,
    se_ratio=0.25,
    nrows=1, 
    **kwargs,
    ):

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.flip = flip
        self.d_model = d_model
        self.nrows = nrows
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = 4

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                self.d_inner, self.d_inner, groups=self.d_inner, bias=conv_bias,
                kernel_size=d_conv, padding=(d_conv - 1) // 2, **factory_kwargs,
            )
            self.act = nn.SiLU()

        self.x_proj_weight = nn.Parameter(torch.stack([
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs).weight
            for _ in range(self.K)
        ], dim=0))
        dt_projs = [self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs) for _ in range(self.K)]
        self.dt_projs_weight = nn.Parameter(torch.stack([p.weight for p in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([p.bias for p in dt_projs], dim=0))
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True)
        
        self.intra_fusion_convs = nn.ModuleList([nn.Conv2d(self.K * self.d_inner, self.d_inner, 1, bias=False) for _ in range(self.K)])
        self.inter_fusion_conv = nn.Conv2d(self.K * self.d_inner, self.d_inner, 1, bias=False)
        
        se_hidden_dim = int(self.d_inner * se_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.d_inner, se_hidden_dim, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(se_hidden_dim, self.d_inner, bias=False),
            nn.Sigmoid(),
        )

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.scan_paths = {}
        

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant": nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random": nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad(): dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge: A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log); A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge: D = D.flatten(0, 1)
        D = nn.Parameter(D); D._no_weight_decay = True
        return D
    
    @staticmethod
    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

    def forward(self, x: torch.Tensor, guide_map_4_channel: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        device = x.device

        x_permuted = x.permute(0, 2, 3, 1).contiguous() 
        x_proj = self.in_proj(x_permuted)              
        x_proj_ch_first = x_proj.permute(0, 3, 1, 2).contiguous() 

        ssm_input = x_proj
        se_input = x_proj_ch_first 
        
        if self.d_conv > 1:
            x_conv = self.act(self.conv2d(x_proj_ch_first))
            ssm_input = x_conv.permute(0, 2, 3, 1).contiguous() 
            se_input = x_conv                                    
        inter_fusion_inputs = [] 
        for i in range(self.K):
            guide_map_i = guide_map_4_channel[:, i:i+1, :, :]
            intra_fusion_inputs = []
            for j in range(self.K):
                path_key = f"{H}x{W}_{j}"
                path_indices = self.scan_paths.setdefault(path_key, get_scan_path_indices(H, W, j, device))
                ssm_input_permuted = ssm_input.permute(0, 3, 1, 2).contiguous()
                reordered_x, restore_indices = reorder_by_path_and_saliency(ssm_input_permuted, guide_map_i, path_indices)
                reordered_x = reordered_x.permute(0, 2, 1).contiguous()
                x_dtsbc = F.linear(reordered_x, self.x_proj_weight[j])
                dt, B_seq, C_seq = torch.split(x_dtsbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                dt = F.linear(dt, self.dt_projs_weight[j])
                u = reordered_x.permute(0, 2, 1).contiguous()
                dt = dt.permute(0, 2, 1).contiguous()
                B_seq = B_seq.permute(0, 2, 1).contiguous()
                C_seq = C_seq.permute(0, 2, 1).contiguous()
                A_log_j, D_j = self.A_logs[j*self.d_inner:(j+1)*self.d_inner, :], self.Ds[j*self.d_inner:(j+1)*self.d_inner]
                A_j = -torch.exp(A_log_j.float())
                delta_bias_j = self.dt_projs_bias[j]
                y_seq = self.selective_scan(
                    u, dt, A_j, B_seq, C_seq, D=D_j, 
                    delta_bias=delta_bias_j, 
                    delta_softplus=True, 
                    nrows=self.nrows
                )
                if self.flip:
                    y_seq_flipped = self.selective_scan(
                        u.flip(dims=[2]), dt.flip(dims=[2]), A_j, 
                        B_seq.flip(dims=[2]), C_seq.flip(dims=[2]), D=D_j, 
                        delta_bias=delta_bias_j,
                        delta_softplus=True, 
                        nrows=self.nrows
                    )
                    y_seq = (y_seq + y_seq_flipped.flip(dims=[2])) 
                y_seq = y_seq.permute(0, 2, 1).contiguous()
                y_restored = y_seq.view(B, L, self.d_inner)
                y_restored_flat = torch.gather(y_restored.permute(0, 2, 1), -1, restore_indices.expand(-1, self.d_inner, -1))
                y_restored_2d = y_restored_flat.view(B, self.d_inner, H, W)
                intra_fusion_inputs.append(y_restored_2d)
            intra_stacked = torch.cat(intra_fusion_inputs, dim=1)
            fused_for_this_guidemap = self.intra_fusion_convs[i](intra_stacked)
            inter_fusion_inputs.append(fused_for_this_guidemap)

        inter_stacked = torch.cat(inter_fusion_inputs, dim=1)
        y_scanned = self.inter_fusion_conv(inter_stacked) 
        squeezed = self.avg_pool(se_input).view(B, self.d_inner)
        excitation = self.fc(squeezed).view(B, self.d_inner, 1, 1)
        y_scaled = y_scanned * excitation 
        y = y_scaled.permute(0, 2, 3, 1).contiguous() 
        y_norm = self.out_norm(y)
        out = self.dropout(self.out_proj(y_norm))    
        out = out.permute(0, 3, 1, 2).contiguous()
        return out


class DPSM(nn.Module):
    def __init__(
        self,
        d_model=96,
        norm_layer=nn.LayerNorm,
        mlp_ratio=4.0,
        drop_path=0.0,
        **kwargs
    ):
        super().__init__()
        
        self.norm1 = norm_layer(d_model)
        self.ssm_engine = HierarchicalSE_SSM(
            d_model=d_model,
            **kwargs
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(d_model)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = Mlp(in_features=d_model, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.0)

    def forward(self, x: torch.Tensor, guide_map_4_channel: torch.Tensor):
        B, C, H, W = x.shape

        shortcut = x 
        x_norm = self.norm1(x)
        x_norm = x_norm.permute(0, 3, 1, 2).contiguous()   
                
        fused_output = self.ssm_engine(x_norm, guide_map_4_channel)
        # print(fused_output.shape)
        x = shortcut + self.drop_path(fused_output.permute(0,3,2,1))
        ffn_shortcut = x
        x_norm2 = self.norm2(x)
        x_ffn = self.mlp(x_norm2)
        x = ffn_shortcut + self.drop_path(x_ffn)
        return x
