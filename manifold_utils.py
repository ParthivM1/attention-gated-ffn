import torch
import torch.nn as nn
import math
from Geo_math import RiemannianMetric, SpectralAlgebra

def clip_grad_norm_riemannian_(parameters, max_norm, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    
    total_norm = 0.0
    
    for p in parameters:
        param_norm = 0.0
        grad = p.grad
        
        if p.ndim == 2 and p.shape[0] >= p.shape[1]: 
             UT_g = torch.matmul(p.transpose(-1,-2), grad)
             sym_UT_g = 0.5 * (UT_g + UT_g.transpose(-1,-2))
             grad_tangent = grad - torch.matmul(p, sym_UT_g)
             param_norm = torch.norm(grad_tangent, norm_type)
        else:
             param_norm = torch.norm(grad, norm_type)
             
        if norm_type == float('inf'):
            total_norm = max(total_norm, param_norm.item())
        else:
            total_norm += param_norm.item() ** norm_type

    total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
            
    return total_norm

def geometric_xavier_init_(tensor, gain=1.0):
    with torch.no_grad():
        rows, cols = tensor.shape
        q, r = torch.linalg.qr(torch.randn(rows, cols, device=tensor.device))
        d = torch.diagonal(r, 0)
        ph = d.sign()
        q *= ph
        tensor.view_as(q).copy_(q)
    return tensor