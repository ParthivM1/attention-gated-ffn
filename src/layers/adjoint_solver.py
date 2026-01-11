import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd # <--- NEW IMPORT
from typing import Tuple, Optional, List, Callable, Union, Dict
import math
import logging

class ManifoldConfig:
    def __init__(self, tol=1e-6, max_sv=10.0, drift_correction=True):
        self.tolerance = tol
        self.spectral_limit = max_sv
        self.use_drift_correction = drift_correction

class SpectralAlgebra:
    @staticmethod
    def skew(X: torch.Tensor) -> torch.Tensor:
        return 0.5 * (X - X.transpose(-1, -2))

    @staticmethod
    def sym(X: torch.Tensor) -> torch.Tensor:
        return 0.5 * (X + X.transpose(-1, -2))

    @staticmethod
    def lie_bracket(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B) - torch.matmul(B, A)

    @staticmethod
    def cayley_map(W: torch.Tensor) -> torch.Tensor:
        B, N, _ = W.shape
        Id = torch.eye(N, dtype=W.dtype, device=W.device).unsqueeze(0).expand(B, -1, -1)
        return torch.linalg.solve(Id - 0.5 * W, Id + 0.5 * W)

    @staticmethod
    def inverse_cayley(Q: torch.Tensor) -> torch.Tensor:
        B, N, _ = Q.shape
        Id = torch.eye(N, dtype=Q.dtype, device=Q.device).unsqueeze(0).expand(B, -1, -1)
        return 2.0 * torch.linalg.solve(Id + Q, Q - Id)

    @staticmethod
    def spectral_clip(A: torch.Tensor, limit: float) -> torch.Tensor:
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        S_clamped = torch.clamp(S, max=limit)
        return torch.matmul(U, torch.matmul(torch.diag_embed(S_clamped), Vh))

    @staticmethod
    def complex_spectral_filter(A: torch.Tensor, threshold: float) -> torch.Tensor:
        vals, vecs = torch.linalg.eigh(1.0j * A)
        real_vals = vals.real
        mask = torch.abs(real_vals) <= threshold
        filtered = real_vals * mask + threshold * torch.sign(real_vals) * (~mask)
        recon = torch.matmul(vecs, torch.matmul(torch.diag_embed(filtered.to(torch.complex64)), vecs.mH))
        return recon.imag

class RiemannianMetric:
    @staticmethod
    def inner_product(U: torch.Tensor, D1: torch.Tensor, D2: torch.Tensor) -> torch.Tensor:
        euc = torch.sum(D1 * D2, dim=(-1, -2))
        UT_D2 = torch.matmul(U.transpose(-1, -2), D2)
        D1T_U = torch.matmul(D1.transpose(-1, -2), U)
        correction = -0.5 * torch.sum(D1T_U * UT_D2.transpose(-1, -2), dim=(-1, -2))
        return euc + correction

    @staticmethod
    def euclidean_to_riemannian(U: torch.Tensor, grad_euc: torch.Tensor) -> torch.Tensor:
        Gt_U = torch.matmul(grad_euc.transpose(-1, -2), U)
        U_Gt_U = torch.matmul(U, Gt_U)
        return grad_euc - U_Gt_U

    @staticmethod
    def project(U: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        UT_Z = torch.matmul(U.transpose(-1, -2), Z)
        sym_UT_Z = SpectralAlgebra.sym(UT_Z)
        return Z - torch.matmul(U, sym_UT_Z)

    @staticmethod
    def orthogonality_error(U: torch.Tensor) -> torch.Tensor:
        B, _, P = U.shape
        Gram = torch.matmul(U.transpose(-1, -2), U)
        Id = torch.eye(P, device=U.device).unsqueeze(0).expand(B, -1, -1)
        return torch.norm(Gram - Id)

class VectorFields:
    def __init__(self, A: torch.Tensor, G: torch.Tensor, config: ManifoldConfig):
        self.A = A
        self.G = G
        self.config = config

    def forward_velocity(self, U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Omega = SpectralAlgebra.skew(self.A)
        vertical = torch.matmul(U, Omega)
        UT_G = torch.matmul(U.transpose(-1, -2), self.G)
        horizontal = self.G - torch.matmul(U, UT_G)
        velocity = vertical + horizontal
        W_lift = torch.matmul(velocity, U.transpose(-1, -2)) - torch.matmul(U, velocity.transpose(-1, -2))
        return velocity, W_lift

    def adjoint_velocity(self, lam: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        term_rot = torch.matmul(lam, self.A)
        G_T = self.G.transpose(-1, -2)
        lam_T = lam.transpose(-1, -2)
        coupling = torch.matmul(lam, G_T) + torch.matmul(self.G, lam_T)
        term_exp = torch.matmul(coupling, U)
        return term_rot + term_exp

class AbstractIntegrator:
    def step(self, U: torch.Tensor, dt: float, dynamics: Callable) -> torch.Tensor:
        raise NotImplementedError

class EulerIntegrator(AbstractIntegrator):
    def step(self, U: torch.Tensor, dt: float, dynamics: Callable) -> torch.Tensor:
        vel, W = dynamics(U)
        op = SpectralAlgebra.cayley_map(dt * W)
        return torch.matmul(op, U)

class MidpointIntegrator(AbstractIntegrator):
    def step(self, U: torch.Tensor, dt: float, dynamics: Callable) -> torch.Tensor:
        v1, w1 = dynamics(U)
        half_step_op = SpectralAlgebra.cayley_map(0.5 * dt * w1)
        u_mid = torch.matmul(half_step_op, U)
        v2, w2 = dynamics(u_mid)
        step_op = SpectralAlgebra.cayley_map(dt * w2)
        return torch.matmul(step_op, U)

class MuntheKaasRK4Integrator(AbstractIntegrator):
    def step(self, U: torch.Tensor, dt: float, dynamics: Callable) -> torch.Tensor:
        v1, w1 = dynamics(U)
        
        op1 = SpectralAlgebra.cayley_map(0.5 * dt * w1)
        u2 = torch.matmul(op1, U)
        v2, w2 = dynamics(u2)
        
        op2 = SpectralAlgebra.cayley_map(0.5 * dt * w2)
        u3 = torch.matmul(op2, U)
        v3, w3 = dynamics(u3)
        
        op3 = SpectralAlgebra.cayley_map(dt * w3)
        u4 = torch.matmul(op3, U)
        v4, w4 = dynamics(u4)
        
        w_avg = (dt / 6.0) * (w1 + 2*w2 + 2*w3 + w4)
        step_op = SpectralAlgebra.cayley_map(w_avg)
        return torch.matmul(step_op, U)

class GeodynamicSolver(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # <--- CRITICAL FIX: Ensures solver runs in FP32
    def forward(ctx, U0, A, G, steps, method_str, config_dict):
        config = ManifoldConfig(**config_dict)
        
        if method_str == 'rk4':
            integrator = MuntheKaasRK4Integrator()
        elif method_str == 'midpoint':
            integrator = MidpointIntegrator()
        else:
            integrator = EulerIntegrator()
            
        fields = VectorFields(A, G, config)
        
        ctx.steps = steps
        ctx.dt = 1.0 / steps
        ctx.integrator = integrator
        ctx.fields = fields
        ctx.config = config
        
        current_U = U0
        trajectory = [U0]
        generators = []
        
        for _ in range(steps):
            def closure(u): return fields.forward_velocity(u)
            
            if config.use_drift_correction:
                v, w = closure(current_U)
                omega_skew = SpectralAlgebra.skew(A)
                comm = SpectralAlgebra.lie_bracket(omega_skew, w)
                corr = 0.5 * ctx.dt * torch.matmul(current_U, comm)
                v_corr = v + corr
                w_corr = torch.matmul(v_corr, current_U.transpose(-1, -2)) - torch.matmul(current_U, v_corr.transpose(-1, -2))
                step_op = SpectralAlgebra.cayley_map(ctx.dt * w_corr)
                current_U = torch.matmul(step_op, current_U)
                generators.append(w_corr)
            else:
                current_U = integrator.step(current_U, ctx.dt, closure)
                generators.append(None)
                
            trajectory.append(current_U)
            
        ctx.save_for_backward(U0, A, G)
        ctx.trajectory = trajectory
        ctx.generators = generators
        
        return current_U

    @staticmethod
    @custom_bwd # <--- CRITICAL FIX: Handles AMP casting for gradients
    def backward(ctx, grad_output):
        U0, A, G = ctx.saved_tensors
        trajectory = ctx.trajectory
        generators = ctx.generators
        config = ctx.config
        dt = ctx.dt
        steps = ctx.steps
        
        lambda_curr = RiemannianMetric.euclidean_to_riemannian(trajectory[-1], grad_output)
        
        grad_A_accum = torch.zeros_like(A)
        grad_G_accum = torch.zeros_like(G)
        
        for i in range(steps - 1, -1, -1):
            U_t = trajectory[i]
            
            if config.use_drift_correction and generators[i] is not None:
                W_t = generators[i]
                inv_step = SpectralAlgebra.cayley_map(-dt * W_t)
                lambda_curr = torch.matmul(inv_step.transpose(-1, -2), lambda_curr)
            else:
                lambda_curr = RiemannianMetric.euclidean_to_riemannian(U_t, lambda_curr)
                
            lambda_proj = RiemannianMetric.project(U_t, lambda_curr)
            
            raw_A_sens = torch.matmul(U_t.transpose(-1, -2), lambda_proj)
            d_grad_A = SpectralAlgebra.skew(raw_A_sens)
            
            UT_lam = torch.matmul(U_t.transpose(-1, -2), lambda_proj)
            d_grad_G = lambda_proj - torch.matmul(U_t, UT_lam)
            
            grad_A_accum += d_grad_A * dt
            grad_G_accum += d_grad_G * dt
            
            d_lam = ctx.fields.adjoint_velocity(lambda_proj, U_t)
            lambda_curr = lambda_curr - dt * d_lam
            
        return lambda_curr, grad_A_accum, grad_G_accum, None, None, None