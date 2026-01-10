import torch
import torch.nn as nn
from torch.autograd import Function
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

class GeodynamicController(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim_A, out_dim_G):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim_A + out_dim_G)
        )
        self.init_weights()
        
    def init_weights(self):
        with torch.no_grad():
            self.net[-1].weight.mul_(0.001)
            self.net[-1].bias.zero_()
            
    def forward(self, x):
        return self.net(x)

class GeodynamicSpectralLayer(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 steps: int = 8, 
                 method: str = 'rk4',
                 tol: float = 1e-6,
                 spectral_limit: float = 10.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.steps = steps
        self.method = method
        self.config = {'tol': tol, 'max_sv': spectral_limit, 'drift_correction': True}
        
        base_W = torch.randn(out_features, in_features)
        Q, _ = torch.linalg.qr(base_W)
        self.U0 = nn.Parameter(Q)
        
        self.size_A = in_features * in_features
        self.size_G = out_features * in_features
        
        self.controller = GeodynamicController(
            in_dim=in_features,
            hidden_dim=in_features // 2,
            out_dim_A=self.size_A,
            out_dim_G=self.size_G
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_ctx = x.mean(dim=1)
        
        params = self.controller(x_ctx)
        
        raw_A = params[:, :self.size_A]
        raw_G = params[:, self.size_A:]
        
        A = raw_A.view(B, self.in_features, self.in_features)
        G = raw_G.view(B, self.out_features, self.in_features)
        
        A = SpectralAlgebra.skew(A)
        
        if self.training:
            A = SpectralAlgebra.complex_spectral_filter(A, self.config['max_sv'])
        
        U0_batch = self.U0.unsqueeze(0).expand(B, -1, -1)
        
        W_dynamic = GeodynamicSolver.apply(
            U0_batch, 
            A, 
            G, 
            self.steps, 
            self.method, 
            self.config
        )
        
        return torch.einsum('bsi,boi->bso', x, W_dynamic)

class TransportLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, vec, u_start, u_end):
        p1 = RiemannianMetric.project(u_start, vec)
        p2 = RiemannianMetric.project(u_end, p1)
        return p2

class ManifoldRegularizer(nn.Module):
    def __init__(self, strength=0.01):
        super().__init__()
        self.strength = strength
        
    def forward(self, A):
        spec_norm = torch.linalg.norm(A, ord=2, dim=(-2, -1))
        return self.strength * spec_norm.mean()

def unit_test_geodynamic():
    B, N, P = 2, 64, 32
    
    x = torch.randn(B, 10, P)
    
    layer = GeodynamicSpectralLayer(P, N, steps=4, method='rk4')
    y = layer(x)
    
    err = 0.0
    if torch.isnan(y).any():
        err = 1.0
        
    loss = y.sum()
    loss.backward()
    
    grad_norm = layer.U0.grad.norm().item()
    
    print(f"Test Complete. Error: {err}, Grad Norm: {grad_norm}")

class AugmentedODE(nn.Module):
    def __init__(self, base_layer):
        super().__init__()
        self.base = base_layer
        
    def forward(self, t, state):
        u, lam = state
        fields = VectorFields(self.base.A_curr, self.base.G_curr, ManifoldConfig())
        du, _ = fields.forward_velocity(u)
        dlam = fields.adjoint_velocity(lam, u)
        return du, dlam

class StiefelFlow:
    def __init__(self, p, n):
        self.p = p
        self.n = n
        
    def random_point(self):
        w = torch.randn(self.n, self.p)
        q, _ = torch.linalg.qr(w)
        return q
    
    def distance(self, u1, u2):
        m = torch.matmul(u1.transpose(-1, -2), u2)
        return self.p - torch.trace(m)

class LieGroupIntegrator:
    def __init__(self, order=4):
        self.order = order
        
    def integrate(self, generator_fn, y0, t_span, steps):
        dt = (t_span[1] - t_span[0]) / steps
        y = y0
        t = t_span[0]
        for _ in range(steps):
            k1 = generator_fn(t, y)
            y_pred = torch.matmul(SpectralAlgebra.cayley_map(0.5*dt*k1), y)
            k2 = generator_fn(t + 0.5*dt, y_pred)
            w = dt * k2 
            y = torch.matmul(SpectralAlgebra.cayley_map(w), y)
            t += dt
        return y

def jacobian_check(layer, x, epsilon=1e-4):
    u0_flat = layer.U0.view(-1)
    
    base_out = layer(x).sum()
    base_out.backward()
    analytic_grad = layer.U0.grad.view(-1).clone()
    layer.zero_grad()
    
    idx = torch.randint(0, u0_flat.shape[0], (1,)).item()
    
    with torch.no_grad():
        layer.U0.view(-1)[idx] += epsilon
        pos_out = layer(x).sum()
        layer.U0.view(-1)[idx] -= 2*epsilon
        neg_out = layer(x).sum()
        layer.U0.view(-1)[idx] += epsilon
        
    numeric_grad = (pos_out - neg_out) / (2 * epsilon)
    
    diff = torch.abs(analytic_grad[idx] - numeric_grad)
    return diff.item() < 1e-3

class SectionalCurvature:
    @staticmethod
    def compute(U, A, B):
        cov_A = torch.matmul(U, SpectralAlgebra.skew(A))
        cov_B = torch.matmul(U, SpectralAlgebra.skew(B))
        commutator = SpectralAlgebra.lie_bracket(cov_A, cov_B)
        return torch.norm(commutator)

if __name__ == "__main__":
    unit_test_geodynamic()