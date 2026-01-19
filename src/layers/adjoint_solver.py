import torch
import torch.nn as nn
from torch.autograd import Function
import torch

# --- CROSS-VERSION COMPATIBILITY FOR AMP ---
# PyTorch 2.4+ requires device_type='cuda'. Older versions crash with it.
try:
    from torch.amp import custom_fwd as _cfwd, custom_bwd as _cbwd
    NEW_AMP_API = True
except ImportError:
    from torch.cuda.amp import custom_fwd as _cfwd, custom_bwd as _cbwd
    NEW_AMP_API = False

def safe_custom_fwd(**kwargs):
    # Strip device_type if we are on an older PyTorch that doesn't support it
    # We detect this loosely by checking if we had to import from torch.cuda.amp
    # OR explicit version check. 
    if 'device_type' in kwargs:
        # Check if the function accepts it? inspect is slow.
        # Simple heuristic: torch 2.4+ needs it.
        # But safest is just:
        if torch.__version__ < '2.4':
            kwargs.pop('device_type')
    return _cfwd(**kwargs)

def safe_custom_bwd(**kwargs):
    if 'device_type' in kwargs and torch.__version__ < '2.4':
        kwargs.pop('device_type')
    return _cbwd(**kwargs)
# -------------------------------------------
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
    def cayley_map(W: torch.Tensor, chunk: int = 2) -> torch.Tensor:
        """
        Cayley map: (I - 0.5 W)^{-1} (I + 0.5 W)
        Chunked solve reduces peak workspace; chunk=1 or 2 is fastest-safe for big matrices.
        """
        *batch, N, _ = W.shape
        Wf = W.reshape(-1, N, N)  # [Bflat, N, N]
        Bflat = Wf.shape[0]

        I = torch.eye(N, device=W.device, dtype=W.dtype).unsqueeze(0)  # [1, N, N]
        out = torch.empty_like(Wf)

        for s in range(0, Bflat, chunk):
            Ws = Wf[s : s + chunk]
            Is = I.expand(Ws.shape[0], N, N)

            # 1. NaN Safety Check
            if torch.isnan(Ws).any():
                raise RuntimeError(f"Cayley map input 'W' contains NaNs at chunk {s}!")

            # Cast to float64 for stability
            Ws_d = Ws.double()
            Is_d = Is.double()
            
            # Tikhonov Regularization
            # norm_W = Ws.norm(dim=(-2,-1), keepdim=True).double() # Check this too if needed?
            # Safe norm to avoid infs in extreme cases
            norm_W = torch.linalg.matrix_norm(Ws_d, ord='fro', dim=(-2,-1), keepdim=True)
            eps = 1e-6 + 1e-4 * norm_W
            
            # A = (1+eps)I - 0.5*W
            # B = (1)I + 0.5*W
            A = Is_d * (1.0 + eps) - 0.5 * Ws_d
            B = Is_d + 0.5 * Ws_d
            
            # 2. Robust Solve with Fallback
            try:
                out_d = torch.linalg.solve(A, B)
            except RuntimeError as e:
                # If solve fails (singular), try Least Squares (lstsq) which handles rank-deficient cases
                # or Pseudo-Inverse.
                print(f"⚠️ Cayley linalg.solve failed (Singular?): {e}. Fallback to lstsq.", flush=True)
                solution = torch.linalg.lstsq(A, B).solution
                out_d = solution
            
            # Cast back
            out[s : s + chunk] = out_d.to(dtype=W.dtype)

        return out.reshape(*batch, N, N)


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
    @safe_custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, U0, A, G, steps, method_str, config_dict):
        config = ManifoldConfig(**config_dict)
        steps = int(steps)
        dt = 1.0 / max(1, steps)

        # Save minimal tensors for backward.
        ctx.steps = steps
        ctx.dt = dt
        ctx.config = config
        ctx.method_str = str(method_str)

        ctx.save_for_backward(U0, A, G)

        # Forward integrate (no Python trajectory retained on ctx)
        fields = VectorFields(A, G, config)
        current_U = U0
        for _ in range(steps):
            v, W = fields.forward_velocity(current_U)
            step_op = SpectralAlgebra.cayley_map(dt * W, chunk=1)
            current_U = torch.matmul(step_op, current_U)

        return current_U

    @staticmethod
    @safe_custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        # IMPORTANT: prevent backward from building its own graph (VRAM creep fix)
        with torch.no_grad():
            U0, A, G = ctx.saved_tensors
            steps = ctx.steps
            dt = ctx.dt
            config = ctx.config

            # Rebuild fields locally (do not keep ctx.fields references)
            fields = VectorFields(A, G, config)

            # Recompute states U_t (detached) for adjoint loop
            # We keep a local list, but it is NOT stored on ctx and uses no_grad -> stable memory.
            U_states = [U0]
            current_U = U0
            for _ in range(steps):
                v, W = fields.forward_velocity(current_U)
                step_op = SpectralAlgebra.cayley_map(dt * W, chunk=1)
                current_U = torch.matmul(step_op, current_U)
                U_states.append(current_U)

            # Initialize adjoint at final time
            lambda_curr = RiemannianMetric.euclidean_to_riemannian(U_states[-1], grad_output)

            grad_A_accum = torch.zeros_like(A)
            grad_G_accum = torch.zeros_like(G)

            # Backward integrate adjoint
            for i in range(steps - 1, -1, -1):
                U_t = U_states[i]

                # Project to tangent
                lambda_proj = RiemannianMetric.project(U_t, lambda_curr)

                # Sensitivities wrt A, G (same as your current logic)
                raw_A_sens = torch.matmul(U_t.transpose(-1, -2), lambda_proj)
                d_grad_A = SpectralAlgebra.skew(raw_A_sens)

                UT_lam = torch.matmul(U_t.transpose(-1, -2), lambda_proj)
                d_grad_G = lambda_proj - torch.matmul(U_t, UT_lam)

                grad_A_accum += d_grad_A * dt
                grad_G_accum += d_grad_G * dt

                # Adjoint dynamics step
                d_lam = fields.adjoint_velocity(lambda_proj, U_t)
                lambda_curr = lambda_curr - dt * d_lam

            # Return grads for (U0, A, G, steps, method_str, config_dict)
            return lambda_curr, grad_A_accum, grad_G_accum, None, None, None
