import torch
from torch.autograd import Function

try:
    from torch.amp import custom_fwd as _cfwd, custom_bwd as _cbwd
except ImportError:
    from torch.cuda.amp import custom_fwd as _cfwd, custom_bwd as _cbwd

class SpectralAlgebra:
    @staticmethod
    def cayley_map(W):
        """
        Fast Cayley Map with Fallback logic for Singularity.
        Architecture invariant: (I - 0.5W)^-1 (I + 0.5W)
        """
        W = W.to(torch.float32)
        # Force skew-symmetry to reduce numerical noise
        W = 0.5 * (W - W.transpose(-1, -2))
        
        N = W.shape[-1]
        I = torch.eye(N, device=W.device, dtype=torch.float32)
        
        # Add epsilon (Tikhonov) for stability
        A = I - 0.5 * W + (1e-4 * I)
        B = I + 0.5 * W
        
        try:
            return torch.linalg.solve(A, B)
        except RuntimeError:
            # Fallback to Least Squares if singular
            return torch.linalg.lstsq(A, B).solution

class GeodynamicSolver(Function):
    @staticmethod
    @_cfwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, U0, A, G):
        Omega = 0.5 * (A - A.transpose(-1, -2))
        vel = torch.matmul(U0, Omega) + (G - torch.matmul(U0, torch.matmul(U0.transpose(-1, -2), G)))
        W = torch.matmul(vel, U0.transpose(-1, -2)) - torch.matmul(U0, vel.transpose(-1, -2))
        
        step_op = SpectralAlgebra.cayley_map(W)
        U_next = torch.matmul(step_op, U0)
        
        ctx.save_for_backward(U0, U_next, step_op)
        return U_next

    @staticmethod
    @_cbwd(device_type="cuda")
    def backward(ctx, grad_output):
        U0, U_next, step_op = ctx.saved_tensors
        lam = torch.matmul(step_op.transpose(-1, -2), grad_output)
        
        # Project adjoint
        UT_lam = torch.matmul(U0.transpose(-1, -2), lam)
        lam_p = lam - torch.matmul(U0, 0.5 * (UT_lam + UT_lam.transpose(-1, -2)))
        
        # Gradients
        raw_A = torch.matmul(U0.transpose(-1, -2), lam_p)
        grad_A = 0.5 * (raw_A - raw_A.transpose(-1, -2))
        grad_G = lam_p - torch.matmul(U0, torch.matmul(U0.transpose(-1, -2), lam_p))
        
        return lam, grad_A, grad_G