import torch

class StiefelManifold:

    @staticmethod
    def skew(X):

        return 0.5 * (X - X.transpose(-1, -2))
    
    @staticmethod
    def sym(X):

        return 0.5 * (X + X.transpose(-1, -2))
    
    @staticmethod
    def check_orthogonality(U, atol=1e-5):

        B, N, P = U.shape
        I = torch.eye(P, device=U.device).unsqueeze(0).expand(B, -1, -1)
        gram = torch.matmul(U.transpose(-1, -2), U)
        diff = torch.norm(gram - I)
        return diff < atol


    @staticmethod
    def project_tangent(U, Z):

        UT_Z = torch.matmul(U.transpose(-1, -2), Z)
        sym_UT_Z = StiefelManifold.sym(UT_Z)
        return Z - torch.matmul(U, sym_UT_Z)

    @staticmethod
    def euclidean_to_riemannian_gradient(U, grad_euc):

        Gt_U = torch.matmul(grad_euc.transpose(-1, -2), U)
        U_Gt_U = torch.matmul(U, Gt_U)
        grad_riemann = grad_euc - U_Gt_U
        return grad_riemann


    @staticmethod
    def canonical_inner_product(U, Delta1, Delta2):

        euclidean_dot = torch.sum(Delta1 * Delta2, dim=(-1, -2))

        UT_D2 = torch.matmul(U.transpose(-1, -2), Delta2)
        D1T_U = torch.matmul(Delta1.transpose(-1, -2), U)
        
        curvature_term = -0.5 * torch.sum(D1T_U * UT_D2.transpose(-1, -2), dim=(-1, -2))
        
        return euclidean_dot + curvature_term


    @staticmethod
    def retraction_qr(U):

        Q, R = torch.linalg.qr(U)
        
        diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
        sign_R = torch.sign(diag_R)
        sign_R[sign_R == 0] = 1 
        
        sign_R = sign_R.unsqueeze(1) 
        Q_corrected = Q * sign_R
        
        return Q_corrected

    @staticmethod
    def retraction_cayley(U, V, dt=1.0):
        #ill do soon
        pass 


    @staticmethod
    def get_geodynamic_field(U, A, G):
        Omega = StiefelManifold.skew(A) 
        
        vertical = torch.matmul(U, Omega)
        
        UT_G = torch.matmul(U.transpose(-1, -2), G)
        U_UT_G = torch.matmul(U, UT_G)
        horizontal = G - U_UT_G
        
        return vertical + horizontal

    
    @staticmethod
    def parallel_transport_projection(U_start, U_end, Vector):
        return StiefelManifold.project_tangent(U_end, Vector)