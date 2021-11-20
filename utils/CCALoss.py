import torch

def GCCA_loss(H1, H2):
    r = 1e-4
    eps = 1e-8

    # H1, H2, H3 = H1.t(), H2.t(), H3.t()

    # print(f'H1 shape ( N X feature) : {H1.shape}')

    assert torch.isnan(H1).sum().item() == 0 
    assert torch.isnan(H2).sum().item() == 0
    # assert torch.isnan(H3).sum().item() == 0

    o1 = H1.size(0)  # N
    o2 = H2.size(0)
    m = H1.size(1)   # out_dim

    top_k = 3

    H1bar = H1 - H1.mean(dim=1).repeat(m, 1).view(-1, m)
    H2bar = H2 - H2.mean(dim=1).repeat(m, 1).view(-1, m)
    assert torch.isnan(H1bar).sum().item() == 0
    assert torch.isnan(H2bar).sum().item() == 0

    A1, S1, B1 = H1bar.svd(some=True, compute_uv=True)
    A2, S2, B2 = H2bar.svd(some=True, compute_uv=True)

    A1, A2 = A1[:, :top_k], A2[:, :top_k]

    assert torch.isnan(A1).sum().item() == 0
    assert torch.isnan(A2).sum().item() == 0

    S_thin_1, S_thin_2 = S1[:top_k], S2[:top_k]


    S2_inv_1 = 1. / (torch.mul( S_thin_1, S_thin_1 ) + eps)
    S2_inv_2 = 1. / (torch.mul( S_thin_2, S_thin_2 ) + eps)

    assert torch.isnan(S2_inv_1).sum().item() == 0
    assert torch.isnan(S2_inv_2).sum().item() == 0

    T2_1 = torch.mul( torch.mul( S_thin_1, S2_inv_1 ), S_thin_1 )
    T2_2 = torch.mul( torch.mul( S_thin_2, S2_inv_2 ), S_thin_2 )

    assert torch.isnan(T2_1).sum().item() == 0
    assert torch.isnan(T2_2).sum().item() == 0

    T2_1 = torch.where(T2_1>eps, T2_1, (torch.ones(T2_1.shape)*eps).to(H1.device).float())
    T2_2 = torch.where(T2_2>eps, T2_2, (torch.ones(T2_2.shape)*eps).to(H2.device).float())


    T_1 = torch.diag(torch.sqrt(T2_1))
    T_2 = torch.diag(torch.sqrt(T2_2))

    assert torch.isnan(T_1).sum().item() == 0
    assert torch.isnan(T_2).sum().item() == 0

    T_unnorm_1 = torch.diag( S_thin_1 + eps )
    T_unnorm_2 = torch.diag( S_thin_2 + eps )

    assert torch.isnan(T_unnorm_1).sum().item() == 0
    assert torch.isnan(T_unnorm_2).sum().item() == 0

    AT_1 = torch.mm(A1, T_1)
    AT_2 = torch.mm(A2, T_2)

    M_tilde = torch.cat([AT_1, AT_2], dim=1)

    # print(f'M_tilde shape : {M_tilde.shape}')

    assert torch.isnan(M_tilde).sum().item() == 0

    Q, R = M_tilde.qr()

    assert torch.isnan(R).sum().item() == 0
    assert torch.isnan(Q).sum().item() == 0

    U, lbda, _ = R.svd(some=False, compute_uv=True)

    assert torch.isnan(U).sum().item() == 0
    assert torch.isnan(lbda).sum().item() == 0

    G = Q.mm(U[:,:top_k])
    assert torch.isnan(G).sum().item() == 0


    U = [] # Mapping from views to latent space

    # Get mapping to shared space
    views = [H1, H2]
    F = [o1, o2] # features per view
    for idx, (f, view) in enumerate(zip(F, views)):
        _, R = torch.qr(view)
        Cjj_inv = torch.inverse( (R.T.mm(R) + eps * torch.eye( view.shape[1], device=view.device)) )
        assert torch.isnan(Cjj_inv).sum().item() == 0
        pinv = Cjj_inv.mm( view.T)
            
        U.append(pinv.mm( G ))

    U1, U2  = U[0], U[1]
    _, S, _ = M_tilde.svd(some=True)

    assert torch.isnan(S).sum().item() == 0
    use_all_singular_values = False
    if not use_all_singular_values:
        S = S.topk(top_k)[0]
    corr = torch.sum(S )
    assert torch.isnan(corr).item() == 0
    # loss = 14.1421-corr
    loss = corr
    return loss

class cca_loss():
    def __init__(self, outdim_size, use_all_singular_values, device):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device
        # print(device)

    def loss(self, H1, H2):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()
        # assert torch.isnan(H1).sum().item() == 0
        # assert torch.isnan(H2).sum().item() == 0

        o1 = o2 = H1.size(0)

        m = H1.size(1)
#         print(H1.size())

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        # assert torch.isnan(H1bar).sum().item() == 0
        # assert torch.isnan(H2bar).sum().item() == 0

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=self.device)
        # assert torch.isnan(SigmaHat11).sum().item() == 0
        # assert torch.isnan(SigmaHat12).sum().item() == 0
        # assert torch.isnan(SigmaHat22).sum().item() == 0

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)
        # assert torch.isnan(D1).sum().item() == 0
        # assert torch.isnan(D2).sum().item() == 0
        # assert torch.isnan(V1).sum().item() == 0
        # assert torch.isnan(V2).sum().item() == 0

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
        # print(posInd1.size())
        # print(posInd2.size())

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)
#         print(Tval.size())

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
            # assert torch.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(self.device)) # regularization for more stability
            U, V = torch.symeig(trace_TT, eigenvectors=True)
            U = torch.where(U>eps, U, (torch.ones(U.shape).float()*eps).to(self.device))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
            # corr.requires_grad = True
            
        return -corr