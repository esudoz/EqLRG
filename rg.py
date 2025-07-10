from lib import *
from net import *
import sys

netspec = sys.argv[1]
f = float(sys.argv[2])
kappa = float(sys.argv[3])

nn,nN = generate_network_from_string(netspec)

lap = make_lap_sym(nn)
N = lap.shape[0]

ev = evs(lap)

lev0 = np.full(N, N**(-0.5)) # eigenvectors with zero-eigenvalue of a given graph Laplacian is simple
rev0 = lev0.copy()

IS_HYP = N!=nN          # determine whether a given graph is a hypergraph or a pairwise network
tauarr = 10**np.arange(-3,9,0.01)

res = []
l = 0
t_r = 1.
t_ratio = 1.
single_node_at_next = False
clus = None
prev_eq_flow = None
psize = {n:1 for n in range(N)}
eq_flow = contruct_eq_flow(lap,lev0,rev0,prev_eq_flow,clus)
deg = np.diag(eq_flow)*len(lap)
N = len(lap)

for l in range(101):
    if (IS_HYP and N<=10) or (not IS_HYP and N<=10): # stop LRG, if size become too small
        break
    
    dia_deg = np.diag(lap)
    t_r /= t_ratio
    Z, s = cal_Z_S(tauarr,ev)
    np.savetxt(f'ZS/{netspec}_f{f}k{kappa}l{l}.d', np.array([tauarr*t_r]+[Z]+[s]).T, delimiter='\t')
    
    if IS_HYP:
        N_next = max(int((1-f)*N),2)
    else:
        N_next  = max(int((1-f)*N),1)
    if N_next==N:
        N_next = N-1

    EQ_N = (1.-f*kappa)*N
    s_c = np.log(EQ_N)/np.log(N)
    tau_ret = estimate_x_for_y_threshold(tauarr, s, s_c)
    tau = float(tau_ret)
    clist,elist = find_clusters(lap,nN,tau,N_next)

    if IS_HYP:
        res.append((l,N,nN,N-nN,np.mean(deg),np.mean(deg[:nN]),np.mean(deg[nN:]),tau*t_r))
        np.savetxt(f'deg_e/{netspec}_f{f}k{kappa}l{l}.d', np.round(deg[nN:],decimals=12), delimiter='\t')
    else:        
        res.append((l,N,np.mean(deg),tau*t_r))
    np.savetxt(f'deg_n/{netspec}_f{f}k{kappa}l{l}.d', np.round(deg[:nN],decimals=12), delimiter='\t')

    if IS_HYP:
        lap,nN,lev0,rev0,t_ratio,ev,clus = Lprime_hyp(clist,lap,nN,ev[1],lev0,rev0)
        N = len(lap)
    else:
        lap,lev0,rev0,t_ratio,ev,clus = Lprime_pair(clist,lap,ev[1],lev0,rev0)
        N,nN = len(lap),len(lap)

    i_to_ip = dict()
    nsize = dict()
    for ci in clus:
        nsize[ci] = 0
        for i in clus[ci]:
            i_to_ip[i] = ci
            nsize[ci] += psize[i]

    eq_flow = contruct_eq_flow(lap,lev0,rev0,eq_flow,clus)
    deg = np.diag(eq_flow)*len(lap)

    size_n = np.array([[nsize[k],deg[k]] for k in clus if k<nN])
    np.savetxt(f'size_n/{netspec}_f{f}k{kappa}l{l}.d', size_n, delimiter='\t')
    if IS_HYP:
        size_e = np.array([[nsize[k],deg[k]] for k in clus if k>=nN])
        np.savetxt(f'size_e/{netspec}_f{f}k{kappa}l{l}.d', size_e, delimiter='\t')
                
    psize = nsize

np.savetxt(f'res/{netspec}_f{f}k{kappa}.d', np.round(np.array(res),decimals=12), delimiter='\t')
