import numpy as np
import numpy.random as rd
from scipy.linalg import eigvals,eigh,expm,eig
import sys
from net import findroot
import networkx as nx
from scipy.interpolate import interp1d

"""Construct new equlibrium flow"""
def contruct_eq_flow(lap,lev0,rev0,prev_eq_flow=None,clus=None):
    
    if prev_eq_flow is None:
        f = np.diag(lev0) @ lap @ np.diag(rev0)
    else:
        f = np.zeros((len(lap),len(lap)))
        for i in range(len(lap)):
            for j in range(len(lap)):
                f[i,j] = np.sum([prev_eq_flow[n1,n2] for n1 in clus[i] for n2 in clus[j]])
            f[i,i] -= np.sum([prev_eq_flow[n1,n2] for n1 in clus[i] for n2 in clus[i] if n1!=n2])
    return f

"""Compute eigenvalues (and optionally eigenvectors) of Laplacian."""
def evs(lap,with_ev=False):
    if with_ev:
        ev, rev = eigh(lap)
        sort_index = np.argsort(ev)
        return ev[sort_index].real,rev[:,sort_index].real
    else:
        ev = eigvals(lap).real
        sort_index = np.argsort(ev)
        return ev[sort_index]

"""Cluster nodes in a hypergraph by analyzing the diffusion kernel."""
def find_clusters(lap,nN,tau,N_next):
    rho = expm(-tau*lap)
    N = len(lap)

    selected_pairs = []
    pairs = []
    values = []
    for n1 in range(N):
        eN = nN if n1<nN else N
        for n2 in range(n1+1,eN):
            pairs.append((n1,n2))
            values.append(rho[n1,n2]*rho[n2,n1]/rho[n1,n1]/rho[n2,n2])
    idx = np.argsort(-np.array(values))
    pair_sorted = np.array(pairs)[idx]

    root = [-1]*len(lap)
    NC = len(lap)
    idd = 0
    while NC!=N_next:
        n1,n2 = pair_sorted[idd]
        n1r = findroot(root, n1)
        n2r = findroot(root, n2)
        if n1r!=n2r:
            if n1r<n2r:
                root[n1r] += root[n2r]
                root[n2r] = n1r
            else:
                root[n2r] += root[n1r]
                root[n1r] = n2r
            NC -= 1
        idd += 1
    return [list(component) for component in nx.connected_components(nx.Graph(list(pair_sorted[:idd])))],pair_sorted[:idd]

def group_vec(c,N,lev0,rev0):
    lv,rv = np.zeros(N),np.zeros(N)
    CLCR = 0
    for i in c:
        lv[i] = lev0[i]
        rv[i] = rev0[i]
        CLCR += lev0[i]*rev0[i]
    return lv,rv,CLCR


"""Reorganize vertex clusters from a given connection list."""
def reorganize_cluster_pair(nN,clist):
    flattened = sum(clist, [])
    nc = [[n] for n in range(nN) if n not in flattened]

    for c in clist:
        nc.append(list(c))
    new_nN = len(nc)
    return nc,new_nN


"""Compute renormalized Laplacian for pairwise graph."""
def Lprime_pair(clist,lap,eval_2,lev0,rev0):
    tlst = []
    nN = len(lap)
    nc,new_nN = reorganize_cluster_pair(nN,clist)
    
    cl_def = lev0[0]
    lv = dict()
    rv = dict()
    CLCR = dict()
    CR = dict()
    CL = dict()

    is_single = [True] * new_nN
    for i,c in enumerate(nc):
        if len(c)!=1:
            lv[i],rv[i],CLCR[i] = group_vec(c,nN,lev0,rev0)

            is_single[i] = False
    for i,c in enumerate(nc):
        if len(c)!=1:
            CR[i] = CLCR[i]**0.5
            CL[i] = CR[i]

    newlap = np.zeros((new_nN,new_nN))

    for v,cv in enumerate(nc):
        for e,ce in enumerate(nc):
            if v>e:
                if is_single[v] and is_single[e]:
                    newlap[v,e] = lap[cv[0],ce[0]]
                    newlap[e,v] = lap[ce[0],cv[0]]
                elif is_single[v]==False and is_single[e]:
                    newlap[v,e] = np.sum(lap[cv[i],ce[0]] * lv[v][cv[i]] for i in range(len(cv))) / CL[v]
                    newlap[e,v] = np.sum(lap[ce[0],cv[i]] * rv[v][cv[i]] for i in range(len(cv))) / CR[v]
                elif is_single[e]==False and is_single[v]:
                    newlap[v,e] = np.sum(lap[cv[0],ce[i]] * rv[e][ce[i]] for i in range(len(ce))) / CR[e]
                    newlap[e,v] = np.sum(lap[ce[i],cv[0]] * lv[e][ce[i]] for i in range(len(ce))) / CL[e]
                elif is_single[e]==False and is_single[v]==False:
                    newlap[v,e] = np.sum(lap[cv[j],ce[i]] * rv[e][ce[i]] * lv[v][cv[j]] for i in range(len(ce)) for j in range(len(cv))) / CR[e] / CL[v]
                    newlap[e,v] = np.sum(lap[ce[i],cv[j]] * lv[e][ce[i]] * rv[v][cv[j]] for i in range(len(ce)) for j in range(len(cv))) / CL[e] / CR[v]
                else:
                    print('error')
                    sys.exit()
        if is_single[v]:
            newlap[v,v] = lap[cv[0],cv[0]]
        else:
            newlap[v,v] = np.sum(lap[cv[i],cv[j]] * lv[v][cv[i]] * rv[v][cv[j]] for i in range(len(cv)) for j in range(len(cv))) / CL[v] / CR[v]


    clus = dict()
    new_lev0,new_rev0 = np.zeros(new_nN),np.zeros(new_nN)
    for v,cv in enumerate(nc):
        if is_single[v]:
            new_lev0[v] = lev0[cv[0]]
            new_rev0[v] = rev0[cv[0]]
        else:
            new_lev0[v] = CL[v]
            new_rev0[v] = CR[v]
        clus[v] = cv

    nev = evs(newlap)
    
    return newlap,new_lev0,new_rev0,eval_2/nev[1],nev,clus

"""Reorganize vertex clusters and hyperedge clusters from a given connection list."""
def reorganize_cluster_hyp(nN,nE,clist):
    flattened = sum(clist, [])
    new_nN = nN
    nc = [[n] for n in range(nN) if n not in flattened]
    ec = [[n] for n in range(nN,nN+nE) if n not in flattened]
    new_nN = len(nc)

    for c in clist:
        if c[0]<nN:
            nc.append(list(c))
        else:
            ec.append(list(c))
    new_nN = len(nc)
    return nc,ec,new_nN

"""Compute renormalized Laplacian for hypergraph."""
def Lprime_hyp(clist,lap,nN,eval_2,lev0,rev0):
    tlst = []
    N = len(lap)
    nc,ec,new_nN = reorganize_cluster_hyp(nN,N-nN,clist)

    new_nE = len(ec)
    new_N = new_nN + new_nE
    
    cl_def = lev0[0]
    lv = dict()
    rv = dict()
    CLCR = dict()
    CR = dict()
    CL = dict()

    is_single = [True] * new_N
    for i,c in enumerate(nc+ec):
        if len(c)!=1:
            lv[i],rv[i],CLCR[i] = group_vec(c,N,lev0,rev0)

            is_single[i] = False
    for i,c in enumerate(nc+ec):
        if len(c)!=1:
            CR[i] = CLCR[i]**0.5
            CL[i] = CR[i]

    newlap = np.zeros((new_N,new_N))

    for v,cv in enumerate(nc):
        for te,ce in enumerate(ec):
            e = te + new_nN
            if is_single[v] and is_single[e]:
                newlap[v,e] = lap[cv[0],ce[0]]
                newlap[e,v] = lap[ce[0],cv[0]]
            elif is_single[v]==False and is_single[e]:
                newlap[v,e] = np.sum(lap[cv[i],ce[0]] * lv[v][cv[i]] for i in range(len(cv))) / CL[v]
                newlap[e,v] = np.sum(lap[ce[0],cv[i]] * rv[v][cv[i]] for i in range(len(cv))) / CR[v]
            elif is_single[e]==False and is_single[v]:
                newlap[v,e] = np.sum(lap[cv[0],ce[i]] * rv[e][ce[i]] for i in range(len(ce))) / CR[e]
                newlap[e,v] = np.sum(lap[ce[i],cv[0]] * lv[e][ce[i]] for i in range(len(ce))) / CL[e]
            elif is_single[e]==False and is_single[v]==False: #?????????????????
                newlap[v,e] = np.sum(lap[cv[j],ce[i]] * rv[e][ce[i]] * lv[v][cv[j]] for i in range(len(ce)) for j in range(len(cv))) / CR[e] / CL[v]
                newlap[e,v] = np.sum(lap[ce[i],cv[j]] * lv[e][ce[i]] * rv[v][cv[j]] for i in range(len(ce)) for j in range(len(cv))) / CL[e] / CR[v]
            else:
                print('error')
                sys.exit()

            if newlap[v,e] is np.nan or newlap[e,v] is np.nan:
                print("nan occurs!!!!!!!!!!!")
                sys.exit()

        if is_single[v]:
            newlap[v,v] = lap[cv[0],cv[0]]
        else:
            newlap[v,v] = np.sum(lap[cv[i],cv[i]] * lv[v][cv[i]] * rv[v][cv[i]] for i in range(len(cv))) / CL[v] / CR[v]

    for te,ce in enumerate(ec):
        e = te + new_nN
        if is_single[e]:
            newlap[e,e] = lap[ce[0],ce[0]]
        else:
            newlap[e,e] = np.sum(lap[ce[i],ce[i]] * lv[e][ce[i]] * rv[e][ce[i]] for i in range(len(ce))) / CL[e] / CR[e]


    new_lev0,new_rev0 = np.zeros(new_N),np.zeros(new_N)
    clus = dict()
    for v,cv in enumerate(nc):
        if is_single[v]:
            new_lev0[v] = lev0[cv[0]]
            new_rev0[v] = rev0[cv[0]]
        else:
            new_lev0[v] = CL[v]
            new_rev0[v] = CR[v]
        clus[v] = cv
    for te,ce in enumerate(ec):
        e = te + new_nN
        if is_single[e]:
            new_lev0[e] = lev0[ce[0]]
            new_rev0[e] = rev0[ce[0]]
        else:
            new_lev0[e] = CL[e]
            new_rev0[e] = CR[e]
        clus[e] = ce

    nev = evs(newlap)
        
    return newlap,new_nN,new_lev0,new_rev0,eval_2/nev[1],nev,clus
    


"""Calculate partition function Z through eigenvalues."""
def cal_Zt(tauarr,eval):
    npexp = np.exp(-eval)
    ret = np.array([np.sum(npexp**tau) for tau in tauarr])
    return ret

"""Calculate partition function Z and normalized entropy s through eigenvalues."""
def cal_Z_S(tauarr,ev):
    exptau = np.zeros(len(tauarr))
    Z = cal_Zt(tauarr,ev)
    for i,t in enumerate(tauarr):
        exptau[i] = t*np.sum(np.exp(-t*ev)*ev)
        exptau[i] = exptau[i] if exptau[i]>1.e-12 else 0
    S = (np.log(Z)+exptau/Z)/np.log(len(ev))
    return Z, S


"""Estimate x where y crosses a threshold, using interpolation."""
def estimate_x_for_y_threshold(x, y, y_c):
    if np.min(y) > y_c or np.max(y) < y_c:
        return None  # x_c is out of range, cannot estimate x
    
    interp_func = interp1d(y, x, kind='linear', bounds_error=False, fill_value="extrapolate")
    
    estimated_x = interp_func(y_c)
    return estimated_x

def make_lap_sym(nn):
    N = len(nn)
    lap = np.full((N,N),0.)
    for n1 in nn:
        for n2 in nn[n1]:
            lap[n1,n2] = -1.
    for j in range(N):
        lap[j,j] = -np.sum(lap[:,j])
    return lap