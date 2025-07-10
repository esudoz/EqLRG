import re
import networkx as nx
import numpy.random as rd
import numpy as np
import heapq

def is_nei(nn,n1,n2):
    for nei1 in nn[n1]:
        if nei1==n2:
            return True
    return False

def findroot(root,n):
    if root[n]<0:
        return n
    else: 
        return findroot(root,root[n])

"""Parse a network specification string and generate the corresponding graph or hypergraph.
usages         
nn = generate_network_from_string('BA_m1_N1000') to make pairwise BA network with m=1 
nn = generate_network_from_string('RN_N1000') to make pairwise random tree from Prufer sequence 
nn = generate_network_from_string('ER_k2_N1000') to make pairwise ER network with <k>=3 
nn = generate_network_from_string('ST_k3_g2.5_N1000') to make pairwise Static network with <k>=3 and P(k) = A k^{-2.5} 

nn = generate_network_from_string('HBA_m1_N1000') to make pairwise BA hypergraph with m=1 
nn = generate_network_from_string('HRN_N1000') to make pairwise random hypergraph from Prufer sequence 
nn = generate_network_from_string('HER_k2_N1000') to make pairwise ER hypergraph with <k>=3 
nn = generate_network_from_string('HST_k3_g2.5_N1000') to make pairwise Static hypergraph with <k>=3 and P(k) = A k^{-2.5} and P(c) is Poissonian 
nn = generate_network_from_string('HST_k3_g2.5_g3_N1000') to make pairwise Static hypergraph with <k>=3 and P(k) = A k^{-2.5} and P(c) = B c^{-3}  

nn = generate_network_from_string('BABI_m1_N1000') to make hypergraph by mapping nodes as vertices and edges as hyperedges for pairwise BA network with m=1 
nn = generate_network_from_string('RNBI_N1000') to make hypergraph by mapping nodes as vertices and edges as hyperedges for pairwise random tree from Prufer sequence 
nn = generate_network_from_string('ERBI_k2_N1000') to make hypergraph by mapping nodes as vertices and edges as hyperedges for pairwise ER network with <k>=3 
nn = generate_network_from_string('STBI_k3_g2.5_N1000') to make hypergraph by mapping nodes as vertices and edges as hyperedges for pairwise Static network with <k>=3 and P(k) = A k^{-2.5} 
"""

def generate_network_from_string(config_str):
    print(config_str)
    if match := re.match(r'^(BA|BABI)_m(\d+)_N(\d+)I(\d+)$', config_str):
        prefix, m, N, seed = match.groups()
        m, N, seed = int(m), int(N), int(seed)
        if prefix == "BA":
            return ba_net(N, m, seed)
        else:
            return ba_bi_net(N, m, seed)
    
    elif match := re.match(r'^(ST|STBI)_k([0-9.]+)_g([0-9.]+)_N(\d+)I(\d+)$', config_str):
        prefix, k, g, N, seed = match.groups()
        k, g, N, seed = float(k), float(g), int(N), int(seed)
        if prefix == "ST":
            return st_net(N, k, g, seed)
        else:
            return st_bi_net(N, k, g, seed)
    
    elif match := re.match(r'^(ER|ERBI)_k([0-9.]+)_N(\d+)I(\d+)$', config_str):
        prefix, k, N, seed = match.groups()
        k, N, seed = float(k), int(N), int(seed)
        if prefix == "ER":
            return er_net(N, k, seed)
        else:
            return er_bi_net(N, k, seed)

    elif match := re.match(r'^HST_k([0-9.]+)_g([0-9.]+)_N(\d+)I(\d+)$', config_str):
        k, g, N, seed = match.groups()
        return st_hyp(int(N), float(k), float(g), int(seed))
    
    elif match := re.match(r'^HST_k([0-9.]+)_g([0-9.]+)_g([0-9.]+)_N(\d+)I(\d+)$', config_str):
        k, g1, g2, N, seed = match.groups()
        return st_hyp_both(int(N), float(k), float(g1), float(g2), int(seed))
    
    elif match := re.match(r'^HBA_m(\d+)_N(\d+)I(\d+)$', config_str):
        m, N, seed = map(int, match.groups())
        return ba_hyp(N, m, seed)
    
    elif match := re.match(r'^HER_k([0-9.]+)_N(\d+)I(\d+)$', config_str):
        k, N, seed = match.groups()
        return er_hyp(int(N), float(k), int(seed))

    elif match := re.match(r'^HRN_N(\d+)I(\d+)$', config_str):
        N, seed = match.groups()
        return Prufer_random_tree_hyp(int(N), int(seed))

    elif match := re.match(r'^(RN|RNBI)_N(\d+)I(\d+)$', config_str):
        prefix, N, seed = match.groups()
        if prefix == "RN":
            return Prufer_random_tree(int(N), int(seed))
        else:
            return Prufer_random_tree_bi(int(N), int(seed))

    else:
        raise ValueError(f"Unrecognized configuration string: {config_str}")


def Prufer_random_tree(nN, seed=None):
    if seed is not None:
        rd.seed(seed)
    n = nN
    if n < 2:
        raise ValueError("Tree must have at least 2 nodes.")
    
    # Step 1: Generate a random Prufer sequence
    prufer = rd.randint(1, n + 1, size=n-2)
    degree = np.ones(n + 1, dtype=int)

    for node in prufer:
        degree[node] += 1
    
    # Step 2: Construct the tree using a priority queue (min-heap)
    min_heap = [i for i in range(1, n + 1) if degree[i] == 1]
    heapq.heapify(min_heap)
    
    edges = []
    for node in prufer:
        leaf = heapq.heappop(min_heap)
        edges.append((leaf, node))
        
        degree[node] -= 1
        degree[leaf] -= 1
        
        if degree[node] == 1:
            heapq.heappush(min_heap, node)
    
    u, v = heapq.heappop(min_heap), heapq.heappop(min_heap)
    edges.append((u, v))
    
    # Step 3: Construct adjacency dictionary
    nn = {i: [] for i in range(n)}
    for u, v in edges:
        nn[u-1].append(v-1)
        nn[v-1].append(u-1)

    return nn, nN


def Prufer_random_tree_bi(nN, seed=None):
    if seed is not None:
        rd.seed(seed)
    n = nN
    if n < 2:
        raise ValueError("Tree must have at least 2 nodes.")
    
    # Step 1: Generate a random Prufer sequence
    prufer = rd.randint(1, n + 1, size=n-2)
    degree = np.ones(n + 1, dtype=int)

    for node in prufer:
        degree[node] += 1
    
    # Step 2: Construct the tree using a priority queue (min-heap)
    min_heap = [i for i in range(1, n + 1) if degree[i] == 1]
    heapq.heapify(min_heap)
    
    edges = []
    for node in prufer:
        leaf = heapq.heappop(min_heap)
        edges.append((leaf, node))
        
        degree[node] -= 1
        degree[leaf] -= 1
        
        if degree[node] == 1:
            heapq.heappush(min_heap, node)
    
    u, v = heapq.heappop(min_heap), heapq.heappop(min_heap)
    edges.append((u, v))
    
    # Step 3: Construct adjacency dictionary in bipartite form

    nE = len(edges)
    nn = {n:[] for n in range(nN+nE)}
    for re,l in enumerate(edges):
        n1 = l[0]-1
        n2 = l[1]-1
        e = re+nN
        nn[e].append(n1)
        nn[n1].append(e)
        nn[e].append(n2)
        nn[n2].append(e)


    return nn, nN

def Prufer_random_tree_hyp(nN, seed=None):
    if seed is not None:
        rd.seed(seed)

    n = nN * 2
    if n < 2:
        raise ValueError("Tree must have at least 2 nodes.")
    
    # Step 1: Generate a random Prufer sequence
    prufer = rd.randint(1, n + 1, size=n-2)
    degree = np.ones(n + 1, dtype=int)

    for node in prufer:
        degree[node] += 1
    
    # Step 2: Construct the tree using a priority queue (min-heap)
    min_heap = [i for i in range(1, n + 1) if degree[i] == 1]
    heapq.heapify(min_heap)
    
    edges = []
    for node in prufer:
        leaf = heapq.heappop(min_heap)
        edges.append((leaf, node))
        
        degree[node] -= 1
        degree[leaf] -= 1
        
        if degree[node] == 1:
            heapq.heappush(min_heap, node)
    
    u, v = heapq.heappop(min_heap), heapq.heappop(min_heap)
    edges.append((u, v))
    
    # Step 3: Construct adjacency dictionary
    nn = {i: [] for i in range(1, n+1)}
    for u, v in edges:
        nn[u].append(v)
        nn[v].append(u)
    
    # Step 4: Bipartite coloring
    colors = {}  # 0 for one set, 1 for another
    queue = [1]
    colors[1] = 0
    
    while queue:
        node = queue.pop()
        for neighbor in nn[node]:
            if neighbor not in colors:
                colors[neighbor] = 1 - colors[node]
                queue.append(neighbor)
    
    # Step 5: Define vertices and hyperedges
    vertices = sorted({k for k, v in colors.items() if v == 0})
    hyperedges = sorted({k for k, v in colors.items() if v == 1})
    
    # Step 6: Reindexing
    vertex_map = {v: i for i, v in enumerate(vertices)}
    hyperedge_map = {h: i + len(vertices) for i, h in enumerate(hyperedges)}
    
    # Step 7: Construct full hypergraph adjacency dictionary
    hyper_nn = {**{vertex_map[v]: [] for v in vertices}, **{hyperedge_map[h]: [] for h in hyperedges}}
    for edge in edges:
        u, v = edge
        if u in vertices and v in hyperedges:
            hyper_nn[vertex_map[u]].append(hyperedge_map[v])
            hyper_nn[hyperedge_map[v]].append(vertex_map[u])
        elif v in vertices and u in hyperedges:
            hyper_nn[vertex_map[v]].append(hyperedge_map[u])
            hyper_nn[hyperedge_map[u]].append(vertex_map[v])
    
    nN = len(vertices)
    
    return hyper_nn, nN

def ba_net(nN, m, seed=None):
    links = list(nx.barabasi_albert_graph(nN, m, seed=seed).edges)
    nn = {n:[] for n in range(nN)}
    for n1,n2 in links:
        nn[n1].append(n2)
        nn[n2].append(n1)
    return nn,nN

def ba_bi_net(nN, m, seed=None):
    links = list(nx.barabasi_albert_graph(nN, m, seed=seed).edges)
    nE = len(links)
    nn = {n:[] for n in range(nN+nE)}
    for re,l in enumerate(links):
        e = re+nN
        nn[e].append(l[0])
        nn[l[0]].append(e)
        nn[e].append(l[1])
        nn[l[1]].append(e)
    return nn,nN

def st_net(nN, k, g, seed=None):
    if seed is not None:
        rd.seed(seed)
    nE = int(k*nN/2)

    id = np.arange(1., nN + 1) 
    p = id**(-1./(g-1))
    sump = np.sum(p)
    p /= sump

    nn = {n:[] for n in range(nN)}
    root = [-1 for n in range(nN)]
    size,sizeroot,maxsize = 1,0,1
    for _ in range(nE):
        while True:
            n1, n2 = rd.choice(nN, size=2, replace=False, p=p)
            if not is_nei(nn,n1,n2):
                break
        nn[n1].append(n2)
        nn[n2].append(n1)

        n1r = findroot(root, n1)
        n2r = findroot(root, n2)
        if n1r!=n2r:
            if n1r<n2r:
                root[n1r] += root[n2r]
                root[n2r] = n1r
                size = -root[n1r]
                sizeroot = n1r
            else:
                root[n2r] += root[n1r]
                root[n1r] = n2r
                size = -root[n2r]
                sizeroot = n2r
            if size>maxsize:
                maxsize = size
                maxsizeroot = sizeroot

    nid = [-1]*nN
    newnN = 0
    for n in range(nN):
        r = findroot(root, n)
        if r==maxsizeroot:
            nid[n] = newnN
            newnN += 1
            
    newnn = {n:[] for n in range(newnN)}
    for n1 in range(nN):
        if nid[n1]!=-1:
            for n2 in nn[n1]:
                newnn[nid[n1]].append(nid[n2])
                newnn[nid[n2]].append(nid[n1])
    return newnn,newnN

def st_bi_net(nN, k, g, seed=None):
    if seed is not None:
        rd.seed(seed)
    nE = int(k*nN/2)
    N = nN+nE

    id = np.arange(1., nN + 1) 
    p = id**(-1./(g-1))
    sump = np.sum(p)
    p /= sump

    nn = {n:[] for n in range(N)}
    node_nn = {n:[] for n in range(nN)}
    root = [-1 for n in range(N)]
    size,sizeroot,maxsize = 1,0,1
    for re in range(nE):
        while True:
            n1, n2 = rd.choice(nN, size=2, replace=False, p=p)
            if not is_nei(node_nn,n1,n2):
                break
        e = re+nN
        node_nn[n1].append(n2)
        node_nn[n2].append(n1)
        nn[n1].append(e)
        nn[n2].append(e)
        nn[e].append(n1)
        nn[e].append(n2)

        n1r = findroot(root, n1)
        n2r = findroot(root, e)
        if n1r!=n2r:
            if n1r<n2r:
                root[n1r] += root[n2r]
                root[n2r] = n1r
                size = -root[n1r]
                sizeroot = n1r
            else:
                root[n2r] += root[n1r]
                root[n1r] = n2r
                size = -root[n2r]
                sizeroot = n2r
            if size>maxsize:
                maxsize = size
                maxsizeroot = sizeroot

        n1r = findroot(root, n2)
        n2r = findroot(root, e)
        if n1r!=n2r:
            if n1r<n2r:
                root[n1r] += root[n2r]
                root[n2r] = n1r
                size = -root[n1r]
                sizeroot = n1r
            else:
                root[n2r] += root[n1r]
                root[n1r] = n2r
                size = -root[n2r]
                sizeroot = n2r
            if size>maxsize:
                maxsize = size
                maxsizeroot = sizeroot

    nid, eid = [-1]*nN, {e:-1 for e in range(nN,N)}
    newnN = 0
    for n in range(nN):
        r = findroot(root, n)
        if r==maxsizeroot:
            nid[n] = newnN
            newnN += 1
    neweN = newnN
    for e in range(nN,N):
        r = findroot(root, e)
        if r==maxsizeroot:
            eid[e] = neweN
            neweN += 1
            
    newnn = {n:[] for n in range(neweN)}
    for n1 in range(nN):
        if nid[n1]!=-1:
            for n2 in nn[n1]:
                newnn[nid[n1]].append(eid[n2])
                newnn[eid[n2]].append(nid[n1])
    return newnn,newnN

def er_net(nN, k, seed=None):
    links = list(nx.erdos_renyi_graph(nN, k/(nN-1), seed=seed).edges)
    nn = {n:[] for n in range(nN)}
    root = [-1 for n in range(nN)]
    size,sizeroot,maxsize = 1,0,1
    for n1,n2 in links:
        nn[n1].append(n2)
        nn[n2].append(n1)
        n1r = findroot(root, n1)
        n2r = findroot(root, n2)
        if n1r!=n2r:
            if n1r<n2r:
                root[n1r] += root[n2r]
                root[n2r] = n1r
                size = -root[n1r]
                sizeroot = n1r
            else:
                root[n2r] += root[n1r]
                root[n1r] = n2r
                size = -root[n2r]
                sizeroot = n2r
            if size>maxsize:
                maxsize = size
                maxsizeroot = sizeroot

    nid = [-1]*nN
    newnN = 0
    for n in range(nN):
        r = findroot(root, n)
        if r==maxsizeroot:
            nid[n] = newnN
            newnN += 1
            
    newnn = {n:[] for n in range(newnN)}
    for n1 in range(nN):
        if nid[n1]!=-1:
            for n2 in nn[n1]:
                newnn[nid[n1]].append(nid[n2])
                newnn[nid[n2]].append(nid[n1])
    return newnn,newnN

def er_bi_net(nN, k, seed=-1):
    links = list(nx.erdos_renyi_graph(nN, k/(nN-1), seed=seed).edges)
    nE = len(links)
    N = nN+nE

    nn = {n:[] for n in range(N)}
    root = [-1 for n in range(N)]
    size,sizeroot,maxsize = 1,0,1
    for re,l in enumerate(links):
        e = re + nN
        nn[e].append(l[0])
        nn[l[0]].append(e)
        nn[e].append(l[1])
        nn[l[1]].append(e)

        n1r = findroot(root, l[0])
        n2r = findroot(root, e)
        if n1r!=n2r:
            if n1r<n2r:
                root[n1r] += root[n2r]
                root[n2r] = n1r
                size = -root[n1r]
                sizeroot = n1r
            else:
                root[n2r] += root[n1r]
                root[n1r] = n2r
                size = -root[n2r]
                sizeroot = n2r
            if size>maxsize:
                maxsize = size
                maxsizeroot = sizeroot

        n1r = findroot(root, l[1])
        n2r = findroot(root, e)
        if n1r!=n2r:
            if n1r<n2r:
                root[n1r] += root[n2r]
                root[n2r] = n1r
                size = -root[n1r]
                sizeroot = n1r
            else:
                root[n2r] += root[n1r]
                root[n1r] = n2r
                size = -root[n2r]
                sizeroot = n2r
            if size>maxsize:
                maxsize = size
                maxsizeroot = sizeroot


    nid, eid = [-1]*nN, {e:-1 for e in range(nN,N)}
    newnN = 0
    for n in range(nN):
        r = findroot(root, n)
        if r==maxsizeroot:
            nid[n] = newnN
            newnN += 1
    neweN = newnN
    for e in range(nN,N):
        r = findroot(root, e)
        if r==maxsizeroot:
            eid[e] = neweN
            neweN += 1
            
    newnn = {n:[] for n in range(neweN)}
    for n1 in range(nN):
        if nid[n1]!=-1:
            for n2 in nn[n1]:
                newnn[nid[n1]].append(eid[n2])
                newnn[eid[n2]].append(nid[n1])
    return newnn,newnN

def st_hyp(nN, k, g, seed=-1):
    eN = nN
    if seed!=-1:
        rd.seed(seed)
    L = int(k*nN)
    N = nN+eN

    id = np.arange(1., nN + 1) 
    p1 = id**(-1./(g-1))
    p1 /= np.sum(p1)

    nn = {n:[] for n in range(N)}
    
    root = [-1 for n in range(N)]
    size,sizeroot,maxsize = 1,0,1
    for _ in range(L):
        while True:
            n = rd.choice(nN, p=p1)
            e = nN + rd.choice(eN)
            if not is_nei(nn,n,e):
                break
        nn[n].append(e)
        nn[e].append(n)
        n1r = findroot(root, n)
        n2r = findroot(root, e)
        if n1r!=n2r:
            if n1r<n2r:
                root[n1r] += root[n2r]
                root[n2r] = n1r
                size = -root[n1r]
                sizeroot = n1r
            else:
                root[n2r] += root[n1r]
                root[n1r] = n2r
                size = -root[n2r]
                sizeroot = n2r
            if size>maxsize:
                maxsize = size
                maxsizeroot = sizeroot


    nid, eid = [-1]*nN, {e:-1 for e in range(nN,N)}
    newnN = 0
    for n in range(nN):
        r = findroot(root, n)
        if r==maxsizeroot:
            nid[n] = newnN
            newnN += 1
    neweN = newnN
    for e in range(nN,N):
        r = findroot(root, e)
        if r==maxsizeroot:
            eid[e] = neweN
            neweN += 1
            
    newnn = {n:[] for n in range(neweN)}
    for n1 in range(nN):
        if nid[n1]!=-1:
            for n2 in nn[n1]:
                newnn[nid[n1]].append(eid[n2])
                newnn[eid[n2]].append(nid[n1])
    return newnn,newnN

def st_hyp_both(nN, k, g1, g2, seed=-1):
    eN = nN
    if seed!=-1:
        rd.seed(seed)
    L = int(k*nN)
    N = nN+eN

    id = np.arange(1., nN + 1) 
    p1 = id**(-1./(g1-1))
    p1 /= np.sum(p1)

    id = np.arange(1., eN + 1) 
    p2 = id**(-1./(g2-1))
    p2 /= np.sum(p2)

    nn = {n:[] for n in range(N)}
    
    root = [-1 for n in range(N)]
    size,sizeroot,maxsize = 1,0,1
    for _ in range(L):
        while True:
            n = rd.choice(nN, p=p1)
            e = nN + rd.choice(eN, p=p2)
            if not is_nei(nn,n,e):
                break
        nn[n].append(e)
        nn[e].append(n)
        n1r = findroot(root, n)
        n2r = findroot(root, e)
        if n1r!=n2r:
            if n1r<n2r:
                root[n1r] += root[n2r]
                root[n2r] = n1r
                size = -root[n1r]
                sizeroot = n1r
            else:
                root[n2r] += root[n1r]
                root[n1r] = n2r
                size = -root[n2r]
                sizeroot = n2r
            if size>maxsize:
                maxsize = size
                maxsizeroot = sizeroot


    nid, eid = [-1]*nN, {e:-1 for e in range(nN,N)}
    newnN = 0
    for n in range(nN):
        r = findroot(root, n)
        if r==maxsizeroot:
            nid[n] = newnN
            newnN += 1
    neweN = newnN
    for e in range(nN,N):
        r = findroot(root, e)
        if r==maxsizeroot:
            eid[e] = neweN
            neweN += 1
            
    newnn = {n:[] for n in range(neweN)}
    for n1 in range(nN):
        if nid[n1]!=-1:
            for n2 in nn[n1]:
                newnn[nid[n1]].append(eid[n2])
                newnn[eid[n2]].append(nid[n1])
    return newnn,newnN

def ba_hyp(nN, m, seed=-1):
    if seed!=-1:
        rd.seed(seed)
    N = nN*2

    node_turn = True
    curN,curE = 1,1
    elist = [[0,nN]]
    npool,epool = [0],[nN]
    nn = {n:[] for n in range(N)}
    while True:
        if node_turn:
            n1 = curN
            for e in range(nN,nN+curE):
                elist.append([n1,e])
                nn[n1].append(e)
                nn[e].append(n1)
                npool.append(n1)
                epool.append(e)
            curN += 1
            node_turn = False
        else:
            e1 = nN+curE
            for n in range(curN):
                elist.append([n,e1])
                nn[n].append(e1)
                nn[e1].append(n)
                npool.append(n)
                epool.append(e1)
            curE += 1
            node_turn = True
        if len(elist)==(curN+curE)*m:
            break
    while curN!=nN or curE!=nN: 
        suc = 0
        if node_turn:
            while suc!=m:
                rn = rd.randint(0,len(epool))
                nei = epool[rn]
                if curN in nn[nei]:
                    continue
                npool.append(curN)
                epool.append(nei)
                elist.append([curN,nei])
                nn[curN].append(nei)
                nn[nei].append(curN)
                suc += 1
            curN += 1
        else:
            while suc!=m:
                rn = rd.randint(0,len(npool))
                nei = npool[rn]
                if curE+nN in nn[nei]:
                    continue
                epool.append(curE+nN)
                npool.append(nei)
                elist.append([nei,curE+nN])
                nn[nei].append(curE+nN)
                nn[curE+nN].append(nei)
                suc += 1
            curE += 1

        node_turn = not node_turn

    return nn,nN 

def er_hyp(nN, k, seed=-1):
    eN = nN
    if seed!=-1:
        rd.seed(seed)
    N = nN+eN    
    L = int(k*nN)

    nn = {n:[] for n in range(N)}

    root = [-1 for n in range(N)]
    size,sizeroot,maxsize = 1,0,1
    for _ in range(L):
        while True:
            n = rd.choice(nN)
            e = nN + rd.choice(eN)
            if not is_nei(nn,n,e):
                break
        nn[n].append(e)
        nn[e].append(n)
        n1r = findroot(root, n)
        n2r = findroot(root, e)
        if n1r!=n2r:
            if n1r<n2r:
                root[n1r] += root[n2r]
                root[n2r] = n1r
                size = -root[n1r]
                sizeroot = n1r
            else:
                root[n2r] += root[n1r]
                root[n1r] = n2r
                size = -root[n2r]
                sizeroot = n2r
            if size>maxsize:
                maxsize = size
                maxsizeroot = sizeroot


    nid, eid = [-1]*nN, {e:-1 for e in range(nN,N)}
    newnN = 0
    for n in range(nN):
        r = findroot(root, n)
        if r==maxsizeroot:
            nid[n] = newnN
            newnN += 1
    neweN = newnN
    for e in range(nN,N):
        r = findroot(root, e)
        if r==maxsizeroot:
            eid[e] = neweN
            neweN += 1

    newnn = {n:[] for n in range(neweN)}
    for n1 in range(nN):
        if nid[n1]!=-1:
            for n2 in nn[n1]:
                newnn[nid[n1]].append(eid[n2])
                newnn[eid[n2]].append(nid[n1])
    return newnn,newnN

