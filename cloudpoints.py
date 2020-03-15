import numpy as np
from ply import read_ply, write_ply
from sklearn.neighbors import KDTree
from numba import njit

def load_points(fname):
    cloud_ply = read_ply(fname)
    points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
    if('class' in cloud_ply.dtype.fields):
        labels = cloud_ply['class']
        return points, labels
    else:
        return points
    
def drop_duplicated_rows(data):
    # Perform lex sort and get sorted data
    sorted_idx = np.lexsort(data.T)
    sorted_data =  data[sorted_idx,:]
    
    # Get unique row mask
    row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))

    # Get unique rows
    return sorted_data[row_mask]

@njit
def local_PCA(points):
    delta_p = points - np.array([points[:, 0].mean(), points[:, 1].mean(), points[:, 2].mean()])
    cov = (delta_p.T @ delta_p) / points.shape[0]
    eigvals, eigvects = np.linalg.eigh(cov)
    return eigvals[::-1], eigvects[:, ::-1]

class Neighborhoods:
    
    def __init__(self, points, neigh_idxs=None, distances=None, k=None):
        self.points = points
        self.n = points.shape[0]
        if(k is None):
            self.neigh_idxs = neigh_idxs
            self.distances = distances
        else:
            kdt = KDTree(points)
            self.distances, self.neigh_idxs = kdt.query(points, k=k)
    
    def restrict(self, k=10000, radius=1e9, min_k=0, inplace=False):
        whs = [ds < radius for ds in self.distances]
        for wh in whs:
            wh[k:] = False
            wh[:min_k] = True
        res_neigh_idxs = []
        res_distances = []
        for wh, ns, ds in zip(whs, self.neigh_idxs, self.distances):
            res_neigh_idxs.append(ns[wh])
            res_distances.append(ds[wh])
        if(inplace):
            self.neigh_idxs = res_neigh_idxs
            self.distances = res_distances
            return self
        else:
            return Neighborhoods(self.points, res_neigh_idxs, res_distances)
    
    def compute_local_PCAs(self):
        self.eigenvalues = np.zeros((self.n, 3))
        self.eigenvectors = np.zeros((self.n, 3, 3))
        for i, ns in enumerate(self.neigh_idxs):
            neighs = self.points[ns, :]
            eigvals, eigvects = local_PCA(neighs)
            self.eigenvalues[i, :] = eigvals
            self.eigenvectors[i, :] = eigvects
            
    def get_features(self, feature_functions):
        features = []
        neighs = [self.points[ns, :] for ns in self.neigh_idxs]
        for func in feature_functions:
            f = func(neighs, self.eigenvalues, self.eigenvectors)
            if(f.ndim == 1):
                f = f.reshape(-1, 1)
            features.append(f)
        return np.concatenate(features, axis=1)
    
    def get_edges(self):
        m = np.sum([ns.size for ns in self.neigh_idxs])
        edges = np.zeros((m, 2), dtype=int)
        j = 0
        for i, ns in enumerate(self.neigh_idxs):
            edges[j:j + ns.size, 0] = i
            edges[j:j + ns.size, 1] = ns
            j += ns.size
        edges = edges[:j, :]
        edges = edges[edges[:, 0] != edges[:, 1], :]
        edges = drop_duplicated_rows(edges)
        distances = np.linalg.norm(self.points[edges[:, 0], :] - self.points[edges[:, 1], :], axis=1)
        return edges, distances



