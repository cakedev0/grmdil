import numpy as np
from pathlib import Path
from cloudpoints import Neighborhoods
import pandas as pd
from ply import write_ply

label_names = {0: 'Unclassified', 1: 'Ground', 2: 'Building', 3: 'Poles',
               4: 'Pedestrians', 5: 'Cars', 6: 'Vegetation'}

class CloudDataset:
    
    def __init__(self, name, points, labels=None, data_path=Path("3d_data/")):
        self.name = name
        self.points = points
        self.labels = labels
        self.data_path = data_path
    
    def compute_neighborhoods(self, params):
        k0 = params[0][0]
        nb = Neighborhoods(self.points, k=k0)
        self.neighborhoods_params = params
        self.neighborhoods = []
        for k, radius in params:
            nb = nb.restrict(k=k, radius=radius, inplace=False)
            self.neighborhoods.append(nb)
        for nb in self.neighborhoods:
            nb.compute_local_PCAs()
    
    def get_edges(self, grm_k=8, grm_radius=0.08):
        smallest_nb = None
        for (k, radius), nb in zip(self.neighborhoods_params, self.neighborhoods):
            if(k >= grm_k and radius >= grm_radius):
                smallest_nb = nb
        return nb.restrict(k=grm_k, radius=grm_radius).get_edges()
    
    def compute_features(self, feature_functions):
        features = []
        for nb in self.neighborhoods:
            features.append(nb.get_features(feature_functions))
        self.features = np.concatenate(features, axis=1)
       
    @property
    def X(self):
        if(self.labels is None):
            return self.features
        else:
            return self.features[self.labels > 0, :]
    
    @property
    def y(self):
        if(self.labels is None):
            return -np.ones(self.points.shape[0])
        else:
            return self.labels[self.labels > 0] - 1
        
    def unlabel(self):
        if(self.labels is not None):
            self.mem_labels = self.labels
            self.labels = None
        
    def relabel(self):
        if(self.labels is None):
            self.labels = self.mem_labels
        
    def __repr__(self):
        labeled = "unlabeled" if self.labels is None else "labeled"
        return f"CloudDataset<{self.name}, {self.points.shape[0]} points, {labeled}>"
    
    def write_GRM(self, clf_scores, path, grm_k=8, grm_radius=0.05):
        edges, distances = self.get_edges(grm_k, grm_radius)
        n, m = self.points.shape[0], distances.size
        with open(path / f"{self.name}_GRM.in", "w") as f:
            f.write(f"{n} {m}\n")
            for scores in clf_scores:
                f.write(" ".join([str(s) for s in scores]) + "\n")
            for edge, d in zip(edges, distances):
                u, v = edge
                f.write(f"{u} {v} {d}\n")
        
        labels = self.mem_labels if self.labels is None else self.labels
        with open(path / f"{self.name}_labels.txt", "w") as f:
            f.write("\n".join([str(l) for l in labels]))
            
    def write_ply_with_labels(self, fname, labels=None):
        if(labels is None):
            labels = self.mem_labels if self.labels is None else self.labels
        write_ply(fname, [self.points, labels], ['x', 'y', 'z', 'class'])
    
class MultiDataset:
    
    def __init__(self, datasets, test_idx=1):
        self.datasets = datasets
        self.test_idx = test_idx
        
    def train_test_split(self):
        self.testset.unlabel()
        ys, Xs = [], []
        for dataset in self.trainsets:
            dataset.relabel()
            Xs.append(dataset.X)
            ys.append(dataset.y)
        
        X_train = np.concatenate(Xs, axis=0)
        y_train = np.concatenate(ys)
        return X_train, self.testset.X, y_train, self.testset.mem_labels
    
    @property
    def testset(self):
        return self.datasets[self.test_idx]
    
    @property
    def trainsets(self):
        datasets = self.datasets[:]
        datasets.pop(self.test_idx)
        return datasets
    
def df_results(labels_test, labels_pred):
    precisions = []
    recalls = []
    IoUs = []
    pd_index = []
    for c in range(1, 7):
        if(np.sum(labels_test == c) > 0):
            precisions.append(np.mean(labels_test[labels_pred == c] == c))
            recalls.append(np.mean(labels_pred[labels_test == c] == c))
            IoUs.append(np.sum((labels_pred == c) & (labels_test == c)) / \
                        np.sum(((labels_pred == c) & (labels_test != 0)) | (labels_test == c)))
            pd_index.append(f"{c} - {label_names[c]}")
    return pd.DataFrame({"precision": precisions, "recall": recalls, "IoU": IoUs}, index=pd_index)