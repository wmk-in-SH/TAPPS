import hdbscan
from sklearn.mixture import BayesianGaussianMixture
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.decomposition import PCA
REGISTRY = {}


class HDBSCANCluster:
    def __init__(self, args):
        self.n_agents = getattr(args, "n_agents", 1)
        self.min_cluster_size_static = getattr(args, "hdbscan_min_cluster_size", None)
        self.min_samples_static = getattr(args, "hdbscan_min_samples", None)
        self.task_latent_dim = getattr(args, "rnn_hidden_dim", None)

        self.min_cluster_size = None
        self.min_samples = None

    def fit(self, hidden_states):
        if hidden_states.shape[1] > self.task_latent_dim:
            hidden_states = PCA(n_components=self.task_latent_dim).fit_transform(hidden_states)
        n_samples = hidden_states.shape[0]

        if self.min_cluster_size_static is not None:
            self.min_cluster_size = self.min_cluster_size_static
        else:
            self.min_cluster_size = max(5, n_samples // (self.n_agents * 2))

        if self.min_samples_static is not None:
            self.min_samples = self.min_samples_static
        else:
            self.min_samples = max(2, self.min_cluster_size // 2)

        clustering = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean'
        )

        labels = clustering.fit_predict(hidden_states)
        label_counts = Counter(labels)

        valid_labels = [label for label, count in label_counts.items()
                        if label != -1 and count >= self.min_cluster_size]

        if len(valid_labels) == 0:
            return None, None, None, None

        filtered_indices = [i for i, lbl in enumerate(labels) if lbl in valid_labels]
        filtered_labels_raw = [labels[i] for i in filtered_indices]
        filtered_hidden_states = hidden_states[filtered_indices]

        cluster_centers = [np.mean(filtered_hidden_states[np.array(filtered_labels_raw) == label], axis=0)
                           for label in valid_labels]

        label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_labels)}
        filtered_labels = np.array([label_mapping[lbl] for lbl in filtered_labels_raw])

        return np.array(cluster_centers), filtered_labels, np.array(filtered_indices), filtered_hidden_states


class DPGMMCluster:
    def __init__(self, args):
        self.n_agents = getattr(args, "n_agents", 1)
        self.n_components = getattr(args, "dpgmm_n_components", self.n_agents * 2)
        self.reg_covar = getattr(args, "dpgmm_reg_covar", 1e-3)
        self.min_cluster_size_static = getattr(args, "dpgmm_min_cluster_size", None)
        self.task_latent_dim = getattr(args, "rnn_hidden_dim", None)
        self.model = BayesianGaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.01,
            reg_covar=self.reg_covar,
            max_iter=500,
            random_state=42
        )

        self.min_cluster_size = None

    def fit(self, hidden_states):
        n_samples = hidden_states.shape[0]

        if self.min_cluster_size_static is not None:
            self.min_cluster_size = self.min_cluster_size_static
        else:
            est_clusters = self.n_components
            self.min_cluster_size = max(5, n_samples // est_clusters // 2)

        if hidden_states.shape[1] > self.task_latent_dim:
            hidden_states = PCA(n_components=self.task_latent_dim).fit_transform(hidden_states)

        try:
            labels = self.model.fit_predict(hidden_states)
        except ValueError as e:
            print(f"[DPGMMCluster] Clustering failed: {e}")
            return None, None, None

        label_counts = Counter(labels)
        valid_labels = [label for label, count in label_counts.items() if count >= self.min_cluster_size]

        if len(valid_labels) == 0:
            return None, None, None

        filtered_indices = [i for i, lbl in enumerate(labels) if lbl in valid_labels]
        filtered_labels_raw = [labels[i] for i in filtered_indices]

        cluster_centers = [np.mean(hidden_states[labels == label], axis=0) for label in valid_labels]

        label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_labels)}
        filtered_labels = np.array([label_mapping[lbl] for lbl in filtered_labels_raw])
        filtered_hidden_states = hidden_states[filtered_indices]

        return np.array(cluster_centers), filtered_labels, np.array(filtered_indices), filtered_hidden_states


class KMeansCluster:
    def __init__(self, args):
        self.n_clusters = getattr(args, "kmeans_n_clusters", 5)
        self.task_latent_dim = getattr(args, "rnn_hidden_dim", 64)

    def fit(self, hidden_states):
        clustering = KMeans(n_clusters=self.n_clusters, random_state=42)
        if hidden_states.shape[1] > self.task_latent_dim:
            hidden_states = PCA(n_components=self.task_latent_dim).fit_transform(hidden_states)
        labels = clustering.fit_predict(hidden_states)

        label_counts = Counter(labels)
        valid_labels = [label for label, count in label_counts.items() if count >= 1]

        if len(valid_labels) == 0:
            return None, None, None, None

        filtered_indices = [i for i, lbl in enumerate(labels) if lbl in valid_labels]
        filtered_labels_raw = [labels[i] for i in filtered_indices]
        filtered_hidden_states = hidden_states[filtered_indices]

        cluster_centers = [np.mean(filtered_hidden_states[np.array(filtered_labels_raw) == label], axis=0)
                           for label in valid_labels]

        label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_labels)}
        filtered_labels = np.array([label_mapping[lbl] for lbl in filtered_labels_raw])

        return np.array(cluster_centers), filtered_labels, np.array(filtered_indices), filtered_hidden_states


REGISTRY["hdbscan"] = HDBSCANCluster
REGISTRY["dpgmm"] = DPGMMCluster
REGISTRY["kmeans"] = KMeansCluster
