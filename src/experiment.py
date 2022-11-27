from imports.ml import *
from imports.general import *
from src.gp_lvm import GPLVM
from src.dataset import Dataset
from src.parameters import Parameters
from postprocessing.figures import Results


class Experiment(Results):
    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
        self.dataset = Dataset(parameters)
        self.model_gplvm = GPLVM(parameters)

    def wav2vec(self):
        cp_path = "/path/to/wav2vec.pt"
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [cp_path]
        )
        self.model_wav2vec = model[0]
        self.model_wav2vec.eval()
        # wav_input_16khz = torch.randn(1,10000)
        self.dataset.X = self.model_wav2vec.feature_extractor(
            self.dataset.X.transpose()
        )

    def run_gplvm(self):
        self.model_gplvm.fit(self.dataset)
        self.Z = self.model_gplvm.z_final
        self.k_means = KMeans(n_clusters=self.latent_dim, n_init=20).fit(self.Z)
        self.y_preds = self.k_means.predict(self.Z)
        self.nmi = nmi(self.dataset.y.astype(int), self.y_preds.astype(int))
        self.summary = {
            "nmi": self.nmi,
            "Z_final": self.Z.tolist(),
            "loss_history": self.model_gplvm.loss_history.tolist(),
        }
        self.save_summary("GPLVM")

    def run_pca(self):
        self.model_pca = PCA(n_components=self.latent_dim, random_state=self.seed)
        self.model_pca.fit(self.dataset.X.transpose())
        self.Z = self.model_pca.transform(self.dataset.X.transpose())
        self.k_means = KMeans(n_clusters=self.latent_dim, random_state=self.seed)
        self.y_preds = self.k_means.fit_predict(self.Z)
        self.nmi = nmi(self.dataset.y.astype(int), self.y_preds.astype(int))
        # self.summary = {
        #    "nmi": self.nmi,
        #    "Z_final": self.Z.tolist(),
        #    "Explained_variance": self.model_pca.explained_variance_ratio_.tolist(),
        #    "Eigen_values": self.model_pca.singular_values_.tolist(),
        # }
        # self.save_summary("PCA")
        return self.nmi

    def run_tsne(self):
        self.model_tsne = TSNE(
            n_components=self.latent_dim,
            n_iter=self.n_iterations,
            init="random",
            perplexity=int(self.n_train / 3),
            random_state=self.seed,
        )
        self.Z = self.model_tsne.fit_transform(self.dataset.X.transpose())
        self.k_means = KMeans(n_clusters=self.latent_dim, n_init=20).fit(self.Z)
        self.y_preds = self.k_means.predict(self.Z)
        self.nmi = nmi(self.dataset.y.astype(int), self.y_preds.astype(int))
        self.summary = {
            "nmi": self.nmi,
            "Z_final": self.Z.tolist(),
            "embedding": self.model_tsne.embedding_.tolist(),
            "kl_divergence": self.model_tsne.kl_divergence_,
        }
        self.save_summary("TSNE")

    def run_umap(self):
        self.model_umap = umap.UMAP(
            n_components=self.latent_dim, min_dist=0.0, random_state=self.seed
        ).fit(self.dataset.X.transpose())
        self.Z = self.model_umap.transform(self.dataset.X.transpose())
        # test_embedding = trans.transform(X_test)
        self.k_means = KMeans(n_clusters=self.latent_dim, n_init=20).fit(self.Z)
        self.y_preds = self.k_means.predict(self.Z)
        self.nmi = nmi(self.dataset.y.astype(int), self.y_preds.astype(int))
        self.summary = {
            "nmi": self.nmi,
            "Z_final": self.Z.tolist(),
            "graph": self.model_umap.graph_.data.tolist(),
            "a": self.model_umap._a,
            "b": self.model_umap._b,
        }
        self.save_summary("UMAP")

    def run_trimap(self):
        print()
        self.model_trimap = trimap.TRIMAP(
            n_dims=self.latent_dim,
            n_inliers=int(self.n_train / 4),
            n_outliers=int((self.n_train / 5) / 2),
            n_random=int((self.n_train / 5) * 0.3),
            n_iters=self.n_iterations,
        ).fit(self.dataset.X.transpose())
        self.Z = self.model_trimap.fit_transform(self.dataset.X.transpose())
        self.k_means = KMeans(n_clusters=self.latent_dim, n_init=20).fit(self.Z)
        self.y_preds = self.k_means.predict(self.Z)
        self.nmi = nmi(self.dataset.y.astype(int), self.y_preds.astype(int))
        self.summary = {
            "nmi": self.nmi,
            "Z_final": self.Z.tolist(),
            "triplets": self.model_trimap.triplets.tolist(),
            "weights": self.model_trimap.weights.tolist(),
        }
        self.save_summary("TRIMAP")

    def save_summary(self, experiment_name) -> None:
        json_dump = json.dumps(self.summary)
        with open(self.savepth + f"results_{experiment_name}.json", "w") as f:
            f.write(json_dump)
