from imports.general import *
from imports.ml import *
from src.parameters import Parameters
from src.dataset import Dataset

# Dependencies:

# TODO:
# Should model be fine-tuned?


class WAV2VEC(object):
    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
        np.random.seed(self.seed)
        self.init_model()
        self.summary = {}

    def init_model(self, cp_path: str = "/imports/wav2vec_large.pt"):
        self.model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [cp_path]
        )
        self.model = self.model[0]
        self.model.eval()

    def fit(self, dataset: Dataset):
        self.X = dataset.X
        self.y = dataset.y
        self.train()

    def train(self):
        self.z_final = np.zeros((self.n.train, self.latent_dim), np.float64)
        for i in range(len(self.X)):
            self.z_final[:, i] = self.model.feature_extractor(self.X[i])
