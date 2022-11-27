from imports.ml import *
from imports.general import *
from src.parameters import Parameters
from src.load_data_from_file import load_data_from_file
from scipy.io.wavfile import read


class Dataset(object):
    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
        np.random.seed(self.seed)
        self.generate()

    def generate(self):
        if self.dataset_name == "default":
            c_1 = np.ones((int(self.n_train / 2), self.data_dim)) / np.sqrt(
                self.data_dim
            )
            c_2 = -np.ones((int(self.n_train / 2), self.data_dim)) / np.sqrt(
                self.data_dim
            )
            self.X = np.append(c_1, c_2, axis=0)
            self.y = np.append(
                np.zeros((int(self.n_train / 2),)),
                np.ones((int(self.n_train / 2),)),
                axis=0,
            )
            if self.cluster_std is None:
                self.cluster_std = 1
            self.X = self.X + np.random.normal(0, self.cluster_std, size=self.X.shape)
            self.X = self.X.transpose()
        elif self.dataset_name == "make_blobs":
            if self.cluster_std is None:
                self.cluster_std = 1 / (10 * self.data_dim)
            self.X, self.y = make_blobs(
                n_samples=self.n_train,
                n_features=self.data_dim,
                centers=2,
                cluster_std=self.cluster_std,
            )
            self.X = self.X - np.mean(self.X, axis=0)
            self.X = self.X.transpose()
        elif self.dataset_name == "audio_mnist_synthetic":
            paths = ["data/audio_mnist/1/0_01_0.wav", "data/audio_mnist/1/1_01_0.wav"]

            f_1, s_1 = read(paths[0])
            f_2, s_2 = read(paths[1])

            if f_1 == f_2:
                self.sample_frequency = f_1
            else:
                print("Sample frequencies are not equal")

            if len(s_1) < len(s_2):
                s_1 = np.append(s_1, np.zeros(len(s_2) - len(s_1)))
            elif len(s_1) > len(s_2):
                s_2 = np.append(s_2, np.zeros(len(s_1) - len(s_2)))

            F = int(self.sample_frequency / 16e3)
            s_1 = s_1[::F]
            s_2 = s_2[::F]
            self.sample_frequency = 16e3
            self.data_dim = len(s_1)

            if self.SNR is None:
                self.SNR = 1

            c_1 = s_1 + np.random.normal(
                0,
                np.sqrt((np.var(s_1)) / self.SNR),
                size=(int(self.n_train / 2), len(s_1)),
            )
            c_2 = s_2 + np.random.normal(
                0,
                np.sqrt((np.var(s_2)) / self.SNR),
                size=(int(self.n_train / 2), len(s_2)),
            )

            self.X = np.append(c_1, c_2, axis=0)
            self.y = np.append(
                np.zeros((int(self.n_train / 2),)),
                np.ones((int(self.n_train / 2),)),
                axis=0,
            )
            self.X = self.X.transpose()
        elif self.dataset_name == "audio_mnist_restaurant":
            paths = ["data/audio_mnist/1/0_01_0.wav", "data/audio_mnist/1/1_01_0.wav"]

            f_1, s_1 = read(paths[0])
            f_2, s_2 = read(paths[1])

            if f_1 == f_2:
                self.sample_frequency = f_1
            else:
                print("Sample frequencies are not equal")

            if len(s_1) < len(s_2):
                s_1 = np.append(s_1, np.zeros(len(s_2) - len(s_1)))
            elif len(s_1) > len(s_2):
                s_2 = np.append(s_2, np.zeros(len(s_1) - len(s_2)))

            F = int(self.sample_frequency / 16e3)
            s_1 = s_1[::F]
            s_2 = s_2[::F]
            self.sample_frequency = 16e3
            self.data_dim = len(s_1)

            if self.SNR is None:
                self.SNR = 1

            frequency, noise = read("data/restaurant-1.wav")
            noise = np.mean(noise, axis=1)

            noise = noise[::F]

            noise_sampling = np.linspace(0, int(9e5), int(self.n_train / 2))
            noises = np.zeros((int(self.n_train / 2), self.data_dim))

            for i in range(int(self.n_train / 2)):
                noises[i, :] = noise[
                    int(noise_sampling[i]) : int(noise_sampling[i] + self.data_dim)
                ]

            var_signal = np.var(np.vstack((s_1, s_2)))

            k_1 = np.sqrt(np.var(s_1) / (np.var(noises) * self.SNR))
            k_2 = np.sqrt(np.var(s_2) / (np.var(noises) * self.SNR))

            c_1 = s_1 + k_1 * noises
            c_2 = s_2 + k_2 * noises

            self.X = np.append(c_1, c_2, axis=0)

            self.y = np.append(
                np.zeros((int(self.n_train / 2),)),
                np.ones((int(self.n_train / 2),)),
                axis=0,
            )
            self.X = self.X.transpose()
        else:
            # Load the MNIST data set and isolate a subset of it.
            (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
            self.X = x_train[: self.n_train, ...].astype(np.float64) / 256.0
            self.y = y_train[: self.n_train]
            self.X = self.X.reshape(self.n_train, -1).transpose()
