from imports.general import *
from imports.ml import *
from src.parameters import Parameters
from src.dataset import Dataset


class GPLVM(object):
    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
        np.random.seed(self.seed)
        self.trainable_variables = []
        self.init_hyperparameters()
        self.optimizer = tf.optimizers.Adam(learning_rate=self.gplvm_learning_rate)
        self.summary = {}

    def fit(self, dataset: Dataset):
        """X must be shape [D,N]"""
        self.X = dataset.X
        self.y = dataset.y
        self.init_z()
        self.train()

    def init_hyperparameters(self):
        self.unconstrained_amplitude = tf.Variable(np.float64(1.0), name="amplitude")
        self.unconstrained_length_scale = tf.Variable(
            np.float64(1.0), name="length_scale"
        )
        self.unconstrained_observation_noise = tf.Variable(
            np.float64(1.0), name="observation_noise"
        )
        self.trainable_variables.append(self.unconstrained_amplitude)
        self.trainable_variables.append(self.unconstrained_length_scale)
        self.trainable_variables.append(self.unconstrained_observation_noise)

    def init_z(self):
        if self.gp_latent_init_pca:
            self.z_init = PCA(self.latent_dim).fit_transform(self.X.transpose())
        else:
            self.z_init = np.random.normal(size=(self.n_train, self.latent_dim))

        self.latent_index_points = tf.Variable(self.z_init, name="latent_index_points")
        self.trainable_variables.append(self.latent_index_points)

    def create_kernel(self) -> tf.Tensor:
        amplitude = tf.math.softplus(1e-8 + self.unconstrained_amplitude)
        length_scale = tf.math.softplus(1e-8 + self.unconstrained_length_scale)
        kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)
        return kernel

    def loss_fn(self) -> tf.Tensor:
        observation_noise_variance = tf.math.softplus(
            1e-8 + self.unconstrained_observation_noise
        )
        gp = tfd.GaussianProcess(
            kernel=self.create_kernel(),
            index_points=self.latent_index_points,
            observation_noise_variance=observation_noise_variance,
        )
        log_probs = gp.log_prob(self.X, name="log_prob")
        return -tf.reduce_mean(log_probs)

    @tf.function(autograph=False, jit_compile=True)
    def train_step(self):
        with tf.GradientTape() as tape:
            loss_value = self.loss_fn()
        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss_value

    def train(self):
        lips = np.zeros((self.n_iterations, self.n_train, self.latent_dim), np.float64)
        self.loss_history = np.zeros((self.n_iterations,), np.float64)
        for i in range(self.n_iterations):
            loss = self.train_step()
            lips[i] = self.latent_index_points.numpy()
            self.loss_history[i] = loss.numpy()

        self.z_final = lips[[np.argmin(self.loss_history)]].squeeze()
