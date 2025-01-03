import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class ESN:
    def __init__(self, n_inputs=1, n_reservoir=4, n_outputs=1,
                 spectral_radius=0.95, sparsity=0.5, noise=1e-6,
                 random_state=None, alpha=1e-6, leaking_rate=1.0):
        """
        Initialize the Echo State Network (ESN).

        Parameters:
        - n_inputs: Number of input neurons.
        - n_reservoir: Number of reservoir neurons.
        - n_outputs: Number of output neurons.
        - spectral_radius: Scaling factor for the reservoir's weight matrix.
        - sparsity: Fraction of non-zero connections in the reservoir.
        - noise: Noise added to the reservoir states.
        - random_state: Seed for reproducibility.
        - alpha: Regularization parameter for the readout.
        - leaking_rate: Leaking rate (α) controlling state update speed.
        """
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.alpha = alpha
        self.leaking_rate = leaking_rate

        # Set random seed
        if random_state:
            np.random.seed(random_state)

        # Initialize input weights (W_in)
        self.W_in = np.random.uniform(-0.5, 0.5, (self.n_reservoir, self.n_inputs))

        # Initialize reservoir weights (W_res) with sparsity
        W = np.random.uniform(-1, 1, (self.n_reservoir, self.n_reservoir))
        W[np.random.rand(*W.shape) > self.sparsity] = 0

        # Compute the spectral radius
        eigenvalues = np.linalg.eigvals(W)
        e_max = max(abs(eigenvalues))
        self.W_res = W * (self.spectral_radius / e_max)

        # Initialize readout weights
        self.W_out = None

    def _update(self, state, input_signal):
        """
        Update the reservoir state.

        Parameters:
        - state: Current state of the reservoir.
        - input_signal: Current input signal.

        Returns:
        - Updated state.
        """
        pre_activation = np.dot(self.W_in, input_signal) + np.dot(self.W_res, state)
        updated_state = (1 - self.leaking_rate) * state + self.leaking_rate * np.tanh(
            pre_activation + self.noise * np.random.randn(self.n_reservoir))
        return updated_state

    def fit(self, inputs, labels, washout=100):
        """
        Train the ESN.

        Parameters:
        - inputs: Input sequence (time_steps x n_inputs).
        - labels: Desired output sequence (time_steps x n_outputs).
        - washout: Number of initial states to discard.
        """
        states = np.zeros((inputs.shape[0], self.n_reservoir))
        state = np.zeros(self.n_reservoir)

        # Collect reservoir states
        for t in range(inputs.shape[0]):
            state = self._update(state, inputs[t])
            states[t] = state

        # Discard washout
        states_washout = states[washout:]
        labels_washout = labels[washout:]

        # Train classifier using RidgeClassifier
        clf = RidgeClassifier(alpha=self.alpha, fit_intercept=False)
        clf.fit(states_washout, labels_washout.ravel())
        self.W_out = clf

    def predict(self, inputs, initial_state=None):
        """
        Generate predictions using the trained ESN.

        Parameters:
        - inputs: Input sequence (time_steps x n_inputs).
        - initial_state: Optional initial state for the reservoir.

        Returns:
        - Predicted output sequence.
        """
        if self.W_out is None:
            raise ValueError("The model has not been trained yet.")

        states = []
        if initial_state is None:
            state = np.zeros(self.n_reservoir)
        else:
            state = initial_state

        for t in range(inputs.shape[0]):
            state = self._update(state, inputs[t])
            states.append(state)

        states = np.array(states)
        predictions = self.W_out.predict(states)
        return predictions


def load_csv(csv_path):
    data = pd.read_csv(csv_path, header=0)
    X = data.iloc[:, 1:].values / 255.0
    X = X.reshape(-1, 28 * 28)
    y = data.iloc[:, 0].values
    return X, y


# Read gesture data
X, y = load_csv('American_Sign_Language_Recognition.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

leaking_rate = 1.0
sparsity = 0.5
spectral_radius = 0.95
# Initialize ESN
esn = ESN(n_inputs=28 * 28,
          n_reservoir=2000,
          n_outputs=1,
          spectral_radius=spectral_radius,
          sparsity=sparsity,
          noise=1e-6,
          random_state=42,
          alpha=1e-6,
          leaking_rate=leaking_rate)

# Train ESN
esn.fit(X_train, y_train, washout=0)

# Predict and evaluate
predictions = esn.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy:.3f}")

# fig, axes = plt.subplots(2, 5, figsize=(10, 5))
# for i in range(10):
#     ax = axes[i // 5, i % 5]
#     ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
#     ax.set_title(f"Predicted: {predictions[i]}, True: {y_test[i]}")
# plt.tight_layout()
# plt.show()

# results = []
# for n_res in range(10, 10001,10):
#     for rnd_state in range(10, 1001):
#         esn = ESN(
#             n_inputs=28*28,
#             n_reservoir=n_res,
#             n_outputs=1,
#             spectral_radius=0.95,
#             sparsity=0.5,
#             noise=1e-6,
#             random_state=rnd_state,
#             alpha=1e-6,
#             leaking_rate=1.0
#         )
#         esn.fit(X_train, y_train, washout=0)
#         preds = esn.predict(X_test)
#         acc = accuracy_score(y_test, preds)
#         print(f"n_reservoir={n_res}, random_state={rnd_state}, accuracy={acc}")
#         results.append((n_res, rnd_state, acc))
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# n_res_list = [r[0] for r in results]
# rnd_state_list = [r[1] for r in results]
# acc_list = [r[2] for r in results]
# ax.scatter(n_res_list, rnd_state_list, acc_list)
# ax.set_xlabel('n_reservoir')
# ax.set_ylabel('random_state')
# ax.set_zlabel('accuracy')
# plt.show()
#
# df = pd.DataFrame(results, columns=['n_reservoir', 'random_state', 'accuracy'])
# df.to_csv('results.csv', index=False)
