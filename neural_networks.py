import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # Learning rate
        self.activation_fn = activation  # Activation function

        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

        # Store activations and gradients for visualization
        self.activations = {}
        self.gradients = {}

    def activation(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation_fn == 'sigmoid':
            sig = self.activation(x)
            return sig * (1 - sig)

    def forward(self, X):
        # Input to hidden layer
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.activation(self.z1)

        # Hidden to output layer
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = 1 / (1 + np.exp(-self.z2))  # Sigmoid activation for output layer

        # Store activations for visualization
        self.activations['input'] = X
        self.activations['hidden'] = self.a1
        self.activations['output'] = self.a2

        return self.a2

    def backward(self, X, y):
        m = X.shape[0]

        # Compute the loss derivative w.r.t. output using binary cross-entropy loss
        dz2 = self.a2 - y  # Derivative for binary cross-entropy with sigmoid activation
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Backpropagate to hidden layer
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.activation_derivative(self.z1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights with gradient descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        # Store gradients for visualization
        self.gradients['W1'] = dW1
        self.gradients['W2'] = dW2

    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)

def generate_data(n_samples=200):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 1).astype(int)  # Inside circle is class 1
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y, xx, yy, grid, steps_per_frame):
    # Clear axes
    ax_hidden.cla()
    ax_input.cla()
    ax_gradient.cla()

    # Perform multiple training steps per frame
    for _ in range(steps_per_frame):
        mlp.forward(X)
        mlp.backward(X, y)

    # Plot hidden features
    hidden_features = mlp.activations['hidden']
    colors = ['red' if label == 1 else 'blue' for label in y.ravel()]
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
                      c=colors, alpha=0.7)
    ax_hidden.set_title('Hidden Layer Feature Space')
    ax_hidden.set_xlabel('Neuron 1 Activation')
    ax_hidden.set_ylabel('Neuron 2 Activation')
    ax_hidden.set_zlabel('Neuron 3 Activation')
    # Keep the view fixed
    ax_hidden.view_init(elev=30, azim=45)

    # Visualize decision hyperplane in hidden space
    W = mlp.W2.ravel()
    b = mlp.b2.ravel()
    if np.all(W != 0):
        # Plane equation: W[0]*x + W[1]*y + W[2]*z + b = 0
        xx_hidden, yy_hidden = np.meshgrid(
            np.linspace(hidden_features[:, 0].min(), hidden_features[:, 0].max(), 10),
            np.linspace(hidden_features[:, 1].min(), hidden_features[:, 1].max(), 10)
        )
        zz_hidden = (-W[0]*xx_hidden - W[1]*yy_hidden - b[0]) / W[2]
        ax_hidden.plot_surface(xx_hidden, yy_hidden, zz_hidden, alpha=0.3, color='green')

    # Plot decision boundary in input space
    Z = mlp.forward(grid)
    Z = Z.reshape(xx.shape)

    # Animate the decision boundary over time
    ax_input.contourf(xx, yy, Z, levels=np.linspace(0, 1, 20), cmap='bwr', alpha=0.3)
    ax_input.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    ax_input.scatter(X[:, 0], X[:, 1], c=colors, edgecolor='k')
    ax_input.set_title(f'Decision Boundary at Step {(frame+1)*steps_per_frame}')
    ax_input.set_xlabel('Feature 1')
    ax_input.set_ylabel('Feature 2')

    # Visualize gradients
    ax_gradient.set_xlim(0, 3)
    ax_gradient.set_ylim(0, 3)
    ax_gradient.axis('off')
    ax_gradient.set_title('Network Gradients')

    # Nodes positions
    nodes_input = [(0.5, 2.5), (0.5, 1.5)]
    nodes_hidden = [(1.5, 2.5), (1.5, 1.5), (1.5, 0.5)]  # 3 hidden neurons
    node_output = (2.5, 1.5)

    # Plot nodes
    for idx, pos in enumerate(nodes_input):
        ax_gradient.add_patch(Circle(pos, 0.1, color='lightblue'))
        ax_gradient.text(pos[0], pos[1], f'$x_{idx+1}$', ha='center', va='center')

    for idx, pos in enumerate(nodes_hidden):
        ax_gradient.add_patch(Circle(pos, 0.1, color='lightblue'))
        ax_gradient.text(pos[0], pos[1], f'$h_{idx+1}$', ha='center', va='center')

    ax_gradient.add_patch(Circle(node_output, 0.1, color='lightblue'))
    ax_gradient.text(node_output[0], node_output[1], '$y$', ha='center', va='center')

    # Update edge connections according to the network size
    max_grad = max(np.abs(mlp.gradients['W1']).max(), np.abs(mlp.gradients['W2']).max())

    # Input to hidden edges
    for i, input_pos in enumerate(nodes_input):
        for j, hidden_pos in enumerate(nodes_hidden):
            grad = np.abs(mlp.gradients['W1'][i, j])
            linewidth = (grad / max_grad) * 5 if max_grad != 0 else 0.1
            ax_gradient.plot([input_pos[0], hidden_pos[0]], [input_pos[1], hidden_pos[1]],
                             'purple', linewidth=linewidth)

    # Hidden to output edges
    for j, hidden_pos in enumerate(nodes_hidden):
        grad = np.abs(mlp.gradients['W2'][j, 0])
        linewidth = (grad / max_grad) * 5 if max_grad != 0 else 0.1
        ax_gradient.plot([hidden_pos[0], node_output[0]], [hidden_pos[1], node_output[1]],
                         'purple', linewidth=linewidth)

def visualize(activation, lr, step_num):
    X, y = generate_data()
    hidden_dim = 3  # Using 3 hidden neurons as per your request
    mlp = MLP(input_dim=2, hidden_dim=hidden_dim, output_dim=1, lr=lr, activation=activation)

    # Prepare grid for decision boundary visualization
    xx, yy = np.meshgrid(np.linspace(-3, 3, 120), np.linspace(-3, 3, 120))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    steps_per_frame = 10
    total_frames = step_num // steps_per_frame

    # Create animation
    ani = FuncAnimation(
        fig,
        partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden,
                ax_gradient=ax_gradient, X=X, y=y, xx=xx, yy=yy, grid=grid,
                steps_per_frame=steps_per_frame),
        frames=total_frames,
        repeat=False
    )

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.05
    step_num = 200
    visualize(activation, lr, step_num)