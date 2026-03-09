import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import plotly.graph_objects as go

# ----------------------------
# Model
# ----------------------------
class Neuron:
    def __init__(self, size, low=-1, high=1, alpha=0.1, seed=None):
        self.alpha = alpha
        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(low, high, size).astype(float)
        self.bias = rng.uniform(low, high)

        self.activation = 0.0
        self.h = 0.0
        self.y = 0.0

    def calculate_activation(self, data_vector):
        self.activation = float(np.dot(self.weights, data_vector) + self.bias)
        return self.activation

    def leaky_relu(self):
        a = self.activation
        self.h = a if a >= 0 else self.alpha * a
        return self.h

    def sigmoid(self):
        a = self.activation
        self.y = 1.0 / (1.0 + np.exp(-a))
        return self.y

def generate_octant_data(n=100, low=-1.0, high=1.0, seed=0, radius=0.65):
    rng = np.random.default_rng(seed)
    X = rng.uniform(low, high, size=(n, 3)).astype(float)
    

    center1 = np.array([0.5, 0.5, 0.5])
    center2 = np.array([-0.5, -0.5, -0.5])
    
    # Define how big the bubbles should be
    radius = 0.4
    

    dist1 = np.linalg.norm(X - center1, axis=1)
    dist2 = np.linalg.norm(X - center2, axis=1)
    

    t = ((dist1 < radius) | (dist2 < radius)).astype(float)
         
    return X, t

def forward_batch(X_batch, hidden_layer, output_neuron):
    W1 = np.stack([n.weights for n in hidden_layer], axis=0)  
    b1 = np.array([n.bias for n in hidden_layer], dtype=float)  
    a1 = np.array([n.alpha for n in hidden_layer], dtype=float) 

    z1 = X_batch @ W1.T + b1  
    h = np.where(z1 >= 0, z1, a1 * z1)

    z2 = h @ output_neuron.weights + output_neuron.bias  
    z2_clip = np.clip(z2, -60, 60)  
    y = 1.0 / (1.0 + np.exp(-z2_clip))

    return y, z1, h, z2

def plot_network_plotly(X, t, hidden_layer, output_neuron, lim=1.2, grid=25, title="Network Space"):
    """
    Interactive 3D visualization using Plotly.
    Renders ONLY the probability space as a volume and the data points.
    """
    fig = go.Figure()

    # ----- 1. Evaluate Probability Space for Volume -----
    coords = np.linspace(-lim, lim, grid)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    
    p, _, _, _ = forward_batch(pts, hidden_layer, output_neuron)
    
    # Add 3D Volume for the probability space
    fig.add_trace(go.Volume(
        x=xx.ravel(), y=yy.ravel(), z=zz.ravel(), value=p.ravel(),
        isomin=0.0,
        isomax=1.0,
        opacity=0.2, 
        surface_count=15, 
        colorscale='RdBu_r', 
        opacityscale=[[0, 1], [0.4, 0], [0.6, 0], [1, 1]], 
        name="Probability Volume",
        showscale=True,
        colorbar=dict(title="p(y=1)")
    ))

    # ----- 2. Scatter Original Training Data -----
    m0 = (t == 0)
    m1 = (t == 1)
    
    fig.add_trace(go.Scatter3d(
        x=X[m0, 0], y=X[m0, 1], z=X[m0, 2],
        mode='markers', marker=dict(size=6, color='blue', line=dict(width=2, color='black')),
        name="Target 0"
    ))
    
    fig.add_trace(go.Scatter3d(
        x=X[m1, 0], y=X[m1, 1], z=X[m1, 2],
        mode='markers', marker=dict(size=6, color='red', symbol='diamond', line=dict(width=2, color='black')),
        name="Target 1"
    ))

    # ----- 3. Layout Aesthetics -----
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X', range=[-lim, lim]),
            yaxis=dict(title='Y', range=[-lim, lim]),
            zaxis=dict(title='Z', range=[-lim, lim])
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig.show()
# ----------------------------
# Training Loop
# ----------------------------
def train(X, t, hidden_layer, output_neuron, lr=0.05, epochs=50, plot_every=0):
    n_hidden = len(hidden_layer)

    for epoch in range(epochs):
        idx = np.random.permutation(len(X))
        Xs, ts = X[idx], t[idx]

        for x_i, t_i in zip(Xs, ts):
            h = np.empty(n_hidden, dtype=float)
            z1 = np.empty(n_hidden, dtype=float)

            for j, neuron in enumerate(hidden_layer):
                z1[j] = neuron.calculate_activation(x_i)
                h[j] = neuron.leaky_relu()

            output_neuron.calculate_activation(h)
            y_hat = output_neuron.sigmoid()

            # ----- backward -----
            delta2 = y_hat - t_i
            v_old = output_neuron.weights.copy()

            output_neuron.weights -= lr * delta2 * h
            output_neuron.bias    -= lr * delta2

            for j, neuron in enumerate(hidden_layer):
                slope = 1.0 if z1[j] >= 0 else neuron.alpha  
                delta1 = delta2 * v_old[j] * slope        
                
                neuron.weights -= lr * delta1 * x_i
                neuron.bias    -= lr * delta1

      
        if plot_every > 0 and (epoch % plot_every == 0 or epoch == epochs - 1):
            plot_network_plotly(
                X, t, hidden_layer, output_neuron,
                lim=1.2, grid=20,  
                title=f"Network Space at Epoch {epoch}"
            )

    return hidden_layer, output_neuron

if __name__ == "__main__":
    X, t = generate_octant_data(n=300, seed=0)

    n_hidden = 100

    hidden_layer = [Neuron(size=3, alpha=0.1, seed=s) for s in range(n_hidden)]
    output_neuron = Neuron(size=n_hidden, alpha=0.1, seed=4)

    train(X, t, hidden_layer, output_neuron, lr=0.05, epochs=10000, plot_every=1000)