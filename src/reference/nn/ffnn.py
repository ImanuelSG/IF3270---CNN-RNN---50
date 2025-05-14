from .loss import LossFunction
from .initializers import *
from engine.autodiff import Value
import nn.loss as loss
import matplotlib.pyplot as plt
import networkx as nx
from .layer import Layer
from .module import Module
from numpy import random
import torch
from tqdm import tqdm
import pickle
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

class FFNN(Module):
    """
    API for Feed Forward Neural Network.
    The interfaces are inspired by TensorFlow, PyTorch, and micrograd
    ===================================================================

    Some general notes regarding input variation
    Usually, X (batch_size, input_size) and the calculation is X @ W.T() + b --> (batch_size, output_size)

    Handled cases:
    1. Single output problem (regression/classification, MSE/CE)
        a. actually outputs single value and output layer has a single neuron
            - at the last layer, the dimension is (batch_size, 1)
            - calculating the loss, the result is of size []
        b. multiclass classification problem with multiple neuron (softmax, CCE)
            - at the last layer, the dimension is (batch_size, n_classes)
            - calculating the loss, the result is of size []
    2. multiple output problem (e.g. practice problem)
        - at the last layer, the dimension is (batch_size, output)
        - definitely, the result of the loss is of size []
            1) for each label, loss for that label must be calculated [dimension still (batch_size, output)]
            2) aggregate the loss for each label through average [resulting dim: (batch_size, 1)]
        - after that, reduce it to [] by taking the average (sum all element then divide by batch_size)
        
    """
    def __init__(self, layers_list: list[Layer], random_seed: int = 42, learning_rate: float = 0.01, verbose: int = 1, epochs: int = 10, batch_size: int = None):
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size
        self.layers = layers_list
        self.loss_function = None
        self.losses = None
        self.val_losses = None
        self.regularization = None
        self.reg_coef = None

        if not (layers_list[0].input_shape is not None and all(layer.input_shape is None for layer in layers_list[1:])):
            raise ValueError("The first layer of the FFNN and only the first layer must have the input_shape attribute defined")

        for i, _ in enumerate(self.layers):
            if (i != 0):
                self.layers[i].input_shape = self.layers[i-1].output_shape

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    ## Just to mimick the compile function in keras and so that we dont need to re-init if we want to change the loss func
    def compile(self, loss_function: str | LossFunction, regularization: str = None, reg_coef: float = 0.0001):
        loss_map = {
            "MSE": loss.MSELoss(),
            "BCE": loss.BinaryCrossEntropyLoss(),
            "CCE": loss.CategoricalCrossEntropyLoss()
        }

        if isinstance(loss_function, str):
            if loss_function not in loss_map:
                raise ValueError(f"Invalid loss function '{loss_function}'. Choose from {list(loss_map.keys())}")

            self.loss_function = loss_map[loss_function]
        else:
            self.loss_function = loss_function

        for i, _ in enumerate(self.layers):
            initializer = self.layers[i].init_method
            if (initializer is not None):
                initializer.initialize(self.layers[i])

        # Store regularization settings
        self.regularization = regularization
        self.reg_coef = reg_coef

    def fit(self, X: Value, y: Value, X_val: Value = None, y_val: Value = None):
        random.seed(self.random_seed)

        if self.loss_function is None:
            raise ValueError("Compile the model before fitting it")

        # Ensures the dimensions are correct
        if X.data.dim() != 2:
            raise ValueError(f"X must be a 2-dimensional tensor, but got {X.data.dim()}-dimensional tensor")

        if y.data.dim() != 1 and y.data.dim() !=2 :
            raise ValueError(f"y must be a 1-dimensional or 2-dimensional tensor, but got {y.data.dim()}-dimensional tensor")
        
        if (X_val is not None and y_val is None) or (X_val is None and y_val is not None):
            raise ValueError("X_val and y_val must both be None or specified at the same time")

        self.losses = []
        self.val_losses = []
        n_samples = X.data.shape[0]

        learning_rate = self.learning_rate
        epochs = self.epochs
        verbose = self.verbose

        self.batch_size = self.batch_size if self.batch_size is not None else min(200, n_samples)

        if self.layers[-1].activation == "softmax":
            num_classes = y.data.unique().shape[0]
            y = torch.nn.functional.one_hot(y.data.to(torch.long), num_classes=num_classes)
            if y_val is not None:
                y_val = torch.nn.functional.one_hot(y_val.data.to(torch.long), num_classes=num_classes)

        for epoch in range(epochs):
            # Shuffle the data before each epoch
            indices = list(range(n_samples))
            random.shuffle(indices)
            X.data = X.data[indices]
            y.data = y.data[indices]

            progress_bar = tqdm(
                total=n_samples,
                desc=f"Epoch {epoch+1}/{epochs}",
                unit="sample",
                bar_format="{desc}:\t{percentage:3.0f}%|{bar:80}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}]{postfix}",
                disable=not verbose,
            )
            progress_bar.set_postfix({"Train Loss": "N/A", "Val Loss": "N/A"})

            # Iterate through batches
            for i in range(0, n_samples, self.batch_size):
                X_batch = X.data[i:i+self.batch_size]
                y_batch = y.data[i:i+self.batch_size]

                y_pred = self.predict(Value(X_batch))

                loss = self.loss_function(y_pred, Value(y_batch))

                if self.regularization == "L1":
                    l1_penalty = Value(0.0)
                    for layer in self.layers:
                        l1_penalty += layer.W.abs().sum()
                    loss += self.reg_coef * l1_penalty
                elif self.regularization == "L2":
                    l2_penalty = Value(0.0)
                    for layer in self.layers:
                        l2_penalty += (layer.W ** 2).sum()
                    loss += self.reg_coef * l2_penalty
                
                self.zero_grad()
                loss.backward()
                
                for param in self.parameters():
                    if param.grad is not None:
                        param.data -= learning_rate * param.grad.data

                progress_bar.update(len(X_batch))

            y_train_pred = self.predict(X)
            train_loss = self.loss_function(y_train_pred, y)
            self.losses.append(train_loss.data.item())

            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                val_loss = self.loss_function(y_val_pred, y_val)
                self.val_losses.append(val_loss.data.item())

                progress_bar.set_postfix({"Train Loss": f"{self.losses[-1]:.4f}", "Val Loss": f"{self.val_losses[-1]:.4f}"})
            else:
                progress_bar.set_postfix({"Train Loss": f"{self.losses[-1]:.4f}"})

            progress_bar.close()

        if verbose:
            print("------------------------------------------------------------")

        print("Training finished!")

        if verbose:
            print("Final Train Loss:", self.losses[-1])
            if X_val is not None and y_val is not None:
                print("Final Val Loss:", self.val_losses[-1])

    def predict(self, X: Value):
        for layer in self.layers:
            X = layer(X)
        return X 

    def score(self, X_test : Value, y_test : Value):
        predictions = self.predict(X_test).data

        if self.layers[-1].activation == "softmax":
            predictions= torch.argmax(predictions, dim=1)
        
        return (predictions == y_test.data).float().mean().item()

    def evaluate_model(model, X_test, y_test):
        predictions = model.predict(X_test).data
        if model.layers[-1].activation == "softmax":
            predictions = torch.argmax(predictions, dim=1)
        accuracy = (predictions == y_test.data).float().mean()
        print(f"Model Accuracy: {accuracy:.4f}")
        return accuracy

    def plot_loss(model):
        plt.plot(model.losses, label="Training Loss")
        if model.val_losses:
            plt.plot(model.val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.show()

    def plot_weight_distribution(model, layer_shown):
        selected_layers = [(i, layer.W.data.flatten()) for i, layer in enumerate(model.layers) if i in layer_shown]

        if not selected_layers:
            print("No valid layers selected.")
            return

        num_layers = len(selected_layers)
        rows = (num_layers + 1) // 2
        cols = 1 if num_layers == 1 else 2

        fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))

        if num_layers == 1:
            axes = [axes]

        for ax, (layer_idx, weights) in zip(axes.flatten(), selected_layers):
            ax.hist(weights, bins=30, alpha=0.75, color="blue")
            ax.set_xlabel("Weight Value")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Weight Distribution - Layer {layer_idx+1}")

        plt.tight_layout()
        plt.show()

    def plot_weight_gradient_distribution(model, layer_shown):
        selected_layers = [(i, layer.W.grad.flatten()) for i, layer in enumerate(model.layers) if i in layer_shown]

        if not selected_layers:
            print("No valid layers selected.")
            return

        num_layers = len(selected_layers)
        rows = (num_layers + 1) // 2
        cols = 1 if num_layers == 1 else 2

        fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))

        if num_layers == 1:
            axes = [axes]

        for ax, (layer_idx, gradients) in zip(axes.flatten(), selected_layers):
            ax.hist(gradients, bins=30, alpha=0.75, color="red")
            ax.set_xlabel("Gradient Value")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Gradient Distribution - Layer {layer_idx+1}")

        plt.tight_layout()
        plt.show()

    def save(self, filename: str):
        if self.loss_function is None:
            raise ValueError("Compile the model before saving it")
        
        execution_dir = os.getcwd()
        models_dir = os.path.join(execution_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        filename = os.path.basename(filename)

        # Final save path: always inside models/ directory
        filepath = os.path.join(models_dir, filename)

        model_data = {
            "layers": [
                (
                    layer.output_shape,
                    layer.activation,
                    layer.init_method,
                    layer.input_shape if i == 0 else None
                ) for i, layer in enumerate(self.layers)
            ],
            "weights": [layer.W.data for layer in self.layers],
            "biases": [layer.b.data for layer in self.layers],
            "random_seed": self.random_seed,
            "learning_rate": self.learning_rate,
            "verbose": self.verbose,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "loss_function": self.loss_function,
            "losses": self.losses,
            "val_losses": self.val_losses,
            "regularization": self.regularization,
            "reg_coef": self.reg_coef
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved successfully to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        execution_dir = os.getcwd()
        filepath = os.path.abspath(os.path.join(execution_dir, filepath))

        # Ensure it is a file, not a directory
        if os.path.isdir(filepath):
            raise ValueError(f"Expected a file but got a directory: {filepath}")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No such file: '{filepath}'")

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        layers = []

        for (output_shape, activation, init_method, input_shape) in model_data["layers"]:
            layer = Layer(output_shape=output_shape, activation=activation, init_method=init_method, input_shape=input_shape)
            layers.append(layer)

        # Create model instance
        model = cls(layers_list=layers,
                    random_seed=model_data["random_seed"],
                    learning_rate=model_data["learning_rate"],
                    verbose=model_data["verbose"],
                    epochs=model_data["epochs"],
                    batch_size=model_data["batch_size"])

        # Restore weights and biases
        for layer, W_data, b_data in zip(model.layers, model_data["weights"], model_data["biases"]):
            layer.W = Value(W_data, requires_grad=True)
            layer.b = Value(b_data, requires_grad=True)

        model.loss_function = model_data["loss_function"]
        model.regularization = model_data["regularization"]
        model.reg_coef = model_data["reg_coef"]
        model.losses = model_data["losses"]
        model.val_losses = model_data["val_losses"]

        print(f"Model loaded successfully from {filepath}")
        return model

    def plot_network(self):
        layer_sizes = []
        weights = {}
        gradients = {}
        biases = {}

        for i in range(len(self.layers)):
            layer_sizes.append(self.layers[i].W.data.shape[0])
            key_init = 'layer_' + str(i + 1)
            weights[key_init] = self.layers[i].W.data.numpy()
            gradients[key_init] = self.layers[i].W.grad.numpy()
            biases[key_init] = self.layers[i].b.data.numpy()

        G = nx.DiGraph()
        pos = {}
        node_labels = {}
        node_count = 0
        edge_matrix = []

        input_node = node_count
        G.add_node(input_node)
        pos[input_node] = (-3, 0)  
        node_labels[input_node] = f"Input\n n={self.layers[0].input_shape}"
        node_count += 1

        node_data = {}  # Store data for tooltips

        for layer_idx, layer_size in enumerate(layer_sizes):
            layer_width = (len(self.layers) - 1) * 3
            x = layer_idx * layer_width
            curr_layer_nodes = []

            for neuron_idx in range(layer_size):
                y = neuron_idx - (layer_size - 1) / 2
                node = node_count
                G.add_node(node)
                pos[node] = (x, y)

                layer_key = f'layer_{layer_idx+1}'
                weight_data = weights[layer_key][neuron_idx] if layer_key in weights else []
                gradient_data = gradients[layer_key][neuron_idx] if layer_key in gradients else []
                bias_data = biases[layer_key][0][neuron_idx] if layer_key in biases else 0

                node_labels[node] = f"Neuron {neuron_idx}"
                node_data[node] = (
                    f"Neuron {neuron_idx}\n"
                    f"Weights: {weight_data}\n"
                    f"Gradients: {gradient_data}\n"
                    f"Bias: {bias_data:.2f}"
                )

                curr_layer_nodes.append(node)
                node_count += 1

            edge_matrix.append(curr_layer_nodes)

        if edge_matrix:
            for neuron in edge_matrix[0]:
                G.add_edge(input_node, neuron)

        for i in range(len(edge_matrix) - 1):
            for j in range(len(edge_matrix[i])): 
                for k in range(len(edge_matrix[i+1])):
                    G.add_edge(edge_matrix[i][j], edge_matrix[i+1][k])

        node_colors = []
        for node in G.nodes():
            if node == 0:
                node_colors.append('#327dd9')
            elif node in edge_matrix[-1]:
                node_colors.append('#6edaeb')
            else:
                node_colors.append('#34ebcc')

        fig, ax = plt.subplots(figsize=(20, 12))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=15, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=6, font_weight="bold", ax=ax)

        ax.set_title("Feed Forward Neural Network")
        ax.set_xlim(min(x for x, y in pos.values()) - 5, max(x for x, y in pos.values()) + 5)
        ax.set_ylim(min(y for x, y in pos.values()) - 5, max(y for x, y in pos.values()) + 5)
        ax.set_axis_off()

        # Wider tooltip annotation
        tooltip = ax.annotate(
            "",
            xy=(0, 0), xytext=(15, 15),  # Increase offset for better positioning
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="black", lw=1.5),  # Wider box
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10, fontweight="bold",
            visible=False
        )

        def on_hover(event):
            if event.inaxes == ax:
                for node, (px, py) in pos.items():
                    if abs(event.xdata - px) < 0.5 and abs(event.ydata - py) < 0.5:  # If cursor is near a node
                        tooltip.set_text(node_data.get(node, ""))
                        tooltip.xy = (px, py)
                        tooltip.set_visible(True)
                        fig.canvas.draw_idle()
                        return
            tooltip.set_visible(False)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_hover)

        # Create a Tkinter window
        root = tk.Tk()
        root.title("Neural Network Visualization")

        frame = tk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(frame, width=1000, height=700)  # Larger default size
        scrollbar_x = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
        scrollbar_y = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(xscrollcommand=scrollbar_x.set, yscrollcommand=scrollbar_y.set)

        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        figure_canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
        figure_canvas_widget = figure_canvas.get_tk_widget()
        figure_canvas_widget.pack()

        root.mainloop()