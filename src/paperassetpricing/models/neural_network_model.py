import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from .base_model import BaseModel
from .model_registry import register_model


torch.manual_seed(0)


@register_model("nn_relu")
class NeuralNetworkModel(BaseModel):
    """
    Feedforward neural network with ReLU activations.

    Architecture: input -> [hidden_layers...] -> output (linear)
    Trained with MSE loss and Adam optimizer.
    """

    def __init__(
        self,
        hidden_layers: list[int] = [64, 32, 16, 8, 4],
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model: nn.Module | None = None

    def _build_model(self, input_dim: int) -> nn.Module:
        layers: list[nn.Module] = []
        in_dim = input_dim
        # hidden layers with ReLU
        for h in self.hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        # output layer (linear)
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y) -> "NeuralNetworkModel":
        # X: numpy array (n_samples, n_features), y: numpy array (n_samples,)
        X_t = torch.from_numpy(X.astype("float32")).to(self.device)
        y_t = torch.from_numpy(y.astype("float32")).view(-1, 1).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # build model if needed
        if self.model is None:
            self.model = self._build_model(X.shape[1]).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        # X: numpy array
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        self.model.eval()
        X_t = torch.from_numpy(X.astype("float32")).to(self.device)
        with torch.no_grad():
            out = self.model(X_t).cpu().numpy().flatten()
        return out

    def save(self, path: Path) -> None:
        # persist state_dict for inference
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "params": {
                    "hidden_layers": self.hidden_layers,
                    "lr": self.lr,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "device": str(self.device),
                },
                "state_dict": self.model.state_dict(),
            },
            path,
        )
