import numpy as np


class PriceForecaster:
    """
    Simple linear autoregressive forecaster trained with least squares.
    Uses a fixed lookback window to predict the next step.
    """

    def __init__(self, lookback=48):
        self.lookback = int(lookback)
        self.coef_ = None

    def fit(self, series):
        series = np.asarray(series, dtype=float).flatten()
        if len(series) <= self.lookback:
            raise ValueError("Price series too short for the chosen lookback.")

        X = []
        y = []
        for i in range(self.lookback, len(series)):
            X.append(series[i - self.lookback:i])
            y.append(series[i])
        X = np.asarray(X)
        y = np.asarray(y)

        # Add bias term
        ones = np.ones((X.shape[0], 1))
        Xb = np.concatenate([ones, X], axis=1)
        self.coef_, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        return self

    def predict_next(self, window):
        if self.coef_ is None:
            raise ValueError("Forecaster not fitted.")
        window = np.asarray(window, dtype=float).flatten()
        if len(window) < self.lookback:
            pad = np.full(self.lookback - len(window), window[0] if len(window) else 0.0)
            window = np.concatenate([pad, window])
        else:
            window = window[-self.lookback:]
        x = np.concatenate([[1.0], window])
        return float(x @ self.coef_)


class LSTMPriceForecaster:
    """
    Lightweight LSTM forecaster for price series.
    """

    def __init__(self, lookback=48, hidden_size=32, epochs=30, lr=1e-3):
        self.lookback = int(lookback)
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.model = None
        self.mean_ = 0.0
        self.std_ = 1.0

    def _build_model(self, input_size=1):
        import torch
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self, hidden):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
                self.fc = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]
                return self.fc(out)

        return _Model(self.hidden_size)

    def fit(self, series):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        series = np.asarray(series, dtype=float).flatten()
        if len(series) <= self.lookback:
            raise ValueError("Price series too short for the chosen lookback.")

        # Normalize for stability
        self.mean_ = float(series.mean())
        self.std_ = float(series.std()) if float(series.std()) > 0 else 1.0
        series = (series - self.mean_) / self.std_

        X = []
        y = []
        for i in range(self.lookback, len(series)):
            X.append(series[i - self.lookback:i])
            y.append(series[i])
        X = np.asarray(X, dtype=np.float32)[:, :, None]
        y = np.asarray(y, dtype=np.float32)[:, None]

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        self.model = self._build_model()
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

        return self

    def predict_next(self, window):
        if self.model is None:
            raise ValueError("Forecaster not fitted.")
        import torch

        window = np.asarray(window, dtype=float).flatten()
        if len(window) < self.lookback:
            pad = np.full(self.lookback - len(window), window[0] if len(window) else 0.0)
            window = np.concatenate([pad, window])
        else:
            window = window[-self.lookback:]

        window = (window - self.mean_) / self.std_
        x = torch.from_numpy(window.astype(np.float32)[None, :, None])
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x).item()
        return pred * self.std_ + self.mean_


def generate_price_forecast(series, horizon=20, lookback=48, model="linear", **kwargs):
    """
    Generate rolling multi-step forecasts for the whole simulation.
    Returns array of shape (len(series), horizon).
    """
    series = np.asarray(series, dtype=float).flatten()
    if model == "lstm":
        forecaster = LSTMPriceForecaster(
            lookback=lookback,
            hidden_size=kwargs.get("hidden_size", 32),
            epochs=kwargs.get("epochs", 30),
            lr=kwargs.get("lr", 1e-3),
        ).fit(series)
    else:
        forecaster = PriceForecaster(lookback=lookback).fit(series)

    forecasts = np.zeros((len(series), horizon), dtype=float)
    for t in range(len(series)):
        history = series[:t + 1]
        preds = []
        window = history.copy()
        for _ in range(horizon):
            next_val = forecaster.predict_next(window)
            preds.append(next_val)
            window = np.append(window, next_val)
        forecasts[t, :] = preds
    return forecasts
