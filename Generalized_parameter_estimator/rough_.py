import torch
from torchdiffeq import odeint  # for ODE-based models (install via: pip install torchdiffeq)

# ============================================================
# 1. GENERALIZED PARAMETER ESTIMATOR CLASS
# ============================================================

class ParameterEstimator:
    def __init__(self, model_func, param_init, optimizer_cls=torch.optim.Adam, lr=0.01, loss_fn=None):
        """
        Generalized parameter estimation using PyTorch.

        Args:
            model_func: Callable model f(x, **params)
            param_init: Dict or list of initial parameter guesses
            optimizer_cls: PyTorch optimizer class (default Adam)
            lr: Learning rate
            loss_fn: Custom loss function (default MSE)
        """
        self.model_func = model_func
        self.params = self._init_params(param_init)
        self.optimizer = optimizer_cls(self.params.values(), lr=lr)
        self.loss_fn = loss_fn if loss_fn else torch.nn.MSELoss()

    def _init_params(self, param_init):
        """Initialize trainable parameters."""
        import numpy as np
        if isinstance(param_init, dict):
            return {name: torch.tensor(val, requires_grad=True, dtype=torch.float32)
                    for name, val in param_init.items()}
        elif isinstance(param_init, (list, tuple, np.ndarray)):
            return {f"p{i}": torch.tensor(float(val), requires_grad=True, dtype=torch.float32)
                    for i, val in enumerate(param_init)}
        elif isinstance(param_init, (float, int)):
            return {"p0": torch.tensor(param_init, requires_grad=True, dtype=torch.float32)}
        else:
            raise ValueError("param_init must be a float, list, tuple, np.ndarray, or dict")

    def fit(self, x_data, y_data, epochs=1000, verbose=True):
        """Fit the model to experimental data."""
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            y_pred = self.model_func(x_data, **self.params)
            loss = self.loss_fn(y_pred, y_data)
            loss.backward()
            self.optimizer.step()

            if verbose and epoch % max(1, (epochs // 10)) == 0:
                loss_val = loss.item()
                param_str = ", ".join(f"{k}={v.item():.4f}" for k, v in self.params.items())
                print(f"Epoch {epoch:4d} | Loss={loss_val:.6f} | {param_str}")

        return {k: v.item() for k, v in self.params.items()}

# ============================================================
# 2. MODEL LIBRARY
# ============================================================

# ---------- KINETIC MODELS ----------

def zero_order_model(t, C0, k):
    return C0 - k * t

def first_order_model(t, C0, k):
    return C0 * torch.exp(-k * t)

def second_order_model(t, C0, k):
    return 1 / (1/C0 + k * t)

def arrhenius_model(T, A, Ea):
    R = 8.314
    return A * torch.exp(-Ea / (R * T))

def power_law_rate(CA, CB, k, m, n):
    return k * (CA ** m) * (CB ** n)

def michaelis_menten(S, Vmax, Km):
    return (Vmax * S) / (Km + S)

def monod_model(S, mumax, Ks):
    return (mumax * S) / (Ks + S)

def pseudo_first_order(t, qe, k1):
    return qe * (1 - torch.exp(-k1 * t))

def pseudo_second_order(t, qe, k2):
    return (k2 * qe**2 * t) / (1 + k2 * qe * t)

# ---------- ADSORPTION ISOTHERMS ----------

def langmuir_isotherm(Ce, qmax, K):
    return (qmax * K * Ce) / (1 + K * Ce)

def freundlich_isotherm(Ce, KF, n):
    return KF * Ce ** (1 / n)

def temkin_isotherm(Ce, A, b, T=298):
    R = 8.314
    return (R * T / b) * torch.log(A * Ce)

def dubinin_radushkevich(Ce, qs, B):
    return qs * torch.exp(-B * (torch.log(1 + 1/Ce) ** 2))

# ---------- REACTOR MODELS ----------

def batch_reactor_model(t, C0, k):
    return C0 * torch.exp(-k * t)

def cstr_steady(CA0, k, tau):
    return CA0 / (1 + k * tau)

def pfr_conversion(tau, k):
    return 1 - torch.exp(-k * tau)

def ode_first_order(t, C0, k):
    def rhs(t, C): return -k * C
    C_pred = odeint(rhs, C0, t).squeeze()
    return C_pred

def cstr_dynamic(t, CA0, tau, k):
    def rhs(t, CA):
        return (CA0 - CA) / tau - k * CA
    CA_pred = odeint(rhs, CA0, t).squeeze()
    return CA_pred

# ---------- DRYING MODELS ----------

def newton_drying(t, M0, Me, k):
    return Me + (M0 - Me) * torch.exp(-k * t)

def henderson_pabis(t, a, k):
    return a * torch.exp(-k * t)

def page_model(t, k, n):
    return torch.exp(-k * t ** n)

# ============================================================
# 3. EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example: first-order kinetics
    t_data = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.float32)
    C_data = torch.tensor([1.0, 0.82, 0.67, 0.55, 0.45, 0.37], dtype=torch.float32)

    estimator = ParameterEstimator(
        model_func=first_order_model,
        param_init={"C0": 1.0, "k": 0.2},
        lr=0.05
    )

    params = estimator.fit(t_data, C_data, epochs=500)
    print("\nEstimated Parameters:", params)

    # Plot results
    with torch.no_grad():
        C_fit = first_order_model(t_data, **{k: torch.tensor(v) for k, v in params.items()})
        plt.scatter(t_data, C_data, label="Experimental", color="red")
        plt.plot(t_data, C_fit, label="Model Fit", color="blue")
        plt.xlabel("Time")
        plt.ylabel("Concentration")
        plt.legend()
        plt.show()
