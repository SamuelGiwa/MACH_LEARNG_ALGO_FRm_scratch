# ---------- KINETIC MODELS ----------

def zero_order_model(t, C0, k):    #single parameter
    return C0 - k * t

def first_order_model(t, C0, k):   #single parameter
    return C0 * torch.exp(-k * t)

def second_order_model(t, C0, k):  #single parameter
    return 1 / (1/C0 + k * t)

def arrhenius_model(T, A, Ea):     #single parameter
    R = 8.314
    return A * torch.exp(-Ea / (R * T))

def power_law_rate(CA, CB, k, m, n): #3 parameter
    return k * (CA ** m) * (CB ** n)

def michaelis_menten(S, Vmax, Km):   #1 parameter
    return (Vmax * S) / (Km + S)

def monod_model(S, mumax, Ks):     #1 parameter
    return (mumax * S) / (Ks + S)

def pseudo_first_order(t, qe, k1):   #1 parameter
    return qe * (1 - torch.exp(-k1 * t))

def pseudo_second_order(t, qe, k2): #2 Parameters
    return (k2 * qe**2 * t) / (1 + k2 * qe * t)


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