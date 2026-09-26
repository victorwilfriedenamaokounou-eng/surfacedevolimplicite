import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

df = pd.read_csv(r"D:\stochastique\informationsoption2023-09-21.csv")

df2 = df[df["option_type"] == "P"].copy()
print(df2.groupby("underlying_symbol")["trade_volume"].mean())
df2 = df2[
    (df2["trade_volume"] > 10)

]
print(df2["trade_volume"])
subset = df2[df2["underlying_symbol"] == "^SPX"]
C = subset["bid"].values
K = subset["strike"].values

second_diff = np.diff(C, n=2)
print(second_diff)

np.where(second_diff < 0)

df2["quoteday"]      = pd.to_datetime(df2["quote_datetime"].str.split(" ").str[0], format='%m/%d/%Y')
df2["expirationday"] = pd.to_datetime(df2["expiration"], format='%m/%d/%Y')
df2["maturity"]      = (df2["expirationday"] - df2["quoteday"]).dt.days / 365

df3 = df2[["strike", "implied_volatility", "maturity","close"]].rename(
    columns={"implied_volatility": "impliedvolatility"}
)
x=df3["impliedvolatility"]
y=df3["maturity"]
plt.scatter(y,x)
plt.xlabel("maturite")
plt.ylabel("volatiliteimplicite")
plt.show()

# ✅ Diagnostic : afficher la distribution réelle
print(df3["impliedvolatility"].describe())

# ✅ Correction d'échelle si IV est en %
if df3["impliedvolatility"].median() > 2:
    df3["impliedvolatility"] = df3["impliedvolatility"] / 100

# ✅ Filtres renforcés
# Filtres
df3 = df3[
    (df3["maturity"] > 14/365) &
    (df3["impliedvolatility"] > 0.01) &
    (df3["impliedvolatility"] < 2.0)
].dropna()

# Grille et interpolation
mat_grid = np.linspace(df3["maturity"].min(), df3["maturity"].max(), 60)
str_grid = np.linspace(df3["strike"].min(),   df3["strike"].max(),   60)
M, S = np.meshgrid(mat_grid, str_grid)

Z = griddata(
    (df3["maturity"], df3["strike"]),
    df3["impliedvolatility"],
    (M, S),
    method='linear'
)
Z = np.clip(Z, 0.01, 2.0)

# Graphique
fig = plt.figure(figsize=(13, 8))
ax  = fig.add_subplot(projection='3d')
surf = ax.plot_surface(M, S, Z, cmap='RdYlGn_r', alpha=0.92)
fig.colorbar(surf, ax=ax, shrink=0.5, label="Volatilité implicite")
ax.set_xlabel("Maturité (années)", labelpad=10)
ax.set_ylabel("Strike",            labelpad=10)
ax.set_zlabel("Volatilité implicite", labelpad=10)  # ✅ set_zlabel
ax.view_init(elev=25, azim=-60)
plt.title("Surface de volatilité implicite — Puts SPX (21/09/2023)")
plt.tight_layout()
plt.show()
df3["rendement"] = np.log(df3["close"] / df3["close"].shift(1))
df3["RV_21"] = (
    df3["rendement"]
    .rolling(window=21)
    .std()
    * np.sqrt(252)
)
horizon = 21

df3["RV_future"] = np.sqrt(
    252 *
    (
        df3["rendement"]**2
    )
    .shift(-1)
    .rolling(horizon)
    .mean()
)
plt.figure(figsize=(12,6))

plt.plot(df3["maturity"], df3["impliedvolatility"], label="Volatilité implicite")
plt.plot(df3["maturity"], df3["RV_future"], label="Volatilité réalisée")

plt.legend()
plt.xlabel("Date")
plt.ylabel("Volatilité")
plt.title("Volatilité implicite vs volatilité réalisée")
plt.show()
"""from scipy.optimize import curve_fit

S0=173.4

def svi(k, a, b, rho, m, sigma):
    #Raw SVI parametrization — Gatheral 2004
    return a + b * (rho*(k - m) + np.sqrt((k - m)**2 + sigma**2))

# log-moneyness
df3["logmoneyness"] = np.log(df3["strike"] / S0)  # S0 = prix SPX spot

# Fitter SVI pour chaque maturité
maturities = df3["maturity"].unique()
iv_fitted  = {}

for T in maturities:
    sub = df3[df3["maturity"] == T]
    try:
        popt, _ = curve_fit(
            svi, sub["logmoneyness"], sub["impliedvolatility"]**2,
            p0=[0.04, 0.1, -0.7, 0.0, 0.2],
            maxfev=10000
        )
        iv_fitted[T] = popt
    except:
        pass

from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
def breeden_litzenberger(strikes, call_prices, r, T):
    #Dérivée seconde par différences finies centrées
    dK = strikes[1] - strikes[0]
    d2C_dK2 = np.gradient(np.gradient(call_prices, dK), dK)
    density  = np.exp(r * T) * d2C_dK2
    # Normalisation
    density  = np.maximum(density, 0)          # densité positive
    density /= np.trapz(density, strikes)      # intégrale = 1
    return density

# Pour une maturité donnée T
T   = 0.25
r   = 0.05   # taux sans risque (Fed funds ~5% en sept 2023)
S0  = 4500 
# Afficher les maturités disponibles après le fit
print("Maturités fittées disponibles :")
print(sorted(iv_fitted.keys()))

# Choisir la plus proche de 0.25
T_cible = 0.25
T = min(iv_fitted.keys(), key=lambda x: abs(x - T_cible))
print(f"Maturité sélectionnée : {T:.6f} ans ({T*365:.1f} jours)")  # SPX spot approximatif

K_grid = np.linspace(2000, 7000, 500)
popt   = iv_fitted[T]
k_grid = np.log(K_grid / S0)

iv_svi  = np.sqrt(svi(k_grid, *popt))
calls   = np.array([black_scholes_call(S0, K, T, r, iv) 
                    for K, iv in zip(K_grid, iv_svi)])

density = breeden_litzenberger(K_grid, calls, r, T)

plt.plot(K_grid, density)
plt.xlabel("Strike K")
plt.ylabel("q(K)")
plt.title(f"Densité risk-neutre — Maturité {T:.2f} ans")
plt.axvline(S0, color='red', linestyle='--', label='ATM')
plt.legend()
plt.show()"""   

"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D  

# ── Black-Scholes ──────────────────────────────────────────────
def bs_call(S, K, T, r, sigma):
    sigma = max(sigma, 1e-8)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# ── SVI variance totale w(k) = sigma_BS^2 * T ─────────────────
def svi_var(k, a, b, rho, m, sigma):
    return a + b*(rho*(k-m) + np.sqrt((k-m)**2 + sigma**2))

# ── Fit SVI avec contraintes strictes ─────────────────────────
def fit_svi(logm, iv, T):
    w_market = iv**2 * T   # variance totale observée
    bounds = (
        [-0.5,  1e-4, -0.999, -1.0, 1e-4],
        [ 1.0,  2.0,   0.999,  1.0, 2.0 ]
    )
    best, best_err = None, np.inf
    # Plusieurs points de départ pour éviter minima locaux
    for p0 in [
        [0.04, 0.2, -0.7,  0.0, 0.3],
        [0.01, 0.1, -0.5, -0.1, 0.2],
        [0.05, 0.3, -0.9,  0.1, 0.1],
    ]:
        try:
            popt, _ = curve_fit(
                svi_var, logm, w_market,
                p0=p0, bounds=bounds, maxfev=20000
            )
            err = np.mean((svi_var(logm, *popt) - w_market)**2)
            if err < best_err:
                best, best_err = popt, err
        except:
            pass
    return best

# ── Refaire le fit sur toutes les maturités ───────────────────
r  = 0.05
S0 = 4500

# Grouper par maturité arrondie à 3 décimales
df3["maturity_r"] = df3["maturity"].round(3)
iv_fitted = {}

for T_r, sub in df3.groupby("maturity_r"):
    if len(sub) < 8:   # trop peu de points → ignorer
        continue
    logm = np.log(sub["strike"]/ S0)
    iv   = sub["impliedvolatility"].values
    popt = fit_svi(logm, iv, T_r)
    if popt is not None:
        iv_fitted[T_r] = popt

print(f"{len(iv_fitted)} maturités fittées")
print("Maturités :", sorted(iv_fitted.keys()))

# ── Sélection maturité + densité ──────────────────────────────
T_cible = 0.25
T = min(iv_fitted.keys(), key=lambda x: abs(x - T_cible))
print(f"\nMaturité utilisée : {T:.3f} ans ({T*365:.0f} jours)")

K_grid = np.linspace(2000, 7000, 1000)
k_grid = np.log(K_grid / S0)

# IV implicite depuis SVI
w_svi  = svi_var(k_grid, *iv_fitted[T])
w_svi  = np.maximum(w_svi, 1e-8)
iv_svi = np.sqrt(w_svi / T)

# ── Vérification visuelle du fit ──────────────────────────────
sub = df3[df3["maturity_r"] == T]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(np.log(sub["strike"]/S0), sub["impliedvolatility"],
                s=15, label="Données brutes", zorder=3)
axes[0].plot(k_grid, iv_svi, 'r-', label="SVI fitté")
axes[0].set_xlabel("Log-moneyness")
axes[0].set_ylabel("IV")
axes[0].set_title(f"Qualité du fit SVI — T={T*365:.0f}j")
axes[0].legend()

# ── Prix de calls et densité ──────────────────────────────────
calls = np.array([bs_call(S0, K, T, r, iv) for K, iv in zip(K_grid, iv_svi)])
plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.view_init(30, 60)  
ax.scatter(k_grid, iv_svi, iv, zdir='z', s=25,c='b', marker='^') 
ax.set_xlabel('strike')
ax.set_ylabel('time-to-maturity')
ax.set_zlabel('implied volatility'); 
dK      = K_grid[1] - K_grid[0]
d2C_dK2 = np.gradient(np.gradient(calls, dK), dK)
density = np.exp(r*T) * d2C_dK2
density = np.maximum(density, 0)
norme   = np.trapezoid(density, K_grid)
density /= norme

print(f"Intégrale densité = {norme:.4f} (doit être ≈ 1 avant normalisation)")

axes[1].plot(K_grid, density, color='darkgreen')
axes[1].axvline(S0, color='red', linestyle='--', label=f'ATM S0={S0}')
axes[1].fill_between(K_grid, density, alpha=0.2, color='green')
axes[1].set_title(f"Densité risk-neutre — T={T*365:.0f} jours")
axes[1].set_xlabel("Strike K")
axes[1].set_ylabel("q(K)")
axes[1].legend()

plt.tight_layout()
plt.show()"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import norm


# ============================================================
# 1. MODELE DE BLACK-SCHOLES
# ============================================================

def bs_call(S, K, T, r, sigma):
    sigma = max(float(sigma), 1e-8)

    if T <= 0:
        return max(S - K, 0.0)

    d1 = (
        np.log(S / K) + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    return (
        S * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )


# ============================================================
# 2. MODELE SVI : VARIANCE TOTALE
# ============================================================

def svi_var(k, a, b, rho, m, sigma):
    return a + b * (
        rho * (k - m)
        + np.sqrt((k - m)**2 + sigma**2)
    )


# ============================================================
# 3. CALIBRATION SVI
# ============================================================

def fit_svi(logm, iv, T):

    logm = np.asarray(logm, dtype=float)
    iv = np.asarray(iv, dtype=float)

    valid = (
        np.isfinite(logm)
        & np.isfinite(iv)
        & (iv > 0)
    )

    logm = logm[valid]
    iv = iv[valid]

    if len(logm) < 8 or T <= 0:
        return None

    w_market = iv**2 * T

    bounds = (
        [-0.5, 1e-4, -0.999, -1.0, 1e-4],
        [1.0, 2.0, 0.999, 1.0, 2.0]
    )

    initial_guesses = [
        [0.04, 0.2, -0.7, 0.0, 0.3],
        [0.01, 0.1, -0.5, -0.1, 0.2],
        [0.05, 0.3, -0.9, 0.1, 0.1],
    ]

    best = None
    best_err = np.inf

    for p0 in initial_guesses:
        try:
            popt, _ = curve_fit(
                svi_var,
                logm,
                w_market,
                p0=p0,
                bounds=bounds,
                maxfev=20000
            )

            fitted = svi_var(logm, *popt)
            err = np.mean((fitted - w_market)**2)

            if err < best_err:
                best = popt
                best_err = err

        except (RuntimeError, ValueError, FloatingPointError):
            continue

    return best


# ============================================================
# 4. PARAMETRES
# ============================================================

r = 0.05
S0 = 4500

T_cible = 0.25

# Vérifier que df3 existe et contient les colonnes requises.
required_columns = {
    "maturity",
    "strike",
    "impliedvolatility"
}

missing = required_columns - set(df3.columns)

if missing:
    raise ValueError(
        f"Colonnes manquantes dans df3 : {missing}"
    )

df3 = df3.copy()

df3["maturity_r"] = df3["maturity"].round(3)


# ============================================================
# 5. CALIBRATION SUR TOUTES LES MATURITES
# ============================================================

iv_fitted = {}

for T_r, sub in df3.groupby("maturity_r"):

    if len(sub) < 8:
        continue

    logm = np.log(
        sub["strike"].to_numpy(dtype=float) / S0
    )

    iv_market = sub["impliedvolatility"].to_numpy(
        dtype=float
    )

    popt = fit_svi(logm, iv_market, T_r)

    if popt is not None:
        iv_fitted[T_r] = popt


print(f"{len(iv_fitted)} maturités fittées")
print("Maturités :", sorted(iv_fitted.keys()))

if not iv_fitted:
    raise RuntimeError(
        "Aucune maturité n'a pu être calibrée."
    )


# ============================================================
# 6. SELECTION DE LA MATURITE
# ============================================================

T = min(
    iv_fitted.keys(),
    key=lambda x: abs(x - T_cible)
)

print(
    f"\nMaturité utilisée : {T:.3f} ans "
    f"({T*365:.0f} jours)"
)


# ============================================================
# 7. CONSTRUCTION DE LA GRILLE DE STRIKES
# ============================================================

K_grid = np.linspace(2000, 7000, 1000)

k_grid = np.log(K_grid / S0)


# ============================================================
# 8. VOLATILITE IMPLICITE ISSUE DE SVI
# ============================================================

popt = iv_fitted[T]

w_svi = svi_var(k_grid, *popt)

w_svi = np.maximum(w_svi, 1e-8)

iv_svi = np.sqrt(w_svi / T)


# ============================================================
# 9. DONNEES DE MARCHE POUR LA MATURITE RETENUE
# ============================================================

sub = df3[df3["maturity_r"] == T].copy()

logm_market = np.log(
    sub["strike"].to_numpy(dtype=float) / S0
)

iv_market = sub["impliedvolatility"].to_numpy(
    dtype=float
)


# ============================================================
# 10. PRIX DES CALLS BLACK-SCHOLES
# ============================================================

calls = np.array([
    bs_call(S0, K, T, r, sigma)
    for K, sigma in zip(K_grid, iv_svi)
])


# ============================================================
# 11. DENSITE RISQUE-NEUTRE
# ============================================================

dK = K_grid[1] - K_grid[0]

# Dérivée seconde du prix du call par rapport au strike.
dC_dK = np.gradient(calls, dK, edge_order=2)

d2C_dK2 = np.gradient(dC_dK, dK, edge_order=2)

# Relation de Breeden-Litzenberger.
density_raw = np.exp(r * T) * d2C_dK2

# Conserver la partie positive pour une visualisation.
density = np.maximum(density_raw, 0.0)

# Intégrale avant normalisation.
norme = np.trapezoid(density, K_grid)

print(
    f"Intégrale de la densité avant normalisation : "
    f"{norme:.6f}"
)

if not np.isfinite(norme) or norme <= 0:
    raise RuntimeError(
        "La densité obtenue est invalide. "
        "Vérifie la surface SVI et les prix des calls."
    )

# Normalisation pour obtenir une intégrale égale à 1.
density /= norme


# ============================================================
# 12. VISUALISATION
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(15, 5)
)


# ------------------------------------------------------------
# GRAPHIQUE 1 : FIT SVI
# ------------------------------------------------------------

axes[0].scatter(
    logm_market,
    iv_market,
    s=20,
    label="Données de marché",
    color="blue",
    alpha=0.7,
    zorder=3
)

axes[0].plot(
    k_grid,
    iv_svi,
    color="red",
    linewidth=2,
    label="SVI calibré"
)

axes[0].set_xlabel("Log-moneyness")
axes[0].set_ylabel("Volatilité implicite")

axes[0].set_title(
    f"Qualité du fit SVI — T={T*365:.0f} jours"
)

axes[0].legend()
axes[0].grid(alpha=0.25)


# ------------------------------------------------------------
# GRAPHIQUE 2 : DENSITE RISQUE-NEUTRE
# ------------------------------------------------------------

axes[1].plot(
    K_grid,
    density,
    color="darkgreen",
    linewidth=2
)

axes[1].axvline(
    S0,
    color="red",
    linestyle="--",
    label=f"Spot S0 = {S0}"
)

axes[1].fill_between(
    K_grid,
    density,
    alpha=0.2,
    color="green"
)

axes[1].set_title(
    f"Densité risque-neutre — T={T*365:.0f} jours"
)

axes[1].set_xlabel("Strike K")
axes[1].set_ylabel("Densité q(K)")

axes[1].legend()
axes[1].grid(alpha=0.25)


plt.tight_layout()
plt.show()