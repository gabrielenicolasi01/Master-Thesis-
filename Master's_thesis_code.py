#librerie e pacchetti
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp

from statsmodels.distributions.empirical_distribution import ECDF 

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

#############################################################################################
            #FUNZIONI DI s^-, s^+, Xi(S^-,K), Xi(S^+,L) V e W e le varie derivate
#############################################################################################


def s_minus(data, Omega):
    arr = np.asarray(data)
    return np.mean(np.maximum(Omega - arr, 0.0))

def lambda_from_s_empirical(s_empirical, s_baseline, eps=0):
    return s_empirical / (s_baseline + eps)

def xi_K(data, lambda_s, Omega, K):
    arr = np.asarray(data)
    scaled_data = Omega + lambda_s * (arr - Omega)
    x_sorted = np.sort(scaled_data)
    n = len(x_sorted)
    F_K = np.sum(x_sorted <= K) / n

    mask = x_sorted <= K
    x_below = x_sorted[mask]
    m = len(x_below)

    if m == 0:
        integral = 0.0
    elif m == 1:
        integral = 0.0
    else:
        widths = np.diff(x_below)
        heights = np.arange(1, m) / n
        integral = np.sum(widths * heights)
    if m > 0:
        integral += (K - x_below[-1]) * (m / n)
    return (Omega - K) * F_K + integral

def xi_K_on_baseline(baseline_data, lambda_s, Omega, K):
    return xi_K(baseline_data, lambda_s, Omega, K)

def vega_tail_baseline(baseline_data, s_target, s_baseline, Omega, K, Delta_s=1e-4):
    lam_plus  = lambda_from_s_empirical(s_target + Delta_s, s_baseline)
    lam_minus = lambda_from_s_empirical(max(s_target - Delta_s, 0.0), s_baseline)
    xi_plus  = xi_K_on_baseline(baseline_data, lam_plus, Omega, K)
    xi_minus = xi_K_on_baseline(baseline_data, lam_minus, Omega, K)
    V = (xi_plus - xi_minus) / (2 * Delta_s)
    return V

def derivata_vega_s_minus(data, s_minus_target, s_baseline, Omega, K, Delta_s):
    lam_plus  = lambda_from_s_empirical(s_minus_target + Delta_s, s_baseline)
    lam_0     = lambda_from_s_empirical(s_minus_target, s_baseline)
    lam_minus = lambda_from_s_empirical(max(s_minus_target - Delta_s, 0.0), s_baseline)
    xi_plus  = xi_K_on_baseline(data, lam_plus, Omega, K)
    xi_0     = xi_K_on_baseline(data, lam_0, Omega, K)
    xi_minus = xi_K_on_baseline(data, lam_minus, Omega, K)
    sec = (xi_plus - 2.0 * xi_0 + xi_minus) / (Delta_s ** 2)
    return sec

def derivata_vega_k(data, s_minus_target, Omega, K, Delta_K=1e-4):
    V_plus = vega_tail_baseline(data, s_minus_target, s_minus_target, Omega, K + Delta_K, Delta_K)
    V_minus = vega_tail_baseline(data, s_minus_target, s_minus_target, Omega, K - Delta_K, Delta_K)
    derivata = (V_plus - V_minus) / (2 * Delta_K)
    return derivata

def calculate_vega_vs_k_values(data, s_minus_target, Omega, K_vals, Delta_s=1e-4):
    vega_vals = []
    for K in K_vals:
        vega = vega_tail_baseline(data, s_minus_target, s_minus_target, Omega, K, Delta_s)
        vega_vals.append(vega)
    return np.array(vega_vals)

def integrated_vega_vs_k_discreto(data, s_minus_target, Omega, K_min, K_max, Delta_s=1e-4, n_points=500):
    K_vals = np.linspace(K_min, K_max, n_points)
    vega_vals = calculate_vega_vs_k_values(data, s_minus_target, Omega, K_vals, Delta_s)
    integral = np.trapz(vega_vals, K_vals)
    return integral

def calculate_vega_vs_s_minus(data, s_minus_vals, Omega, K, Delta_s=1e-4):
    vega_vals = []
    for s_minus in s_minus_vals:
        vega = vega_tail_baseline(data, s_minus, s_minus, Omega, K, Delta_s)
        vega_vals.append(vega)
    return np.array(vega_vals)

def integrated_vega_vs_s_minus(data, Omega, K, s_min, s_max, Delta_s=1e-4, n_points=500):
    s_minus_vals = np.linspace(s_min, s_max, n_points)
    vega_vals = calculate_vega_vs_s_minus(data, s_minus_vals, Omega, K, Delta_s)
    integral = np.trapz(vega_vals, s_minus_vals)
    return integral

def s_plus(data, Omega):
    arr = np.asarray(data)
    return np.mean(np.maximum(arr - Omega, 0.0))

def lambda_from_s_plus_empirical(s_empirical, s_baseline, eps=0):
    return s_empirical / (s_baseline + eps)

def xi_plus(data, lambda_s, Omega, L, H):
    data = np.asarray(data)
    scaled_data = Omega + lambda_s * (data - Omega)
    x_sorted = np.sort(scaled_data)
    ecdf = ECDF(x_sorted)

    mask = (x_sorted >= L) & (x_sorted <= H)
    x_in_range = x_sorted[mask]
    
    if len(x_in_range) == 0:
        return 0.0
    
    surv = 1 - ecdf(x_in_range)
    
    diff = np.diff(x_in_range)

    heights = surv[:-1]
    
    integral = np.sum(diff * (x_in_range[:-1] - Omega) * heights)
    
    return integral

def xi_plus_on_baseline(baseline_data, lambda_s, Omega, L, H):
    return xi_plus(baseline_data, lambda_s, Omega, L, H)
   
def w_vega_right_baseline(baseline_data, s_target, s_baseline, Omega, L, H, Delta_s=1e-4):
    lam_plus = lambda_from_s_plus_empirical(s_target + Delta_s, s_baseline)
    lam_minus = lambda_from_s_plus_empirical(max(s_target - Delta_s, 0.0), s_baseline)
    xi_plus_val = xi_plus_on_baseline(baseline_data, lam_plus, Omega, L, H)
    xi_minus_val = xi_plus_on_baseline(baseline_data, lam_minus, Omega, L, H)
    W = (xi_plus_val - xi_minus_val) / (2 * Delta_s)
    return W


#############################################################################################
            #FUNZIONI PER LE METRICHE DI RISCHIO STANDARD
#############################################################################################

# Max drawdown
def max_drawdown(x):
    cum_returns = (1 + x).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()

# Value at Risk storico 5%
def var_5(x, alpha=0.05):
    return np.quantile(x, alpha)

# CVaR (Expected Shortfall) 
def cvar_historical(returns, alpha=0.05):
    x = returns.dropna().values
    if len(x) == 0:
        return np.nan
    q = np.quantile(x, alpha)
    tail = x[x <= q]
    if len(tail) == 0:
        return np.nan
    return tail.mean() 

#############################################################################################
            #FUNZIONI PER L'OTTIMIZZAZIONE DEL PORTAFOGLIO 
#############################################################################################

# Funzioni per ottimizzazione del portafoglio
def solve_portfolio(constraints_func, objective_func):
    w = cp.Variable(n_assets)
    prob = cp.Problem(objective_func(w), constraints_func(w))
    prob.solve(solver=cp.SCS)
    if w.value is None:
        print("Problema non risolto. Pesatura = NaN")
        return pd.Series(np.nan, index=assets_returns.columns)
    w_val = np.array(w.value)
    w_val = w_val / w_val.sum()  # normalizza per sicurezza
    return pd.Series(w.value * risk_asset_weight, index=assets_returns.columns)

#Minima varianza
def objective_minvar(w): return cp.Minimize(cp.quad_form(w, Sigma))
def constraints_minvar(w): return [cp.sum(w) == 1, w >= min_weight_per_asset]

#############################################################################################
            #Scaricamento e analisi dei dati
#############################################################################################

# Scarica dati settimanali S&P 500 
ticker = yf.Ticker("^GSPC") #modificare il ticker se si vuole cambiare asset
data = ticker.history(start="1950-01-01", end="2025-01-01", interval="1wk") #periodo temporle e rendimenti settimanali
close = data['Close']

# Calcola log returns
log_ret = np.log(close / close.shift(1)).dropna()

# Parametri Put
strike_ratio = 0.97  # Modificato a 1 per una protezione completa

premium_ratio = 0.0002  # Modifica qui se vuoi includere un premio

# Calcolo per log rendimenti portafoglio coperto
strike = strike_ratio * close
premium_put = premium_ratio * close

payoff_put = np.maximum(strike - close.shift(-1), 0)
final_with_put = close.shift(-1) + payoff_put - premium_put
final_with_put = final_with_put.dropna()

log_ret_with_put = np.log(final_with_put / close.iloc[:-1].values)
log_ret_with_put = pd.Series(log_ret_with_put, index=final_with_put.index)

# Stress Test
#calcolo log rendimenti dopo stress test
extreme_moves = [-0.05, -0.3, -0.2, -0.1, -0.2, -0.3,-0.05, -0.3, -0.2, -0.1, -0.2, -0.3]
last_price = close.iloc[-1]
new_prices = [last_price * (1 + extreme_moves[0])]
for r in extreme_moves[1:]:
    new_prices.append(new_prices[-1] * (1 + r))
new_dates = [data.index[-1] + pd.Timedelta(weeks=i) for i in range(1, len(new_prices) + 1)]
data_ext = pd.DataFrame({'Close': new_prices}, index=new_dates)
data_stress = pd.concat([data, data_ext])

close_stress = data_stress['Close']
log_ret_stress = np.log(close_stress / close_stress.shift(1)).dropna()

strike_stress = strike_ratio * close_stress
payoff_put_stress = np.maximum(strike_stress - close_stress.shift(-1), 0)
premium_put_stress = premium_ratio * close_stress

final_with_put_stress = close_stress.shift(-1) + payoff_put_stress - premium_put_stress
final_with_put_stress = final_with_put_stress.dropna()
log_ret_with_put_stress = np.log(final_with_put_stress / close_stress.iloc[:-1].values)
log_ret_with_put_stress = pd.Series(log_ret_with_put_stress, index=final_with_put_stress.index)

print(log_ret_with_put_stress)


# Funzione sintetica per metriche
def stats(series):
    return {
        'Mean': series.mean(),
        'Std': series.std(),
        'Variance': series.var(),
        'Quantile_0.05': series.quantile(0.05)
    }

# Crea DataFrame riepilogativo
df_summ = pd.DataFrame([
    stats(log_ret),
    stats(log_ret_with_put),
    stats(log_ret_stress),
    stats(log_ret_with_put_stress)
],
    index=['Orig_NonProt', 'Orig_Put', 'Stress_NonProt', 'Stress_Put']
)

print(df_summ)


# Gafico delle distribuzioni empiriche
def plot_pdf_cdf(returns, title_pdf, title_cdf):
    x = np.sort(returns)
    ecdf = ECDF(returns)
    y = ecdf(x)

    Omega = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # PDF (istogramma normalizzato)
    axes[0].hist(returns, bins=50, density=True, alpha=0.6,
                 color='steelblue', edgecolor="black")
    axes[0].set_title(title_pdf)
    axes[0].set_xlabel("Rendimento logaritmico")
    axes[0].set_ylabel("Densità")

    # CDF (ECDF)
    axes[1].step(x, y, where="post", color="blue")
    axes[1].set_title(title_cdf)
    axes[1].set_xlabel("Rendimento logaritmico")
    axes[1].set_ylabel("P(R ≤ r)")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Evidenzia area sotto Omega = 0 nella CDF usando fill_between
    mask = x <= Omega
    if np.any(mask):
        # Per fill_between, estrai x e y fino a Omega inclusa
        x_fill = np.concatenate((x[mask], [Omega]))
        y_fill = np.concatenate((y[mask], [y[mask][-1]]))
        axes[1].fill_between(x_fill, 0, y_fill, color='lightblue', alpha=0.3)

    axes[1].axvline(Omega, color='black', linestyle='--')

    plt.tight_layout()
    plt.show()

# Visualizza tutti i portafogli 
plot_pdf_cdf(log_ret, "PDF portafoglio passivo non protetto", "CDF portafoglio passivo non protetto")
plot_pdf_cdf(log_ret_with_put, "PDF portafoglio coperto", "CDF portafoglio coperto")
plot_pdf_cdf(log_ret_stress, "PDF con stress test ", "CDF con stress test ")
plot_pdf_cdf(log_ret_with_put_stress, "PDF con stress test portafoglio coperto", "CDF con stress test portafoglio coperto")

#############################################################################################
            #Risultati analisi di V 
#############################################################################################

#Calcolo delle funzioni 
Omega = 0
Delta_s = 1e-3

# Calcolo dei quantili K specifici
K_orig = -0.032 #modificare questo K per replicare i risultati

# Calcolo s^- per ogni distribuzione
s_orig = s_minus(log_ret, Omega)
s_put_orig = s_minus(log_ret_with_put, Omega)
s_nohedge = s_minus(log_ret_stress, Omega)
s_put_stress = s_minus(log_ret_with_put_stress, Omega)

# Calcolo lambda senza baseline comune
lambda_orig = lambda_from_s_empirical(s_orig, s_orig)
lambda_put_orig = lambda_from_s_empirical(s_put_orig, s_orig)
lambda_nohedge = lambda_from_s_empirical(s_nohedge, s_orig)
lambda_put_stress = lambda_from_s_empirical(s_put_stress, s_orig)

# Calcolo xi_K e Vega con K specifici e lambda personalizzati
xi_orig = xi_K_on_baseline(log_ret, lambda_orig, Omega, K_orig)
V_orig = vega_tail_baseline(log_ret, s_orig, s_orig, Omega, K_orig, Delta_s)

xi_put_orig = xi_K_on_baseline(log_ret_with_put, lambda_put_orig, Omega, K_orig)
V_put_orig = vega_tail_baseline(log_ret_with_put, s_put_orig, s_orig, Omega, K_orig, Delta_s)

xi_nohedge = xi_K_on_baseline(log_ret_stress, lambda_nohedge, Omega, K_orig)
V_nohedge = vega_tail_baseline(log_ret_stress, s_nohedge,s_orig, Omega, K_orig, Delta_s)

xi_put_stress = xi_K_on_baseline(log_ret_with_put_stress, lambda_put_stress, Omega, K_orig)
V_put_stress = vega_tail_baseline(log_ret_with_put_stress, s_put_stress, s_orig, Omega, K_orig, Delta_s)

# Creazione DataFrame riepilogativo
df_metriche_nobase = pd.DataFrame({
    'Portafoglio': [
        'Originale non protetto',
        'Originale con Put',
        'Stress test non protetto',
        'Stress test con Put'
    ],
    'Semi-deviazione s^-': [
        s_orig,
        s_put_orig,
        s_nohedge,
        s_put_stress
    ],
    'Lambda': [
        lambda_orig,
        lambda_put_orig,
        lambda_nohedge,
        lambda_put_stress
    ],
    'Xi(K, s^-)': [
        xi_orig,
        xi_put_orig,
        xi_nohedge,
        xi_put_stress
    ],
    'Vega': [
        V_orig,
        V_put_orig,
        V_nohedge,
        V_put_stress
    ]
})


print("Quantile 0.05 per la il portafoglio senza put", K_orig)
print("")
with pd.option_context('display.float_format', '{:,.6f}'.format):
    print(df_metriche_nobase.to_string(index=False))

#############################################################################################
            #Grafici per il lato sinistro
#############################################################################################

x = np.sort(log_ret.values)
ecdf = ECDF(x)
y = ecdf(x)

Omega = 0.0
K = np.quantile(x, 0.05)

# Maschera per lato sinistro
mask = x <= Omega

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Primo grafico: CDF completa e troncata (lato sinistro)
mask_K = x <= K
y_cdf = y
y_truncated = np.minimum(y_cdf, y_cdf[np.searchsorted(x, K)])

axs[0].step(x[mask], y_cdf[mask], where='post', color='blue', label=r"CDF completa $F_\lambda(x)$")
axs[0].step(x[mask], y_truncated[mask], where='post', color='red', label=r"CDF troncata $F^K_\lambda(x)$")
axs[0].axvline(Omega, color='black', linestyle='--')
axs[0].axvline(K, color='black', linestyle='--')

ylim_min, ylim_max = axs[0].get_ylim()
y_pos = ylim_min - 0.05 * (ylim_max - ylim_min)
axs[0].text(Omega, y_pos, r"$\Omega$", ha='center', va='top', fontsize=12)
axs[0].text(K, y_pos, r"$K$", ha='center', va='top', fontsize=12)

axs[0].set_xlabel("Rendimento logaritmico")
axs[0].set_ylabel("P(R ≤ r)")
axs[0].set_title("CDF completa e troncata")
axs[0].legend()
axs[0].grid(True, linestyle="--", alpha=0.5)

# Secondo grafico: CDF scalata e troncata per diversi λ
lambdas = [lambda_put_orig,lambda_orig, lambda_nohedge]
colors = ['red', 'blue', 'orange']

for lam, c in zip(lambdas, colors):
    scaled_data = Omega + lam * (x - Omega)
    x_scaled = np.sort(scaled_data)
    ecdf_scaled = ECDF(x_scaled)
    y_scaled = ecdf_scaled(x_scaled)

    mask_Omega = x_scaled <= Omega
    F_K = ecdf_scaled(K)
    y_truncated_scaled = np.minimum(y_scaled, F_K)

    axs[1].step(x_scaled[mask_Omega], y_scaled[mask_Omega], where='post', linestyle='--', color=c, alpha=0.7, label=f"CDF scalata λ={lam.round(2)}")
    axs[1].step(x_scaled[mask_Omega], y_truncated_scaled[mask_Omega], where='post', color=c, )

axs[1].axvline(Omega, color='black', linestyle='--')
axs[1].axvline(K, color='black', linestyle='--')

axs[1].set_xlabel("Rendimento logaritmico")
axs[1].set_ylabel("P(R ≤ r)")
axs[1].set_title("Effetto di λ sulla CDF troncata e ξ(K, s^-)")
axs[1].legend()
axs[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()


#grafici per variazioni di omega
# Parametri di input
Omegas = np.linspace(-0.1, 0.1, 500)    # Range Omegas
K = log_ret.quantile(0.05)              # Soglia K
Delta_s = 1e-3
Omega_current = 0.0

# Liste per raccolta risultati
s_list, lambda_list, xi_list, vega_list = [], [], [], []

for Om in Omegas:
    s = s_minus(log_ret, Om)
    lam = lambda_from_s_empirical(s, s_minus(log_ret, Omega_current))
    xi  = xi_K_on_baseline(log_ret, lam, Om, K)
    V   = vega_tail_baseline(log_ret, s, s_minus(log_ret, Omega_current), Om, K, Delta_s)
    
    s_list.append(s)
    lambda_list.append(lam)
    xi_list.append(xi)
    vega_list.append(V)

# Converti in array per facilità
s_array      = np.array(s_list)
lambda_array = np.array(lambda_list)
xi_array     = np.array(xi_list)
vega_array   = np.array(vega_list)

# Creazione grafici
fig, axs = plt.subplots(2, 2, figsize=(12, 6))

# Semi-deviazione s^- vs Omega
axs[0,0].plot(Omegas, s_array)
axs[0,0].axvline(Omega_current, color='red', linestyle='--', label=r"$\Omega$ attuale")
axs[0,0].set_title("Semi-deviazione s⁻ vs Ω")
axs[0,0].set_xlabel("Ω")
axs[0,0].set_ylabel("s⁻")
axs[0,0].legend()
axs[0,0].grid(True, linestyle='--', alpha=0.5)

# Lambda vs Omega
axs[0,1].plot(Omegas, lambda_array, color = 'orange')
axs[0,1].axvline(Omega_current, color='red', linestyle='--', label=r"$\Omega$ attuale")
axs[0,1].set_title("λ vs Ω")
axs[0,1].set_xlabel("Ω")
axs[0,1].set_ylabel("λ")
axs[0,1].legend()
axs[0,1].grid(True, linestyle='--', alpha=0.5)

# Xi vs Omega
axs[1,0].plot(Omegas, xi_array, color='blue')
axs[1,0].axvline(Omega_current, color='red', linestyle='--', label=r"$\Omega$ attuale")
axs[1,0].set_title(f"ξ(K,s⁻) vs Ω  (K={K:.3f})")
axs[1,0].set_xlabel("Ω")
axs[1,0].set_ylabel("ξ")
axs[1,0].legend()
axs[1,0].grid(True, linestyle='--', alpha=0.5)

# Vega vs Omega
axs[1,1].plot(Omegas, vega_array, color='purple')
axs[1,1].axvline(Omega_current, color='red', linestyle='--', label=r"$\Omega$ attuale")
axs[1,1].set_title(f"Vega vs Ω  (K={K:.3f})")
axs[1,1].set_xlabel("Ω")
axs[1,1].set_ylabel("Vega")
axs[1,1].legend()
axs[1,1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

#grafici per variazioni di K
# Parametri dati
Omega = 0.0
Ks = np.linspace(-0.3, 0.0, 500)  # Range per K
K_current = log_ret.quantile(0.05)  # K attuale fisso
Delta_s = 1e-3

# Liste per raccogliere risultati
s_list, xi_list, vega_list = [], [], []

for Ki in Ks:
    s = s_minus(log_ret, Omega)  # semi-deviazione sinistra fissa rispetto a Omega
    lam = lambda_from_s_empirical(s, s_minus(log_ret, Omega))  # lambda calcolato rispetto baseline fissa
    xi = xi_K_on_baseline(log_ret, lam, Omega, Ki)
    V = vega_tail_baseline(log_ret, s, s_minus(log_ret, Omega), Omega, Ki, Delta_s)

    s_list.append(s)
    xi_list.append(xi)
    vega_list.append(V)

# Conversione in array per plot
xi_array = np.array(xi_list)
vega_array = np.array(vega_list)

#cambiare questi per visualizzare atri K nei grafici
K_1 = -0.02
K_2 = -0.03
K_3 = log_ret.quantile(0.05)

# Creazione figura e subplot
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Grafico 1: ξ(K,s⁻) vs K
axs[0].plot(Ks, xi_array, color='blue', lw=2)
axs[0].axvline(K_1, color='red', linestyle='--', lw=2, label='K = -0.02')
axs[0].axvline(K_2, color='red', linestyle='--', lw=2, label='K = -0.03')
axs[0].axvline(K_3, color='red', linestyle='--', lw=2, label='K = -0.032')
axs[0].set_title(r'$\xi(K,s^-)$ vs K', fontsize=14)
axs[0].set_xlabel('K', fontsize=12)
axs[0].set_ylabel(r'$\xi$', fontsize=12)
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.5)

# Grafico 2: Vega vs K
axs[1].plot(Ks, vega_array, color='purple', lw=2)
axs[1].axvline(K_1, color='red', linestyle='--', lw=2, label='K = -0.02')
axs[1].axvline(K_2, color='red', linestyle='--', lw=2, label='K = -0.03')
axs[1].axvline(K_3, color='red', linestyle='--', lw=2, label='K = -0.032')
axs[1].set_title('Vega vs K', fontsize=14)
axs[1].set_xlabel('K', fontsize=12)
axs[1].set_ylabel('Vega', fontsize=12)
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

#Analisi di Xi e V rispetto a variare di K per i portafogli coperto

Omega = 0.0
Ks = np.linspace(-0.3, 0.0, 5000)   # Range per K
K_current = -0.02 # K attuale fisso
Delta_s = 1e-3

baseline_data = log_ret   # oppure: log_ret_put

s_list, xi_list, vega_list = [], [], []

for Ki in Ks:
    # semi-deviazione sinistra fissa rispetto a Omega
    s = s_minus(log_ret_with_put, Omega)

    # lambda calcolato rispetto alla baseline
    lam = lambda_from_s_empirical(s, s_minus(baseline_data, Omega))

    # calcolo di xi e vega sulla baseline
    xi = xi_K_on_baseline(log_ret_with_put, lam, Omega, Ki)
    V = vega_tail_baseline(log_ret_with_put, s, s_minus(baseline_data, Omega), Omega, Ki, Delta_s)

    # accumulo dei risultati
    s_list.append(s)
    xi_list.append(xi)
    vega_list.append(V)


xi_array = np.array(xi_list)
vega_array = np.array(vega_list)


K_1 = -0.02
K_2 = -0.03
K_3 = log_ret.quantile(0.05)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

axs[0].plot(Ks, xi_array, color='blue', lw=2)
axs[0].axvline(K_1, color='red', linestyle='--', lw=2, label='K = -0.02')
axs[0].axvline(K_2, color='red', linestyle='--', lw=2, label='K = -0.03')
axs[0].axvline(K_3, color='red', linestyle='--', lw=2, label='K = -0.032')
axs[0].set_title(r'$\xi(K,s^-)$ vs K', fontsize=14)
axs[0].set_xlabel('K', fontsize=12)
axs[0].set_ylabel(r'$\xi$', fontsize=12)
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.5)

axs[1].plot(Ks, vega_array, color='purple', lw=2)
axs[1].axvline(K_1, color='red', linestyle='--', lw=2, label='K = -0.02')
axs[1].axvline(K_2, color='red', linestyle='--', lw=2, label='K = -0.03')
axs[1].axvline(K_3, color='red', linestyle='--', lw=2, label='K = -0.032')
axs[1].set_title('Vega vs K', fontsize=14)
axs[1].set_xlabel('K', fontsize=12)
axs[1].set_ylabel('Vega', fontsize=12)
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

K_targets = [-0.02, -0.03,-0.032]
for Kt in K_targets:
    idx = np.argmin(np.abs(Ks - Kt))  # indice del valore di K più vicino
    print(f"K = {Ks[idx]:.3f} → Vega = {vega_array[idx]:.6f}")

#grafici per variazioni di s^-
# Parametri base
Omega = 0.0
Delta_s = 1e-3
K = log_ret.quantile(0.05)

# Calcolo s_left di riferimento (baseline semi-dev sinistra)
s_left = s_minus(log_ret, Omega)

# Variazione di s^-
s_minus_vals = np.linspace(s_left * 0.1, s_left * 3, 500)

lambda_vals = []
xi_vals = []
vega_vals = []

for s_val in s_minus_vals:
    lam = lambda_from_s_empirical(s_val, s_left)  # Calcola lambda rispetto baseline s_left
    lambda_vals.append(lam)

    xi = xi_K(log_ret, lam, Omega, K)
    xi_vals.append(xi)

    vega = vega_tail_baseline(log_ret, s_val, s_left, Omega, K, Delta_s)
    vega_vals.append(vega)

# Grafico 1: s^- e lambda
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

axs[0].plot(s_minus_vals, s_minus_vals, label=r'$s^-$')
axs[0].set_title('Andamento di $s^-$')
axs[0].set_xlabel('Indice')
axs[0].set_ylabel(r'$s^-$')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(s_minus_vals, lambda_vals, color='orange', label=r'$\lambda$')
axs[1].set_title('Andamento di $\lambda$')
axs[1].set_xlabel(r'$s^-$')
axs[1].set_ylabel(r'$\lambda$')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()

# Grafico 2: Xi vs s^-
plt.figure(figsize=(12, 4))
plt.plot(s_minus_vals, xi_vals, label=r'$\xi$ vs $s^-$', color='blue')
plt.xlabel(r'$s^-$')
plt.ylabel(r'$\xi$')
plt.title(r'Andamento di $\xi$ rispetto a $s^-$')
plt.legend()
plt.grid(True)
plt.show()


# Grafici dei Vega rispetto a s^-
Delta_s = 1e-3
s_minus_vals = np.linspace(s_orig * 0.1, s_orig * 3, 500)

vega_vals_orig = [vega_tail_baseline(log_ret, s_val, s_orig, Omega, K_orig, Delta_s) for s_val in s_minus_vals]
vega_vals_put = [vega_tail_baseline(log_ret_with_put, s_val, s_orig, Omega, K_orig, Delta_s) for s_val in s_minus_vals]
vega_vals_stress = [vega_tail_baseline(log_ret_stress, s_val, s_orig, Omega, K_orig, Delta_s) for s_val in s_minus_vals]
vega_vals_stress_put = [vega_tail_baseline(log_ret_with_put_stress, s_val, s_orig, Omega, K_orig, Delta_s) for s_val in s_minus_vals]

fig, axs = plt.subplots(2, 2, figsize=(12, 5), sharex=True)

# Colori comuni
vega_color = 'purple'
line_color = 'red'

axs[0, 0].plot(s_minus_vals, vega_vals_orig, color=vega_color)
axs[0, 0].axvline(s_orig, color=line_color, linestyle='--')
axs[0, 0].set_title('Portafoglio passivo')
axs[0, 0].set_ylabel('Vega')
axs[0, 0].grid(True)

axs[0, 1].plot(s_minus_vals, vega_vals_put, color=vega_color)
axs[0, 1].axvline(s_put_orig, color=line_color, linestyle='--')
axs[0, 1].set_title('Portafoglio coperto')
axs[0, 1].set_ylabel('Vega')
axs[0, 1].grid(True)

axs[1, 0].plot(s_minus_vals, vega_vals_stress, color=vega_color)
axs[1, 0].axvline(s_nohedge, color=line_color, linestyle='--')
axs[1, 0].set_title('Stress test su portafoglio passivo')
axs[1, 0].set_xlabel(r'$s^-$ (Semi-deviazione Sinistra)')
axs[1, 0].set_ylabel('Vega')
axs[1, 0].grid(True)

axs[1, 1].plot(s_minus_vals, vega_vals_stress_put, color=vega_color)
axs[1, 1].axvline(s_put_stress, color=line_color, linestyle='--')
axs[1, 1].set_title('Stress test su portafoglio coperto')
axs[1, 1].set_xlabel(r'$s^-$ (Semi-deviazione Sinistra)')
axs[1, 1].set_ylabel('Vega')
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

#grafici della fragilità di secondo ordine
Delta_s = 1e-3

# Intervalli s^- specifici per ciascun caso
s_minus_vals = np.linspace(s_orig * 0.5, s_orig * 8, 500)

# Calcolo derivate di Vega per ciascun caso
deriv_vega_vals_orig = [derivata_vega_s_minus(log_ret, s,s_orig, Omega, K_orig, Delta_s) for s in s_minus_vals]
deriv_vega_vals_put = [derivata_vega_s_minus(log_ret_with_put, s,s_orig, Omega, K_orig, Delta_s) for s in s_minus_vals]
deriv_vega_vals_stress = [derivata_vega_s_minus(log_ret_stress, s,s_orig, Omega, K_orig, Delta_s) for s in s_minus_vals]
deriv_vega_vals_stress_put = [derivata_vega_s_minus(log_ret_with_put_stress, s,s_orig, Omega, K_orig, Delta_s) for s in s_minus_vals]

# Creazione figura 2x2
fig, axs = plt.subplots(2, 2, figsize=(12, 5), sharex=False)

# Colori
vega_color = 'purple'
line_color = 'red'

# 1. Originale non protetto
axs[0, 0].plot(s_minus_vals, deriv_vega_vals_orig, color=vega_color)
axs[0, 0].axvline(s_orig, color=line_color, linestyle='--')
axs[0, 0].set_title('Derivata Vega - Portafoglio passivo')
axs[0, 0].set_ylabel('Derivata di Vega')
axs[0, 0].grid(True)

# 2. Originale con Put
axs[0, 1].plot(s_minus_vals, deriv_vega_vals_put, color=vega_color)
axs[0, 1].axvline(s_put_orig, color=line_color, linestyle='--')
axs[0, 1].set_title('Derivata Vega - Portafoglio coperto')
axs[0, 1].set_ylabel('Derivata di Vega')
axs[0, 1].grid(True)

# 3. Stress test non protetto
axs[1, 0].plot(s_minus_vals, deriv_vega_vals_stress, color=vega_color)
axs[1, 0].axvline(s_nohedge, color=line_color, linestyle='--')
axs[1, 0].set_title('Derivata Vega - stress test su portafoglio passivo')
axs[1, 0].set_xlabel(r'$s^-$ (Semi-deviazione Sinistra)')
axs[1, 0].set_ylabel('Derivata di Vega')
axs[1, 0].grid(True)

# 4. Stress test con Put
axs[1, 1].plot(s_minus_vals, deriv_vega_vals_stress_put, color=vega_color)
axs[1, 1].axvline(s_put_stress, color=line_color, linestyle='--')
axs[1, 1].set_title('Derivata Vega - stress test su portafoglio coperto')
axs[1, 1].set_xlabel(r'$s^-$ (Semi-deviazione Sinistra)')
axs[1, 1].set_ylabel('Derivata di Vega')
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

#grafici del fragility drift
# Parametri di base
Omega = 0.0
Delta_s = 1e-3
Delta_K = Delta_s
s_orig = s_minus(log_ret, Omega)
K = -0.0318

# Intervalli di variazione

K_vals = np.linspace(-0.06, 0, 500)

# Calcolo derivate di Vega per ogni valore nell'intervallo

deriv_vega_K_vals_orig = [derivata_vega_k(log_ret, s_orig, Omega, k, Delta_K) for k in K_vals]
deriv_vega_K_vals_put = [derivata_vega_k(log_ret_with_put, s_orig, Omega, k, Delta_K) for k in K_vals]
deriv_vega_K_vals_stress = [derivata_vega_k(log_ret_stress, s_orig, Omega, k, Delta_K) for k in K_vals]
deriv_vega_K_vals_stress_put = [derivata_vega_k(log_ret_with_put_stress, s_orig, Omega, k, Delta_K) for k in K_vals]

# Creazione figura 2x2
fig, axs = plt.subplots(2, 2, figsize=(12, 5), sharex=False)

# Colori
vega_color = 'purple'
line_color = 'red'

# 1. Originale non protetto
axs[0, 0].plot(K_vals, deriv_vega_K_vals_orig, label="Derivata Vega rispetto a K", color="purple")
axs[0, 0].axvline(K, color='black', linestyle='--',)
axs[0, 0].set_title('Derivata Vega vs K - Portafoglio passivo')
axs[0, 0].set_ylabel('Derivata di Vega')
axs[0, 0].grid(True)

# 2. Originale con Put
axs[0, 1].plot(K_vals, deriv_vega_K_vals_put, label="Derivata Vega rispetto a K", color="purple")
axs[0, 1].axvline(K, color='black', linestyle='--',)
axs[0, 1].set_title('Derivata Vega vs K - Portafoglio coperto')
axs[0, 1].set_ylabel('Derivata di Vega')
axs[0, 1].grid(True)

# 3. Stress test non protetto
axs[1, 0].plot(K_vals, deriv_vega_K_vals_stress, label="Derivata Vega rispetto a K", color="purple")
axs[1, 0].axvline(K, color='black', linestyle='--',)
axs[1, 0].set_title('Derivata Vega vs K - stress test su portafoglio passivo')
axs[1, 0].set_xlabel(r'K')
axs[1, 0].set_ylabel('Derivata di Vega')
axs[1, 0].grid(True)

# 4. Stress test con Put
axs[1, 1].plot(K_vals, deriv_vega_K_vals_stress_put, label="Derivata Vega rispetto a K", color="purple")
axs[1, 1].axvline(K, color='black', linestyle='--',)
axs[1, 1].set_title('Derivata Vega vs K - stress test su portafoglio coperto')
axs[1, 1].set_xlabel(r'K')
axs[1, 1].set_ylabel('Derivata di Vega')
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()


#############################################################################################
            #analisi di W 
#############################################################################################


#Grafici della funzione di sopravvivenza
def plot_pdf_survival(returns, title_pdf, title_survival):
    x = np.sort(returns)
    ecdf = ECDF(returns)
    y_cdf = ecdf(x)
    y_survival = 1 - y_cdf  # funzione di sopravvivenza

    Omega = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # PDF (istogramma normalizzato)
    axes[0].hist(returns, bins=50, density=True, alpha=0.6,
                 color='steelblue', edgecolor="black")
    axes[0].set_title(title_pdf)
    axes[0].set_xlabel("Rendimento logaritmico")
    axes[0].set_ylabel("Densità")

    # Funzione di sopravvivenza
    axes[1].step(x, y_survival, where="post", color="blue")
    axes[1].set_title(title_survival)
    axes[1].set_xlabel("Rendimento logaritmico")
    axes[1].set_ylabel("P(R > r)")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Evidenzia area sopra Omega (coda destra)
    mask = x >= Omega
    if np.any(mask):
        x_fill = np.concatenate(([Omega], x[mask]))
        y_fill = np.concatenate(([1], y_survival[mask]))
        axes[1].fill_between(x_fill, 0, y_fill, color='lightblue', alpha=0.3)

    axes[1].axvline(Omega, color='black', linestyle='--')

    plt.tight_layout()
    plt.show()

plot_pdf_survival(log_ret, "PDF portafoglio non protetto", "Funzione di sopravvivenza portafoglio non protetto")
plot_pdf_survival(log_ret_with_put, "PDF portafoglio con Put ", "Funzione di sopravvivenza portafoglio con Put ")


def plot_survival_comparison(returns1, returns2, label1, label2, Omega=0.0):
    x1 = np.sort(returns1)
    x2 = np.sort(returns2)
    ecdf1 = ECDF(returns1)
    ecdf2 = ECDF(returns2)
    y_survival1 = 1 - ecdf1(x1)
    y_survival2 = 1 - ecdf2(x2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Funzione di sopravvivenza portafoglio 1
    axes[0].step(x1, y_survival1, where="post", color="blue", label=label1)
    axes[0].fill_between(x1[x1 >= Omega], 0, (1 - ecdf1(x1[x1 >= Omega])),
                         color='lightblue', alpha=0.3)
    axes[0].axvline(Omega, color='black', linestyle='--')
    axes[0].set_title(label1)
    axes[0].set_xlabel("Rendimento logaritmico")
    axes[0].set_ylabel("P(R > r)")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Funzione di sopravvivenza portafoglio 2
    axes[1].step(x2, y_survival2, where="post", color="blue", label=label2)
    axes[1].fill_between(x2[x2 >= Omega], 0, (1 - ecdf2(x2[x2 >= Omega])),
                         color='lightblue', alpha=0.3)
    axes[1].axvline(Omega, color='black', linestyle='--')
    axes[1].set_title(label2)
    axes[1].set_xlabel("Rendimento logaritmico")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


# Esempio di utilizzo
plot_survival_comparison(
    log_ret,
    log_ret_with_put,
    "Portafoglio passivo",
    "Portafoglio coperto"
)


#############################################################################################
            #Risultati analisi di W + Grafico
#############################################################################################

Omega = 0
Delta_s = 1e-3

# Baseline returns originali
baseline_returns = log_ret.copy()

# Calcolo L e H unica baseline (quelle del portafoglio originale)
L_baseline = 0.03 #modificare questo per replicare i risultati
H_baseline = log_ret.max()


# Calcolo s_plus per portafoglio originale e con put
s_plus_orig = s_plus(log_ret, Omega)
s_plus_put = s_plus(log_ret_with_put, Omega)

# Calcolo lambda+ rispetto baseline s_plus originale
lambda_plus_orig = lambda_from_s_plus_empirical(s_plus_orig, s_plus_orig)
lambda_plus_put = lambda_from_s_plus_empirical(s_plus_put, s_plus_orig)

# Calcolo xi+ per i due portafogli con L_baseline e H_baseline fissi
xi_plus_orig = xi_plus_on_baseline(baseline_returns, lambda_plus_orig, Omega, L_baseline, H_baseline)
xi_plus_put = xi_plus_on_baseline(baseline_returns, lambda_plus_put, Omega, L_baseline, H_baseline)

# Calcolo sensibilità W con L e H baseline fissi
W_orig = w_vega_right_baseline(log_ret, s_plus_orig, s_plus_orig, Omega, L_baseline, H_baseline, Delta_s)
W_put = w_vega_right_baseline(log_ret_with_put, s_plus_put, s_plus_orig, Omega, L_baseline, H_baseline, Delta_s)

# Creazione DataFrame riepilogativo senza stress test
df_metriche_destra_semplice = pd.DataFrame({
    'Portafoglio': [
        'Originale non protetto',
        'Originale con Put'
    ],
    'Semi-deviazione s^+': [
        s_plus_orig,
        s_plus_put
    ],
    'Lambda^+': [
        lambda_plus_orig,
        lambda_plus_put
    ],
    'Xi^+(L,H)': [
        xi_plus_orig,
        xi_plus_put
    ],
    'W (Sensitività Vega s^+)': [
        W_orig,
        W_put
    ]
})
with pd.option_context('display.float_format', '{:,.6f}'.format):
    print(df_metriche_destra_semplice.to_string(index=False))

# Intervallo di s^+ da задати
s_plus_vals = np.linspace(s_plus_orig * 0.1, s_plus_orig * 3, 500)

# Calcolo W per ogni s^+ per il portafoglio originale
W_vals_orig = [w_vega_right_baseline(log_ret, s, s_plus_orig, Omega, L_baseline, H_baseline, Delta_s) for s in s_plus_vals]

plt.figure(figsize=(11, 5))

plt.plot(s_plus_vals, W_vals_orig, color='purple')
plt.axvline(s_plus_orig, color='black', linestyle='--', alpha=0.7, label=r'$s^+$ portafoglio passivo')
plt.axvline(s_plus_put, color='red', linestyle='--', alpha=0.7, label=r'$s^+$ portafoglio coperto')

plt.title('W rispetto a $s^+$ ')
plt.xlabel(r'$s^+$ (Semi-deviazione Destra)')
plt.ylabel('Sensibilità W di Vega')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



#############################################################################################
            #PORTAFOGLI BARBELL ANALISI DEI SINGOLI ASSET
#############################################################################################


# Strumenti da includere nel portafoglio
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
crypto_candidates = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD', 'DOGE-USD']
smallcap_candidates = ['NIO', 'TRMD', 'FUBO', 'AAOI', 'SPCE', 'ZM']
start = "2020-01-01"
end = "2025-01-01"
min_weeks = 5 * 52  # ≈260 settimane

# scarica history settimanale e calcola log returns
def weekly_log_returns(ticker, start, end):
    t = yf.Ticker(ticker)
    hist = t.history(start=start, end=end, interval="1wk")[['Close']].rename(columns={'Close': ticker})
    if hist.empty:
        return pd.Series(dtype=float)
    # Normalizza indice (rimuove fusi orari)
    hist.index = pd.to_datetime(hist.index)
    try:
        hist.index = hist.index.tz_convert(None)
    except (TypeError, AttributeError):
        try:
            hist.index = hist.index.tz_localize(None)
        except Exception:
            pass
    lr = np.log(hist[ticker] / hist[ticker].shift(1)).dropna()
    lr.name = ticker
    return lr

# Scarica rendimenti settimanali
all_candidates = assets + crypto_candidates + smallcap_candidates
returns = {}

for tk in all_candidates:
    try:
        lr = weekly_log_returns(tk, start, end)
        if len(lr) >= min_weeks:
            returns[tk] = lr
        else:
            print(f"Scartato {tk}: {len(lr)} settimane (< {min_weeks})")
    except Exception as e:
        print(f"Errore scaricando {tk}: {e}")

if not returns:
    raise RuntimeError("Nessun ticker con almeno 5 anni di dati settimanali.")

# Costruisci DataFrame 
df_rets = pd.concat(returns.values(), axis=1)
df_rets.columns = list(returns.keys())
df_rets.sort_index(inplace=True)

# Normalizza indice e risolvi fusi orari / duplicati
df_rets.index = pd.to_datetime(df_rets.index)
try:
    df_rets.index = df_rets.index.tz_convert(None)
except (TypeError, AttributeError):
    try:
        df_rets.index = df_rets.index.tz_localize(None)
    except Exception:
        pass

# Resample su giovedì per uniformare tutte le serie
df_rets = df_rets.resample('W-THU').last()

# --- 3) Tasso risk-free (da ^IRX) ---
print("Scaricamento tasso risk-free (^IRX)...")
irx = yf.Ticker("^IRX").history(start=start, end=end, interval="1wk")['Close']
irx.index = pd.to_datetime(irx.index).tz_localize(None)
rf_for_weeks = ((1 + irx / 100) ** (1 / 52) - 1).rename('rf_weekly')
rf_for_weeks = rf_for_weeks.reindex(df_rets.index, method='ffill')
print("Esempio tasso risk-free settimanale:")
print(rf_for_weeks.head())

# Preparazione per portafoglio reale
asset_simple = np.exp(df_rets) - 1

# Allinea il risk-free come asset aggiuntivo
rf_aligned = rf_for_weeks.reindex(df_rets.index).ffill()
df_rets_with_rf = pd.concat([df_rets, rf_aligned.rename('RF')], axis=1)

# --- 5) Salvataggio finale ---
final_df = df_rets_with_rf.dropna(how='all')
print("Final DataFrame head (log returns + risk-free):")
print(final_df.head())

final_df.to_csv("weekly_log_returns_with_rf.csv")


# METRICHE RISCHIO/RENDIMENTO PER OGNI ASSET

assets_df = final_df

asset_metrics_df = pd.DataFrame(index=assets_df.columns)

# Media, volatilità, min/max
asset_metrics_df['MeanReturn'] = assets_df.mean()
asset_metrics_df['Volatility'] = assets_df.std()
asset_metrics_df['MinReturns'] = assets_df.min()
asset_metrics_df['MaxReturns'] = assets_df.max()
# Max drawdown
asset_metrics_df['MaxDrawdown'] = assets_df.apply(max_drawdown)
# Value at Risk storico 5%
asset_metrics_df['VaR_5%'] = assets_df.apply(var_5)
#CVaR (Expected Shortfall) 
alpha = 0.05
asset_metrics_df[f'CVaR_{int(alpha*100)}%'] = assets_df.apply(lambda x: cvar_historical(x, alpha))

# Trasponi per leggibilità (righe = metriche, colonne = asset)
asset_metrics_df_T = asset_metrics_df.T

print("Metriche rischio/rendimento per ogni asset:")
print('')
print(asset_metrics_df_T.round(4))

#############################################################################################
            #CALCOLO DI V E W PER OGNI ASSET
#############################################################################################

asset_tail_metrics_df = pd.DataFrame(index=final_df.columns, 
                                     columns=['s_minus', 'xi', 'Vega', 's_plus', 'xi_plus', 'W'])

Omega = 0
quantile = 0.05
Delta_s = 1e-2

for asset in final_df.columns:
    data = final_df[asset].dropna()
    if data.empty:
        continue
    
    # Coda sinistra
    K = -0.03 #np.quantile(data, quantile) #in alternativa
    s_minus_val = s_minus(data, Omega)
    lambda_minus = lambda_from_s_empirical(s_minus_val, s_minus_val)
    xi_minus_val = xi_K_on_baseline(data, lambda_minus, Omega, K)
    V_val = vega_tail_baseline(data, s_minus_val, s_minus_val, Omega, K, Delta_s)
    
    # Coda destra
    L = 0.03 #np.quantile(data, 1 - quantile) = in alternativa
    s_plus_val = s_plus(data, Omega)
    lambda_plus = lambda_from_s_plus_empirical(s_plus_val, s_plus_val)
    xi_plus_val = xi_plus_on_baseline(data, lambda_plus, Omega, L, data.max())
    W_val = w_vega_right_baseline(data, s_plus_val, s_plus_val, Omega, L, data.max(), Delta_s=1e-2)
    
    asset_tail_metrics_df.loc[asset] = [s_minus_val, xi_minus_val, V_val, s_plus_val, xi_plus_val, W_val]

asset_tail_metrics_df = asset_tail_metrics_df


print('')
print("Metriche di coda per ogni asset:")
print('')
print(asset_tail_metrics_df.round(4))

#############################################################################################
            #CREAZIONE DEI PORTAFOGLI BARBELL E CALCOLO DI V E W PER OGNI PORTAFOGLIO
#############################################################################################

#############################################################################################
            #PORTAFOGLO 85% RF
#############################################################################################

rf_weight = 0.85               # 85% capitale in risk-free
risk_asset_weight = 1 - rf_weight
min_weight_per_asset = 0.01    # VINCOLO almeno 1% per ciascun asset rischioso

# Identifica le colonne degli asset rischiosi (escludendo 'RF')
risky_assets_cols = [col for col in final_df.columns if col != 'RF']

# Ridefinisce i dati solo con gli asset rischiosi
assets_returns = final_df[risky_assets_cols]
mu = assets_returns.mean()
Sigma = assets_returns.cov()
n_assets = len(assets_returns.columns) # CORRETTO: n_assets = 15

# Mantieni la serie RF separata
rf_series = final_df['RF']

# COSTRUZIONE PORTAFOGLI
weights_minvar = solve_portfolio(constraints_minvar, objective_minvar)
weights_minvar['RF'] = rf_weight

# HighRisk (ex Barbell)
vols = assets_returns.std().sort_values(ascending=False)
top_risky = vols.index[:5]
weights_highrisk = pd.Series(0.0, index=assets_returns.columns)
weights_highrisk[top_risky] = risk_asset_weight / len(top_risky)
weights_highrisk['RF'] = rf_weight

# Equal Weight
weights_equal = pd.Series(risk_asset_weight / n_assets, index=assets_returns.columns)
weights_equal['RF'] = rf_weight

# Portafogli rischio/rendimento target (SR) 
target_returns = [0.005, 0.007, 0.010]  # basso, medio, alto
weights_sr_list = []

for i, tr in enumerate(target_returns, 1):
    def constraints_sr(w):
        return [cp.sum(w) == 1, 
                w >= min_weight_per_asset, 
                cp.sum(cp.multiply(mu.values, w)) >= tr]
    w_sr = solve_portfolio(constraints_sr, objective_minvar)
    w_sr['RF'] = rf_weight
    weights_sr_list.append((f"SR_{i}", w_sr))

# DataFrame finale portafogli
portfolios_dict = {
    'MinVar': weights_minvar,
    'HighRisk': weights_highrisk,
    'EqualWeight': weights_equal
}
for name, w_sr in weights_sr_list:
    portfolios_dict[name] = w_sr

barbell_85_rf = pd.DataFrame(portfolios_dict)
barbell_85_rf = barbell_85_rf[sorted(barbell_85_rf.columns)]
print("Portafogli barbell 85% RF:")
print(barbell_85_rf.round(4))

# Assumendo che 'barbell_85_rf' sia il tuo DataFrame dei pesi
somma_pesi = barbell_85_rf.sum()
print("Somma dei pesi per ciascun portafoglio:")
print(somma_pesi.round(8))

# CALCOLO SERIE STORICHE DEI RENDIMENTI
port_returns_df = pd.DataFrame(index=final_df.index)

for col in barbell_85_rf.columns:
    weights = barbell_85_rf[col]
    port_returns_df[col] = (assets_returns * weights[assets_returns.columns]).sum(axis=1) + weights['RF'] * rf_series

print("\nSerie storiche dei rendimenti dei portafogli:")
print(port_returns_df.head())

# Salva i rendimenti dei portafogli in un file CSV
port_returns_df.to_csv('portafoglio_rendimenti_85RF.csv')


# METRICHE RISCHIO/RENDIMENTO

metrics_df = pd.DataFrame(index=barbell_85_rf.columns)
metrics_df['MeanReturn'] = port_returns_df.mean()
metrics_df['Volatility'] = port_returns_df.std()
metrics_df['Sharpe'] = (port_returns_df.mean() - rf_series.mean()) / port_returns_df.std()
metrics_df['MinReturns'] = port_returns_df.min()
metrics_df['MaxReturns'] = port_returns_df.max()
metrics_df['MaxDrawdown'] = port_returns_df.apply(max_drawdown)
metrics_df['VaR_5%'] = port_returns_df.apply(var_5)
alpha = 0.05
metrics_df[f'CVaR_{int(alpha*100)}%'] = port_returns_df.apply(lambda x: cvar_historical(x, alpha))
metrics_df_T = metrics_df.T

print("\n Metriche rischio/rendimento (righe/colonne invertite):")
print(metrics_df_T.round(4))

#  METRICHE DI CODA
tail_metrics_df = pd.DataFrame(index=barbell_85_rf.columns, 
                               columns=['s_minus', 'xi', 'Vega', 's_plus', 'xi_plus', 'W'])

Omega = 0
K, L, Delta_s = -0.01, 0.01, 1e-3 #CAMBIARE QUESTI PER REPLICARE I RISULTATI


Delta_s = 1e-2
for port in barbell_85_rf.columns:
    data = port_returns_df[port]
    s_minus_val = s_minus(data, Omega)
    lambda_minus = lambda_from_s_empirical(s_minus_val, s_minus_val)
    xi_minus_val = xi_K_on_baseline(data, lambda_minus, Omega, K)
    V_val = vega_tail_baseline(data, s_minus_val, s_minus_val, Omega, K, Delta_s)
    s_plus_val = s_plus(data, Omega)
    lambda_plus = lambda_from_s_plus_empirical(s_plus_val, s_plus_val)
    xi_plus_val = xi_plus_on_baseline(data, lambda_plus, Omega, L, data.max())
    W_val = w_vega_right_baseline(data, s_plus_val, s_plus_val, Omega, L, data.max(), Delta_s)
    tail_metrics_df.loc[port] = [s_minus_val, xi_minus_val, V_val, s_plus_val, xi_plus_val, W_val]

print("\n Metriche di coda dei portafogli (85% RF):")
print(tail_metrics_df.round(4))


#############################################################################################
            #PORTAFOGLO 90% RF
#############################################################################################

rf_weight = 0.90               # 90% capitale in risk-free
risk_asset_weight = 1 - rf_weight
min_weight_per_asset = 0.01    # almeno 1% per ciascun asset rischioso

# Identifica le colonne degli asset rischiosi (escludendo 'RF')
risky_assets_cols = [col for col in final_df.columns if col != 'RF']

# Ridefinisce i dati solo con gli asset rischiosi
assets_returns = final_df[risky_assets_cols]
mu = assets_returns.mean()
Sigma = assets_returns.cov()
n_assets = len(assets_returns.columns) # CORRETTO: n_assets = 15

# Mantieni la serie RF separata
rf_series = final_df['RF']

# COSTRUZIONE PORTAFOGLI
weights_minvar = solve_portfolio(constraints_minvar, objective_minvar)
weights_minvar['RF'] = rf_weight

# HighRisk (ex Barbell) 
vols = assets_returns.std().sort_values(ascending=False)
top_risky = vols.index[:5]
weights_highrisk = pd.Series(0.0, index=assets_returns.columns)
weights_highrisk[top_risky] = risk_asset_weight / len(top_risky)
weights_highrisk['RF'] = rf_weight

# Equal Weight 
weights_equal = pd.Series(risk_asset_weight / n_assets, index=assets_returns.columns)
weights_equal['RF'] = rf_weight

#  Portafogli rischio/rendimento target (SR) 
target_returns = [0.005, 0.007, 0.010]  # basso, medio, alto
weights_sr_list = []

for i, tr in enumerate(target_returns, 1):
    def constraints_sr(w):
        return [cp.sum(w) == 1, 
                w >= min_weight_per_asset, 
                cp.sum(cp.multiply(mu.values, w)) >= tr]
    w_sr = solve_portfolio(constraints_sr, objective_minvar)
    w_sr['RF'] = rf_weight
    weights_sr_list.append((f"SR_{i}", w_sr))

# DataFrame finale portafogli 
portfolios_dict = {
    'MinVar': weights_minvar,
    'HighRisk': weights_highrisk,
    'EqualWeight': weights_equal
}
for name, w_sr in weights_sr_list:
    portfolios_dict[name] = w_sr

barbell_90_rf = pd.DataFrame(portfolios_dict)
barbell_90_rf = barbell_90_rf[sorted(barbell_90_rf.columns)]
print("Portafogli barbell 90% RF:")
print(barbell_90_rf.round(4))

# Assumendo che 'barbell_85_rf' sia il tuo DataFrame dei pesi
somma_pesi = barbell_90_rf.sum()
print('')
print("Somma dei pesi per ciascun portafoglio:")
print(somma_pesi.round(8))

#CALCOLO SERIE STORICHE DEI RENDIMENTI

port_returns_df = pd.DataFrame(index=final_df.index)

for col in barbell_90_rf.columns:
    weights = barbell_90_rf[col]
    port_returns_df[col] = (assets_returns * weights[assets_returns.columns]).sum(axis=1) + weights['RF'] * rf_series

print("\nSerie storiche dei rendimenti dei portafogli:")
print(port_returns_df.head())

# Salva i rendimenti dei portafogli in un file CSV
port_returns_df.to_csv('portafoglio_rendimenti_90RF.csv')

# METRICHE RISCHIO/RENDIMENTO
metrics_df = pd.DataFrame(index=barbell_90_rf.columns)
metrics_df['MeanReturn'] = port_returns_df.mean()
metrics_df['Volatility'] = port_returns_df.std()
metrics_df['Sharpe'] = (port_returns_df.mean() - rf_series.mean()) / port_returns_df.std()
metrics_df['MinReturns'] = port_returns_df.min()
metrics_df['MaxReturns'] = port_returns_df.max()
metrics_df['MaxDrawdown'] = port_returns_df.apply(max_drawdown)
metrics_df['VaR_5%'] = port_returns_df.apply(var_5)
alpha = 0.05
metrics_df[f'CVaR_{int(alpha*100)}%'] = port_returns_df.apply(lambda x: cvar_historical(x, alpha))
metrics_df_T = metrics_df.T

print("\n Metriche rischio/rendimento (righe/colonne invertite):")
print(metrics_df_T.round(4))

# METRICHE DI CODA
tail_metrics_df = pd.DataFrame(index=barbell_90_rf.columns, 
                               columns=['s_minus', 'xi', 'Vega', 's_plus', 'xi_plus', 'W'])

Omega = 0
K, L, Delta_s = -0.01, 0.01, 1e-3 #CAMBIARE QUESTI PER REPLICARE I RISULTATI

for port in barbell_90_rf.columns:
    data = port_returns_df[port]
    s_minus_val = s_minus(data, Omega)
    lambda_minus = lambda_from_s_empirical(s_minus_val, s_minus_val)
    xi_minus_val = xi_K_on_baseline(data, lambda_minus, Omega, K)
    V_val = vega_tail_baseline(data, s_minus_val, s_minus_val, Omega, K, Delta_s)
    s_plus_val = s_plus(data, Omega)
    lambda_plus = lambda_from_s_plus_empirical(s_plus_val, s_plus_val)
    xi_plus_val = xi_plus_on_baseline(data, lambda_plus, Omega, L, data.max())
    W_val = w_vega_right_baseline(data, s_plus_val, s_plus_val, Omega, L, data.max(), Delta_s)
    tail_metrics_df.loc[port] = [s_minus_val, xi_minus_val, V_val, s_plus_val, xi_plus_val, W_val]

print("\n Metriche di coda dei portafogli (90% RF):")
print(tail_metrics_df.round(4))


#############################################################################################
            #PORTAFOGLO 95% RF
#############################################################################################

rf_weight = 0.95               # 95% capitale in risk-free
risk_asset_weight = 1 - rf_weight
min_weight_per_asset = 0.01    # almeno 1% per ciascun asset rischioso

# Identifica le colonne degli asset rischiosi (escludendo 'RF')
risky_assets_cols = [col for col in final_df.columns if col != 'RF']

# Ridefinisce i dati solo con gli asset rischiosi
assets_returns = final_df[risky_assets_cols]
mu = assets_returns.mean()
Sigma = assets_returns.cov()
n_assets = len(assets_returns.columns) # CORRETTO: n_assets = 15

# Mantieni la serie RF separata
rf_series = final_df['RF']

#COSTRUZIONE PORTAFOGLI
weights_minvar = solve_portfolio(constraints_minvar, objective_minvar)
weights_minvar['RF'] = rf_weight

# HighRisk (ex Barbell) 
vols = assets_returns.std().sort_values(ascending=False)
top_risky = vols.index[:5]
weights_highrisk = pd.Series(0.0, index=assets_returns.columns)
weights_highrisk[top_risky] = risk_asset_weight / len(top_risky)
weights_highrisk['RF'] = rf_weight

# Equal Weight 
weights_equal = pd.Series(risk_asset_weight / n_assets, index=assets_returns.columns)
weights_equal['RF'] = rf_weight

# Portafogli rischio/rendimento target (SR) 
target_returns = [0.005, 0.007, 0.010]
weights_sr_list = []

for i, tr in enumerate(target_returns, 1):
    def constraints_sr(w):
        return [cp.sum(w) == 1, w >= min_weight_per_asset, cp.sum(cp.multiply(mu.values, w)) >= tr]
    w_sr = solve_portfolio(constraints_sr, objective_minvar)
    w_sr['RF'] = rf_weight
    weights_sr_list.append((f"SR_{i}", w_sr))

portfolios_dict = {
    'MinVar': weights_minvar,
    'HighRisk': weights_highrisk,
    'EqualWeight': weights_equal
}
for name, w_sr in weights_sr_list:
    portfolios_dict[name] = w_sr

barbell_95_rf = pd.DataFrame(portfolios_dict)
barbell_95_rf = barbell_95_rf[sorted(barbell_95_rf.columns)]
print("Portafogli barbell 95% RF:")
print(barbell_95_rf.round(4))

# Assumendo che 'barbell_85_rf' sia il tuo DataFrame dei pesi
somma_pesi = barbell_95_rf.sum()
print('')
print("Somma dei pesi per ciascun portafoglio:")
print(somma_pesi.round(8))

# CALCOLO SERIE STORICHE DEI RENDIMENTI
port_returns_df = pd.DataFrame(index=final_df.index)

for col in barbell_95_rf.columns:
    weights = barbell_95_rf[col]
    port_returns_df[col] = (assets_returns * weights[assets_returns.columns]).sum(axis=1) + weights['RF'] * rf_series

print("\nSerie storiche dei rendimenti dei portafogli:")
print(port_returns_df.head())
# Salva i rendimenti dei portafogli in un file CSV
port_returns_df.to_csv('portafoglio_rendimenti_95RF.csv')


# METRICHE RISCHIO/RENDIMENTO
metrics_df = pd.DataFrame(index=barbell_95_rf.columns)
metrics_df['MeanReturn'] = port_returns_df.mean()
metrics_df['Volatility'] = port_returns_df.std()
metrics_df['Sharpe'] = (port_returns_df.mean() - rf_series.mean()) / port_returns_df.std()
metrics_df['MinReturns'] = port_returns_df.min()
metrics_df['MaxReturns'] = port_returns_df.max()
metrics_df['MaxDrawdown'] = port_returns_df.apply(max_drawdown)
metrics_df['VaR_5%'] = port_returns_df.apply(var_5)
alpha = 0.05
metrics_df[f'CVaR_{int(alpha*100)}%'] = port_returns_df.apply(lambda x: cvar_historical(x, alpha))
metrics_df_T = metrics_df.T

print("\n Metriche rischio/rendimento (righe/colonne invertite):")
print(metrics_df_T.round(4))

# METRICHE DI CODA
tail_metrics_df = pd.DataFrame(index=barbell_95_rf.columns, 
                               columns=['s_minus', 'xi', 'Vega', 's_plus', 'xi_plus', 'W'])

Omega = 0
K, L, Delta_s = -0.01, 0.01, 1e-3 #CAMBIARE QUESTI PER REPLICARE I RISULTATI

for port in barbell_95_rf.columns:
    data = port_returns_df[port]
    s_minus_val = s_minus(data, Omega)
    lambda_minus = lambda_from_s_empirical(s_minus_val, s_minus_val)
    xi_minus_val = xi_K_on_baseline(data, lambda_minus, Omega, K)
    V_val = vega_tail_baseline(data, s_minus_val, s_minus_val, Omega, K, Delta_s)
    s_plus_val = s_plus(data, Omega)
    lambda_plus = lambda_from_s_plus_empirical(s_plus_val, s_plus_val)
    xi_plus_val = xi_plus_on_baseline(data, lambda_plus, Omega, L, data.max())
    W_val = w_vega_right_baseline(data, s_plus_val, s_plus_val, Omega, L, data.max(), Delta_s)
    tail_metrics_df.loc[port] = [s_minus_val, xi_minus_val, V_val, s_plus_val, xi_plus_val, W_val]

print("\n Metriche di coda dei portafogli (95% RF):")
print(tail_metrics_df.round(4))

#############################################################################################
        #ANALISI GRAFICA DELLE DISTRIBUZIONI DEI PORTAFOGLI
#############################################################################################

def plot_pdf_cdf(returns, title_pdf, title_cdf):
    x = np.sort(returns)
    ecdf = ECDF(returns)
    y = ecdf(x)

    Omega = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(returns, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor="black")
    axes[0].set_title(title_pdf)
    axes[0].set_xlabel("Rendimento logaritmico")
    axes[0].set_ylabel("Densità")

    axes[1].step(x, y, where="post", color="blue")
    axes[1].set_title(title_cdf)
    axes[1].set_xlabel("Rendimento logaritmico")
    axes[1].set_ylabel("P(R ≤ r)")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Ruota le etichette tick sull'asse x per entrambi i subplot
    for ax in axes:
        ax.tick_params(axis='x', labelrotation=45)

    mask = x <= Omega
    if np.any(mask):
        x_fill = np.concatenate((x[mask], [Omega]))
        y_fill = np.concatenate((y[mask], [y[mask][-1]]))
        axes[1].fill_between(x_fill, 0, y_fill, color='lightblue', alpha=0.3)

    axes[1].axvline(Omega, color='black', linestyle='--')

    plt.tight_layout()
    plt.show()


# Cambia qui i path se necessario
returns_85 = pd.read_csv('portafoglio_rendimenti_85RF.csv', index_col=0)
returns_90 = pd.read_csv('portafoglio_rendimenti_90RF.csv', index_col=0)
returns_95 = pd.read_csv('portafoglio_rendimenti_95RF.csv', index_col=0)

# Nome della colonna per il portafoglio Equal Weight. Adatta se necessario.
equal_weight_col_name = "EqualWeight"  

# Genera i grafici PDF e CDF per i 3 livelli di risk-free 
for rf, returns_df in zip([85, 90, 95], [returns_85, returns_90, returns_95]):
    eqw_returns = returns_df[equal_weight_col_name]
    title_pdf = f"PDF EqualWeight - {rf}% RF"
    title_cdf = f"CDF EqualWeight - {rf}% RF"
    plot_pdf_cdf(eqw_returns, title_pdf, title_cdf)
    
#############################################################################################
        #GRAFICI AGGIUNTIVI PER V DEI PORTAFOGLI BARBELL
#############################################################################################

log_ret = returns_85['EqualWeight'] #CAMBIARE LA COLONNA E DF PER OTTENERE GLI ALTRI GRAFICI
# Parametri dati
Omega = 0.0
Ks = np.linspace(-0.3, 0.0, 500)  # Range per K
K_current = log_ret.quantile(0.05)  # K attuale fisso
Delta_s = 1e-3

# Liste per raccogliere risultati
s_list, xi_list, vega_list = [], [], []

for Ki in Ks:
    s = s_minus(log_ret, Omega)  # semi-deviazione sinistra fissa rispetto a Omega
    lam = lambda_from_s_empirical(s, s_minus(log_ret, Omega))  # lambda calcolato rispetto baseline fissa
    xi = xi_K_on_baseline(log_ret, lam, Omega, Ki)
    V = vega_tail_baseline(log_ret, s, s_minus(log_ret, Omega), Omega, Ki, Delta_s)

    s_list.append(s)
    xi_list.append(xi)
    vega_list.append(V)

# Conversione in array per plot
xi_array = np.array(xi_list)
vega_array = np.array(vega_list)

K_1 = -0.01
K_2 = -0.02
K_3 = -0.03

# Creazione figura e subplot
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Grafico 1: ξ(K,s⁻) vs K
axs[0].plot(Ks, xi_array, color='blue', lw=2)
axs[0].axvline(K_1, color='red', linestyle='--', lw=2, label='K = -0.01')
axs[0].axvline(K_2, color='red', linestyle='--', lw=2, label='K = -0.02')
axs[0].axvline(K_3, color='red', linestyle='--', lw=2, label='K = -0.03')
axs[0].set_title(r'$\xi(K,s^-)$ vs K', fontsize=14)
axs[0].set_xlabel('K', fontsize=12)
axs[0].set_ylabel(r'$\xi$', fontsize=12)
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.5)

# Grafico 2: Vega vs K
axs[1].plot(Ks, vega_array, color='purple', lw=2)
axs[1].axvline(K_1, color='red', linestyle='--', lw=2, label='K = -0.01')
axs[1].axvline(K_2, color='red', linestyle='--', lw=2, label='K = -0.02')
axs[1].axvline(K_3, color='red', linestyle='--', lw=2, label='K = -0.03')
axs[1].set_title('Vega vs K', fontsize=14)
axs[1].set_xlabel('K', fontsize=12)
axs[1].set_ylabel('Vega', fontsize=12)
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


# Parametri base
Omega = 0.0
Delta_s = 1e-3
K = -0.01

# Calcolo s_left di riferimento (baseline semi-dev sinistra)
s_left = s_minus(log_ret, Omega)

# Variazione di s^-
s_minus_vals = np.linspace(s_left * 0.1, s_left * 3, 500)

lambda_vals = []
xi_vals = []
vega_vals = []

for s_val in s_minus_vals:
    lam = lambda_from_s_empirical(s_val, s_left)  # Calcola lambda rispetto baseline s_left
    lambda_vals.append(lam)

    xi = xi_K(log_ret, lam, Omega, K)
    xi_vals.append(xi)

    vega = vega_tail_baseline(log_ret, s_val, s_left, Omega, K, Delta_s)
    vega_vals.append(vega)

# Grafico 1: s^- e lambda
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

axs[0].plot(s_minus_vals, s_minus_vals, label=r'$s^-$')
axs[0].set_title('Andamento di $s^-$')
axs[0].set_xlabel('Indice')
axs[0].set_ylabel(r'$s^-$')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(s_minus_vals, lambda_vals, color='orange', label=r'$\lambda$')
axs[1].set_title('Andamento di $\lambda$')
axs[1].set_xlabel(r'$s^-$')
axs[1].set_ylabel(r'$\lambda$')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()

# Xi vs s^-
plt.figure(figsize=(12, 4))
plt.plot(s_minus_vals, xi_vals, label=r'$\xi$ vs $s^-$', color='blue')
plt.xlabel(r'$s^-$')
plt.ylabel(r'$\xi$')
plt.title(r'Andamento di $\xi$ rispetto a $s^-$')
plt.legend()
plt.grid(True)
plt.show()

# varia s^-
Delta_s = 1e-3
# Calcolo s^- per ogni distribuzione
s_orig = s_minus(log_ret, Omega)
K_orig = -0.01
s_minus_vals = np.linspace(s_orig * 0.1, s_orig * 3, 500)

vega_vals_orig = [vega_tail_baseline(log_ret, s_val, s_orig, Omega, K_orig, Delta_s) for s_val in s_minus_vals]

vega_color = 'purple'
line_color = 'red'

plt.figure(figsize=(8, 6))
plt.plot(s_minus_vals, vega_vals_orig, color=vega_color)
plt.axvline(s_orig, color=line_color, linestyle='--')
plt.title('Portafoglio passivo')
plt.ylabel('Vega')
plt.grid(True)
plt.show()


Delta_s = 1e-3

# Intervalli s^- specifici per ciascun caso
s_minus_vals = np.linspace(s_orig * 0.1, s_orig * 3, 500)

# Calcolo derivate di Vega per ciascun caso
deriv_vega_vals_orig = [derivata_vega_s_minus(log_ret, s,s_orig, Omega, K_orig, Delta_s) for s in s_minus_vals]


vega_color = 'purple'
line_color = 'red'

plt.figure(figsize=(8, 6))
plt.plot(s_minus_vals, deriv_vega_vals_orig, color=vega_color)
plt.axvline(s_orig, color=line_color, linestyle='--')
plt.title('Portafoglio passivo')
plt.ylabel('Vega')
plt.grid(True)
plt.show()


plt.tight_layout()
plt.show()


Omega = 0.0
Delta_s = 1e-3
Delta_K = Delta_s
s_orig = s_minus(log_ret, Omega)
K = -0.02

# Intervalli di variazione

K_vals = np.linspace(-0.06, 0, 500)

# Calcolo derivate di Vega per ogni valore nell'intervallo

deriv_vega_K_vals_orig = [derivata_vega_k(log_ret, s_orig, Omega, k, Delta_K) for k in K_vals]

vega_color = 'purple'
line_color = 'red'

plt.figure(figsize=(8, 6))
plt.plot(s_minus_vals, deriv_vega_K_vals_orig, color=vega_color)
plt.axvline(s_orig, color=line_color, linestyle='--')
plt.title('Portafoglio passivo')
plt.ylabel('Vega')
plt.grid(True)
plt.show()


plt.tight_layout()
plt.show()


