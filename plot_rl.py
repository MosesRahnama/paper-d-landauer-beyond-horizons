#!/usr/bin/env python3
"""
Generate Figure 1: R_L^stat(T/T_c) for QCD flux tubes.
Data from Kaczmarek & Zantow (hep-lat/0506019), N_f = 2, T_c = 200 MeV.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Kaczmarek-Zantow lattice data: [T/Tc, S_inf, U_inf/Tc]
data = np.array([
    [0.79,  5.48,   9.37],
    [0.84,  6.49,  10.19],
    [0.88,  7.78,  11.30],
    [0.93, 12.92,  15.96],
    [0.98, 16.38,  19.27],
    [1.01, 14.83,  17.71],
    [1.04, 12.93,  15.78],
    [1.09,  5.49,   7.84],
    [1.13,  3.95,   6.14],
    [1.19,  2.73,   4.72],
    [1.29,  1.63,   3.37],
    [1.43,  1.25,   2.85],
    [1.57,  1.02,   2.51],
    [1.72,  0.91,   2.32],
    [1.89,  0.87,   2.26],
    [2.99,  0.67,   2.09],
])

T_Tc = data[:, 0]
S_inf = data[:, 1]
U_Tc = data[:, 2]
R_L = U_Tc / (T_Tc * S_inf)

fig, ax = plt.subplots(figsize=(4.5, 3.5))

# Plot R_L data
ax.plot(T_Tc, R_L, 'ko-', markersize=4, linewidth=1.2, label=r'QCD ($N_f\!=\!2$, lattice)')

# Reference lines
ax.axhline(y=2.0, color='#CC0000', linestyle='--', linewidth=0.9,
           label=r'Schwarzschild BH (Smarr)')
ax.axhline(y=1.0, color='#0066CC', linestyle='-.', linewidth=0.9,
           label=r'Cosmological horizon')
ax.axhline(y=0.75, color='#009933', linestyle=':', linewidth=0.9,
           label=r'Photon gas ($3/4$)')

# T_c band
ax.axvspan(0.95, 1.05, alpha=0.12, color='gray', label=r'$T_c$ region')

# Annotations
ax.annotate(r'$\mathcal{R}_L^{\mathrm{stat}} \approx 2.16$',
            xy=(0.79, R_L[0]), xytext=(1.15, 2.25),
            fontsize=8, ha='left',
            arrowprops=dict(arrowstyle='->', color='black', lw=0.7))

ax.annotate(r'min $\approx 1.17$',
            xy=(1.04, R_L[6]), xytext=(1.35, 1.10),
            fontsize=8, ha='left',
            arrowprops=dict(arrowstyle='->', color='black', lw=0.7))

ax.set_xlabel(r'$T / T_c$', fontsize=11)
ax.set_ylabel(r'$\mathcal{R}_L^{\mathrm{stat}} = U/(TS)$', fontsize=11)
ax.set_xlim(0.65, 3.15)
ax.set_ylim(0.5, 2.5)
ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
ax.tick_params(labelsize=9)

fig.tight_layout()
fig.savefig('fig_rl_qcd.pdf', dpi=300, bbox_inches='tight')
fig.savefig('fig_rl_qcd.png', dpi=300, bbox_inches='tight')
print("Saved fig_rl_qcd.pdf and fig_rl_qcd.png")
