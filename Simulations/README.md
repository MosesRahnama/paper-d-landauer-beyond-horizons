# Supplementary Simulation for Paper D

**The Landauer Saturation Diagnostic for Non-Gravitational Systems: QCD Confinement and the Casimir Effect**

Moses Rahnama (2026)

## Overview

This repository contains the companion simulation for Paper D of the Landauer saturation trilogy. The simulation performs 70 independent falsifiable checks across 8 groups, verifying every quantitative claim in the manuscript.

The static Landauer ratio is defined as

$$\mathcal{R}_L^{\mathrm{stat}} \equiv \frac{U}{T\,S}$$

where $U$ is the total internal energy, $T$ the temperature, and $S$ the entropy of the system. This is the Euler homogeneity factor of the equation of state (Callen, Ch. 3).

## Key results verified

| System | $\mathcal{R}_L^{\mathrm{stat}}$ | Classification | Source |
|---|---|---|---|
| QCD, deep confinement | 2.16 | Super-Landauer | Lattice |
| Schwarzschild black hole | 2 | Super-Landauer | Smarr relation |
| QCD, at $T_c$ | 1.17 | Super-Landauer | Lattice |
| QCD, $T/T_c = 2.99$ | 1.04 | Super-Landauer | Lattice |
| Cosmological apparent horizon | 1 | Saturating | Misner-Sharp |
| Photon gas (blackbody) | 3/4 | Sub-Landauer | Stefan-Boltzmann |
| Casimir, high-$T$ | 0 | Sub-Landauer | Exact |

## Check groups

1. **Storage-dimension geometric scaling** (8 checks): $d$-dimensional storage manifold yields $F \sim r^{d-1}$ for $d \geq 1$; $F \sim 1/r^2$ for $d = 0$ (broadcast)
2. **QCD flux tube $\mathcal{R}_L^{\mathrm{stat}}$ from lattice data** (17 checks): Kaczmarek-Zantow free energy, internal energy, entropy; ratio invariance under rescaling (Bazavov)
3. **Casimir $\mathcal{R}_L^{\mathrm{stat}}$** (12 checks): Analytical high-$T$ and low-$T$ limits
4. **Black hole static $\mathcal{R}_L^{\mathrm{stat}}$** (8 checks): Smarr relation gives $\mathcal{R}_L^{\mathrm{stat}} = 2$
5. **Classification table** (7 checks): Full ordering, category assignments, photon gas benchmark
6. **QCD string tension** (6 checks): Dimensional-analysis illustration
7. **Perturbative high-$T$ limit** (6 checks): Asymptotic freedom and $\mathcal{R}_L^{\mathrm{stat}} \to 0$
8. **Cross-consistency with Papers A, B, C** (6 checks): Trilogy coherence

## Running

```bash
python simulation.py
```

All 70 checks should report `[OK]`. No external dependencies beyond NumPy.

## Relation to the trilogy

- **Paper A** (Calorimetric Measurement Bound): Establishes the Landauer bound in quantum measurement
- **Paper B** (Born Rule from Record-Formation Constraints): Derives Born's rule from information-thermodynamic axioms
- **Paper C** (Black Holes as Landauer-Saturating Erasure Channels): Introduces the differential $\mathcal{R}_L^{\mathrm{diff}} = \delta E / (T\,\delta S)$ for black hole evaporation
- **Paper D** (this work): Extends to a static ratio $\mathcal{R}_L^{\mathrm{stat}} = U/(TS)$ for non-gravitational systems

The static ratio used here differs from Paper C's differential ratio by the Smarr homogeneity factor: for Schwarzschild black holes, $\mathcal{R}_L^{\mathrm{stat}} = 2$ while $\mathcal{R}_L^{\mathrm{diff}} = 1$.

## License

MIT
