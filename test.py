# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 21:33:31 2022

@author: Arjun
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import optimize
import warnings

warnings.filterwarnings("ignore")

# %%

plt.style.use("ggplot")

# %%
if not os.path.exists("fig"):
    os.makedirs("fig/png")
    os.makedirs("fig/svg")
    os.makedirs("fig/tif")
    os.makedirs("fig/html")


# %% [markdown]
# ### function to estimate efficiency

# %%
def dGox_func(gamma):
    return 60.3 - 28.5 * (4 - gamma)


def dCG_O2(gamma):
    return dGox_func(gamma) + (gamma / gamma_O2) * dGred_O2


def efficiency_CNED_Inf(gamma, T, eA, CNB, NSource, CNED):

    if gamma > 8:
        print("Error. wrong values of DR of S of P")
        return

    # Ts = 298
    # pre-assign std G of formation from CHNOSZ
    dfGH2O = -237.2  # kJ/mol
    dfGO2aq = 16.5  # kJ/mol
    dfGNH4 = -79.5  # kJ/mol
    # dfGHCO3aq = -586.9 #kJ/mol
    # dfGglu = -919.8 #kJ/mol
    # dfGbio = -67 #kJ/mol
    # dfGeth = -181.8 #kJ/mol
    dfGNO3 = -111  # kJ/mol
    # dfGNO2 = -32.2 #kJ/mol
    # dfGNO = 102 #kJ/mol
    # dfGN2H4 = 159.17 #kJ/mol
    # dfGNH2OH = -43.6 #kJ/mol
    dfGN2 = 18.2  # kJ/mol
    # dfGH2 = 17.7
    gamma_B = 4.2  # e- mol/Cmol

    # define epectron acceptor and some other DR
    if eA == "O2":
        gamma_eA = 4
        dGred_eA = 2 * dfGH2O - dfGO2aq
    elif eA == "NO3":
        gamma_eA = 8
        dGred_eA = dfGNH4 + 3 * dfGH2O - dfGNO3
    elif eA == "Denitrification":
        gamma_eA = 5
        dGred_eA = 0.5 * dfGN2 + 3 * dfGH2O - dfGNO3
    elif eA == "Fe(III)":  # ferrihydrite
        # dGred_eA values are taken from La Rowe 2015
        gamma_eA = 1
        dGred_eA = -100 * gamma_eA
    elif eA == "FeOOH":  # goethite
        gamma_eA = 1
        dGred_eA = -75.58 * gamma_eA
    elif eA == "Mn4":  # goethite
        gamma_eA = 2
        dGred_eA = -120.03 * gamma_eA
    elif eA == "SO4":  # goethite
        gamma_eA = 8
        dGred_eA = -24.04 * gamma_eA

    # define NSource
    if NSource == "NO3":
        gamma_NS = 8
        dGred_NS = dfGNH4 + 3 * dfGH2O - dfGNO3
    elif NSource == "NH4":
        # 0.5N2 + 4e + 5H ->  NH4 + 0.5H2
        gamma_NS = 0
        dGred_NS = 0
        # dGred_NS values are taken from La Rowe 2015

    # growth yield calculations
    if gamma < 4.67:
        dGrX = -(666.7 / gamma + 243.1)  # kJ/Cmol biomass
    else:
        dGrX = -(157 * gamma - 339)  # kJ/Cmol biomass

    # Anbolic reaction
    dCGX = dCG_O2(gamma_B)
    dGana = (gamma_B / gamma) * dCG_O2(gamma) - dCGX
    dGana1 = dGana

    dG_ox = dGox_func(gamma)
    if gamma_NS == 0:  # NSource = NH4
        dGcat = dG_ox + gamma * dGred_eA / gamma_eA
        Y = dGcat / (dGrX - dGana1 + gamma_B / gamma * dGcat)
        xN_cat = 0
        xEA_cat = (gamma) / gamma_eA
        dGrS = dGcat
    else:

        def xN(y1):
            return (y1 / CNB - 1 / CNED) * (1 - y1 * gamma_B / gamma) ** -1

        def dGcat(y2):
            return (
                dG_ox
                + dGred_NS * xN(y2)
                + dGred_eA * (gamma - gamma_NS * xN(y2)) / gamma_eA
            )

        def fun(y3):
            return y3 - dGcat(y3) / (dGrX - dGana1 + gamma_B / gamma * dGcat(y3))

        sol = optimize.root_scalar(fun, bracket=[0, 3], method="brentq")
        Y = sol.root
        xN_cat = xN(Y)
        xEA_cat = (gamma - gamma_NS * xN_cat) / gamma_eA
        TER = CNB / Y
        dGrS = dGcat(Y)
        if CNED < TER:  # true means N rich ED
            xN_cat = 0
            xEA_cat = gamma / gamma_eA
            dGcatNoNS = dG_ox + dGred_eA * gamma / gamma_eA
            Y = dGcatNoNS / (dGrX - dGana1 + gamma_B / gamma * dGcatNoNS)
            if dGcatNoNS > 0:
                print("Error. dGcat POSITIVE")
                return
            dGrS = dGcatNoNS
        else:
            if dGcat(Y) > 0:
                print("Error. dGcat POSITIVE")
                return
    v_EA = (1 - Y * gamma_B / gamma) * xEA_cat
    v_N = Y / CNB - 1 / CNED
    YCO2 = 1 - (Y)

    if xEA_cat < 0:
        print("eTransferEA <0")
        return

    return [Y, YCO2, dGrS, xN_cat, xEA_cat, v_EA, v_N]


# %%
T = 273 + 25
dfGH2O = -237.2  # kJ/mol
dfGO2aq = 16.5  # kJ/mol
# DeltaG of complete oxidation of organic matter
gamma_O2 = 4
dGred_O2 = 2 * dfGH2O - dfGO2aq
eA = ["O2", "NO3", "Fe(III)", "FeOOH", "SO4"]  # Electron acceptors
gammaB = 4.2  # Degree of reduction of biomass
xycolor = np.array([1, 1, 1]) * 0.2
lstyle = ["-", "--", "-.", "."]
labelfont = 16
# backcolor = [0.9412, 0.9412, 0.9412]
backcolor = np.array([1, 1, 1]) * 0.975
GridCol = np.array([1, 1, 1]) * 0.9
# backcolor=[1 1 1]
axisfont = 12
LC = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

xycol = np.array([1, 1, 1]) * 0.4
gamma_B = 4.2
CNB = 5  # CN ratio of biomass
INORM = (
    np.arange(0.001, 0.1 + 0.0005, 0.0005) / 0.5
)  # normalized inorganic N supply rate
delta_psi = 120 * 0.001  # volt # Membrane potential
R = 8.314  # J/ mol K , Gas constant
FaradayC = 96.48534  # Coulomb per mol e or kJ/ volt, faraday constant

# if 7~=exist('/fig', 'dir')
#    mkdir('/fig')
# end


# %% [markdown]
# # Figure 2 matplotllib

# %%
gamma = np.arange(0.5, 8 + 0.05, 0.05)
dGred = -np.array([5, 24.04, 40, 75.6, 85, 122.7])
# dGred = -np.array([122])
dGrX = np.zeros((len(gamma)))
dCGX = dCG_O2(gamma_B)
dGB_ana = (gamma_B / gamma) * dCG_O2(gamma) - dCGX
# viridis = plt.cm.get_cmap('viridis', len(dGred))
# LC = viridis.colors

for i in range(0, len(gamma)):
    if gamma[i] < 4.67:
        dGrX[i] = -(666.7 / gamma[i] + 243.1)  # kJ/Cmol biomass
    else:
        dGrX[i] = -(157 * gamma[i] - 339)  # kJ/Cmol biomass
fig = plt.figure(figsize=(5.5, 4.5), facecolor="w")
axs = fig.subplots(nrows=1, ncols=1)

for i in range(0, len(dGred)):
    dGcat = (60.3 - 28.5 * (4 - gamma)) + gamma * dGred[i]
    Y = dGcat / (dGrX - dGB_ana + (4.2 / gamma) * dGcat)
    Ycat = 1 - (Y * 4.2 / gamma)
    # nTh = (Y * dGB_ana) / (-Ycat * dGcat)
    # nTh = dGB_ana / (dGrX - dGB_ana)
    nTh = dCGX / (dGcat / Y)
    axs.plot(gamma, Y, label=" " + str(-dGred[i]), linewidth=3.0, color=LC[i])
# axs.set_ylim([0, 0.65])
axs.set_ylim(bottom=0)
# axs.set_xlim([0, 3])


axs.tick_params(axis="x", labelsize=axisfont)
axs.tick_params(axis="y", labelsize=axisfont)
axs.set_xlabel(r"$\gamma_{ED}$", fontsize=labelfont)

axs.legend(
    loc="upper left",
    bbox_to_anchor=(-0.125, 1.3),
    frameon=False,
    fontsize=labelfont - 4,
    title=r"$ \frac{-\Delta_{red}G_{EA}}{\gamma_{EA}}$ kJ $\mathrm{(e^- mol)^{-1}}$",
    title_fontsize=labelfont,
)

# axs.text(2.3, 0.6175, r"($\mathrm{{SO_4}^{2-}}$)", fontsize=labelfont - 4)
# axs.text(2.2, 0.5, r"(goethite)", fontsize=labelfont - 4)
# axs.text(2.3, 0.385, r"($\mathrm{O_2}$)", fontsize=labelfont - 4)


axs.set_ylabel(r"$G_{norm} = e$", fontsize=labelfont)

fig.tight_layout()
fig.savefig("fig/png/Figure2.png", dpi=300)
fig.savefig("fig/svg/Figure2.svg", dpi=300)
plt.show()