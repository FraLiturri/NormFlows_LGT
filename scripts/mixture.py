from neumc.utils import grab
import torch
import numpy as np
import neumc
import matplotlib.pyplot as plt
from neumc.utils.mixture import DataLoader, TemperedMixture
from scipy.stats import linregress
from neumc.mc import mixture_metropolize
import neumc.physics.u1 as u1
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "notebooks"))

beta = 4
torch_device = 'cpu'
L = 8
u1_action = u1.U1GaugeAction(beta)
path = r"C:\Users\franc\OneDrive\Desktop\Data\L8_8x8_24_mix"

dataloader = DataLoader(path_to_folder=path, beta_min=1, beta_max=beta, step=1, samples_size=2**12, L=L)
data = dataloader.load_data()
models = dataloader.load_models()

mixture = TemperedMixture(dataloader=dataloader, changes=10000, device = torch_device, L = L)
all_samples = mixture.sampler(power = 1.5)
mix, log_q = mixture.mix_builder(models = models, device = torch_device)

lp_mix = -neumc.utils.batch_function.batch_action(mixture.new_samples, batch_size=1024, action=u1_action, device=torch_device)
u_p, s_p, s_q, accepted = mixture_metropolize(mixture = mix, log_q= log_q, log_p= lp_mix, samples_q = mixture.new_samples)

print("Accept rate is:", float(accepted.count_nonzero()) / len(accepted) * 100, "%")

fit_2x1 = linregress(s_q, s_p)
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect(1)
ax.set_xlabel(r"$\log q$")
ax.set_ylabel(r"$\log P$")
lqs = np.linspace(s_q.min(), s_q.max(), 100)
ax.scatter(s_q, s_p, s=5, alpha=0.25)
ax.plot(lqs, lqs * fit_2x1.slope + fit_2x1.intercept, color="red", zorder=10)
ax.text(0.15, 0.85, f"$\\log P = {fit_2x1.slope:.3}\\log q+{fit_2x1.intercept:.3f}$", transform=ax.transAxes,)

plt.savefig(f"out_u1/S{beta}.png", bbox_inches="tight")

Q = grab(u1.topo_charge(u_p))  #!here plaquettes are computed internally: topo_charge calls compute_u1_plaq;

plt.figure(figsize=(5, 3.5), dpi=125)
np.savetxt(f"out_u1/Q{beta}.txt", Q)
plt.plot(Q)
plt.title(r"$\beta = $" + f"{beta}")
plt.xlabel(r"$t_{MC}$")
plt.ylabel(r"topological charge $Q$")
plt.savefig(f"out_u1/Q{beta}.png", bbox_inches="tight")
plt.close()