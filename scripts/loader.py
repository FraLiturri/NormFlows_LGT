from matplotlib import pyplot as plt
from neumc.utils import grab
from neumc.utils.stats_utils import torch_bootstrap, torch_bootstrapf
import torch
import numpy as np
import sys, os
import neumc, neumc.physics.u1 as u1
from neumc.mc import metropolize
from scipy.stats import linregress
import neumc.nf.cs_coupling as cs_cpl


torch_device = "cuda" if torch.cuda.is_available() else "cpu"
float_type = torch.float32
batch_size = 1024
L = 8
lattice_shape = (L, L)
link_shape = (2, L, L)
beta = float(sys.argv[1])
u1_action = u1.U1GaugeAction(beta)
F_exact = -u1.logZ(L, beta=beta) - 2 * L * L * np.log(2 * np.pi)

# Model parameters;
hidden_channels = [8, 8]
kernel_size = 3
in_channels = 6
dilation = 1
n_layers = 24
n_knots = 9

masks = neumc.nf.gauge_masks.sch_2x1_masks_gen(
    lattice_shape=(L, L), float_dtype=float_type, device=torch_device
)

def make_plaq_coupling(mask):
    out_channels = 3 * (n_knots - 1) + 1
    net = neumc.nf.nn.make_conv_net(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        use_final_tanh=False,
        dilation=dilation,
    )
    net.to(torch_device)
    return cs_cpl.CSCoupling(n_knots=n_knots, net=net, mask=mask)

loops_function = lambda x: [u1.compute_u1_2x1_loops(x)]
layers = neumc.nf.u1_equiv.make_u1_equiv_layers(
    loops_function=loops_function,
    make_plaq_coupling=make_plaq_coupling,
    masks=masks,
    n_layers=n_layers,
    device=torch_device,
)

prior = neumc.nf.prior.MultivariateUniform(
    torch.zeros(link_shape), 2 * torch.pi * torch.ones(1), device=torch_device
)
z = prior.sample_n(12)
plaq = u1.compute_u1_plaq(z, mu=0, nu=1)
model = {"prior": prior, "layers": layers}
history = {"dkl": [], "std_dkl": [], "loss": [], "ess": []}

MODEL_WEIGHTS_PATH = "out_u1/weights_7.0.pt"
if os.path.exists(MODEL_WEIGHTS_PATH):
    try:
        print(f"Loading existing weights from {MODEL_WEIGHTS_PATH}")
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")
        model["layers"].load_state_dict(state_dict)  # or use strict=False if appropriate
        print("Weights loaded successfully!")
    except (RuntimeError, KeyError, IOError) as e:
        print(f"Failed to load weights ({e}). Starting training with random initialization")
else:
    print(f"No existing weights found at {MODEL_WEIGHTS_PATH}. Starting training with random initialization")

u_2x1, lq_2x1 = neumc.nf.flow.sample(
    n_samples=2**16, batch_size=2**10, prior=prior, layers=layers
)
lp_2x1 = -neumc.utils.batch_function.batch_action(
    u_2x1, batch_size=1024, action=u1_action, device=torch_device
)
ess_2x1 = neumc.utils.ess(lp_2x1, lq_2x1)
print(f"ESS: {ess_2x1}")

fit_2x1 = linregress(lq_2x1, lp_2x1)
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect(1)
ax.set_xlabel(r"$\log q$")
ax.set_ylabel(r"$\log P$")
lqs = np.linspace(lq_2x1.min(), lq_2x1.max(), 100)
ax.scatter(lq_2x1, lp_2x1, s=5, alpha=0.25)
ax.plot(lqs, lqs * fit_2x1.slope + fit_2x1.intercept, color="red", zorder=10)
ax.text(
    0.15,
    0.85,
    f"$\\log P = {fit_2x1.slope:.3}\\log q+{fit_2x1.intercept:.3f}$",
    transform=ax.transAxes,
)
plt.savefig(f"out_u1/u1_rs_lr.png", bbox_inches="tight")

lw_2x1 = lp_2x1 - lq_2x1
F_q_2x1, F_q_std_2x1 = torch_bootstrap(-lw_2x1, n_samples=100, binsize=1)
F_nis_2x1, F_nis_std_2x1 = torch_bootstrapf(
    lambda x: -(torch.special.logsumexp(x, 0) - np.log(len(x))),
    lw_2x1,
    n_samples=100,
    binsize=1,
)

u_p, s_p, s_q, accepted = metropolize(u_2x1, lq_2x1, lp_2x1)

print("Accept rate is:", float(accepted.count_nonzero()) / len(accepted) * 100, "%")
print(f"F_q = {F_q_2x1:.4f}+/-{F_q_std_2x1:.4f}  F_q-F_exact = {F_q_2x1 - F_exact:.5f}")
print(
    f"F_NIS = {F_nis_2x1:.3f}+/-{F_nis_std_2x1:.4f} F_NIS-F_exact = {F_nis_2x1-F_exact:.4f}"
)

Q = grab(u1.topo_charge(u_p))
plt.figure(figsize=(5, 3.5), dpi=125)
np.savetxt(f"out_u1/Q{beta}.txt", Q)
plt.plot(Q)
plt.title(r"$\beta = $" + f"{beta}")
plt.xlabel(r"$t_{MC}$")
plt.ylabel(r"topological charge $Q$")
plt.savefig(f"out_u1/u1_rs_Q.png", bbox_inches="tight")


plt.figure(figsize=(5, 3.5), dpi=125)
plt.hist(lq_2x1, bins=1000)
plt.xlabel(r"$\log q$")
plt.ylabel(r"$P(\log q)$")
plt.savefig(f"out_u1/u1_rs_lq.png", bbox_inches="tight")
plt.show()

plt.figure(figsize=(5, 3.5), dpi=125)
plt.hist(lp_2x1, bins=1000)
plt.xlabel(r"$\log P$")
plt.ylabel(r"$P(\log P)$")
plt.savefig(f"out_u1/u1_rs_lp.png", bbox_inches="tight")
plt.show()