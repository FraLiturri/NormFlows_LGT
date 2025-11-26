import torch 
import numpy as np 
import sys
import os
import time
import matplotlib.pyplot as plt
from scipy.special import iv
from scipy.stats import linregress

import neumc 
import neumc.physics.u1 as u1
import neumc.nf.flow as nf
import neumc.nf.u1_equiv as equiv
import neumc.nf.cs_coupling as cs_cpl

from neumc.training.gradient_estimator import RTEstimator, PathGradientEstimator, REINFORCEEstimator
from neumc.utils.stats_utils import torch_bootstrap, torch_bootstrapf
from neumc.utils import grab
import neumc.utils.metrics as um
from  neumc.nf.u1_model_asm import assemble_model_from_dict
from neumc.utils import ess, ess_lw
from neumc.mc import metropolize

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

from mask_plots import loop, plot_plaq_mask 
from  live_plot import  init_live_plot, update_plots

torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
float_type = torch.float32

batch_size = 1024
L = 8 
lattice_shape = (L, L)
link_shape = (2, L, L)
beta = 1
u1_action = u1.U1GaugeAction(beta)
F_exact = -u1.logZ(L, beta=beta) - 2 * L * L * np.log(2 * np.pi)

#Model parameters; 
hidden_channels = [8,8]
kernel_size = 3
in_channels = 6
dilation = 1
n_layers = 16
n_knots = 9

#Training parameters; 
N_era = 1
N_epoch = 1
base_lr = .001
lambda_l2 = 1e-4
print_freq = N_era*N_epoch # epochs
plot_freq = 1 # epochs

masks = neumc.nf.gauge_masks.sch_2x1_masks_gen(lattice_shape=(L,L), float_dtype=float_type, device=torch_device)
def make_plaq_coupling(mask):
    out_channels = 3 * (n_knots - 1) + 1
    net = neumc.nf.nn.make_conv_net(in_channels=in_channels,
                        out_channels=out_channels,
                        hidden_channels=hidden_channels,
                        kernel_size=kernel_size,
                        use_final_tanh=False,
                        dilation=dilation)
    net.to(torch_device)
    return cs_cpl.CSCoupling(n_knots=n_knots, net=net, mask=mask)

loops_function = lambda x: [u1.compute_u1_2x1_loops(x)]
layers = neumc.nf.u1_equiv.make_u1_equiv_layers(loops_function=loops_function,
                                                make_plaq_coupling=make_plaq_coupling,
                                                masks=masks, n_layers=n_layers, device=torch_device)

prior = neumc.nf.prior.MultivariateUniform(torch.zeros(link_shape), 2*torch.pi*torch.ones(1), device=torch_device)
z = prior.sample_n(12)
plaq = u1.compute_u1_plaq(z,mu=0,nu=1)

model={
  'prior':prior,
  'layers': layers
}

history = {
    'dkl' : [],
    'std_dkl': [],
    'loss' : [],
    'ess' : []
}


MODEL_WEIGHTS_PATH = "weights.pt"
# Check if weights file exists and load them
if os.path.exists(MODEL_WEIGHTS_PATH):
    print(f"Loading existing weights from {MODEL_WEIGHTS_PATH}")
    state_dict = torch.load(MODEL_WEIGHTS_PATH)
    model['layers'].load_state_dict(state_dict)
    print("Weights loaded successfully!")
else:
    print(f"No existing weights found at {MODEL_WEIGHTS_PATH}")
    print("Starting training with random initialization")


optimizer = torch.optim.Adam(model['layers'].parameters(), lr=base_lr, weight_decay=lambda_l2)
train_step3 = PathGradientEstimator(prior, layers, u1_action)

[plt.close(plt.figure(fignum)) for fignum in plt.get_fignums()] # close all existing figures
live_plot = init_live_plot(N_era, N_epoch, metric='dkl')
live_plot['fig'].suptitle(r"Training for $\beta = $"+ f"{beta}") # Changed from live_plot['ax'].set_title to live_plot['fig'].suptitle
start_time = time.time()

for era in range(N_era):
    for epoch in range(N_epoch):
        optimizer.zero_grad()
        loss, log_q, log_p = train_step3.step(batch_size=batch_size)
        optimizer.step()

        um.add_metrics(history, {
                "loss": loss.cpu().numpy().item(),
                "ess": neumc.utils.ess(log_p, log_q).cpu().numpy().item(),
                "dkl": neumc.utils.dkl(log_p, log_q).cpu().numpy().item(),
                "std_dkl": (log_p - log_q).std().cpu().numpy().item(),
        })

        if epoch % print_freq == 0:
            avg = um.average_metrics(history, N_epoch, history.keys())
            ellapsed_time = time.time()-start_time
            print(f"Era {era:3d} epoch {epoch:4d} ellapsed time {ellapsed_time:.1f}")
            um.print_dict(avg)


# Save model weights after training
print(f"\nSaving model weights to {MODEL_WEIGHTS_PATH}")
torch.save(model['layers'].state_dict(), MODEL_WEIGHTS_PATH)
print("Model weights saved successfully!")
print(f"File size: {os.path.getsize(MODEL_WEIGHTS_PATH) / (1024**2):.2f} MB")

#Sampling: 
u_2x1, lq_2x1 = neumc.nf.flow.sample(n_samples=2**16, batch_size=2**10, prior=prior, layers=layers)
lp_2x1 = -neumc.utils.batch_function.batch_action(u_2x1,batch_size=1024, action=u1_action,
                                           device=torch_device)
ess_2x1 = neumc.utils.ess(lp_2x1, lq_2x1)
print(f"ESS: {ess_2x1}")

fit_2x1 = linregress(lq_2x1, lp_2x1)
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect(1);
ax.set_xlabel(r"$\log q$");
ax.set_ylabel(r"$\log P$")
lqs = np.linspace(lq_2x1.min(), lq_2x1.max(), 100);
ax.scatter(lq_2x1, lp_2x1, s=5, alpha=0.25);
ax.plot(lqs, lqs * fit_2x1.slope + fit_2x1.intercept, color='red', zorder=10);
ax.text(0.15, .85, f"$\\log P = {fit_2x1.slope:.3}\\log q+{fit_2x1.intercept:.3f}$", transform=ax.transAxes);
plt.savefig(f'u1_rs_reg{beta}_L{L}.png', bbox_inches='tight')

lw_2x1 = lp_2x1-lq_2x1
F_q_2x1, F_q_std_2x1 = torch_bootstrap(-lw_2x1, n_samples=100, binsize=1)
print(f"F_q = {F_q_2x1:.4f}+/-{F_q_std_2x1:.4f}  F_q-F_exact = {F_q_2x1 - F_exact:.5f}")
F_nis_2x1, F_nis_std_2x1 = torch_bootstrapf(
  lambda x: -(torch.special.logsumexp(x, 0) - np.log(len(x))),
                                    lw_2x1, n_samples=100, binsize=1)
print(f"F_NIS = {F_nis_2x1:.3f}+/-{F_nis_std_2x1:.4f} F_NIS-F_exact = {F_nis_2x1-F_exact:.4f}")

u_p, s_p, s_q, accepted = metropolize(u_2x1, lq_2x1, lp_2x1)
Q = grab(u1.topo_charge(u_p))
plt.figure(figsize=(5, 3.5), dpi=125)
plt.plot(Q)

plt.title(r"$\beta = $"+ f"{beta}")
plt.xlabel(r'$t_{MC}$')
plt.ylabel(r'topological charge $Q$')
plt.savefig(f'u1_rs_Q_beta{beta}_L{L}.png', bbox_inches='tight')
plt.show()