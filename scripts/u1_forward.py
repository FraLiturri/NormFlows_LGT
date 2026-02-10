import torch
import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
from scipy.stats import linregress
import neumc
import neumc.physics.u1 as u1
import neumc.nf.cs_coupling as cs_cpl
from neumc.training.forward import ForwardGradientEstimator
from neumc.utils.stats_utils import torch_bootstrap, torch_bootstrapf
from neumc.utils import grab
import neumc.utils.metrics as um
from neumc.mc import metropolize
from neumc.training.gradient_estimator import PathGradientEstimator
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "notebooks"))

torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
float_type = torch.float32

L = 4
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
n_layers = 16
n_knots = 9

# Training parameters;
N_era = 10
N_epoch = int(sys.argv[2])
base_lr = float(sys.argv[3])
beta_step = float(sys.argv[4])
print_freq = 10  # epochs

perform_mix_training = True
masks = neumc.nf.gauge_masks.sch_2x1_masks_gen(lattice_shape=(L, L), float_dtype=float_type, device=torch_device)

def make_plaq_coupling(mask):
    out_channels = 3 * (n_knots - 1) + 1
    net = neumc.nf.nn.make_conv_net(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels, kernel_size=kernel_size, use_final_tanh=False, dilation=dilation)
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

prior = neumc.nf.prior.MultivariateUniform(torch.zeros(link_shape), 2 * torch.pi * torch.ones(1), device=torch_device)
z = prior.sample_n(12)

plaq = u1.compute_u1_plaq(z, mu=0, nu=1)

model = {"prior": prior, "layers": layers}
history = {"dkl": [], "std_dkl": [], "loss": [], "ess": []}

FILE_PATH = f"out_u1/L{L}_{hidden_channels[0]}x{hidden_channels[1]}_{n_layers}"
CHECKPOINT_PATH = f"{FILE_PATH}/checkpoints/checkpoint.pt"
BETA_CHECKPOINT_PATH = f"{FILE_PATH}/checkpoints/checkpoint_{beta}.pt"
DATA_PATH = f"{FILE_PATH}/data.pt"

if not os.path.exists(f"{FILE_PATH}/checkpoints"):
    os.makedirs(f"{FILE_PATH}/checkpoints")

# Carica checkpoint e dati se esistono
checkpoint_loaded = False
data_loaded = False

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=torch_device)
    model["layers"].load_state_dict(checkpoint['model_state_dict'])
    checkpoint_loaded = True
else:
    print(f"No checkpoint found at {CHECKPOINT_PATH}")
    print("Starting training with random initialization")

optimizer = torch.optim.Adam(model["layers"].parameters(), lr=base_lr)
if checkpoint_loaded:
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded.")
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr
        print(f"Learning Rate resettato manualmente a: {base_lr}")


if os.path.exists(DATA_PATH):
    data_obj = torch.load(DATA_PATH, map_location=torch_device)
    if isinstance(data_obj, dict) and 'phi' in data_obj:
        phi_data = data_obj['phi']
    elif isinstance(data_obj, torch.Tensor):
        phi_data = data_obj
    else:
        phi_data = None
        if isinstance(data_obj, dict):
            for v in data_obj.values():
                if isinstance(v, torch.Tensor):
                    phi_data = v
                    break
    if phi_data is None:
        raise RuntimeError(f"Data at {DATA_PATH} has no tensor 'phi' or recognizable tensor values.")
    if phi_data.ndim != 4:
        raise RuntimeError(f"Loaded phi tensor has shape {tuple(phi_data.shape)} but expected 4 dims (n_samples, Nd, L, L).")

    dataset = torch.utils.data.TensorDataset(phi_data)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size= 2**13, shuffle=True, drop_last=True)
    data_iter = iter(dataloader)
    forward_grad = ForwardGradientEstimator(prior=prior, flow=layers, action=u1_action, device=torch_device)
    data_loaded = True

else:
    print(f"No checkpoint found at {DATA_PATH}")
    print("Training with reverse loss.")

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
reverse_grad= PathGradientEstimator(prior = prior, flow = layers, action=u1_action)
start_time = time.time()

for era in range(N_era):
    total_ess = 0
    for epoch in range(N_epoch):
        if perform_mix_training and data_loaded and epoch % 10 == 1 and beta > 1:
            try:
                data_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                data_batch = next(data_iter)
            
            phi_batch = data_batch[0].to(torch_device)
            optimizer.zero_grad()
            loss, log_q, log_p = forward_grad.forward_pass(phi_batch)
            loss.backward()
            optimizer.step()

            um.add_metrics(
                history,
                {
                    "loss": loss.detach().numpy(),
                    "ess": neumc.utils.ess(log_p, log_q).detach().numpy().item(),
                    "dkl": neumc.utils.dkl(log_p, log_q).detach().numpy().item(),
                    "std_dkl": (log_p - log_q).std().detach().numpy().item(),
                },
            )

            avg = um.average_metrics(history, N_epoch, history.keys())
            ellapsed_time = time.time() - start_time
            print(f" <F> Era {era:3d} epoch {epoch:4d} ellapsed time {ellapsed_time:.1f}")
            um.print_dict(avg)
            total_ess += neumc.utils.ess(log_p, log_q).detach().numpy().item()

        else:
            optimizer.zero_grad()
            loss, log_q, log_p = reverse_grad.step(batch_size=1024)
            torch.nn.utils.clip_grad_norm_( model["layers"].parameters(), max_norm=0.8)  # Gradient clipping
            optimizer.step()

            um.add_metrics(
                history,
                {
                    "loss": loss.cpu().numpy().item(),
                    "ess": neumc.utils.ess(log_p, log_q).cpu().numpy().item(),
                    "dkl": neumc.utils.dkl(log_p, log_q).cpu().numpy().item(),
                    "std_dkl": (log_p - log_q).std().cpu().numpy().item(),
                },
            )

            if epoch % print_freq == 0:
                avg = um.average_metrics(history, N_epoch, history.keys())
                ellapsed_time = time.time() - start_time
                print(f" <R> Era {era:3d} epoch {epoch:4d} ellapsed time {ellapsed_time:.1f}")
                um.print_dict(avg)
                total_ess += neumc.utils.ess(log_p, log_q).cpu().numpy().item()
        
    scheduler.step()

with open(f"{FILE_PATH}/ess.txt", "ab") as f:
    np.savetxt(f, np.array([[beta, total_ess]]), delimiter = ' ')

checkpoint = {
    'model_state_dict': model["layers"].state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'model_config': {
        'hidden_channels': hidden_channels,
        'kernel_size': kernel_size,
        'in_channels': in_channels,
        'dilation': dilation,
        'n_layers': n_layers,
        'n_knots': n_knots,
        'lattice_shape': lattice_shape,
    },
    'training_config': {
        'batch_size': 1024,
        'base_lr': base_lr,
    }
}

torch.save(checkpoint, BETA_CHECKPOINT_PATH)
torch.save(checkpoint, CHECKPOINT_PATH)

print(f"Checkpoint saved to {BETA_CHECKPOINT_PATH}")
print(f"Latest checkpoint saved to {CHECKPOINT_PATH}")
print(f"File size: {os.path.getsize(CHECKPOINT_PATH) / (1024**2):.2f} MB")

# Sampling: #!Note that u_2x1 are NOT the plaquettes, but the link variables (angles);
u_2x1, lq_2x1 = neumc.nf.flow.sample(n_samples=2**17, batch_size=2**12, prior=prior, layers=layers)  # shape of u_2x1: (n_samples, 2, L, L);
lp_2x1 = -neumc.utils.batch_function.batch_action(u_2x1, batch_size=1024, action=u1_action, device=torch_device)

lw_2x1 = lp_2x1 - lq_2x1
F_q_2x1, F_q_std_2x1 = torch_bootstrap(-lw_2x1, n_samples=100, binsize=1)
F_nis_2x1, F_nis_std_2x1 = torch_bootstrapf(
    lambda x: -(torch.special.logsumexp(x, 0) - np.log(len(x))),
    lw_2x1,
    n_samples=1000,
    binsize=16,
)

u_p, s_p, s_q, accepted = metropolize(u_2x1, lq_2x1, lp_2x1)
with open(f"{FILE_PATH}/acc.txt", "ab") as f:
    np.savetxt(f, np.array([[beta, float(accepted.count_nonzero()) / len(accepted) * 100]]), delimiter = ' ')

print("Accept rate is:", float(accepted.count_nonzero()) / len(accepted) * 100, "%")
print(f"F_q = {F_q_2x1:.4f}+/-{F_q_std_2x1:.4f}  F_q-F_exact = {F_q_2x1 - F_exact:.5f}")
print(f"F_NIS = {F_nis_2x1:.3f}+/-{F_nis_std_2x1:.4f} F_NIS-F_exact = {F_nis_2x1-F_exact:.4f}")

Q = grab(u1.topo_charge(u_p))  #!here plaquettes are computed internally: topo_charge calls compute_u1_plaq;
#np.savetxt(f"{FILE_PATH}/Q{beta}.txt", Q)

calculator = u1.U1TopologicalSusceptibility(vol = L**L, beta = beta)
chi_values = calculator.compute_chi_theory()

Q_samples = u1.topo_charge(u_2x1).double()
Q2 = Q_samples ** 2

n_boot = 1000
binsize = 128
lw_double = lw_2x1.double()
Q_nis, Q_nis_std = torch_bootstrap(Q2, n_samples=n_boot, binsize=binsize, logweights=lw_double)

chi_nis = (Q_nis / (L*L)).item()
chi_nis_std = (Q_nis_std / (L*L)).item()

print(f"Chi NIS: {chi_nis:.6e} +/- {chi_nis_std:.6e}")
print(f"Chi teorico: {chi_values:.6e}")

data_row = np.array([[beta, chi_nis, chi_nis_std, chi_values]])
with open(f"{FILE_PATH}/chi.txt", "ab") as f:
    np.savetxt(f, data_row, fmt='%.6e', delimiter=' ')

if perform_mix_training:
    next_beta = round(beta + beta_step, 2)
    next_action = u1.U1GaugeAction(next_beta)
    next_u_p, next_s_p, next_s_q, next_accepted = metropolize(u_2x1, lq_2x1, lp_2x1)

    data_to_save = {"phi": next_u_p.cpu()}
    torch.save(data_to_save, f'{FILE_PATH}/data.pt')
