# Normalizing Flows for $U(1)$ LGT

Here's a list of observations and facts useful for my personal understanding of Normalizing Flows for Lattice Gauge Theories (LGT). 

- [Normalizing Flows for $U(1)$ LGT](#normalizing-flows-for-u1-lgt)
  - [How is the training done?](#how-is-the-training-done)
  - [How many samples from prior distribution are needed?](#how-many-samples-from-prior-distribution-are-needed)
  - [How the masking works?](#how-the-masking-works)

## How is the training done?
The optimal parameters are found by minimizing the D-KL (forward or reverse) loss:
$$
\begin{equation}
\begin{align*}
        \text{reverse:} \ \ \ D_{KL}(q|p) &= \int_{\Omega} q(\phi|\theta) \big( \log q(\phi|\theta) - \log p(\phi) \big) \ d\phi \\
        \text{forward:} \ \ \ D_{KL}(p|q) &= \int_{\Omega} q(\phi|\theta) \big( \log p(\phi) - \log q(\phi|\theta) \big) \ d\phi
\end{align*}
\end{equation}
$$
## How many samples from prior distribution are needed?

## How the masking works? 

