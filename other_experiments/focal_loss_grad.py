# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import matplotlib.pyplot as plt
import numpy as np

# Visualize Focal Loss and Its Gradient

# +
def focal_loss(pt, *, alpha=0.5, gamma=2.):
    return - alpha * np.power(1 - pt, gamma) * np.log(pt)


def focal_loss_grad(pt, *, alpha=0.5, gamma=2.):
    return alpha * np.power(1 - pt, gamma - 1) * (gamma * np.log(pt) - (1 - pt) / pt)


def cross_entropy_loss(pt, *, alpha=0.5):
    return -alpha * np.log(pt)


def cross_entropy_loss_grad(pt, *, alpha=0.5):
    return -alpha / pt


fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(12, 8))
pt = np.linspace(0.001, 1., 1000)

ax = axes[0]
loss = cross_entropy_loss(pt)
ax.plot(pt, loss, label='Cross Entropy', c='black')

# Cross entropy loss grad.
ax = axes[1]
grad = cross_entropy_loss_grad(pt)
ax.plot(pt, grad, label='Cross Entropy', c='black')

for gamma in [0.1, 0.5, 1., 2., 2.5, 3., 3.5, 4., 5., 10.]:    
    # Plot loss.
    ax = axes[0]
    loss = focal_loss(pt, gamma=gamma)
    ax.plot(pt, loss, label=f'{gamma=}')

    # Plot Grad.
    ax = axes[1]
    grad = focal_loss_grad(pt, gamma=gamma)
    ax.plot(pt, grad, label=f'{gamma=}')


axes[0].legend()
axes[1].legend()
axes[0].set_title('Focal Loss')
axes[1].set_title('Focal Loss Gradient')
fig.tight_layout()
# -

grad_plot = axes[1]
grad_plot.set_ylim(-300, 0)
grad_plot.set_xlim(0., .02)
fig

grad_plot = axes[1]
grad_plot.set_ylim(-50, 0)
grad_plot.set_xlim(0., .1)
fig

grad_plot = axes[1]
grad_plot.set_ylim(-10, 0)
grad_plot.set_xlim(.1, .5)
fig

grad_plot = axes[1]
grad_plot.set_ylim(-5, 0)
grad_plot.set_xlim(.5, .8)
fig

grad_plot = axes[1]
grad_plot.set_ylim(-5, 0)
grad_plot.set_xlim(.3, .6)
fig


# # Negative Focal Loss

# +
def neg_focal_loss(pt, *, gamma=1.):
    # return -np.log(pt) / ((1 - pt) ** gamma)
    return -np.exp(1 - pt) * np.log(pt)

def neg_focal_loss_grad(pt, *, gamma=1.):
    # return -1/(pt * (1 - pt)**gamma) + gamma * np.log(pt)/((1 - pt)**gamma)
    return np.exp(1 - pt) * np.log(pt) - np.exp(1 - pt) / pt

fig, axes = plt.subplots(nrows=2)
ax = axes[0]
ax.plot(pt, neg_focal_loss(pt, gamma=0.5), label='Neg focal loss')
ax.plot(pt, cross_entropy_loss(pt), label='Cross Entropy')
ax.legend()

ax = axes[1]
ax.plot(pt, neg_focal_loss_grad(pt, gamma=0.5), label='Neg focal loss grad')
ax.plot(pt, cross_entropy_loss_grad(pt), label='Cross Entropy Grad')
ax.legend()
fig.tight_layout()
# -

axes[1].set_xlim(0., 0.1)
fig

axes[1].set_xlim(0.1, 0.5)
axes[1].set_ylim(-50, 0)
fig

axes[1].set_xlim(0.25, 0.8)
axes[1].set_ylim(-10, 0)
fig

axes[1].set_xlim(0.55, 1.0)
axes[1].set_ylim(-10, 0)
fig


# # Charbonnier Loss

# +
def charbonnier_loss(pt, *, alpha=2, epsilon=1e-6):
    return (np.ones_like(pt) - pt + epsilon) ** (1. / alpha)


def charbonnier_loss_grad(pt, *, alpha=2, epsilon=1e-6):
    alpha_inv = 1. / alpha
    return -alpha_inv * (np.ones_like(pt) - pt + epsilon) ** (alpha_inv - 1)


fig, axes = plt.subplots(nrows=2)
ax = axes[0]
ax.plot(pt, square_root(pt), label='Charbonnier')
ax.plot(pt, cross_entropy_loss(pt), label='Cross Entropy')
ax.legend()

ax = axes[1]
ax.plot(pt, charbonnier_loss_grad(pt), label='Charbonnier')
ax.plot(pt, cross_entropy_loss_grad(pt), label='Cross Entropy Grad')
ax.set_ylim(-10, 0.1)
ax.legend()
fig.tight_layout()
