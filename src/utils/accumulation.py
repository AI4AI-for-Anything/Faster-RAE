"""Bounded-memory backward passes for one effective generator update."""
from contextlib import nullcontext
import torch


def sync_context(model, final):
    return nullcontext() if final or not hasattr(model, 'no_sync') else model.no_sync()


def adaptive_weight(recon_grads, gan_grads, maximum):
    return (recon_grads.norm() / (gan_grads.norm() + 1e-6)).clamp(0, maximum).detach()


def generator_backward(model, batches, forward_losses, last_layer, *, use_gan,
                       disc_weight, max_d_weight, scaler=None):
    """Mean gradients over equal microbatches; caller steps optimizer/EMA once.

    With accumulation+GAN, first sum last-layer gradients, then replay each
    microbatch with the same torch RNG state for the weighted backward pass.
    Only one microbatch graph is retained. Weight remains rank-local, as upstream.
    forward_losses must not mutate model buffers (G train has no running norms;
    D stays eval). It returns scalar recon/lpips/gan/recon_total losses.
    """
    count = len(batches)
    weight, states = None, []
    device = last_layer.device
    if use_gan and count > 1:
        recon_sum, gan_sum = torch.zeros_like(last_layer), torch.zeros_like(last_layer)
        raw_model = getattr(model, 'module', model)
        for batch in batches:
            states.append((torch.get_rng_state(),
                           torch.cuda.get_rng_state(device) if device.type == 'cuda' else None))
            # No DDP forward/reducer for this diagnostic autograd.grad pass.
            with sync_context(model, False):
                losses = forward_losses(raw_model, batch)
                rg = torch.autograd.grad(losses['recon_total'], last_layer, retain_graph=True)[0]
                gg = torch.autograd.grad(losses['gan'], last_layer)[0]
            recon_sum.add_(rg.detach(), alpha=1 / count)
            gan_sum.add_(gg.detach(), alpha=1 / count)
            del losses, rg, gg
        weight = adaptive_weight(recon_sum, gan_sum, max_d_weight)
        del recon_sum, gan_sum
    metrics = {}
    for index, batch in enumerate(batches):
        if states:
            torch.set_rng_state(states[index][0])
            if states[index][1] is not None:
                torch.cuda.set_rng_state(states[index][1], device)
        # Both forward and backward belong inside no_sync, per the DDP contract.
        with sync_context(model, index == count - 1):
            losses = forward_losses(model, batch)
            if weight is None:
                if use_gan:
                    rg = torch.autograd.grad(losses['recon_total'], last_layer, retain_graph=True)[0]
                    gg = torch.autograd.grad(losses['gan'], last_layer, retain_graph=True)[0]
                    weight = adaptive_weight(rg, gg, max_d_weight)
                    del rg, gg
                else:
                    weight = losses['recon_total'].new_zeros(())
            total = losses['recon_total'] + disc_weight * weight * losses['gan']
            loss = total / count
            (scaler.scale(loss) if scaler else loss).backward()
        for key, value in (losses | {'total': total}).items():
            metrics[key] = metrics.get(key, 0) + value.detach() / count
        del losses, total, loss
    return metrics, weight
