"""
SoftOpt (PyTorch): a COMPLETE, standalone torch.optim.Optimizer -- not a
wrapper. Use it exactly like torch.optim.Adam:

    opt = SoftOpt(model.parameters(), model, loss_fn, lr=1e-3)
    loss = opt.step(batch)

One step() call does the Adam-style base update AND the Newton-style
correction, using genuine forward-mode automatic differentiation
(torch.func.jvp) to get an exact directional derivative and curvature of
your loss along a random probe direction -- mathematically the same
quantity a hand-written Klein-Maimon soft-number propagation would give
(dual numbers are isomorphic to single-axis soft numbers per the book's
own Appendix A.2), computed via PyTorch's own AD engine instead of
hand-written soft-number code.

Requirement (the "known computation graph" criterion, validated across
nine domains): loss_fn(model, batch) must be an exactly re-evaluatable,
differentiable function -- e.g. a full-batch loss over a fixed dataset,
a physics/circuit simulator, a known projection model. If your loss is
a small stochastic minibatch sample that changes meaning between calls
(the classic deep-RL moving-target problem), this mechanism is not
validated to help -- use plain Adam/SGD instead.
"""
import numpy as np
import torch
import torch.func as tfunc
from torch.optim import Optimizer
from torch.nn.utils.stateless import _reparametrize_module


class SoftOpt(Optimizer):
    def __init__(self, params, model, loss_fn, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 newton_lo=-0.5, newton_hi=0.5, eta_fallback=0.3, h_fd=1e-4, seed=None,
                 _test_frozen_constant=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.model = model
        self.loss_fn = loss_fn
        self.newton_lo = newton_lo
        self.newton_hi = newton_hi
        self.eta_fallback = eta_fallback
        self.h_fd = h_fd
        self.rng = np.random.default_rng(seed)
        self._test_frozen_constant = _test_frozen_constant

    def _adam_update(self):
        for group in self.param_groups:
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] += 1
                state['exp_avg'].mul_(b1).add_(p.grad, alpha=1 - b1)
                state['exp_avg_sq'].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                bc1 = 1 - b1 ** state['step']
                bc2 = 1 - b2 ** state['step']
                denom = (state['exp_avg_sq'] / bc2).sqrt().add_(group['eps'])
                p.data.addcdiv_(state['exp_avg'], denom, value=-group['lr'] / bc1)

    def step(self, batch, _test_magnitude_source=None, _test_foreign_sampler=None):
        """batch: whatever loss_fn(model, batch) needs. Returns the loss
        (a float) from the pre-correction forward pass.
        _test_* are TEST-ONLY hooks for the causal-validation harness --
        normal usage never sets these.
        """
        all_params = [p for group in self.param_groups for p in group['params']]

        # --- standard forward/backward for the Adam step ---
        with torch.enable_grad():
            self.zero_grad()
            loss = self.loss_fn(self.model, batch)
            loss.backward()
        loss_value = float(loss.detach())

        with torch.no_grad():
            self._adam_update()

        if _test_magnitude_source == 'plain':
            return loss_value

        # --- Newton-style correction via genuine forward-mode AD ---
        with torch.no_grad():
            named = {id(p): name for name, p in self.model.named_parameters()}
            all_named = dict(self.model.named_parameters())
            names = [n for n, p in all_named.items() if any(p is q for q in all_params)]
            flat0 = torch.cat([all_named[n].detach().reshape(-1) for n in names])
            n_total = flat0.numel()
            delta = torch.tensor(self.rng.choice([-1.0, 1.0], size=n_total).astype(np.float32))

            fixed_vals = {n: p.detach() for n, p in all_named.items() if n not in names}

            def make_scalar_loss(the_batch):
                def scalar_loss(flat):
                    pd = {}
                    idx = 0
                    for n in names:
                        p = all_named[n]
                        nn_ = p.numel()
                        pd[n] = flat[idx:idx + nn_].reshape(p.shape)
                        idx += nn_
                    pd.update(fixed_vals)
                    with _reparametrize_module(self.model, pd):
                        return self.loss_fn(self.model, the_batch)
                return scalar_loss

            def D1_D2_at(point_theta, point_delta, point_batch):
                scalar_loss = make_scalar_loss(point_batch)
                _, d1 = tfunc.jvp(scalar_loss, (point_theta,), (point_delta,))
                def d_along(fp):
                    _, d = tfunc.jvp(scalar_loss, (fp,), (point_delta,))
                    return d
                _, d2 = tfunc.jvp(d_along, (point_theta,), (point_delta,))
                return float(d1), float(d2)

            D1, D2_real = D1_D2_at(flat0, delta, batch)
            activate = D2_real > 1e-8

            if _test_magnitude_source in (None, 'real'):
                D2_used = D2_real
            elif _test_magnitude_source == 'frozen':
                D2_used = self._test_frozen_constant
            elif _test_magnitude_source in ('foreign_point_and_dir', 'foreign_point_same_dir'):
                theta_f, batch_f = _test_foreign_sampler(self.rng)
                delta_f = (torch.tensor(self.rng.choice([-1.0, 1.0], size=n_total).astype(np.float32))
                           if _test_magnitude_source == 'foreign_point_and_dir' else delta)
                _, D2_f = D1_D2_at(theta_f, delta_f, batch_f)
                D2_used = abs(D2_f)
            else:
                raise ValueError(_test_magnitude_source)

            if activate and D2_used > 1e-9:
                t_star = float(np.clip(-D1 / D2_used, self.newton_lo, self.newton_hi))
            else:
                t_star = float(np.clip(-self.eta_fallback * D1, self.newton_lo, self.newton_hi))

            new_flat = flat0 + t_star * delta
            idx = 0
            for n in names:
                p = all_named[n]
                nn_ = p.numel()
                p.data.copy_(new_flat[idx:idx + nn_].reshape(p.shape))
                idx += nn_

        return loss_value
