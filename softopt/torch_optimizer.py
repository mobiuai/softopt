"""
SoftOptTorch: the SoftOpt step as a standalone torch.optim.Optimizer, for
problems whose known model is written in PyTorch.

    from softopt import SoftOptTorch

    def loss_fn(model, batch):
        t, measured = batch
        return ((model(t) - measured) ** 2).mean()

    opt = SoftOptTorch(model.parameters(), model, loss_fn, lr=1e-2)
    for step in range(num_steps):
        loss = opt.step((t, measured))   # forward + backward + update, in one call

Each step() does exactly what the numpy SoftOpt does:

  1. an Adam base update from the gradient, then
  2. a bounded Newton correction along a random +-1 probe direction delta.

The correction needs two numbers about the loss along delta, at the
post-Adam point: the directional derivative D1 and the curvature D2.
Both come from PyTorch's forward-mode AD (torch.func.jvp), nested once:
the outer JVP returns D1 as its value and D2 as its tangent, in a single
forward pass. That is the single-axis soft-number propagation of
Foundations of Soft Logic (dual numbers are isomorphic to single-axis
soft numbers, Appendix A.2), executed by PyTorch's own engine -- so it
runs on whatever device the model is on (CPU, CUDA, MPS). D2 here is
exact, not a finite difference.

Requirement: loss_fn(model, batch) must be a fixed objective that can be
re-evaluated exactly -- the "known computation graph" criterion in the
README. A target that changes between calls is outside SoftOpt's scope.

Scale: in the correction each parameter moves by at most `newton_bound`,
which defaults to the learning rate.

BatchNorm: the correction pass uses batch statistics, exactly as the
training forward does, but never updates the running mean/var. PyTorch's
own forward-over-forward rule for batch_norm is about 1% off in the
second derivative (D1 stays exact); for BN-free models D2 is exact.
MaxPool layers are routed through the with-indices kernel during the
correction pass, which has a forward-AD rule on every backend.
"""
from contextlib import contextmanager

import torch
from torch.func import functional_call, jvp
from torch.optim import Optimizer


class _LossModule(torch.nn.Module):
    """Lets functional_call drive loss_fn(model, batch) with substituted
    parameters, using only public torch.func API."""

    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self._loss_fn = loss_fn

    def forward(self, batch):
        return self._loss_fn(self.model, batch)


@contextmanager
def _freeze_bn_stats(model):
    bns = [m for m in model.modules()
           if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
           and m.track_running_stats]
    for m in bns:
        m.track_running_stats = False
    try:
        yield
    finally:
        for m in bns:
            m.track_running_stats = True


_MAXPOOL_FN = {
    torch.nn.MaxPool1d: torch.nn.functional.max_pool1d,
    torch.nn.MaxPool2d: torch.nn.functional.max_pool2d,
    torch.nn.MaxPool3d: torch.nn.functional.max_pool3d,
}


@contextmanager
def _forward_ad_safe_pooling(model):
    """On some backends (e.g. Apple MPS) plain max_pool has no forward-AD
    rule, while max_pool_with_indices does on every backend. Route MaxPool
    modules through the with-indices kernel for the correction pass only.
    The output is identical; only the kernel that computes it changes."""
    patched = []
    for m in model.modules():
        fn = _MAXPOOL_FN.get(type(m))
        if fn is None or m.return_indices:
            continue

        def fwd(x, m=m, fn=fn):
            return fn(x, m.kernel_size, m.stride, m.padding, m.dilation,
                      ceil_mode=m.ceil_mode, return_indices=True)[0]
        m.forward = fwd
        patched.append(m)
    try:
        yield
    finally:
        for m in patched:
            del m.forward          # back to the class's own forward


class SoftOptTorch(Optimizer):
    def __init__(self, params, model, loss_fn, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0, newton_bound=None,
                 eta_fallback=0.3, curvature_floor=1e-8, correct_every=1,
                 seed=None, _test_frozen_constant=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.model = model
        self.loss_fn = loss_fn
        self.newton_bound = lr if newton_bound is None else newton_bound
        self.eta_fallback = eta_fallback
        self.curvature_floor = curvature_floor
        self.correct_every = max(1, int(correct_every))
        self._wrapper = _LossModule(model, loss_fn)
        self._gen = {}           # one torch.Generator per device
        self._seed = seed
        self._n_steps = 0
        self._prev_D2 = None
        self._test_frozen_constant = _test_frozen_constant
        # diagnostics from the most recent correction
        self.last = {}

    # ------------------------------------------------------------------
    def _generator(self, device):
        key = str(device)
        if key not in self._gen:
            g = torch.Generator(device=device)
            g.manual_seed(self._seed if self._seed is not None
                          else torch.seed() % (2 ** 63))
            self._gen[key] = g
        return self._gen[key]

    def _adam_update(self):
        for group in self.param_groups:
            b1, b2 = group['betas']
            lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] += 1
                if wd != 0.0:                       # decoupled (AdamW)
                    p.mul_(1 - lr * wd)
                state['exp_avg'].mul_(b1).add_(p.grad, alpha=1 - b1)
                state['exp_avg_sq'].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                bc1 = 1 - b1 ** state['step']
                bc2 = 1 - b2 ** state['step']
                denom = (state['exp_avg_sq'] / bc2).sqrt_().add_(eps)
                p.addcdiv_(state['exp_avg'], denom, value=-lr / bc1)

    def _directional_D1_D2(self, batch, delta):
        """Exact D1 = dL/dt and D2 = d2L/dt2 of L(theta + t*delta) at t=0,
        from one nested forward-mode pass."""
        own = {f"model.{n}": p.detach() for n, p in self._names_params}
        rest = {f"model.{n}": b for n, b in self.model.named_buffers()}
        frozen = {f"model.{n}": p.detach() for n, p in self.model.named_parameters()
                  if f"model.{n}" not in own}
        tang = {f"model.{n}": d for (n, _), d in zip(self._names_params, delta)}

        def L(pd):
            return functional_call(self._wrapper, {**pd, **frozen, **rest}, (batch,))

        def dL(pd):
            return jvp(L, (pd,), (tang,))[1]

        with _freeze_bn_stats(self.model), _forward_ad_safe_pooling(self.model):
            D1, D2 = jvp(dL, (own,), (tang,))
        return float(D1), float(D2)

    # ------------------------------------------------------------------
    def step(self, batch, _test_magnitude_source=None):
        """One complete step on `batch`. Returns the training loss (float)
        from the forward pass that produced the gradient.

        _test_magnitude_source is a TEST-ONLY hook for ablations:
          'plain'  -> Adam update only, through this exact code path
          'frozen' -> correction uses a constant in place of the real D2
          'lagged' -> correction uses the previous step's real D2 (same
                      scale, wrong batch/point/direction)
        Normal use never sets it.
        """
        if not hasattr(self, '_names_params'):
            ids = {id(p) for g in self.param_groups for p in g['params']}
            self._names_params = [(n, p) for n, p in self.model.named_parameters()
                                  if id(p) in ids]

        with torch.enable_grad():
            self.zero_grad(set_to_none=True)
            loss = self.loss_fn(self.model, batch)
            loss.backward()
        loss_value = float(loss.detach())

        with torch.no_grad():
            self._adam_update()
        self._n_steps += 1

        if _test_magnitude_source == 'plain' or self._n_steps % self.correct_every:
            return loss_value

        params = [p for _, p in self._names_params]
        delta = []
        for p in params:
            g = self._generator(p.device)
            r = torch.randint(0, 2, p.shape, generator=g, device=p.device)
            delta.append(r.to(p.dtype).mul_(2).sub_(1))

        D1, D2_real = self._directional_D1_D2(batch, delta)

        if _test_magnitude_source in (None, 'real'):
            D2_used = D2_real
        elif _test_magnitude_source == 'frozen':
            D2_used = self._test_frozen_constant
        elif _test_magnitude_source == 'lagged':
            # genuine curvature, same scale, but from the PREVIOUS step's
            # batch, point and direction: right size, wrong information
            D2_used = self._prev_D2 if self._prev_D2 is not None else D2_real
        else:
            raise ValueError(_test_magnitude_source)

        b = self.newton_bound
        self._prev_D2 = D2_real
        if D2_real > self.curvature_floor and D2_used > self.curvature_floor:
            t_star = -D1 / D2_used
            mode = 'newton'
        else:
            t_star = -self.eta_fallback * D1
            mode = 'fallback'
        t_raw = t_star
        t_star = max(-b, min(b, t_star))

        with torch.no_grad():
            for p, d in zip(params, delta):
                p.add_(d, alpha=t_star)

        self.last = dict(D1=D1, D2=D2_real, t_raw=t_raw, t_star=t_star, mode=mode)
        return loss_value
