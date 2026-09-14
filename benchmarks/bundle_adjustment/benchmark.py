"""
DEEP BENCHMARK 2/3: Bundle adjustment at REALISTIC scale vs Levenberg-Marquardt.

WHY THIS SCRIPT: at toy scale (8 cameras/25 points = 123 params), scipy's LM
converges almost instantly and there's no room for any correction to help.
This script scales up to a size closer to real structure-from-motion
problems (default: 50 cameras x 200 points = ~900 params) where LM's
per-iteration cost (solving an n x n normal-equations system) becomes
genuinely expensive -- the honest test of whether SoftOpt gives real value
in the regime where switching to LM is no longer cheap/practical.

RUNTIME WARNING: at N_CAMS=50, N_PTS=200 this can take a long time (LM's
cost grows steeply with the parameter count). Start with the SMALL_TEST
config below to confirm everything runs, then increase to REALISTIC.

USAGE:
    pip install numpy scipy
    python3 script2_bundle_adjustment_realistic_scale.py --config small_test
    python3 script2_bundle_adjustment_realistic_scale.py --config realistic
"""
import numpy as np, sys, time, argparse
from scipy.optimize import least_squares
from scipy.stats import wilcoxon
sys.path.insert(0, '.')
from softopt import SoftOpt

CONFIGS = {
    'small_test':  dict(n_cams=8,  n_pts=25,  seeds=10, total_nfev=15, n_rounds=5),
    'medium':      dict(n_cams=20, n_pts=80,  seeds=10, total_nfev=30, n_rounds=6),
    'realistic':   dict(n_cams=50, n_pts=200, seeds=5,  total_nfev=60, n_rounds=6),
}

def build_problem(n_cams, n_pts, seed=55):
    rng0 = np.random.default_rng(seed)
    focal = 500.0
    true_cams = np.concatenate([
        rng0.uniform(-0.2, 0.2, (n_cams, 3)),
        rng0.uniform(-1, 1, (n_cams, 2)),
        rng0.uniform(4, 7, (n_cams, 1)),
    ], axis=1).ravel()
    true_pts = rng0.uniform(-1.5, 1.5, size=(n_pts, 3)).ravel()
    true_theta = np.concatenate([true_cams, true_pts])
    n_params = 6 * n_cams + 3 * n_pts

    def unpack(theta):
        cams = theta[:6*n_cams].reshape(n_cams, 6)
        pts = theta[6*n_cams:].reshape(n_pts, 3)
        return cams, pts

    def rotate_exact(p, thx, thy, thz):
        Rx = np.array([[1,0,0],[0,np.cos(thx),-np.sin(thx)],[0,np.sin(thx),np.cos(thx)]])
        Ry = np.array([[np.cos(thy),0,np.sin(thy)],[0,1,0],[-np.sin(thy),0,np.cos(thy)]])
        Rz = np.array([[np.cos(thz),-np.sin(thz),0],[np.sin(thz),np.cos(thz),0],[0,0,1]])
        return Rz @ Ry @ Rx @ p

    def project_all_exact(theta):
        cams, pts = unpack(theta)
        uv = np.zeros((n_cams, n_pts, 2))
        for c in range(n_cams):
            thx, thy, thz, tx, ty, tz = cams[c]
            R = None
            for p in range(n_pts):
                pc = rotate_exact(pts[p], thx, thy, thz) + np.array([tx, ty, tz])
                uv[c, p, 0] = focal * pc[0] / pc[2]
                uv[c, p, 1] = focal * pc[1] / pc[2]
        return uv

    obs_sigma = 1.0
    noise_rng = np.random.default_rng(seed + 1)
    observed = project_all_exact(true_theta) + noise_rng.normal(0, obs_sigma, size=(n_cams, n_pts, 2))

    def energy_exact(theta):
        pred = project_all_exact(theta)
        return float(np.mean((pred - observed) ** 2))

    def smul(x, y):
        a1,b1=x; a2,b2=y
        return (a1*b2+a2*b1, b1*b2)
    def sadd(x, y): return (x[0]+y[0], x[1]+y[1])
    def ssub(x, y): return (x[0]-y[0], x[1]-y[1])
    def scos(x): a,b=x; return (-a*np.sin(b), np.cos(b))
    def ssin(x): a,b=x; return (a*np.cos(b), np.sin(b))
    def sinv(x): a,b=x; return (-a/b**2, 1.0/b)
    def sdiv(x,y): return smul(x, sinv(y))

    def rotate_point_soft(p, thx_s, thy_s, thz_s):
        cx, sx = scos(thx_s), ssin(thx_s)
        cy, sy = scos(thy_s), ssin(thy_s)
        cz, sz = scos(thz_s), ssin(thz_s)
        x, y, z = p
        y1 = ssub(smul(cx, y), smul(sx, z)); z1 = sadd(smul(sx, y), smul(cx, z)); x1 = x
        x2 = sadd(smul(cy, x1), smul(sy, z1))
        z2 = sadd((-smul(sy, x1)[0], -smul(sy, x1)[1]), smul(cy, z1)); y2 = y1
        x3 = ssub(smul(cz, x2), smul(sz, y2)); y3 = sadd(smul(sz, x2), smul(cz, y2)); z3 = z2
        return x3, y3, z3

    def g_delta(theta0, delta):
        cams0, pts0 = unpack(theta0)
        camsD, ptsD = unpack(delta)
        total_pot = 0.0
        n_obs = n_cams * n_pts
        for c in range(n_cams):
            thx=(camsD[c,0],cams0[c,0]); thy=(camsD[c,1],cams0[c,1]); thz=(camsD[c,2],cams0[c,2])
            tx=(camsD[c,3],cams0[c,3]); ty=(camsD[c,4],cams0[c,4]); tz=(camsD[c,5],cams0[c,5])
            for p in range(n_pts):
                px=(ptsD[p,0],pts0[p,0]); py=(ptsD[p,1],pts0[p,1]); pz=(ptsD[p,2],pts0[p,2])
                x3,y3,z3 = rotate_point_soft((px,py,pz), thx, thy, thz)
                xc = sadd(x3, tx); yc = sadd(y3, ty); zc = sadd(z3, tz)
                u = smul((0.0, focal), sdiv(xc, zc)); v = smul((0.0, focal), sdiv(yc, zc))
                du = ssub(u, (0.0, observed[c,p,0])); dv = ssub(v, (0.0, observed[c,p,1]))
                total_pot += 2*du[0]*du[1] + 2*dv[0]*dv[1]
        return total_pot / (2*n_obs)

    def residuals(theta):
        pred = project_all_exact(theta)
        return (pred - observed).ravel()

    return dict(n_params=n_params, true_theta=true_theta, energy_exact=energy_exact,
                g_delta=g_delta, residuals=residuals)

def run_benchmark(cfg_name):
    cfg = CONFIGS[cfg_name]
    print(f"Config '{cfg_name}': {cfg['n_cams']} cameras, {cfg['n_pts']} points, "
          f"{6*cfg['n_cams']+3*cfg['n_pts']} params, {cfg['seeds']} seeds, "
          f"total budget {cfg['total_nfev']} evals\n")

    prob = build_problem(cfg['n_cams'], cfg['n_pts'])
    n_params = prob['n_params']
    nfev_per_round = cfg['total_nfev'] // cfg['n_rounds']

    def lm_round(theta0, max_nfev):
        res = least_squares(prob['residuals'], theta0, method='lm', max_nfev=max_nfev)
        return res.x

    def run_pure_lm(seed):
        theta = prob['true_theta'] + np.random.default_rng(seed).normal(0, 0.05, n_params)
        theta = lm_round(theta, cfg['total_nfev'])
        return prob['energy_exact'](theta)

    def run_softopt(seed):
        theta = prob['true_theta'] + np.random.default_rng(seed).normal(0, 0.05, n_params)
        opt = SoftOpt(n_params, prob['g_delta'], lr=0.02, seed=seed + 777)
        for _ in range(cfg['n_rounds']):
            theta = lm_round(theta, nfev_per_round)
            grad_dummy = np.zeros(n_params)  # LM already did the "gradient" step;
                                              # SoftOpt's Adam term is a no-op here,
                                              # only its Newton correction fires
            theta = opt.step(theta, grad_dummy)
        return prob['energy_exact'](theta)

    t0 = time.time()
    lm_scores = np.array([run_pure_lm(s) for s in range(cfg['seeds'])])
    print(f"pure LM:      mean={lm_scores.mean():.4f}  ({time.time()-t0:.0f}s elapsed)")
    softopt_scores = np.array([run_softopt(s) for s in range(cfg['seeds'])])
    print(f"SoftOpt+LM:   mean={softopt_scores.mean():.4f}  ({time.time()-t0:.0f}s elapsed)")

    wins = int(np.sum(softopt_scores < lm_scores))
    p = wilcoxon(softopt_scores, lm_scores).pvalue if cfg['seeds'] >= 6 else float('nan')
    print(f"\nSoftOpt+LM beats pure LM: {wins}/{cfg['seeds']}  p={p:.4g}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='small_test', choices=list(CONFIGS.keys()))
    args = parser.parse_args()
    run_benchmark(args.config)
