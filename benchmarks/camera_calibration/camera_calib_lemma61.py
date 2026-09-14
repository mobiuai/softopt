"""
Camera calibration / bundle adjustment: a genuinely DIFFERENT domain
(computer vision, not physics/chemistry) with the same "known computation
graph + noisy measurement" structure as VQE/FakeFez.

Known: N 3D world points, known focal length. Unknown: camera pose
(3 Euler rotation angles + 3 translation) = 6 params. Objective: sum of
squared REPROJECTION errors between the known-model's predicted pixel
coordinates and noisy observed pixels.

Genuine Lemma-6.1 soft-number propagation through: 3 sequential axis
rotations (sin/cos, same machinery as quantum gates), translation
(addition), and perspective division (the book's own soft-inverse,
(a,b)^-1 = (-a/b^2, 1/b)).
"""
import numpy as np

N_POINTS = 20
FOCAL = 500.0
N_PARAMS = 6  # thetax, thetay, thetaz, tx, ty, tz

def smul(x, y):
    a1,b1=x; a2,b2=y
    return (a1*b2+a2*b1, b1*b2)
def sadd(x, y):
    return (x[0]+y[0], x[1]+y[1])
def ssub(x, y):
    return (x[0]-y[0], x[1]-y[1])
def scos(x):
    a,b=x; return (-a*np.sin(b), np.cos(b))
def ssin(x):
    a,b=x; return (a*np.cos(b), np.sin(b))
def sinv(x):
    a,b=x; return (-a/b**2, 1.0/b)
def sdiv(x, y):
    return smul(x, sinv(y))

rng0 = np.random.default_rng(42)
WORLD_PTS = rng0.uniform(-1, 1, size=(N_POINTS, 3)) + np.array([0,0,5.0])  # in front of camera
TRUE_THETA = np.array([0.15, -0.1, 0.05, 0.3, -0.2, 0.1])

def rotate_point_exact(p, thx, thy, thz):
    Rx = np.array([[1,0,0],[0,np.cos(thx),-np.sin(thx)],[0,np.sin(thx),np.cos(thx)]])
    Ry = np.array([[np.cos(thy),0,np.sin(thy)],[0,1,0],[-np.sin(thy),0,np.cos(thy)]])
    Rz = np.array([[np.cos(thz),-np.sin(thz),0],[np.sin(thz),np.cos(thz),0],[0,0,1]])
    return Rz @ Ry @ Rx @ p

def project_exact(theta, pts=WORLD_PTS):
    thx,thy,thz,tx,ty,tz = theta
    uv = []
    for p in pts:
        pc = rotate_point_exact(p, thx,thy,thz) + np.array([tx,ty,tz])
        u = FOCAL * pc[0]/pc[2]; v = FOCAL * pc[1]/pc[2]
        uv.append((u,v))
    return np.array(uv)

OBS_NOISE_SIGMA = 1.0  # pixels
_noise_rng = np.random.default_rng(7)
OBSERVED = project_exact(TRUE_THETA) + _noise_rng.normal(0, OBS_NOISE_SIGMA, size=(N_POINTS,2))

def energy_exact(theta):
    pred = project_exact(theta)
    return float(np.mean((pred - OBSERVED)**2))

E0 = 0.0  # true params give ~noise-floor error, not exactly 0, but this is the ideal target

def rotate_point_soft(p, thx_s, thy_s, thz_s):
    """p: real 3-vector (fixed point, not perturbed). thx_s etc: soft numbers."""
    cx, sx = scos(thx_s), ssin(thx_s)
    cy, sy = scos(thy_s), ssin(thy_s)
    cz, sz = scos(thz_s), ssin(thz_s)
    x,y,z = (0.0,p[0]),(0.0,p[1]),(0.0,p[2])  # constants as soft numbers (potential=0)
    # Rx
    y1 = ssub(smul(cx,y), smul(sx,z))
    z1 = sadd(smul(sx,y), smul(cx,z))
    x1 = x
    # Ry
    x2 = sadd(smul(cy,x1), smul(sy,z1))
    z2 = sadd((-smul(sy,x1)[0], -smul(sy,x1)[1]), smul(cy,z1))
    y2 = y1
    # Rz
    x3 = ssub(smul(cz,x2), smul(sz,y2))
    y3 = sadd(smul(sz,x2), smul(cz,y2))
    z3 = z2
    return x3, y3, z3

def g_delta(theta0, delta):
    thx,thy,thz,tx,ty,tz = [(delta[i],theta0[i]) for i in range(6)]
    total_pot = 0.0
    n = N_POINTS
    for p, obs in zip(WORLD_PTS, OBSERVED):
        x3,y3,z3 = rotate_point_soft(p, thx, thy, thz)
        xc = sadd(x3, tx); yc = sadd(y3, ty); zc = sadd(z3, tz)
        u = smul((0.0,FOCAL), sdiv(xc, zc))
        v = smul((0.0,FOCAL), sdiv(yc, zc))
        du = ssub(u, (0.0, obs[0]))
        dv = ssub(v, (0.0, obs[1]))
        sq_u_pot = 2*du[0]*du[1]
        sq_v_pot = 2*dv[0]*dv[1]
        total_pot += (sq_u_pot + sq_v_pot)
    return total_pot / (2*n)

def D1_D2(theta0, delta, h=1e-4):
    d1 = g_delta(theta0, delta)
    gp = g_delta(theta0 + h*delta, delta)
    gm = g_delta(theta0 - h*delta, delta)
    d2 = (gp - gm) / (2*h)
    return d1, d2

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    theta0 = TRUE_THETA + rng.normal(0, 0.1, 6)
    delta = rng.choice([-1.,1.], size=6)
    d1 = g_delta(theta0, delta)
    h = 1e-6
    d1_fd = (energy_exact(theta0+h*delta) - energy_exact(theta0-h*delta)) / (2*h)
    print(f"D1 (soft, exact)      = {d1:.6f}")
    print(f"D1 (finite-diff check)= {d1_fd:.6f}")
    print(f"diff = {abs(d1-d1_fd):.2e}")
    print(f"energy at true theta = {energy_exact(TRUE_THETA):.4f} (noise floor)")
    print(f"energy at theta0     = {energy_exact(theta0):.4f}")
