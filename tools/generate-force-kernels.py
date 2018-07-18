from pmesh.pm import ParticleMesh
import numpy

class NDiff:
    D = {
        #http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/
        'snrd2_5': (8, (2, 1)),
        'snrd2_7': (32, (5, 4, 1)),
        'snrd2_9': (128, (14, 14, 6, 1)),
        'snrd2_11': (512, (42, 48, 27, 8, 1)), 
        'snrd4_7': (96, (39, 12, -5)),
        'snrd4_9': (96, (27, 16, -1, -2)),
        'snrd4_11': (1536, (322, 256, 39, -32, -11)),
        #http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/lanczos-low-noise-differentiators/
        'lnld2_5': (10, (1, 2)),
        'lnld2_7': (28, (1, 2, 3)),
        'lnld2_7': (60, (1, 2, 3, 4)),
        'lnld2_9': (110, (1, 2, 3, 4, 5)),
        'lnld4_5': (12, (8, -1,)),    # default in MP-Gadget3
        'lnld4_7': (252, (58, 67, -22)),
        'lnld4_9': (1188, (126, 193,142, -86)),
        'lnld4_11': (5148, (296, 503, 532, 294, -300)),
        #http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/central-differences/
        'nscd_3': (2, (1, )),
        'nscd_5': (12, (8, -1)),
        'nscd_7': (60, (45, -9, 1)),
        'nscd_9': (840, (672, -168, 32, -3)),

    }
    def __init__(self, name):
        self.name = name
        self.norm = self.D[name][0]
        self.coeffs = numpy.array(self.D[name][1], 'f8')

        s = sum((i+1) * c for i, c in enumerate(self.coeffs))
        #print(self.name, self.norm, s)
        assert self.norm  == 2.0 * s

    def __call__(self, w):
        return 2 * 1.0 / self.norm * numpy.sum(numpy.sin((i+1) * w) * c for i, c in enumerate(self.coeffs))

def decic(k, v):
    """ Deconvolves the CIC window, once"""
    f = 1
    for dir in range(3):
        cellsize = (v.BoxSize[dir] / v.Nmesh[dir])
        wd = k[dir] * cellsize
        tmp = numpy.sinc(wd * 0.5 / numpy.pi)
        f = f * 1 / tmp ** 2
    return v * f

# low noise differentiator
def gradient(dir, dfkernel):
    def kernel(k, v, df=dfkernel, dir=dir):
        cellsize = (v.BoxSize[dir] / v.Nmesh[dir])
        w = k[dir] * cellsize
        a = dfkernel(w) / cellsize
        return v * (1j * a)
    return kernel

def longrange(r_split):
    if r_split != 0:
        def kernel(k, v):
            kk = sum(ki ** 2 for ki in k)
            return v * numpy.exp(-kk * r_split**2)
    else:
        def kernel(k, v):
            return v
    return kernel

def laplace(k, v):
    kk = sum(ki ** 2 for ki in k)
    mask = (kk == 0).nonzero()
    kk[mask] = 1
    b = v / kk
    b[mask] = 0
    return b

def force_pm(pm, x, y, dfkernel=NDiff('lnld4_5'), split=1.25, compensate_cic=True):
    """ split is in unit of mesh cells, 1.25 is Gadget Asmth.
        lnld4_5 is gadget differentiation.
    """
    rho = pm.paint(y)
    x = numpy.array(x)
    p = rho.r2c()\
        .apply(laplace, out=Ellipsis)

    if compensate_cic:
        p = p.apply(decic, out=Ellipsis) \
             .apply(decic, out=Ellipsis) # twice, once for paint, once of readout

    if split:
        p = p.apply(longrange(split * pm.BoxSize[0] / pm.Nmesh[0]), out=Ellipsis)

    r = []
    for d in range(3):
        f = p.apply(gradient(d, dfkernel=dfkernel))
        f = f.c2r(out=Ellipsis)

        r.append(f.readout(x))

    p = p.c2r(out=Ellipsis)
    p = p.readout(x)

    # Add 4 pi t omatch the direct sum.
    return numpy.array(r).T * (4 * numpy.pi), p * (4 * numpy.pi)

# brute force with summation of images,

from itertools import product

def gravity_spline(dist, a):
    # copied from Gadget / check for typos? But we really only care the very outside
    r = numpy.einsum('...j,...j->...', dist, dist) ** 0.5
    fac = 1 / r ** 3
    pot = - 1 / r

    u = r / a

    inner = u < 0.5
    mid   = (u >= 0.5) & (u < 1.0)
    fac[inner] = (a ** -3 * ( 32. / 3 + u * u * (32.0 * u - 38.4)))[inner]
    fac[mid] = (a ** -3 * ( 64. / 3 - 48.0 * u
                        + 38.4 * u * u
                        - 32.0 / 3 * u * u * u
                        - 0.2/3 / (u * u * u)))[mid]

    pot[inner] = (a**-1 * (
            -2.8 + u * u * (16./3 + u * u * (6.4 * u - 0.6))
        ))[inner]
    pot[mid]  = (a**-1 * (
            -3.2 + 0.2 / 3 / u + u * u * (32. / 3 + u * (-16.0 + u *(9.6 - 6.4 / 3 * u)))
        ))[mid]

    return dist * fac[..., None], -pot

def gravity_plummer(dist, a):
    r2 = numpy.einsum('...j, ...j->...', dist, dist)
    r2 = r2 + a**2

    return (1.0 / (r2 * r2 ** 0.5)[..., None] * dist,
           1.0 / (r2 ** 0.5))

from scipy.special import erfc
# the 'correct' kernel from gadget paper
def w(u, split):
    u = u * 0.5 / split
    u = abs(u)
    return erfc(u) + 2 * u / numpy.pi ** 0.5 * numpy.exp(-u * u)
def v(u, split):
    u = u * 0.5 / split
    u = abs(u)
    return erfc(u)


def force_direct(pm, x, y, a=1/20., kernel=gravity_plummer, split=False, Nimg=4):
    """ Returns
        f, p : force and potential
    """
    # fix me : unreasonably slow this is.

    def dochunk(x1):
        dist = []

        for ix, iy, iz in product(*([range(-Nimg, Nimg+1)] * 3)):
            nL = pm.BoxSize * [ix, iy, iz]

            dist1 = (y + nL)[:, None, :] - x1[None, :, :]

            dist.append(dist1)

            # 0 is the axis of images
        dist = numpy.concatenate(dist, axis=0)

        fa, pa = kernel(dist, a)

        if split:
            r = (dist **2).sum(axis=-1) ** 0.5
            u = r / (pm.BoxSize[0] / pm.Nmesh[0])
            fa = fa * w(u, split)[..., None]
            pa = pa * v(u, split)

        f = (fa.sum(axis=0))
        p = (pa.sum(axis=0))
        assert p.ndim == 1

        return f, p

    F = []
    P = []
    chunksize = 1024*1024 // (2 * Nimg + 1)**3 + 1
    for i in range(0, len(x), chunksize):
        x1 = x[i:i+chunksize]

        f, p = dochunk(x1)
        F.append(f)
        P.append(p)

    F = numpy.concatenate(F, axis=0)
    P = numpy.concatenate(P, axis=0)

    assert len(F) == len(x)
    assert len(P) == len(x)
    assert F.ndim == 2
    assert P.ndim == 1
    return F, P

def main(ns):
    pm = ParticleMesh(BoxSize=512., Nmesh=[512, 512, 512], dtype='f4')

    Q = numpy.array([
        pm.BoxSize * [0.5, 0.5, 0.5],
    ])

    # test charges -- penetrates the page thought the source charge
    Ntest = 512 # segments in radial
    Nsample = 48      # estimating the variance at different directions; anisotropic-ness
    Nshift = 48 # number of shifts

    Split = ns.split       # should try smaller split if the variance doesn't go up then we are good; in mesh units.
    Smoothing = 1./ 20 # shouldn't be very sensitive to this; in distance units.

    Rmax = 10.0 # max r to go, in mesh units

    # generate points on a sphere
    test = numpy.random.uniform(0, 1.0, size=(Ntest * Nsample, 3))
    unitvectors = test / (test**2).sum(axis=-1)[:, None] ** 0.5

    # add distance uniformly
    r = numpy.linspace(0, Rmax * pm.BoxSize[0] / pm.Nmesh[0], Ntest)
    r = numpy.repeat(r, Nsample)

    test = unitvectors * r[:, None]
    test = test + pm.BoxSize * 0.5 # at the center.

    # r is in mesh units
    r = ((test[:] - Q[0]) ** 2).sum(axis=-1) ** 0.5 / (pm.BoxSize[0] / pm.Nmesh[0])

    def compute(test, Q):
        f_longrange, p_longrange = force_pm(pm, test, Q, split=Split, compensate_cic=ns.decic)
        f_plummer, p_plummer = force_direct(pm, test, Q, a=Smoothing, kernel=gravity_plummer)
        f_spline, p_spline = force_direct(pm, test, Q, a=Smoothing, kernel=gravity_spline, Nimg=4)
        f_erf, p_erf = force_direct(pm, test, Q, a=Smoothing, kernel=gravity_spline, split=Split, Nimg=0)

        f_longrange = numpy.einsum('ij,ij->i', f_longrange, unitvectors)
        f_plummer = numpy.einsum('ij,ij->i', f_plummer, unitvectors)
        f_spline = numpy.einsum('ij,ij->i', f_spline, unitvectors)
        f_erf = numpy.einsum('ij,ij->i', f_erf, unitvectors)

        # renormalize
        p_plummer -= p_plummer[-1] - p_longrange[-1]
        p_spline -= p_spline[-1] - p_longrange[-1]

        p_spline[r == 0] = 0
        p_plummer[r == 0] = 0
        p_longrange[r == 0] = 0
        p_erf[r == 0] = 0

        f_longrange[r == 0] = 0
        f_plummer[r == 0] = 0
        f_spline[r == 0] = 0
        f_erf[r == 0] = 0
        return (f_longrange, f_plummer, f_spline, f_erf,
                p_longrange, p_plummer, p_spline, p_erf, )

    terms = [[],] * 8

    for junk in range(Nshift):
        shift = numpy.random.uniform(low=-0.5, high=0.5, size=(1, 3))
        for i, result in enumerate(compute(test + shift, Q + shift)):
            terms[i] = numpy.concatenate([terms[i], result], axis=0)

    for i, result in enumerate(terms):
        terms[i] = terms[i].reshape(Nshift, -1, Nsample).transpose((1, 2, 0)).ravel()

    (f_longrange, f_plummer, f_spline, f_erf,
     p_longrange, p_plummer, p_spline, p_erf, ) = terms
    print(len(terms[0]))

    def stat(x, size):
        x = x.reshape(-1, size)
        mean = numpy.mean(x, axis=-1)
        std = numpy.std(x, axis=-1)
        return mean, std

    rx, junk = stat(r, Nsample)
    rp_1d, rp_1d_s = stat((p_spline - p_longrange) / p_spline, Nsample * Nshift)
    rf_1d, rf_1d_s = stat((f_spline - f_longrange) / f_spline, Nsample * Nshift)
    rp_erf, junk = stat((p_erf) / p_spline, Nsample * Nshift)
    rf_erf, junk = stat((f_erf) / f_spline, Nsample * Nshift)

    rp_1d[rx==0] = 1
    rf_1d[rx==0] = 1
    rp_erf[rx==0] = 1
    rf_erf[rx==0] = 1


    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = Figure(figsize=(8, 8))
    figure.text(0.5, 0.91, "Split=%g" % Split, ha='center', va='center')
    canvas = FigureCanvasAgg(figure)
    ax = figure.add_subplot(221)
    ax.set_title('potential')
    l, = ax.plot(rx, rp_1d, '-', label='actual')
    ax.fill_between(rx, rp_1d - 3 * rp_1d_s, rp_1d + 3 * rp_1d_s, color=l.get_color(), alpha=0.4, label='3-sigma')
    ax.plot(rx, rp_erf, label='erf, spline', ls=':')
    ax.legend()
    ax = figure.add_subplot(222)
    ax.set_title('force')
    l, = ax.plot(rx, rf_1d, '-', label='actual')
    ax.fill_between(rx, rf_1d - 3 * rf_1d_s, rf_1d + 3 * rf_1d_s, color=l.get_color(), alpha=0.4, label='3-sigma')
    ax.plot(rx, rf_erf, label='erf, spline', ls=':')
    ax.legend()

    ax = figure.add_subplot(223)
    ax.set_title('relative and zoom')
    l, = ax.plot(rx, rp_1d, '-', label='actual, zero-crossing')
    ax.fill_between(rx, -3 * rp_1d_s,  3 * rp_1d_s, color=l.get_color(), alpha=0.4, label='3-sigma')
    ax.plot(rx, rp_erf / rp_1d - 1, label='erf, spline', ls=':')
    ax.plot(rx, rp_erf, '--', label='erf, zero-crossing', color='gray')
    ax.set_ylim(-0.02, 0.02)
    ax.grid()
    ax.legend()
    ax = figure.add_subplot(224)
    ax.set_title('relative and zoom')
    l, = ax.plot(rx, rf_1d, '-', label='actual, zero-crossing')
    ax.fill_between(rx,  - 3 * rf_1d_s,  3 * rf_1d_s, color=l.get_color(), alpha=0.4, label='3-sigma')
    ax.plot(rx, rf_erf / rf_1d - 1, label='erf, spline', ls=':')
    ax.plot(rx, rf_erf, '--', label='erf, zero-crossing', color='gray')
    ax.set_ylim(-0.03, 0.03)
    ax.grid()
    ax.legend()

    figure.savefig(ns.prefix + 'diagonstics-%.2f.png' % Split, dpi=200)


    table = numpy.array([rx, rp_1d, rf_1d, rp_erf, rf_erf]).T

    numpy.savetxt('shortrange-force-kernels.txt',
        table,
        header='x(in mesh units) w_pot_1d(x) w_force_1d(x) [erfc + other terms] w_pot_erf(x) w_force_erf(x) split=%.2f' % Split
    )

    # simply multiple these numbers to the unwindowed short-range force.
    s = toc(table,
            "shortrange_force_kernels",
         header='x(in mesh units) w_pot_1d(x)  w_force_1d(x) [erfc + other terms] w_pot_erf(x) w_force_erf(x) split=%.2f' % Split
        )

    with open('shortrange-kernels.c', 'w') as ff:
        ff.write(s)

def toc(array, arrayname, header):
    template = """
# %(header)s
const double %(name)s[][%(size)d] = {

%(text)s

    };
    
    """
    t = []
    for row in array:
        t.append(','.join('%.15e' % i for i in row))
    t = ['{ %s},' % ti for ti in t]
    return template % dict(
        header=header,
        name=arrayname,
        size=array.shape[1],
        text='\n'.join(t)
    )
    
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('split', type=float, help='Split of range, in mesh units')
ap.add_argument('--no-decic', action='store_false', dest='decic', default=True, help='deconvolve cic window')
ap.add_argument('prefix', default="")
ns = ap.parse_args()

main(ns)
