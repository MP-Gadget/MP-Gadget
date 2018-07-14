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

def force_pm(pm, x, y, dfkernel=NDiff('lnld4_5'), split=1.25):
    """ split is in unit of mesh cells, 1.25 is Gadget Asmth.
        lnld4_5 is gadget differentiation.
    """
    rho = pm.paint(y)
    x = numpy.array(x)
    p = rho.r2c()\
        .apply(laplace) \
        .apply(decic) \
        .apply(decic) # twice, once for paint, once of readout

    if split:
        p = p.apply(longrange(split * pm.BoxSize[0] / pm.Nmesh[0]))

    r = []
    for d in range(3):
        f = p.apply(gradient(d, dfkernel=dfkernel))
        f = f.c2r()

        r.append(f.readout(x))

    p = p.c2r()
    p = p.readout(x)

    # Add 4 pi t omatch the direct sum.
    return numpy.array(r).T * (4 * numpy.pi), p * (4 * numpy.pi)

# brute force with summation of images,
# smoothed by a plummer kernel r**2 / (r**2 + a**2)
# FIXME: change to a spline

from itertools import product

def gravity_spline(dist, a):
    r = numpy.einsum('ij,ij->i', dist, dist) ** 0.5
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

    return dist * fac[:, None], -pot

def gravity_plummer(dist, a):
    r2 = numpy.einsum('ij, ij->i', dist, dist)
    r2 = r2 + a**2

    return (1.0 / (r2 * r2 ** 0.5)[:, None] * dist,
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

    F = []
    P = []

    for x1 in x:
        dist = []
        shift = []
        for ix, iy, iz in product(*([range(-Nimg, Nimg+1)] * 3)):
            nL = pm.BoxSize * [ix, iy, iz]
            dist.append(y - x1 + nL)
            L = (nL ** 2).sum(axis=-1) ** 0.5

            if L != 0:
                shift.append(- 1. / numpy.repeat(L, len(y)))
            else:
                shift.append(0 * numpy.repeat(L, len(y)))

        dist = numpy.concatenate(dist, axis=0)
        shift = numpy.concatenate(shift, axis=0)

        fa, pa = kernel(dist, a)

        if split:
            r = (dist **2).sum(axis=-1) ** 0.5
            u = r / (pm.BoxSize[0] / pm.Nmesh[0])
            fa = fa * w(u, split)
            pa = pa * v(u, split)

        pa = pa - shift

        f = (fa.sum(axis=0))
        p = (pa.sum(axis=0))


        F.append(f)
        P.append(p)

    F = numpy.array(F)
    P = numpy.array(P)

    return F, P

def main():
    pm = ParticleMesh(BoxSize=128., Nmesh=[128, 128, 128])

    Q = numpy.array([
        pm.BoxSize * [0.5, 0.5, 0.5],
    ])

    # test charges -- penetrates the page thought the source charge
    Ntest = 2048
    # along x
    test = numpy.vstack(
    (
     numpy.linspace(0, 10, Ntest * 2 + 1),
     numpy.linspace(0, 0, Ntest * 2 + 1),
     numpy.linspace(0, 0, Ntest * 2 + 1),
    )).T + pm.BoxSize * 0.5

    # mesh units
    r = ((test[:] - Q[0]) ** 2).sum(axis=-1) ** 0.5 / (pm.BoxSize[0] / pm.Nmesh[0])

    f_longrange, p_longrange = force_pm(pm, test, Q, split=1.25)
    f_plummer, p_plummer = force_direct(pm, test, Q, a=1./20, kernel=gravity_plummer)
    f_spline, p_spline = force_direct(pm, test, Q, a=1./20, kernel=gravity_spline, Nimg=4)
    f_erf, p_erf = force_direct(pm, test, Q, a=1./20, kernel=gravity_spline, split=1.25, Nimg=0)

    f_longrange = f_longrange[:, 0]
    f_plummer = f_plummer[:, 0]
    f_spline = f_spline[:, 0]
    f_erf = f_erf[:, 0]

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

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(8, 4))
    canvas = FigureCanvasAgg(figure)
    ax = figure.add_subplot(131)
#    ax.plot(r, f_longrange, label='Longrange')
#    ax.plot(r, f_plummer - f_longrange, label='plummer - longrange')
    ax.plot(r, (f_spline - f_longrange) / f_spline, label='actual', ls='--')
    ax.plot(r, f_erf / f_spline, label='erf, spline', ls=':')
#    ax.set_ylim(-0.10, 0.10)
    #ax.set_xlim(-4, 4)
    ax.legend()
    ax = figure.add_subplot(132)
#    ax.plot(r, p_longrange, label='Longrange')
#    ax.plot(r, p_plummer - p_longrange, label='plummer - longrange', )
    ax.plot(r, (p_spline - p_longrange) / p_spline, label='actual', ls='--')
    ax.plot(r, p_erf / p_spline, label='erf', ls=':')
#    ax.set_ylim(-0.10, 0.10)
    #ax.set_xlim(-4, 4)
    ax.legend()

    ax = figure.add_subplot(133)
    ax.plot(r, f_spline, label='Spline')
    ax.plot(r, f_longrange, label='Longrange')
    #ax.plot(r, f_erf, label='erf')
    ax.set_ylim(-0.01, 0.00)
    ax.legend()
    figure.savefig('diagonstics.png', dpi=200)

    rx = r
    rp_1d = (p_spline - p_longrange) / p_spline
    rf_1d = (f_spline - f_longrange) / f_spline
    rp_erf = (p_erf - p_longrange) / p_spline
    rf_erf = (f_erf - f_longrange) / f_spline
    rp_1d[r==0] = 1
    rf_1d[r==0] = 1
    rp_erf[r==0] = 1
    rf_erf[r==0] = 1

    table = numpy.array([rx, rp_1d, rf_1d, rp_erf, rf_erf]).T

    numpy.savetxt('shortrange-force-kernels.txt',
        table,
        header='x(in mesh units) w_pot_1d(x) w_force_1d(x) [erfc + other terms] w_pot_erf(x) w_force_erf(x)')

    # simply multiple these numbers to the unwindowed short-range force.
    s = toc(table,
            "shortrange_force_kernels",
         header='x(in mesh units) w_pot_1d(x)  w_force_1d(x) [erfc + other terms] w_pot_erf(x) w_force_erf(x)'
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
    
main()
