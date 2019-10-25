"""Script to generate the short-range force kernel tables for Gadget.
These have to account for smoothing on large scales to match directly onto the PM force."""

import os.path
from itertools import product
import numpy
from scipy.special import erfc
#This comes from nbodykit
from pmesh.pm import ParticleMesh

class NDiff:
    """Long-range differentiation kernels."""
    Defs = {
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
        'lnld2_9': (60, (1, 2, 3, 4)),
        'lnld2_11': (110, (1, 2, 3, 4, 5)),
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
        self.norm = self.Defs[name][0]
        self.coeffs = numpy.array(self.Defs[name][1], 'f8')

        s = sum((i+1) * c for i, c in enumerate(self.coeffs))
        #print(self.name, self.norm, s)
        assert self.norm  == 2.0 * s

    def __call__(self, ww):
        return 2 * 1.0 / self.norm * numpy.sum([numpy.sin((i+1) * ww) * c for i, c in enumerate(self.coeffs)])

def decic(k, v):
    """ Deconvolves the CIC window, once"""
    f = 1
    for axis in range(3):
        cellsize = (v.BoxSize[axis] / v.Nmesh[axis])
        wd = k[axis] * cellsize
        tmp = numpy.sinc(wd * 0.5 / numpy.pi)
        f = f * 1 / tmp ** 2
    return v * f

def gradient(axis, dfkernel):
    """low noise differentiator"""
    def kernel(k, v):
        """Kernel function for long-range gravity gradient operator"""
        cellsize = (v.BoxSize[axis] / v.Nmesh[axis])
        w = k[axis] * cellsize
        a = dfkernel(w) / cellsize
        return v * (1j * a)
    return kernel

def longrange(r_split):
    """Smoothing kernel for the long range force."""
    if r_split != 0:
        def kernel(k, v):
            """Kernel function for smoothing kernel"""
            kk = sum(ki ** 2 for ki in k)
            return v * numpy.exp(-kk * r_split**2)
    else:
        def kernel(k, v):
            """Null smoothing kernel"""
            _ = k
            return v
    return kernel

def laplace(k, v):
    """Multiplies the fourier grid by a Laplacian to compute force."""
    kk = sum(ki ** 2 for ki in k)
    mask = (kk == 0).nonzero()
    kk[mask] = 1
    b = v / kk
    b[mask] = 0
    return b

def force_pm(pm, x, y, dfkernel=NDiff('lnld4_5'), split=1.25, compensate_cic=True):
    """
    pm: PM grid to use.
    x, y: vectors of particles for which positions are computed.
    dfkernel: Force kernel to use. lnld4_5 is gadget differentiation.
    split: in unit of mesh cells, 1.25 is Gadget Asmth.
    Signifies there is a short-range force and thus the kernel should be smoothed.
    Higher values reduce mesh anisotropy.
    compensate_cic: Should we compensate for the window function from adding particles to the grid?
    Gadget does.
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

def gravity_spline(dist, a):
    """This computes the direct gravitational force for a particle softened with a spline, as is done in Gadget."""
    # copied from Gadget / check for typos? But we really only care the very outside
    r = numpy.einsum('...j,...j->...', dist, dist) ** 0.5
    u = r / a
    fac = numpy.zeros_like(r)

    inner = u < 0.5
    mid   = (u >= 0.5) & (u < 1.0)
    outer = u >= 1.0
    fac[inner] = (a ** -3 * ( 32. / 3 + u * u * (32.0 * u - 38.4)))[inner]
    fac[mid] = (a ** -3 * ( 64. / 3 - 48.0 * u[mid]
                            + 38.4 * u[mid]**2
                            - 32.0 / 3 * u[mid]**3
                            - 0.2/3 / (u[mid]**3)))
    fac[outer] = 1 / r[outer] ** 3

    pot = numpy.zeros_like(r)
    pot[inner] = (a**-1 * (-2.8 + u * u * (16./3 + u * u * (6.4 * u - 0.6))))[inner]
    pot[mid]  = (a**-1 * (-3.2 + 0.2 / 3 / u[mid] + u[mid]**2 * (32. / 3 + u[mid] * (-16.0 + u[mid] *(9.6 - 6.4 / 3 * u[mid])))))
    pot[outer] = - 1 / r[outer]

    return dist * fac[..., None], -pot

def gravity_plummer(dist, a):
    """The computes the direct force for a point particle"""
    r2 = numpy.einsum('...j, ...j->...', dist, dist)
    r2 = r2 + a**2

    return (1.0 / (r2 * r2 ** 0.5)[..., None] * dist, 1.0 / (r2 ** 0.5))

# the 'correct' kernel from gadget paper
def g2_force_kern(u, split):
    """Gadget kernel for the force."""
    u = u * 0.5 / split
    u = abs(u)
    return erfc(u) + 2 * u / numpy.pi ** 0.5 * numpy.exp(-u * u)

def g2_pot_kern(u, split):
    """Gadget kernel for the potential."""
    u = u * 0.5 / split
    u = abs(u)
    return erfc(u)


def force_direct(pm, x, y, a=1/20., kernel=gravity_plummer, split=False, Nimg=4):
    """
    split: if true, force is split into long and short-range parts, as in Gadget,
           and the kernel is multiplied by the erfc functions above.
    pm: only used if split=True. Used to compute the erf window in units of the PM grid.
    x, y: vectors of particle coordinates
    a: softening length in units of grid.
    kernel: force kernel to use: either 1/r^2 or softened.
    Nimg: Number of images to go to mimic the periodic boundary condition.
    Returns
        f, p : force and potential
    """
    # fix me : unreasonably slow this is.

    def dochunk(x1):
        """Evaluate the potential for a chunk of the total."""
        dist = []

        # create image particles outside the primary box;
        # approximating the periodic boundary
        for ix, iy, iz in product(*([range(-Nimg, Nimg+1)] * 3)):
            nL = pm.BoxSize * [ix, iy, iz]

            dist1 = (y + nL)[:, None, :] - x1[None, :, :]

            dist.append(dist1)

        # 0 is the axis of images
        dist = numpy.concatenate(dist, axis=0)

        fa, pa = kernel(dist, a)

        if split:
            # short range only?
            r = (dist **2).sum(axis=-1) ** 0.5
            u = r / (pm.BoxSize[0] / pm.Nmesh[0])
            fa = fa * g2_force_kern(u, split)[..., None]
            pa = pa * g2_pot_kern(u, split)

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
    """Main routine that makes the plots."""
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

    Rmax = Split * 10.0 # max r to go, in mesh units

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

    def radial_only(force_potential):
        """Do computations for a force kernel and project to the radial direction."""
        f, p = force_potential
        f = numpy.einsum('ij,ij->i', f, unitvectors)
        p[r == 0] = 0
        f[r == 0] = 0
        return f, p

    def compute(test, Q):
        """Do the computations for the various different types of force kernels."""
        # This is the long-range PM force
        f_longrange, p_longrange = radial_only(force_pm(pm, test, Q,
                                        dfkernel=NDiff(ns.diffkernel),
                                        split=Split, compensate_cic=ns.decic))

        # This is the full gravity force for a particle with spline softening;
        # computed brute force; periodic boundary is handled with 4 images each side.
        f_spline, p_spline = radial_only(force_direct(pm, test, Q, a=Smoothing,
                            kernel=gravity_spline, Nimg=4))

        # This is the short range gravity force for a particle with spline softening;
        # without the periodic boundary condition, since this is short range.
        # this is roughly the shortrange force used in Gadget2, but notice the actual
        # force in Gadget2 is truncated at 4.5 * Split.
        f_erf, p_erf = radial_only(force_direct(pm, test, Q, a=Smoothing,
                            kernel=gravity_spline, split=Split, Nimg=0))

        # This is the full gravity force for a particle with plummer softening;
        # for comparison with spline and not actually used.
        # notice that the softening parameter is inconsistent between plummer and spline:
        # c.f. http://iopscience.iop.org/article/10.1088/1749-4699/1/1/015003/pdf
        f_plummer, p_plummer = radial_only(force_direct(pm, test, Q, a=Smoothing, kernel=gravity_plummer))

        # renormalize; summation of images has a 1 / n term that must be removed.
        p_spline -= p_spline[-1] - p_longrange[-1]
        p_plummer -= p_plummer[-1] - p_longrange[-1]

        return (f_longrange, f_plummer, f_spline, f_erf,
                p_longrange, p_plummer, p_spline, p_erf, )

    terms = [[],] * 8

    # Compute forces;
    # to include effect of off-mesh,
    # randomly shift the test particle and source particle set,
    # keeping the distance.

    for _ in range(Nshift):
        shift = numpy.random.uniform(low=-0.5, high=0.5, size=(1, 3))
        for i, result in enumerate(compute(test + shift, Q + shift)):
            terms[i] = numpy.concatenate([terms[i], result], axis=0)

    for i, result in enumerate(terms):
        terms[i] = terms[i].reshape(Nshift, -1, Nsample).transpose((1, 2, 0)).ravel()

    (f_longrange, _, f_spline, f_erf,
     p_longrange, _, p_spline, p_erf, ) = terms
    print(len(terms[0]))

    def stat(x, size):
        """Compute mean and sd for different force kernels"""
        x = x.reshape(-1, size)
        mean = numpy.mean(x, axis=-1)
        std = numpy.std(x, axis=-1)
        return mean, std

    rx, _ = stat(r, Nsample)
    #Variance between long-range force and direct force. Measures mesh anisotropy.
    rp_1d, rp_1d_s = stat((p_spline - p_longrange) / p_spline, Nsample * Nshift)
    rf_1d, rf_1d_s = stat((f_spline - f_longrange) / f_spline, Nsample * Nshift)
    rp_erf, _ = stat((p_erf) / p_spline, Nsample * Nshift)
    rf_erf, _ = stat((f_erf) / f_spline, Nsample * Nshift)

    rp_1d[rx==0] = 1
    rf_1d[rx==0] = 1
    rp_erf[rx==0] = 1
    rf_erf[rx==0] = 1

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = Figure(figsize=(8, 8))
    figure.text(0.5, 0.91, "Split=%g" % Split, ha='center', va='center')
    _ = FigureCanvasAgg(figure)
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
    ax.fill_between(rx, -3 * rp_1d_s,  3 * rp_1d_s, color=l.get_color(), alpha=0.4, label='3-sigma')
    ax.plot(rx, rp_erf / rp_1d - 1, label='erf/actual ratio (spline)', ls=':')
    ax.set_ylim(-0.02, 0.02)
    ax.grid()
    ax.legend()
    ax = figure.add_subplot(224)
    ax.set_title('relative and zoom')
    ax.fill_between(rx,  - 3 * rf_1d_s,  3 * rf_1d_s, color=l.get_color(), alpha=0.4, label='3-sigma')
    ax.plot(rx, rf_erf / rf_1d - 1, label='erf/actual ratio (spline)', ls=':')
    ax.set_ylim(-0.03, 0.03)
    ax.grid()
    ax.legend()

    figure.savefig(os.path.join(ns.prefix, 'diagnostics-%.2f.png' % Split), dpi=200)


    table = numpy.array([rx, rp_1d, rf_1d, rp_erf, rf_erf]).T
    # These numbers now contain the short-range kernels.
    # One may simply multiply the unwindowed short-range force by whichever column one wants
    # to get the corrected force kernel.
    # Most accurate appears to be to use the 'exact' kernels as a table: columns 1 and 2.
    # columns 3 and 4 are provided for comparison with Gadget-2.
    numpy.savetxt('shortrange-force-kernels-%.2f.txt' % Split,
                  table, header='x(in mesh units) w_pot_1d(x) w_force_1d(x) [erfc + other terms] w_pot_erf(x) w_force_erf(x) split=%.2f' % Split
                 )

    s = toc(table, "shortrange_force_kernels",
            header='x(in mesh units) w_pot_1d(x)  w_force_1d(x) [erfc + other terms] w_pot_erf(x) w_force_erf(x) split=%.2f' % Split
           )

    with open('shortrange-kernels.c', 'w') as ff:
        ff.write(s)

def toc(array, arrayname, header):
    """ Function to save the tables to a carefully formatted C file
        which can be linked against MP-Gadget's gravity.c.
        The main purpose is to add {} around array rows."""
    template = """
// # %(header)s
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

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('split', type=float, help='Split of range, in mesh units')
    ap.add_argument('--no-decic', action='store_false', dest='decic', default=True, help='deconvolve cic window')
    ap.add_argument('--diffkernel', type=str, choices=NDiff.Defs.keys(), default='lnld4_5',
                    help='diffkernel to use in pm')
    ap.add_argument('prefix', default="", help="Directory in which to save figures.")
    opts = ap.parse_args()

    main(opts)
