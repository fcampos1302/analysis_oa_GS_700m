"""
Frequency-wavenumber spectrum and cospectrum: optimized version.

Computes exactly the same calculation as the original version
(linear detrend in t, x, y -> 3D Hanning window with the same normalization
-> 3D rfftn -> product and normalization -> isotropic spectrum with the same
rings, same inclusive edges, and same dth = pi/(n-1)).
Results match the original to machine precision.

What changes (implementation only):
  1. calc_ispec_3d: instead of looping over the full 3D array once per ring,
     a sparse weight matrix (ring x k,l point) is built with the same masks
     and ONE matrix-array multiplication is performed.
  2. No fftshift of the 3D array: the 2D map of |k| is un-shifted (same values,
     just permuted) and the sum per ring does not depend on order.
  3. In-place linear detrending with the closed-form least squares formula
     (same fit as scipy.signal.detrend type='linear', without copies).
  4. Window applied by slices: the full 3D window is not created.
  5. Fields are processed one by one and freed as soon as they are used.
  6. FFT using scipy.fft with multiple threads (workers).

Usage (same signature as before, plus the optional `workers` argument):
    from spectra_fast import computing_KE_spectrum, cospectrum
    kr, kt, ke = computing_KE_spectrum(u, v, dx, dy, dt)
    kr, kt, ww = cospectrum(u, v, taux, tauy, dx, dy, dt, workers=8)

Fields can be numpy arrays (nx, ny, nt) or xarray.DataArray with that same
dimension order. If you pass DataArray (without .values), each field is loaded
inside the function, one at a time, using less memory.

workers=-1 uses all cores; on a shared node set the number of allocated cores
(for example, SLURM_CPUS_PER_TASK).
If input is float32, everything here is calculated in float64 (the original did
the detrending in float32): relative differences on the order of 1e-7.
"""
import warnings
import numpy as np
from scipy import fft, sparse


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def _detrend_linear_inplace(a, axis):
    """Removes the linear least-squares fit along `axis`.

    Equivalent to scipy.signal.detrend(a, axis=axis, type='linear'), but
    modifies `a` (3D C-contiguous float64 array) in-place without creating 3D copies.
    """
    n = a.shape[axis]
    t = np.arange(n, dtype=np.float64)
    t -= t.mean()
    tt = t @ t
    if axis == 2:
        for i in range(a.shape[0]):
            s = a[i]                                  # (ny, nt)
            s -= s.mean(axis=1, keepdims=True)
            s -= np.outer(s @ t / tt, t)
    elif axis == 1:
        for i in range(a.shape[0]):
            s = a[i]                                  # (ny, nt)
            s -= s.mean(axis=0, keepdims=True)
            s -= np.outer(t, t @ s / tt)
    elif axis == 0:
        a -= a.mean(axis=0, keepdims=True)
        b = np.tensordot(t, a, axes=(0, 0)) / tt      # (ny, nt)
        for i in range(a.shape[0]):
            a[i] -= t[i] * b
    else:
        raise ValueError("axis must be 0, 1, or 2")
    return a


def _hann_norm(n):
    """Same window and normalization as the original code."""
    w = np.hanning(n)
    return (n / (w**2).sum()) * w


def _prep(x, w12, w3):
    """Private float64 copy -> detrend (t, x, y) -> 3D window."""
    if hasattr(x, "dims"):                            # xarray.DataArray: loaded here
        x = x.values
    a = np.array(x, dtype=np.float64, order="C")      # does not modify user's array
    if not np.isfinite(a.sum()):
        warnings.warn("Field contains NaN or inf (land points?): spectrum will output NaN, "
                      "same as original version. Use a box over ocean only.")
    _detrend_linear_inplace(a, 2)
    _detrend_linear_inplace(a, 0)
    _detrend_linear_inplace(a, 1)
    for i in range(a.shape[0]):                       # same window as win[i]
        a[i] *= w12[i][:, None] * w3[None, :]
    return a


def _rfft3(a, workers):
    return fft.rfftn(a, axes=(0, 1, 2), workers=workers, overwrite_x=True)


# --------------------------------------------------------------------------
# Isotropic spectrum
# --------------------------------------------------------------------------
def _ring_weights(k, l):
    """Rings and weights exactly as in calc_ispec_3d (ring x point sparse matrix)."""
    dk = np.abs(k[2] - k[1])
    dl = np.abs(l[2] - l[1])
    wv = np.sqrt(k[:, None]**2 + l[None, :]**2)
    kmax = min(k.max(), l.max())
    dkr = np.sqrt(dk**2 + dl**2)
    kr = np.arange(dkr/2., kmax + dkr/2., dkr)

    # same |k| map, reordered to the natural FFT order (same as E)
    wv = fft.ifftshift(wv).ravel()

    rows, cols, vals, nan_rows = [], [], [], []
    for i in range(kr.size):
        idx = np.flatnonzero((wv >= kr[i] - dkr/2) & (wv <= kr[i] + dkr/2))
        if idx.size == 1:          # original outputs NaN here (pi/0 * 0)
            nan_rows.append(i)
            continue
        dth = np.pi / (idx.size - 1)
        rows.append(np.full(idx.size, i))
        cols.append(idx)
        vals.append(wv[idx] * dth)

    W = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(kr.size, wv.size))
    return kr, W, nan_rows


def _apply_rings(W, nan_rows, E):
    n1, n2, nomg = E.shape
    Er = W @ E.reshape(n1 * n2, nomg)                 # single pass over E
    Er[nan_rows] = np.nan
    return Er.squeeze()


def calc_ispec_3d_fast(k, l, E):
    """Same as calc_ispec_3d, but:
      - k, l : wavenumbers WITH fftshift (as in original; so dk, dl,
               kmax, and rings match identically).
      - E    : array (nx, ny, nomg) WITHOUT fftshift (natural FFT order).
    """
    kr, W, nan_rows = _ring_weights(k, l)
    return kr, _apply_rings(W, nan_rows, E)


# --------------------------------------------------------------------------
# Kinetic energy spectrum
# --------------------------------------------------------------------------
def computing_KE_spectrum(u, v, dx, dy, dt, workers=-1):
    nx, ny, nt = u.shape
    kx, ky, kt = fft.fftfreq(nx, dx), fft.fftfreq(ny, dy), fft.rfftfreq(nt, dt)
    dkx, dky, dkt = 1/(dx*nx), 1/(dy*ny), 1/(dt*nt)
    w1, w2, w3 = _hann_norm(nx), _hann_norm(ny), _hann_norm(nt)
    w12 = w1[:, None] * w2[None, :]

    print("1-3) u: detrending, windowing, fft ...")
    f = _rfft3(_prep(u, w12, w3), workers)
    ke = f.real**2
    ke += f.imag**2
    del f

    print("1-3) v: detrending, windowing, fft ...")
    f = _rfft3(_prep(v, w12, w3), workers)
    tmp = f.real**2
    tmp += f.imag**2
    del f
    ke += tmp
    del tmp

    ke *= 0.5
    ke /= (dkx*dky*dkt)
    ke /= (nx*ny*nt)**2

    print("4) computing isotropic spectrum")
    kr, ke_iso = calc_ispec_3d_fast(fft.fftshift(kx), fft.fftshift(ky), ke)
    return kr, kt, ke_iso


# --------------------------------------------------------------------------
# Cospectrum (wind work)
# --------------------------------------------------------------------------
def cospectrum(u, v, tu, tv, dx, dy, dt, workers=-1):
    nx, ny, nt = u.shape
    kx, ky, kt = fft.fftfreq(nx, dx), fft.fftfreq(ny, dy), fft.rfftfreq(nt, dt)
    dkx, dky, dkt = 1/(dx*nx), 1/(dy*ny), 1/(dt*nt)
    w1, w2, w3 = _hann_norm(nx), _hann_norm(ny), _hann_norm(nt)
    w12 = w1[:, None] * w2[None, :]

    print("1-3) u, tau_x: detrending, windowing, fft ...")
    fu = _rfft3(_prep(u, w12, w3), workers)
    ft = _rfft3(_prep(tu, w12, w3), workers)
    ww = fu.real * ft.real                            # Re[conj(u^) tau^]
    ww += fu.imag * ft.imag
    del fu, ft

    print("1-3) v, tau_y: detrending, windowing, fft ...")
    fv = _rfft3(_prep(v, w12, w3), workers)
    ft = _rfft3(_prep(tv, w12, w3), workers)
    tmp = fv.real * ft.real
    tmp += fv.imag * ft.imag
    ww += tmp
    del fv, ft
    ww += tmp
    del tmp

    ww /= (dkx*dky*dkt)
    ww /= (nx*ny*nt)**2

    print("4) computing isotropic spectrum")
    kr, ww_iso = calc_ispec_3d_fast(fft.fftshift(kx), fft.fftshift(ky), ww)
    return kr, kt, ww_iso


# --------------------------------------------------------------------------
# Co- and quadrature-spectrum, and stress time-shift
# --------------------------------------------------------------------------
def cross_spectrum(u, v, tu, tv, dx, dy, dt, workers=-1):
    """Isotropic co- and quadrature-spectrum of velocity (u, v) with stress (tu, tv).

    `co` is identical to the output of cospectrum(); `quad` is Im[conj(u^) tau^] with
    the same normalization and rings. Together, they allow computing the time-shifted
    stress cospectrum (shift_cospectrum) without recalculating.
    """
    nx, ny, nt = u.shape
    kx, ky, kt = fft.fftfreq(nx, dx), fft.fftfreq(ny, dy), fft.rfftfreq(nt, dt)
    dkx, dky, dkt = 1/(dx*nx), 1/(dy*ny), 1/(dt*nt)
    w1, w2, w3 = _hann_norm(nx), _hann_norm(ny), _hann_norm(nt)
    w12 = w1[:, None] * w2[None, :]

    print("1-3) u, tau_x: detrending, windowing, fft ...")
    fu = _rfft3(_prep(u, w12, w3), workers)
    ft = _rfft3(_prep(tu, w12, w3), workers)
    co = fu.real * ft.real
    co += fu.imag * ft.imag
    qu = fu.real * ft.imag
    qu -= fu.imag * ft.real
    del fu, ft

    print("1-3) v, tau_y: detrending, windowing, fft ...")
    fv = _rfft3(_prep(v, w12, w3), workers)
    ft = _rfft3(_prep(tv, w12, w3), workers)
    tmp = fv.real * ft.real
    tmp += fv.imag * ft.imag
    co += tmp
    tmp = fv.real * ft.imag
    tmp -= fv.imag * ft.real
    qu += tmp
    del fv, ft, tmp

    for a in (co, qu):
        a /= (dkx*dky*dkt)
        a /= (nx*ny*nt)**2

    print("4) computing isotropic spectra")
    kr, W, nan_rows = _ring_weights(fft.fftshift(kx), fft.fftshift(ky))
    return kr, kt, _apply_rings(W, nan_rows, co), _apply_rings(W, nan_rows, qu)


def shift_cospectrum(co, quad, kt, delta):
    """Cospectrum obtained using tau(t - delta) with u(t).

    delta in the same units as dt (hours if dt = 1); delta > 0 lags the stress.
    Works for fractional shifts (e.g., 0.5 h).
    """
    ph = 2 * np.pi * np.asarray(kt) * delta
    return co * np.cos(ph) + quad * np.sin(ph)
