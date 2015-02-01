import numpy as np
import pyfftw
import matplotlib.pyplot as plt

"""
Pseudo-spectral simulations sometimes store "compressed" Fourier space
representations, in the sense that 2/3 dealiasing or something like it
is used, and the files written on disk don't contain the zeros.

This code is supposed to show a valid approach to handling such data.
"""

def generate_data(
        n,
        bytes_per_float = 4,
        p = 1.5):
    """
    generate something that has the proper shape
    """
    assert(n % 2 == 0)
    a = np.zeros((n, n, n/2+1), dtype = 'c{0}'.format(bytes_per_float*2))
    a[:] = np.random.randn(*a.shape) + 1j*np.random.randn(*a.shape)
    k, j, i = np.mgrid[0:n, 0:n, 0:n/2+1]
    a /= (k**2 + j**2 + i**2)**(p/2)
    a[0, 0, 0] = 0.0
    return a

def padd_with_zeros(
        a,
        n,
        odtype = None):
    if (type(odtype) == type(None)):
        odtype = a.dtype
    assert(a.shape[0] <= n)
    b = np.zeros((n, n, n/2 + 1), dtype = odtype)
    m = a.shape[0]
    b[     :m/2,      :m/2, :m/2+1] = a[     :m/2,      :m/2, :m/2+1]
    b[     :m/2, n-m/2:   , :m/2+1] = a[     :m/2, m-m/2:   , :m/2+1]
    b[n-m/2:   ,      :m/2, :m/2+1] = a[m-m/2:   ,      :m/2, :m/2+1]
    b[n-m/2:   , n-m/2:   , :m/2+1] = a[m-m/2:   , m-m/2:   , :m/2+1]
    return b

def main():
    n = 64
    N = 256
    a = generate_data(n, p = 1.8)
    b = padd_with_zeros(a, N)
    c = np.zeros((N, N, N), np.float32)
    t = pyfftw.FFTW(
        b, c,
        axes = (0, 1, 2),
        direction = 'FFTW_BACKWARD',
        flags = ('FFTW_ESTIMATE',),
        threads = 4)
    t.execute()
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_axes([.0, .0, 1., 1.])
    ax.set_axis_off()
    ax.imshow(c[:, :, 0],
              interpolation = 'none')
    fig.savefig('tst2.pdf', format = 'pdf')
    ax.cla()
    ax.set_axis_off()
    ax.imshow(c[:, 0],
              interpolation = 'none')
    fig.savefig('tst1.pdf', format = 'pdf')
    ax.cla()
    ax.set_axis_off()
    ax.imshow(c[0],
              interpolation = 'none')
    fig.savefig('tst0.pdf', format = 'pdf')
    return None

if __name__ == '__main__':
    main()

