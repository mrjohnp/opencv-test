Mathematical model of the image degradation frequency in domain representation:

S = H * U + N

S: SPECTRUM of blurred (DEGRAGED) image
U: SPECTRUM of original true (UNDEGRADED) image
H: FREQUENCY response of POINT SPREAD FUNCTION (PSF) > specified by only one parameter > R: radius
N: SPECTRUM of ADDITIVE NOISE

Restoration formula in frequency domain:

U' = Hw * S

U': Spectrum of estimation of original image U
Hw: Restoration filte > eg. Wiener filter


Wiener filter:
Is a way to restore a blurred image.

Simplified version of the Wiener filter formula:

Hw = H / (|H|pow2 + 1 / SNR)

SNR: Signal to Noise Ratio

In order to recover an out-of-focus image by Wiener filter,
it needs to know the SNR e R of the circural PSF.

_______________________________________________________________

array(
    [
        [[856, 0]],
        [[855, 1]],
        [[855, 2]],
        [[851, 6]],
        [[851, 8]],
        [[852, 9]],
        [[853, 9]],
        [[854, 10]],
        [[855, 9]],
        [[855, 8]],
        [[856, 7]],
        [[856, 6]],
        [[858, 4]],
        [[858, 3]],
        [[856, 1]]
    ], dtype=int32)
