import numpy

import math


def sinebuffer(
    freq_hz: float, length_seconds: float, amplitude: float, samplerate: float
):
    buffer_size = int(math.ceil(length_seconds * samplerate))

    radians_per_sample = 2 * numpy.pi * freq_hz / samplerate

    x = numpy.empty(buffer_size)
    phase = 0
    for i in range(buffer_size):
        x[i] = phase
        phase += radians_per_sample

    result = numpy.multiply(numpy.sin(x), amplitude)

    return x, result


def harmonic_series(
    fundamental_freq_hz, length_seconds: float, harmonics: list, samplerate: float
):
    result: list = []

    for harmonic in harmonics:
        result.append(
            sinebuffer(
                fundamental_freq_hz * harmonic, length_seconds, 1 / harmonic, samplerate
            )
        )

    return result


def generate_odd_harmonics(max: int):
    return numpy.arange(1, max, 2)


def generate_even_harmonics(max: int):
    x = numpy.arange(0, max, 2)
    x[0] = 1
    return x
