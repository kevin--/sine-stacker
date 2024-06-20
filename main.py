#!/usr/bin/env python3

import sinewave
import soundfile
import librosa

import matplotlib.pylab as plt
import numpy
import os
from pathlib import Path


note_fundamental_hz = 440
samplerate = 48000


def make_title(harmonics: numpy.array, plot_depth: int):
    has_fundamental = harmonics[0] == 1

    if harmonics[plot_depth] == 1:
        return "Fundamental"
    elif has_fundamental:
        return "Fundamental + Harmonics " + ", ".join(
            map(str, harmonics[1 : plot_depth + 1])
        )
    else:
        return "Harmonics " + ", ".join(map(str, harmonics[:plot_depth]))


def draw_time_plots(max_harmonic: int):
    output_dir = Path("./output-plots")
    if not output_dir.is_dir():
        output_dir.mkdir()

    harmonics = sinewave.generate_odd_harmonics(max_harmonic)
    # harmonics = sinewave.generate_even_harmonics(100)

    result = sinewave.harmonic_series(
        note_fundamental_hz, 1 / note_fundamental_hz, harmonics, samplerate
    )

    for i in range(len(harmonics)):
        fig, (components, summed, both) = plt.subplots(3)

        x = result[0][0]
        summed_y_data = numpy.empty(len(result[0][1]))
        summed_y_data.fill(0)

        depth_list = range(i + 1)

        for plot_depth in depth_list:
            r = result[plot_depth]

            components.plot(x, r[1])
            both.plot(x, r[1])
            summed_y_data = numpy.add(summed_y_data, r[1])

        both.plot(x, summed_y_data)

        components.set_title("Components")
        components.set(xlabel="Angle [rad]", ylabel="sin(x)")

        summed.set_title("Summed")
        summed.set(xlabel="Angle [rad]", ylabel="sin(x)")
        summed.plot(x, summed_y_data)

        both.set_title("Overlay")
        both.set(xlabel="Angle [rad]", ylabel="sin(x)")

        fig.suptitle(make_title(harmonics, depth_list[-1]))
        fig.set_size_inches(16, 10)
        fig.tight_layout()

        print(f"saving harmonic plot {harmonics[i]}")
        fig.savefig(output_dir / Path(f"{harmonics[i]}.png"))


def draw_freq_plot(buffer, samplerate: float, harmonics, max_harmonic: int):
    output_dir = Path("./output-fft")
    if not output_dir.is_dir():
        output_dir.mkdir()

    fft_resolution = 4096

    S = librosa.stft(buffer, n_fft=fft_resolution)
    D = librosa.amplitude_to_db(numpy.abs(S), ref=numpy.max)
    # average over file
    D_AVG = numpy.mean(D, axis=1)

    frequency_labels = librosa.fft_frequencies(sr=samplerate, n_fft=fft_resolution)

    fig, (chart) = plt.subplots(1)
    chart.set_xscale("log", base=10)

    chart.plot(frequency_labels, D_AVG)

    chart.set(xlabel="Frequency [Hz]", ylabel="Magnitude [dB]")

    fig.suptitle(make_title(harmonics, max_harmonic))
    fig.set_size_inches(16, 10)
    fig.tight_layout()

    print(f"saving harmonic fft {harmonics[max_harmonic]}")
    fig.savefig(output_dir / Path(f"{harmonics[max_harmonic]}.png"))


def write_waves(max_harmonic: int):
    output_dir = Path("./output-audio")
    if not output_dir.is_dir():
        output_dir.mkdir()

    harmonics = sinewave.generate_odd_harmonics(max_harmonic)

    result = sinewave.harmonic_series(note_fundamental_hz, 1, harmonics, samplerate)

    summed_y_data = numpy.empty(len(result[0][1]))
    summed_y_data.fill(0)

    for i in range(len(harmonics)):
        r = result[i]
        summed_y_data = numpy.add(summed_y_data, r[1])

        draw_freq_plot(summed_y_data, samplerate, harmonics, i)

        print(f"saving harmonic audio {harmonics[i]}")
        soundfile.write(
            output_dir / Path(f"harmonic_{harmonics[i]}.wav"), summed_y_data, samplerate
        )


if __name__ == "__main__":
    max_harmonic = 100

    draw_time_plots(max_harmonic)
    write_waves(max_harmonic)
