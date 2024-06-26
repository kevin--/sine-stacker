#!/usr/bin/env python3

import sinewave
import soundfile
import librosa

import matplotlib.pylab as plt
import matplotlib.ticker as mticker
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


def draw_single_plot(
    x, result, harmonics: list, harmonic_index: int, depth_list: list, output_dir: Path
):
    fig, (components) = plt.subplots(1)

    for plot_depth in depth_list:
        r = result[plot_depth]
        components.plot(x, r[1])

    components.set(xlabel="Angle [radians]", ylabel="Amplitude")

    fig.suptitle(make_title(harmonics, harmonic_index))
    fig.set_size_inches(16, 10)
    fig.tight_layout()

    print(f"saving single harmonic plot {harmonics[harmonic_index]}")
    fig.savefig(output_dir / Path(f"{harmonics[harmonic_index]}_single.png"))

    fig.clear()
    plt.close()
    plt.cla()
    plt.clf()


def draw_time_plots(harmonics: list, make_overlay: bool = False):
    output_dir = Path("./output-plots")
    if not output_dir.is_dir():
        output_dir.mkdir()

    result = sinewave.harmonic_series(
        note_fundamental_hz, 1 / note_fundamental_hz, harmonics, samplerate
    )

    for i in range(len(harmonics)):
        is_first_fundamental = i == 0 and harmonics[i] == 1
        is_first_harmonic = i == 1 and harmonics[0] == 1

        x = result[0][0]
        summed_y_data = numpy.empty(len(result[0][1]))
        summed_y_data.fill(0)

        depth_list = range(i + 1)

        if is_first_fundamental or is_first_harmonic:
            draw_single_plot(x, result, harmonics, i, depth_list, output_dir)

        if make_overlay:
            fig, (components, summed, both) = plt.subplots(3)
        else:
            fig, (components, summed) = plt.subplots(2)

        for plot_depth in depth_list:
            r = result[plot_depth]

            components.plot(x, r[1])
            if make_overlay:
                both.plot(x, r[1])
            summed_y_data = numpy.add(summed_y_data, r[1])

        if make_overlay:
            both.plot(x, summed_y_data)

        components.set_title("Partials")
        components.set(xlabel="Angle [radians]", ylabel="Amplitude")

        summed.set_title("Summed")
        summed.set(xlabel="Angle [radians]", ylabel="Amplitude")
        summed.plot(x, summed_y_data)

        if make_overlay:
            both.set_title("Overlay")
            both.set(xlabel="Angle [radians]", ylabel="Amplitude")

        fig.suptitle(make_title(harmonics, depth_list[-1]))
        fig.set_size_inches(16, 10)
        fig.tight_layout()

        print(f"saving harmonic plot {harmonics[i]}")
        fig.savefig(output_dir / Path(f"{harmonics[i]}.png"))

        fig.clear()
        plt.close()
        plt.cla()
        plt.clf()


def format_frequency(x, pos):
    if x < 1000:
        return f"{int(x)}Hz"

    return f"{int(x/1000)}kHz"


def draw_freq_plot(buffer, samplerate: float, harmonics, max_harmonic: int):
    output_dir = Path("./output-fft")
    if not output_dir.is_dir():
        output_dir.mkdir()

    fft_resolution = 4096

    fig, (chart) = plt.subplots(1)
    chart.set_xscale("log", base=10)

    frequency_labels = librosa.fft_frequencies(sr=samplerate, n_fft=fft_resolution)

    S = librosa.stft(buffer, n_fft=fft_resolution)
    D = librosa.amplitude_to_db(numpy.abs(S), ref=numpy.max)
    # average over file
    D_AVG = numpy.mean(D, axis=1)

    chart.plot(frequency_labels, D_AVG)

    chart.set(xlabel="Frequency", ylabel="Magnitude [dB]")
    chart.xaxis.set_major_formatter(mticker.FuncFormatter(format_frequency))

    fig.suptitle(make_title(harmonics, max_harmonic))
    fig.set_size_inches(16, 10)
    fig.tight_layout()

    print(f"saving harmonic fft {harmonics[max_harmonic]}")
    fig.savefig(output_dir / Path(f"{harmonics[max_harmonic]}.png"))

    fig.clear()
    plt.close()
    plt.cla()
    plt.clf()


def write_waves(harmonics: list):
    output_dir = Path("./output-audio")
    if not output_dir.is_dir():
        output_dir.mkdir()

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


def render_single_harmonic_series():
    max_harmonic = 100

    # harmonics = sinewave.generate_odd_harmonics(max_harmonic)
    harmonics = sinewave.generate_even_harmonics(max_harmonic)

    draw_time_plots(harmonics)
    write_waves(harmonics)


def render_stacked_freq_plots():
    max_harmonic = 51
    even_harmonics = sinewave.generate_even_harmonics(max_harmonic)
    odd_harmonics = sinewave.generate_odd_harmonics(max_harmonic)

    waves = []

    # odd
    waves.append(
        sinewave.sum_series(
            sinewave.harmonic_series(note_fundamental_hz, 1, odd_harmonics, samplerate)
        )
    )
    # even
    waves.append(
        sinewave.sum_series(
            sinewave.harmonic_series(note_fundamental_hz, 1, even_harmonics, samplerate)
        )
    )

    output_dir = Path("./output-fft-stacked")
    if not output_dir.is_dir():
        output_dir.mkdir()

    fft_resolution = 4096

    fig, (chart) = plt.subplots(1)
    chart.set_xscale("log", base=10)

    frequency_labels = librosa.fft_frequencies(sr=samplerate, n_fft=fft_resolution)

    for buffer in waves:
        S = librosa.stft(buffer, n_fft=fft_resolution)
        D = librosa.amplitude_to_db(numpy.abs(S), ref=numpy.max)
        # average over file
        D_AVG = numpy.mean(D, axis=1)

        chart.plot(frequency_labels, D_AVG)

    chart.set(xlabel="Frequency", ylabel="Magnitude [dB]")
    chart.xaxis.set_major_formatter(mticker.FuncFormatter(format_frequency))

    fig.suptitle("Odd vs Even Harmonic Series")
    fig.set_size_inches(16, 10)
    fig.tight_layout()

    print(f"saving odd/even harmonic fft")
    fig.savefig(output_dir / Path(f"odd_even_{max_harmonic}.png"))

    fig.clear()
    plt.close()
    plt.cla()
    plt.clf()


if __name__ == "__main__":
    # render_single_harmonic_series()
    render_stacked_freq_plots()
