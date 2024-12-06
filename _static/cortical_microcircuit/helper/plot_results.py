from _static.helper.plot_helper import load_hdf5
from collections import defaultdict
import numpy as np
from elephant.statistics import mean_firing_rate, isi, cv
from quantities import s, ms, Quantity
import matplotlib.pyplot as plt
import logging
import matplotlib
matplotlib.rcParams["font.size"] = "14"
logger = logging.getLogger("cortical_microcircuit_analyse")

labels = ["23e", "23i", "4e", "4i", "5e", "5i", "6e", "6i"]


def get_binwidth(rates):
    """
    Calculates the bin width using the Freedman-Diaconis rule from the recorded firing rates.

    :param rates: The recorded firing rates.
    :return: The calculated bin width.
    """
    # rule of Freedman-Diaconis
    return 2.0 * (np.quantile(rates, 0.75) - np.quantile(rates, 0.25)) / (float(rates.shape[0])**(1.0 / 3.0))


def rate_plot(fig, axs, segments, settings, name, color):
    """
    Adds a rate plot to a Matplotlib figure object for all eight populations.

    :param fig: A Matplotlib figure object to which the plot will be added.
    :param axs: The axis objects for the plot.
    :param segments: Neo `Segment` objects containing the result data.
    :param settings: A dictionary containing the settings used during the experiment.
    :param name: The name of the experiment.
    :param color: The color used to display the results.
    """
    logger.info(f"Start {name} rate plot")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)

    counter = 0
    for pop_key, segment in segments.items():
        spiketrain_list = segment.spiketrains
        # Not spiking neurons with ids larger than the last spiking neuron might be omitted
        # -> use population size for calculation
        lastNeuron = settings[pop_key]["population_size"]
        rates = np.zeros(lastNeuron)
        for spiketrain in spiketrain_list:
            nrn = spiketrain.annotations["source_index"]
            rates[nrn] = mean_firing_rate(spiketrain.rescale(s))

        val, binedges = np.histogram(rates, bins=list(np.arange(0, 100, 1)), density=True)

        layer = pop_key[:-1]
        syn_type = pop_key[-1]
        side = 0 if syn_type == "e" else 1
        c_plot = counter // 2
        axs[c_plot, side].stairs(edges=binedges, values=val, linewidth=2, color=color, label=name)
        axs[c_plot, side].text(0.95, 0.9, layer + syn_type, horizontalalignment='right',
                               verticalalignment='top', transform=axs[c_plot, side].transAxes)
        axs[c_plot, side].set_xlim(0, 25)
        counter += 1

    # set left ylabel
    for ax in axs[:, 0]:
        ax.set_ylabel('p')

    for ax in axs[3]:
        ax.set_xlabel('Rate (Hz)')
    axs[0, 0].legend(bbox_to_anchor=(0., 0.6, 0.8, .102), loc='center right', framealpha=0.0)
    logger.info(f"Finished {name} rate plot")


def irregularity_plot(fig, axs, segments, settings, name, color):
    """
    Adds an irregularity plot to a Matplotlib figure object for all eight populations.

    :param fig: A Matplotlib figure object to which the plot will be added.
    :param axs: The axis objects for the plot.
    :param segments: A dictionary of Neo `Segment` objects containing the result data.
    :param settings: A dictionary containing the settings used during the experiment.
    :param name: The name of the experiment.
    :param color: The color used to display the results.
    """
    logger.info(f"Start {name} irregularity plot")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)

    counter = 0
    for pop_key, segment in segments.items():
        spiketrain_list = segment.spiketrains
        cvs = []
        for spiketrain in spiketrain_list:
            # in newer elephant version spiketrain is no longer sorted before isi is calculated
            # -> sort before using function
            # during this, the duration is overwritten and the result is no longer a spiketrain
            # for the caluclation of the mean firing t_start has to be readded
            spiketrain_sorted = np.sort(spiketrain.times.view(Quantity))
            inter_spike_times = isi(spiketrain_sorted)
            if inter_spike_times.size:
                cvs.append(cv(inter_spike_times))
        cvs = np.array(cvs)
        binwidth = get_binwidth(cvs)
        bins = np.arange(0, 2, binwidth)
        layer = pop_key[:-1]
        syn_type = pop_key[-1]
        side = 0 if syn_type == "e" else 1
        c_plot = counter // 2

        axs[c_plot, side].hist(cvs, label=name, bins=bins, linestyle="-", histtype="step", density=True, color=color)
        axs[c_plot, side].text(0.95, 0.9, layer + syn_type, horizontalalignment='right',
                               verticalalignment='top', transform=axs[c_plot, side].transAxes)
        axs[c_plot, side].set_xlim(0, 2)
        counter += 1

    # set left ylabel
    for ax in axs[:, 0]:
        ax.set_ylabel('p')

    for ax in axs[3]:
        ax.set_xlabel('CV')
    logger.info(f"Finished {name} irregularity plot")


def spiketime_plot(segments, settings, name, iteration, t_min, t_max, frac_neurons):
    """
    Generates a plot showing spike times of a subset of neurons across all eight populations.

    :param segments: A dictionary of Neo `Segment` objects containing the result data.
    :param settings: A dictionary containing the settings used during the experiment.
    :param name: The name of the experiment.
    :param iteration: The experiment iteration during which the data was recorded.
    :param t_min: The start time for the data to be plotted.
    :param t_max: The end time for the data to be plotted.
    :param frac_neurons: The fraction of neurons from each population to display in the plot.
    """
    logger.info('Start Spiketimes plot')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # determine vertical offset
    offset = 0
    n_to_plot = {}
    for key in labels:
        n_to_plot[key] = int(round(settings[key]["population_size"] * frac_neurons))
        offset += n_to_plot[key]

    y_max = offset + 1
    yticks = []
    ytickslocs = []

    for key, segment in segments.items():
        yticks.append(key)
        ytickslocs.append(offset - 0.5 * n_to_plot[key])
        pop_color = '#595289' if key.endswith("e") else '#af143c'
        sliced_segment = segment.time_slice(t_min * ms, t_max * ms)
        for spikes in sliced_segment.spiketrains:
            n = spikes.annotations["source_index"]
            if n > n_to_plot[key]:
                break
            ax.plot(spikes, np.zeros(len(spikes)) + offset - n, '.', color=pop_color, markersize=3)
        offset = offset - n_to_plot[key]

    # label figure
    y_min = offset
    ax.set_xlim([t_min, t_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel('time (ms)', size=16)
    ax.set_ylabel('Population', size=16)
    ax.set_yticks(ytickslocs)
    ax.set_yticklabels(yticks, fontsize='large')
    fig.savefig(f"./plots/cortical_microcircuit/{name}_spiketimes_{iteration}.png")
    plt.close(fig)
    logger.info(f"Finished {name} spiketimes plot")


def load_data(folder, name, measure_spikes_from_time):
    """
    Loads result data from disk.

    :param folder: The path to the folder where the results are stored.
    :param name: The name of the experiment.
    :param measure_spikes_from_time: The start time from which the data should be considered.
    :return: A tuple containing:
        - A dictionary with the settings used during the experiment.
        - A dictionary of Neo `Segment` objects containing the result data.
    """
    logger.info(f"Load {name} results")
    blocks = {}
    for label in labels:
        blocks[label] = load_hdf5(f"{folder}/{label}.h5")

    # delete first spikes (not regular spiking at the beginning)
    segments = defaultdict(dict)
    settings = defaultdict(dict)
    for label, block in blocks.items():
        for n_seg, segment in enumerate(block.segments):
            sim_time = segment.annotations["sim_time"]
            # delete first spikes (not regular spiking at the beginning)
            segments[n_seg][label] = segment.time_slice(measure_spikes_from_time * ms, sim_time * ms)
            population_size = segment.annotations["population_size"]
            settings[n_seg][label] = {"sim_time": sim_time, "population_size": population_size}
    return settings, segments


def plot(result_dir_bss1, result_dir_nest, measure_spikes_from_time=1000, frac_neurons=0.1, t_min=3000, t_max=3500):
    """
    Generates all relevant plots for the cortical microcircuit.

    :param result_dir_bss1: Path to the results obtained from the BrainScaleS system.
    :param result_dir_nest: Path to the results obtained from the HPC cluster.
    :param measure_spikes_from_time: The start time from which the data should be considered.
    :param frac_neurons: The fraction of neurons from each population to display in the spike time plot.
    :param t_min: The start time for the data to be plotted in the spike time plot.
    :param t_max: The end time for the data to be plotted in the spike time plot.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-2s [%(name)s] %(message)s',
                        filename=None,
                        filemode='w')
    figs = defaultdict(list)
    axs = defaultdict(list)
    plot_names = ("Rate", "Irregularity")

    for folder, name, color in [(result_dir_bss1, "BSS-1", "black"), (result_dir_nest, "NEST", "blue")]:
        if folder is None:
            logger.info(f"No {name} data available. Please (re-)execute a job on the respective site.")
            continue
        settings, segments = load_data(folder, name, measure_spikes_from_time)
        for iteration, segment in segments.items():
            setting = settings[iteration]
            for plot_name in plot_names:
                if len(figs[plot_name]) <= iteration:
                    fig, ax = plt.subplots(4, 2, sharex='col', sharey='row')
                    figs[plot_name].append(fig)
                    axs[plot_name].append(ax)
            rate_plot(figs["Rate"][iteration], axs["Rate"][iteration], segment, setting, name, color)
            irregularity_plot(figs["Irregularity"][iteration], axs["Irregularity"][iteration], segment, setting, name, color)
            spiketime_plot(segment, setting, name, iteration, t_min, t_max, frac_neurons)

    for plot_name, figures in figs.items():
        for it, fig in enumerate(figures):
            fig.savefig(f"./plots/cortical_microcircuit/{plot_name}_distribution_{it}.png", bbox_inches='tight')
            plt.close(fig)
    logger.info("plotting done")
