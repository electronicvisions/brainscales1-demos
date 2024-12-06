import numpy as np
import matplotlib.pyplot as plt
from _static.helper.plot_helper import load_hdf5
from neo import SpikeTrain
from quantities import ms, s, Quantity
from elephant.statistics import isi, cv, mean_firing_rate
import os
from collections import defaultdict
import pandas as pd
import logging
logger = logging.getLogger("random_network_analyse")

img_variables = ['neuron_rate', 'neuron_cv', 'global_cv']


def get_correction(val):
    """
    Applies a correction to the relative inhibitory strength based on the weight calibration for inhibitory weights.

    This correction is necessary because inhibitory circuits exhibit slightly different weight calibration.
    However, for simplicity, the same calibration is used for both excitatory and inhibitory weights during experiments.

    :param val: The initial relative inhibitory strength to be corrected.
    :return: The corrected relative inhibitory strength.
    """
    return (0.8693828 * val + 0.28674744)


def evaluation(directory='results/', plot_folder='./plots/'):
    """
    Evaluates the firing behavior of the balanced random network for all different configurations of external firing rate and relative inhibitory weight and generates a spike time plot for each regime.
    Moreover, firing statistics are extracted.

    :param directory: The directory where data to be evaluated is stored.
    :param plot_folder: The directory where result plots should be stored.
    :return: A pandas dataframe containing the network statistics for all different configurations.
    """
    logger.info('Evaluation started')
    populations = ["exc", "inh"]
    # Result storage for overview plots
    results = defaultdict(list)

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    # Open files
    segments = {}

    for pop in populations:
        segments[pop] = load_hdf5(os.path.join(directory, f"{pop}.h5")).segments

    for segment_exc, segment_inh in zip(*[segments[pop] for pop in populations]):
        end_time = segment_exc.t_stop
        start_time = segment_exc.t_start
        simtime = end_time - start_time
        considered_time = simtime - 1000 * ms  # Index where the signal should have reached equlilbrium
        exc_spiketrains = segment_exc.time_slice(considered_time, end_time).spiketrains
        inh_spiketrains = segment_inh.time_slice(considered_time, end_time).spiketrains
        considered_spiketrains = exc_spiketrains + inh_spiketrains
        # Eval metadata
        g = segment_exc.annotations["g"]
        eta = segment_exc.annotations["eta"]
        if g >= 7 and eta <= 1.2:
            # Use all spikes for evaluation
            considered_spiketrains = np.array(considered_spiketrains, dtype="object")[:2083]
        else:
            # Use only 30 special spikes for evaluation
            considered_spiketrains = np.array(considered_spiketrains, dtype="object")[:30]

        # Careful this removes start time annotation
        # Therefore we again make a SpikeTrain out of the result
        considered_spiketrains = [SpikeTrain(np.sort(st.times.view(Quantity)), t_start=considered_time, t_stop=end_time) for st in considered_spiketrains]
        global_spikes = np.sort([sp for st in considered_spiketrains for sp in st])

        name = f"g_{g}_eta_{eta}"
        pltname = os.path.join(plot_folder, name + ".png")
        # Plotting starts here
        start_hist, end_hist = int(simtime) - 1000, int(simtime)
        # Bin size of 0.2 used -> cf. BA Schwarzenböck
        bins = int((end_hist - start_hist) / 0.2)
        start_plot, end_plot = int(simtime) - 200, int(simtime) - 100

        fig = plt.figure(figsize=(5, 4))
        gs = fig.add_gridspec(2, hspace=0., height_ratios=[1, 0.6])
        ax = gs.subplots(sharex=True, sharey=False)

        ax[0].set(ylabel='Neuron Id')
        ax[1].set(ylabel='Spikes', xlabel='Time (ms)')
        ax[1].set_xlim(start_plot, end_plot)

        num_trains = min(30, len(considered_spiketrains))
        for n in range(num_trains):
            spiketrain = considered_spiketrains[n]
            ax[0].plot(spiketrain, n * np.ones(len(spiketrain)), linewidth=0, marker='|', markersize=3, color="black")

        hist, _, _ = ax[1].hist(global_spikes, bins=bins, range=(start_hist, end_hist), rwidth=0.9, color="black")
        ax[0].set_xlim(start_plot, end_plot)
        ax[0].set_yticks([0, 10, 20, 30])
        ax[1].set_xticks([9800, 9825, 9850, 9875, 9900])

        fig.tight_layout()
        fig.savefig(pltname)
        plt.close(fig)

        # Store results for heatmaps
        neuron_isi = [isi(tst) for tst in considered_spiketrains]
        neuron_cv = [cv(n) for n in neuron_isi if n.size]
        neuron_mean_cv = np.mean(neuron_cv)
        # The next step requires neo spiketrains with correct start and stop time
        # If start/stop is provided manually, neo does a different calculation and fails for empty spiketrains
        neuron_freq = [mean_firing_rate(st.rescale(s)) for st in considered_spiketrains]
        neuron_mean_frq = np.mean(neuron_freq)
        hist_cv = cv(hist)

        results["g"].append(float(g))
        results["eta"].append(float(eta))
        results["neuron_rate"].append(neuron_mean_frq)
        results["neuron_cv"].append(neuron_mean_cv)
        results["global_cv"].append(hist_cv)

    return pd.DataFrame.from_dict(results)


def plot_heatmaps(DF, correct_g_values, res_path):
    """
    Generates heatmap plots of the mean firing rate, irregularity, and synchrony of the balanced random network for all different configurations of external firing rate and relative inhibitory weight.

    :param DF: The pandas dataframe that contains the network statistics (extracted via the evaluation function).
    :param correct_g_values: Boolean value indicating whether to apply slightly different calibration values for inhibitory weights.
    :param res_path: The directory where result plots should be stored.
    """
    axis_g_data = np.array(np.sort(list(set(DF["g"]))))
    axis_eta_data = np.array(np.sort(list(set(DF["eta"]))))

    if correct_g_values:
        corrected_g_data = np.array([get_correction(k) for k in axis_g_data])
    else:
        corrected_g_data = axis_g_data

    imgs = {}
    for var in img_variables:
        imgs[var] = {}
        for g, valu in DF.groupby("g"):
            imgs[var][g] = {}
            for eta, val in valu.groupby("eta"):
                imgs[var][g][eta] = np.mean(val[var].values)

    for var, img in imgs.items():
        X, Y = np.meshgrid(corrected_g_data, axis_eta_data)
        fig = plt.figure()
        ax = plt.axes()
        values = np.array([[img[g][eta] for g in axis_g_data] for eta in axis_eta_data])
        cs = ax.pcolormesh(X, Y, values)

        ax.set_xlabel('g')
        ax.set_ylabel('eta')
        fig.tight_layout()
        fig.colorbar(cs, ax=ax)

        if not os.path.exists(res_path):
            os.makedirs(res_path)
        fig.savefig(f"{res_path}/{var}.png")
        plt.close(fig)


def plot(result_dir_bss1, result_dir_nest, correct_g=False):
    """
    Generates all relevant plots for the balanced random network.

    :param result_dir_bss1: Path to the results obtained from the BrainScaleS system.
    :param result_dir_nest: Path to the results obtained from the HPC cluster.
    :param correct_g: Boolean value indicating whether to apply slightly different calibration values for inhibitory weights.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-2s [%(name)s] %(message)s',
                        filename=None,
                        filemode='w')
    for folder, name in [(result_dir_bss1, "BSS-1"), (result_dir_nest, "NEST")]:
        if folder is None:
            logger.info(f"No {name} data available. Please (re-)execute a job on the respective site.")
            continue
        res_path = f"./plots/random_network/{name}/"
        result_df = evaluation(directory=folder, plot_folder=res_path)
        plot_heatmaps(result_df, correct_g, res_path)
    logger.info("plotting done")
