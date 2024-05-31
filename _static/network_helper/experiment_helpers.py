def evaluate_folder(experiment):
    """
    Evaluates the folder where results are stored.
    If the function is called for the first time, a folder is generated with the format
    `./results/{experiment_name}_{counter}` and its path is stored in `result_folder`.

    :param experiment: The experiment object.
    :return: The path to the folder where the results are stored.
    """
    if experiment.result_folder:
        return experiment.result_folder
    experiment.result_folder = f"results/{experiment.name}"
    folder_base = experiment.result_folder
    filecounter = 0
    while os.path.exists(experiment.result_folder):
        filecounter += 1
        experiment.result_folder = f"{folder_base}_{filecounter}"
    experiment.result_folder += '/'
    os.makedirs(experiment.result_folder)
    logger.debug(f"Results will be stored in {experiment.result_folder}")
    return experiment.result_folder


def store_hdf5(experiment):
    """
    Writes results to disk in a custom HDF5 format.

    :param experiment: The experiment object containing result data and the path where results should be stored.
    """
    # Custom format to store results
    for pop_type, block in experiment.results.items():
        with h5py.File(f"{evaluate_folder(experiment)}{pop_type}.h5", "w") as hf:
            block_group = hf.create_group("block")
            block_group.create_dataset("name", data=block.name)
            segments_group = block_group.create_group("segments")
            for num, segment in enumerate(block.segments):
                seg_group = segments_group.create_group(str(num))
                spiketrains_group = seg_group.create_group("spiketrains")
                annotations = seg_group.create_group("annotations")
                for key, val in segment.annotations.items():
                    annotations.create_dataset(key, data=val)
                for n_sp, sp in enumerate(segment.spiketrains):
                    # size of result could be minimized by removing additional group but
                    # requires more logic during loading
                    spiketrain_group = spiketrains_group.create_group(str(n_sp))
                    spiketrain_group.create_dataset("spiketrain", data=sp)
                    for key, val in sp.annotations.items():
                        spiketrain_group.create_dataset(key, data=val)
            logger.debug(f"Saved data of Population {pop_type}")
    logger.debug("Done writing")


def transform_spikes_to_neo(pop, size, data, simulation_time):
    """
    Transforms spike results into the Neo format.

    :param pop: The name of the population.
    :param size: The size of the population.
    :param data: The resulting spike data.
    :param simulation_time: The duration of the simulation.
    :return: A Neo `Segment` object containing the result data.
    """
    result = Segment(description=pop)
    try:
        for n in range(size):
            spikes = data[np.where(data[:, 0] == n)][:, 1]
            sptr = SpikeTrain(spikes * ms, t_start=0 * ms, t_stop=simulation_time * ms)
            sptr.annotate(source_population=pop, source_index=n)
            result.spiketrains.append(sptr)
    except IndexError:
        logger.error("No spike data for Population {pop}")
    return result


def write_results_to_disk(experiment):
    """
    Writes results to disk using the NeoMatlabIO format.

    :param experiment: The experiment object containing result data and the path where results should be stored.
    """
    logger.debug("Write results to disk")
    for key, value in experiment.results.items():
        # NeoMatlabIO is the only Neo IO that allows for writing and reading data
        # and generates resonable compressed files and is stable enough between versions
        # However it does not store annotations
        w = NeoMatlabIO(f'{evaluate_folder(experiment)}{key}.mat')
        w.write(value)
        logger.debug(f"Saved data of Population {key}")
    logger.debug("Done writing")


