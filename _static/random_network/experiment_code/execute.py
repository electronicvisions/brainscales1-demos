def evaluate_sweep_parameters():
    """
    Helper function used during hardware operation to find suitable mappings
    for different configurations of the balanced random network.

    :return: A dictionary where the keys are mapping IDs and
             the values are the associated network parameters.
    """
    mappings = defaultdict(lambda: defaultdict(list))
    for eta in par.etas:
        if eta > 3.2:
            raise RuntimeError("Values above eta = 3.2 are not supported by provided mappings")
        for g in par.gs:
            if g >= 7 and eta <= 1.2:
                mappings["all"][eta].append(g)
            else:
                mappings["30"][eta].append(g)
    return mappings


def run_on_nest():
    """
    Executes the balanced random network experiment on the NEST simulator.

    :return: The experiment object containing the experiment description,
             result data, and the path where results are stored.
    """
    mappings = evaluate_sweep_parameters()
    logger.info("Start simulations")
    experiment = RandomNetwork(pynn)
    for mapping_key, mapping in mappings.items():
        # We need to rebuild the network to map the hardware topology into the simulation
        for eta, gs in mapping.items():
            # Reconfigure external spikes
            for g in gs:
                # It seems that the used pyNN.nest version accumulates information in between runs
                # and becomes exponentially slower at some point.
                # To circumvent this, we reset the state and rebuild the network in between each run.
                start = datetime.now()
                set_nest()
                experiment.build(mapping_key, g)
                experiment.reconfigure_ext_spikes_sim(eta)
                logger.info(f"Start simulation with g={g}, eta={eta}")
                start_exp = datetime.now()
                logger.info(f"Finished setup of experiment with g={g}, eta={eta} in {start_exp-start}")
                experiment.run()
                stop = datetime.now()
                logger.info(f"Finished simulation with g={g}, eta={eta} in {stop-start_exp}")
                experiment.save_results(g=g, eta=eta, mapping=mapping_key)
                logger.info("Stored results")
    logger.info("Finished simulation")

    return experiment


def run_on_hw():
    """
    Emulates the balanced random network experiment on the BrainScaleS-1 hardware.

    :return: The experiment object containing the experiment description,
             result data, and the path where results are stored.
    """
    # prepare mapping
    marocco = PyMarocco()
    runtime = Runtime(marocco.default_wafer)
    set_hardware_settings(marocco, runtime)
    pynn.setup(marocco=marocco, marocco_runtime=runtime)

    # Build Network
    logger.info("Build network")
    experiment = RandomNetwork(pynn)
    experiment.build()

    # run on hardware
    logger.info("Start hardware executions")
    # All loaded mappings are generated with capacity for nu_ext = 3.2
    mappings = evaluate_sweep_parameters()
    for iteration, (mapping_key, mapping) in enumerate(mappings.items()):
        if iteration > 0:
            # We need to init the repeaters of all previously used hicanns to circumvent neighbor injections
            start = datetime.now()
            reticle_init(runtime.wafer(), marocco.defects.path)
            stop = datetime.now()
            logger.info(f"Hardware reset took {stop-start}")
        load_mapping(marocco, runtime, mapping_key)
        # Since we manually reset the wafer, we have to force the configurator to redo the configuration
        marocco.hicann_configurator.set_force()
        for eta, gs in mapping.items():
            # Reconfigure external spikes
            experiment.reconfigure_ext_spikes_hw(runtime, eta)
            last_g = par.mapping_g
            for g in gs:
                experiment.reset_network()
                experiment.modify_weights(last_g, g)
                last_g = g
                start = datetime.now()
                experiment.run()
                stop = datetime.now()
                logger.info(f"Finished hardware execution with g={g}, eta={eta} in {stop-start}")
                experiment.save_results(g=g, eta=eta, mapping=mapping_key)
                logger.info("Stored results")
                # Make the configurator smart again
                marocco.hicann_configurator.set_smart()
    logger.info("Finished hardware executions")

    return experiment


def main():
    """
    Executes the balanced random network experiment on either the BrainScaleS-1
    hardware or the NEST simulator, depending on the value of the `simulator`
    variable, and stores the results to disk.
    """
    startAll = datetime.now()

    # run on NEST
    if simulator == "nest":
        experiment = run_on_nest()
    # run on hardware
    elif simulator == "brainscales":
        experiment = run_on_hw()

    logger.info("Writing results to disk")
    store_hdf5(experiment)
    endAll = datetime.now()
    logger.info(f"Total time: {endAll - startAll}")


if __name__ == '__main__':
    main()
