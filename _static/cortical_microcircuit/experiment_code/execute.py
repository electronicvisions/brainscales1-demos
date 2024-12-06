def run_on_nest():
    """
    Executes the cortical microcircuit experiment on the NEST simulator.

    :return: The experiment object containing the experiment description,
             result data, and the path where results are stored.
    """
    set_nest()
    experiment = CorticalNetwork(pynn)
    experiment.build()
    experiment.run()
    experiment.save_results()
    return experiment


def run_on_hw():
    """
    Emulates the cortical microcircuit experiment on the BrainScaleS-1 hardware.

    :return: The experiment object containing the experiment description,
             result data, and the path where results are stored.
    """
    # prepare hardware configuration
    marocco = PyMarocco()
    runtime = Runtime(marocco.default_wafer)
    set_hardware_settings(marocco, runtime)
    load_mapping(marocco, runtime)
    pynn.setup(marocco=marocco, marocco_runtime=runtime)

    # build network
    logger.info("Build network")
    experiment = CorticalNetwork(pynn)
    experiment.build()

    logger.info("Start hardware execution")
    iterations = 2 if par.record_second_sample else 1
    for iteration in range(iterations):
        if iteration > 0:
            time.sleep(par.wait_time)
            marocco.reset_parameters = False
            marocco.hicann_configurator = pysthal.NOPHICANNConfigurator()
        experiment.reset_network()
        logger.info(f"Start Run {iteration}")
        experiment.run()
        logger.info(f"Transform results of Run {iteration}")
        experiment.save_results()
    logger.info("Finished hardware execution")
    return experiment


def main():
    """
    Executes the cortical microcircuit experiment on either the BrainScaleS-1 hardware or the NEST simulator,
    depending on the value of the `simulator` variable, and stores the results to disk.
    """
    startAll = datetime.now()

    # run on NEST
    if simulator == "nest":
        experiment = run_on_nest()
    # run on hardware
    elif simulator == "brainscales":
        experiment = run_on_hw()

    logger.info(f"Writing results to disk")
    store_hdf5(experiment)
    endAll = datetime.now()
    logger.info(f"Total time: {endAll - startAll}")


if __name__ == '__main__':
    r = main()
