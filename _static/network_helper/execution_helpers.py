def reticle_init(wafer, defects_path):
    """
    Initializes the BrainScaleS system.
    This initialization is necessary when new mappings are loaded to ensure unused reticles do not interfere with the operation.

    :param wafer: An instance of the sthal wafer object representing the BrainScaleS wafer to initialize.
    :param defects_path: The file path to the defect results, used for handling known defects in the system.
    """

    class ConfigFPGAOnlyHICANNConfigurator(pysthal.ParallelHICANNv4Configurator):
        def config(self, fpga_handle, hicann_handle, hicann_data, stage):
            pass

    configurator = ConfigFPGAOnlyHICANNConfigurator()
    wafer.configure(configurator)


def set_sthal_params(wafer):
    """
    Sets hardware-specific settings for all HICANNs in use.
    These settings are predefined and should not be changed.

    :param wafer: An instance of the sthal wafer object representing the BrainScaleS wafer to configure.
    """
    for hicann in wafer.getAllocatedHicannCoordinates():
        fgs = wafer[hicann].floating_gates
        for ii in range(fgs.getNoProgrammingPasses().value()):
            cfg = fgs.getFGConfig(Enum(ii))
            cfg.fg_biasn = 0
            cfg.fg_bias = 0
            fgs.setFGConfig(Enum(ii), cfg)

        for block in iter_all(FGBlockOnHICANN):
            fgs.setShared(block, HICANN.shared_parameter.V_dllres, 275)
            fgs.setShared(block, HICANN.shared_parameter.V_ccas, 800)


def set_hardware_settings(marocco, runtime):
    """
    Sets the hardware configuration for the system required during the experiment.

    :param marocco: An instance of the marocco object used to orchestrate the experiment translation and configure the hardware.
    :param runtime: The duration of the experiment
    """
    marocco.neuron_placement.default_neuron_size(par.neuron_size)
    marocco.defects.backend = Defects.Backend.XML
    marocco.default_wafer = Wafer(30)
    marocco.calib_backend = PyMarocco.CalibBackend.Binary
    marocco.calib_path = par.calib_path
    marocco.defects.path = par.defects_path

    if not par.bigcap:
        # Use smallcap to achive higher weights and smaller synaptic time constants
        marocco.param_trafo.use_big_capacitors = False

    if par.slow:
        # Use i_gl slow
        marocco.param_trafo.i_gl_speedup = pysthal.SpeedUp.SLOW

    # set parameter trafo
    marocco.param_trafo.set_trafo_with_bio_to_hw_ratio(par.max_bio_voltage, par.min_bio_voltage, par.max_hw_voltage, par.min_hw_voltage)

    # Some checks can be skipped if already verified in previous runs
    marocco.verification = PyMarocco.Skip
    marocco.checkl1locking = PyMarocco.SkipCheck
    marocco.ensure_analog_trigger = PyMarocco.SkipEnsure
    marocco.scrutinize_mapping = PyMarocco.SkipScrutinize

    marocco.reset_parameters = True
    marocco.skip_mapping = True
    marocco.backend = PyMarocco.Hardware


def split(elements):
    """
    Helper function to split potential multapses into individual connections.
    Separates the first occurrence of each multapse and keeps the remaining occurrences in a separate list for further processing.

    :param elements: A list of connections that should be split if they represent multapses.
    :return: A tuple containing a list of unique connections and a list of connections that occured more than once.
    """
    added = set()
    res = []
    remaining = []
    for elem in elements:
        idx = (elem[0], elem[1])
        if idx in added:
            remaining.append(elem)
        else:
            res.append(elem)
            added.add(idx)
    return (res, remaining)


def split_multapses(matrix):
    """
    Function to split multapses into individual connections.
    Each connection that occurs more than once is split into separate "chunks," where each chunk contains only a single occurrence of the connection.

    :param matrix: A list of connections that may represent multapses and need to be split.
    :return: A list of lists, where each inner list contains only unique connections (no duplicates).
    """
    res = []
    result, remaining = split(matrix)
    res.append(result)
    max_multapses = 0
    while (len(remaining) > 0):
        result, remaining = split(remaining)
        res.append(result)
        max_multapses += 1
    logger.debug(f"Maximum number of multapses: {max_multapses}")
    return res


def generate_connector_from_matrix(matrix, simulator):
    """
    Generates a PyNN connector from the connectivity matrix.

    :param matrix: The connectivity matrix representing the network's connections.
    :param simulator: The simulator that is used, either "brainscales" or "nest".
    :return: A PyNN connector object configured based on the provided connectivity matrix and simulator.
    """
    if simulator == "brainscales":
        # In brainscales one projection can not implement multapses
        # -> split multapses to several projections
        split_connections = split_multapses(matrix)
        connector = []
        for conn in split_connections:
            connector.append(pynn.FromListConnector(conn))
        return connector
    else:
        connector = pynn.FromListConnector(matrix)
        return connector


def load_mapping(marocco, runtime, tag=None):
    """
    Loads previously generated mapping results from disk.

    :param marocco: An instance of the marocco object used to orchestrate the experiment translation and configure the hardware.
    :param runtime: An instance of the runtime object used to store the results.
    :param tag: A tag to distinguish different result files. Default is None.
    """
    marocco_file = "marocco_results.bin" if tag is None else f"marocco_results_{tag}.bin"
    sthal_file = "marocco_wafer.bin" if tag is None else f"marocco_wafer_{tag}.bin"
    logger.info(f"Load mapping results from {par.mapping_path}")
    runtime.results().load(os.path.join(par.mapping_path, marocco_file))
    runtime.wafer().load(os.path.join(par.mapping_path, sthal_file))
    logger.info("Finished loading mapping results")
    # set hardware specific parameters
    # If already included into mapping this can be skipped
    # set_sthal_params(runtime.wafer())


def set_nest():
    """
    Initializes the PyNN object for simulation.
    """
    master_seed = par.kernel_seed
    pynn.setup(timestep=par.timestep, min_delay=0.1, max_delay=15.0, threads=par.threads, native_rng_baseseed=master_seed)
    n_vp = pynn.nest.GetKernelStatus(['total_num_virtual_procs'])[0]
    logger.debug(f'num_virtual_processes {n_vp}')
    pynn.nest.SetKernelStatus({'print_time': False,
                               'dict_miss_is_error': True,
                               'rng_seeds': range(master_seed + n_vp + 1, master_seed + 2 * n_vp + 1)})


