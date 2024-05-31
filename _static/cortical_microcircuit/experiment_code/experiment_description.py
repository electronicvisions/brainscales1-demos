class CorticalNetwork(object):
    """
    Class to model and execute the cortical microcircuit experiment.

    This class handles the creation of the network structure for the cortical microcircuit,
    execution, and result management.
    """

    def __init__(self, pynn):
        """
        Initialize the experiment and set up bookkeeping variables.

        This constructor creates the experiment object and initializes any internal state required
        for the simulation/emulation.

        :param pynn: A Pynn instance used for running the experiment.
        """
        self.name = "cortical_microcircuit"
        # total connection counter
        self.total_connections = 0

        # set seeds
        self.network_rng = rn.NumpyRNG(seed=par.kernel_seed, parallel_safe=par.parallel_safe)
        self.weight_rng = rn.NumpyRNG(seed=par.weight_seed, parallel_safe=par.parallel_safe)
        self.init_rng = rn.NumpyRNG(seed=par.init_seed, parallel_safe=par.parallel_safe)
        self.delay_rng = rn.NumpyRNG(seed=par.delay_seed, parallel_safe=par.parallel_safe)
        self.variation_rng = rn.NumpyRNG(seed=par.var_seed, parallel_safe=par.parallel_safe)

        # simulation or hardware run
        self.pynn = pynn

        self.simulator = simulator

        self.neuron_parameters = par.neuron_parameters

        # whether parameter variation is added to the model
        self.add_variation = True
        self.variation = par.variation
        self.variation_boundaries = par.variation_boundaries

        # set model
        self.model = self.pynn.IF_cond_exp

        self.result_folder = None
        self.results = {}
        self._init_results()

        # calculate indegrees from connection probability
        self.indegrees = self._get_indegrees()

        self.weights = self._calculate_weights()

    def _generate_populations(self):
        """
        Generates the eight populations of the network.
        """
        # set populations
        self.populations = {}
        for layer, num in par.num_neurons.items():
            self.populations[layer] = self.pynn.Population(
                int(num * par.scale), self.model, cellparams=self.neuron_parameters, label=layer)

    def _apply_variations(self):
        """
        Applies neuron parameter variations to all populations.

        This function adjusts the neuron parameters across all populations to model the hardware behavior
        of the BrainScaleS-1 system. Therefore, it should only be applied during the NEST simulation.
        """
        for param, std in self.variation.items():

            # skip parameters with no variation
            if std == 0:
                continue
            logger.debug(f"{param} is affected by parameter variation")

            # create normal distribution around initial parameters and apply them to populations
            for layer, pop in self.populations.items():

                # reversal potentials are treated separately, as applying the relative width directly does not make sense
                mean = pop.get(param)

                if not np.isscalar(mean):
                    logger.error(f"Parameter variaton requires a uniform parameter dirstribution. "
                                 f"{param} already has different values for individual neurons. "
                                 f"Maybe it is already affected by a prevoius parameter variation.")
                    raise RuntimeError(f"Can not apply variations to {param}.")

                # create gaussian
                normal_dist = rn.RandomDistribution("normal_clipped",
                                                   mu=mean,
                                                   sigma=std,
                                                   low=self.variation_boundaries[param][0],
                                                   high=self.variation_boundaries[param][1],
                                                   rng=self.variation_rng)

                # manually redraw the values outside the boundaries since pynn module is defect, then set them for the population.
                # Clipping does not work, since threshold = reset is not supported.
                initial_vals = normal_dist.next(pop.size)
                vals, new_mean = self._redraw(initial_vals, normal_dist, self.variation_boundaries[param])
                pop.tset(param, vals)
                logger.debug(f"{layer} mean was shifted by {new_mean-mean:.2f} from {mean:.2f} to {new_mean:.2f}")

    def _init_membranes(self):
        """
        Initializes the neuron's membrane potential. This is only possible in the NEST simulation.
        """
        V0_mean = {}
        V0_sd = {}
        V_dist = {}

        for key, pop in self.populations.items():
            V0_mean[key] = (np.mean(self.populations[key].get('v_thresh')) + np.mean(self.populations[key].get('v_rest'))) / 2
            V0_sd[key] = np.abs((np.mean(self.populations[key].get('v_thresh')) - np.mean(self.populations[key].get('v_rest'))) / 3)
            V_dist[key] = rn.RandomDistribution('normal', mu=V0_mean[key], sigma=V0_sd[key], rng=self.init_rng)
            pop.initialize(v=V_dist[key])

    def _generate_recurrent_connections(self):
        """
        Generates the internal connections of the network.
        """
        self.projections = {}
        self.weight_dist = {}
        self.delay_dist = {}

        # recurrent connections
        values = {}
        if self.simulator == "nest" and par.internal_connector == "load_routing_results":
            logger.debug(f"Load routing results from {par.routing_results_path}")
            # Preload routing results
            with open(par.routing_results_path, "rb") as f:
                routing_results = pickle.load(f)

        for target_index, target_pop in enumerate(par.label):
            self.projections[target_pop] = {}
            for source_index, source_pop in enumerate(par.label):

                # set delay and weights
                weight_mean = self.weights[target_index][source_index]
                weight_std = par.synaptic_weight_std * weight_mean

                if source_pop.endswith("e"):
                    target = "excitatory"
                    delay_mean = par.excitatory_delay
                    delay_std = par.excitatory_delay_std
                    lower_bound = 0
                    upper_bound = np.inf

                elif source_pop.endswith("i"):
                    target = "inhibitory"
                    delay_mean = par.inhibitory_delay
                    delay_std = par.inhibitory_delay_std
                    lower_bound = 0
                    upper_bound = np.inf

                weight_distribution = rn.RandomDistribution("normal_clipped",
                                                           mu=weight_mean,
                                                           sigma=weight_std,
                                                           low=lower_bound,
                                                           high=upper_bound,
                                                           rng=self.weight_rng)
                delay_distribution = rn.RandomDistribution("normal_clipped",
                                                          mu=delay_mean,
                                                          sigma=delay_std,
                                                          low=0.1,
                                                          high=15.0,
                                                          rng=self.delay_rng)

                source_size = self.populations[source_pop].size
                target_size = self.populations[target_pop].size

                # Own fixed total number connector with multiple connections -> Potjans and Diesmann
                if self.simulator == "brainscales" or par.internal_connector == "fixedtotalnumber":
                    # In-degree scaling as described in Albada et al. (2015) "Scalability of Asynchronous Networks
                    # Is Limited by One-to-One Mapping between Effective Connectivity and Correlations"
                    # Number of inputs per target neuron (in-degree) for full scale model is scaled with k_scale
                    # To receive total connection number it is multiplied with downscaled target population size (scale)
                    n_connection = int(round(self.indegrees[target_index][source_index] * par.k_scale * target_size))
                    self.total_connections += n_connection
                    if (n_connection == 0):
                        continue

                    # connection matrix [(neuron_pop1,neuron_pop2,weight,delay),(...)]
                    # With autapses
                    matrix = np.zeros((4, n_connection), dtype=float)
                    matrix[0] = self.network_rng.randint(0, source_size, n_connection)
                    matrix[1] = self.network_rng.randint(0, target_size, n_connection)
                    matrix[2] = weight_distribution.next(n_connection)
                    matrix[3] = delay_distribution.next(n_connection)

                    matrix = matrix.T
                    # pyhmf expects int values for neuron numbers.
                    # Using dtype = int, the weights get rounded to zero and no connection is established
                    matrix = [[int(a), int(b), c, d] for a, b, c, d in matrix]
                    values[(source_index, target_index)] = matrix
                    connector = generate_connector_from_matrix(matrix, self.simulator)

                elif par.internal_connector == "load_routing_results":
                    # Use preloaded routing results
                    # Does not check number of neurons in Populations
                    # Only use networks with matching Populations
                    try:
                        matrix = routing_results[(source_index, target_index)]
                    except KeyError:
                        matrix = []
                    if len(matrix) == 0:
                        logger.debug(f"No connections found for {source_pop} to {target_pop}")

                    # translate delays, last column in results stores number of HICANNs in route
                    weights_dummy = weight_distribution.next(len(matrix))
                    matrix = [[pre, post, weights_dummy[n], self._apply_delay_calib(d)] for n, (pre, post, w, d) in enumerate(matrix)]
                    self.total_connections += len(matrix)
                    connector = generate_connector_from_matrix(matrix, self.simulator)
                    values[(source_index, target_index)] = matrix

                else:
                    raise Exception("wrong Connector")

                # implement projection
                logger.debug(f"creating connection from {source_pop} to {target_pop}")
                if self.simulator == "brainscales":
                    if isinstance(connector, list):
                        self.projections[target_pop][source_pop] = []
                        for c in connector:
                            self.projections[target_pop][source_pop].append(self.pynn.Projection(
                                self.populations[source_pop], self.populations[target_pop], c, target=target))
                    else:
                        self.projections[target_pop][source_pop] = self.pynn.Projection(
                            self.populations[source_pop], self.populations[target_pop], connector, target=target)
                else:
                    self.projections[target_pop][source_pop] = self.pynn.Projection(
                        self.populations[source_pop], self.populations[target_pop], connector, pynn.StaticSynapse(), receptor_type=target)

        logger.debug(f"total connections: {self.total_connections}\n")

    def _compensate_external_input(self):
        """
        Compensates for the removed external input by increasing the leak potential.
        """
        for sourceKey, v_rest in par.v_rest_new.items():
            self.populations[sourceKey].tset('v_rest', np.repeat(v_rest, self.populations[sourceKey].size))

    def _set_recording(self):
        """
        Configures the simulator to record spike data.
        """
        for population in self.populations.values():
            if self.simulator == 'nest':
                population.record(["spikes"])
            else:
                population.record()

    def build(self):
        """
        Builds the network structure of the cortical microcircuit using the PyNN API.
        """
        self._generate_populations()

        # Create projections
        self._generate_recurrent_connections()

        # modify resting potential to correct for scaling and replaced external input
        self._compensate_external_input()

        if self.simulator == "nest":
            # add variation of the neuron parameters to the network
            if self.add_variation:
                self._apply_variations()
            # initialze membrane potentials
            # not possible on hardware
            self._init_membranes()

        # record information
        self._set_recording()

        logger.debug("Network successfully build")

    def save_results(self):
        """
        Transforms the result data of the current execution to the Neo format and
        appends it to the results of the experiment object.
        """
        logger.debug("Store results")
        # save spikes
        for key, population in self.populations.items():
            if self.simulator == "nest":
                data = population.get_data().segments[0]
            else:
                data = transform_spikes_to_neo(key, population.size, population.getSpikes(), par.simulation_time)
            # Add annotations
            data.annotate(population_size=population.size, sim_time=par.simulation_time)
            self.results[key].segments.append(data)

        logger.debug("Finished storing results")

    def run(self):
        """
        Executes the network on the selected platform.
        """
        self.pynn.run(par.simulation_time)
        logger.debug("Network ran successfully")

    def reset_network(self):
        """
        Resets the experiment state in between executions.
        """
        self.pynn.reset()
        logger.debug("Network reset")

    def _init_results(self):
        """
        Initializes the result data structure as an empty Neo Block for each population.
        """
        for pop in par.label:
            self.results[pop] = Block(name=pop)

    def _apply_delay_calib(self, distance):
        """
        Translates a physical distance on the wafer into a corresponding delay value.

        :param distance: The distance between the neurons on the wafer.
        :return: Corresponding delay value.
        """
        delay = par.delay_calib[0] * distance + par.delay_calib[1]
        # Apply random deviations with std of 0.1 ms
        delay = self.delay_rng.normal_clipped(mu=delay, sigma=0.1, low=0.1)
        return delay

    def _calculate_weights(self):
        """
        Generates weight matrix.
        Values are optimized for the cortical microcircuit on BrainScaleS-1.
        If time constants are adapted, the weights have to be adjusted as well.
        Two additional factors can be used to independently scale the excitatory and inhibitory weights.

        :return: A connectivity matrix with weight values.
        """
        w = np.zeros([len(par.label), len(par.label)])
        for source_index, source_pop in enumerate(par.label):
            factor = par.exc if source_pop.endswith("e") else par.inh
            for target_index, target_pop in enumerate(par.label):
                w[target_index][source_index] = par.weights[source_pop[-1]][target_pop[:-1]][target_pop[-1]] * factor
        logger.debug(f"Using weights:\n{w}")
        return w

    def _get_indegrees(self):
        """
        Calculates the number of incoming synapses per neuron.

        :return: A connectivity matrix with number of in-degrees.
        """
        K = np.zeros([len(par.label), len(par.label)])
        for target_index, target_pop in enumerate(par.label):
            for source_index, source_pop in enumerate(par.label):
                n_target = par.num_neurons[target_pop]
                n_source = par.num_neurons[source_pop]
                K[target_index][source_index] = np.log(1.
                                                       - par.conn_probs[target_index][source_index]) / np.log(
                    1. - 1. / (n_target * n_source)) / n_target
        return K

    def _redraw(self, array, distribution, boundaries):
        """
        Helper function to randomly draw values following a specific distribution,
        with the ability to redraw values if they fall outside the specified boundaries.

        :param array: The array containing the initial values to be updated.
        :param distribution: A RandomDistribution object used to draw values according to a specific distribution (e.g., normal distribution).
        :param boundaries: A tuple containing the lower and upper bounds for the values.
        :return: A tuple containing:
            - An array with the resulting values after applying the distribution and boundary checks.
            - The mean value of the newly generated values.
        """
        MaxRedraw = 200
        for i in range(len(array)):
            iteration = 0
            while array[i] <= boundaries[0] or array[i] >= boundaries[1]:
                if iteration > MaxRedraw:
                    raise Exception('More than 200 redraws, check distribution parameters')
                array[i] = distribution.next(1)
                iteration += 1
        new_mean = np.mean(array)
        return array, new_mean


