class RandomNetwork():
    """
    Class to model and execute the balanced random network experiment.

    This class handles the creation of the network structure, execution, and result management.
    """

    def __init__(self, pynn):
        """
        Initialize the experiment and set up bookkeeping variables.

        This constructor creates the experiment object and initializes any internal state required
        for the simulation/emulation.

        :param pynn: A Pynn instance used for running the experiment.
        """
        self.name = "random_network"
        self.pynn = pynn
        self.simulator = simulator
        self.result_folder = None
        self._init_results()

    def build(self, mapping=None, g=par.mapping_g):
        """
        Builds the network structure of the balanced random network using the PyNN API.

        :param mapping: The name of the mapping to be used during hardware operation.
        :param g: The relative inhibitory weight for which the experiment description should be built.
        """
        self._generate_weight_dist(g)
        self._generate_populations()
        self._generate_connections(mapping)
        self._set_recording()

    def _init_results(self):
        """
        Initializes the result data structure as an empty Neo Block for each population.
        """
        self.results = {}
        for pop in ["exc", "inh"]:
            self.results[pop] = Block(name=pop)

    def _generate_weight_dist(self, g):
        """
        Generates random distributions for the weights of the three different types of connections.

        :param g: The relative inhibitory weight for which the experiment description should be built.
        """
        self.J = {}
        self.J["inh"] = RD('normal_clipped', low=0, high=np.inf, mu=par.inh_conductance * g, sigma=0.5 * par.inh_conductance * g * par.variation_switch, rng=par.weight_rng)
        self.J["exc"] = RD('normal_clipped', low=0, high=np.inf, mu=par.exc_conductance, sigma=0.5 * par.exc_conductance * par.variation_switch, rng=par.weight_rng)
        self.J["ext"] = RD('normal_clipped', low=0, high=np.inf, mu=par.exc_conductance, sigma=0.5 * par.exc_conductance * par.variation_switch, rng=par.weight_rng)

    def _generate_populations(self):
        """
        Generates the populations of the network.
        """
        self.pops = {}
        if self.simulator == "nest":
            self.pops["exc"] = self.pynn.Population(par.Nex, self.pynn.IF_cond_exp, par.neuron_parameters, label="Excitatory Population")
            self.pops["inh"] = self.pynn.Population(par.Nin, self.pynn.IF_cond_exp, par.neuron_parameters, label="inhibitory Population")
        else:
            self.pops["exc"] = self.pynn.Population(par.Nex, self.pynn.IF_cond_exp, self.get_mean_params(par.neuron_parameters), label="Excitatory Population")
            self.pops["inh"] = self.pynn.Population(par.Nin, self.pynn.IF_cond_exp, self.get_mean_params(par.neuron_parameters), label="inhibitory Population")
        self.network = self.pops["exc"] + self.pops["inh"]
        logger.info("Internal populations initialized")
        # Initialize external Population
        # Use dummy since spikes will be calculated by "reconfigure_ext_spikes" method
        self.pops["ext"] = self.pynn.Population(par.Next, pynn.SpikeSourcePoisson, {'rate': 1}, label="External Population")
        logger.info("External population initialized")

    def _set_recording(self):
        """
        Configures the simulator to record spike data.
        """
        if self.simulator == 'nest':
            self.pops["exc"].record(["spikes"])
            self.pops["inh"].record(["spikes"])
        else:
            self.pops["exc"].record()
            self.pops["inh"].record()

    def _generate_connectors(self, mapping):
        """
        Generates the PyNN connectors for the connections between the two internal populations and for the external input.

        :param mapping: The name of the mapping to be used during hardware operation.
        """
        network_rng = NumpyRNG(par.network_seed, parallel_safe=True)

        # Initialize connectors
        if par.connector == "matrix":
            result_matrix = self.get_matrix(network_rng)
        elif par.connector == "load":
            if self.simulator == 'brainscales':
                raise RuntimeError("Network has to be regenerated on brainscales.")
            with open(f"{par.routing_results_path}/implemented_routes_{mapping}.pickle", "rb") as f:
                result_matrix = pickle.load(f)
            logger.info("Routing matrix loaded. Apply delay calib and weight distribution.")
            for source in ["exc", "inh"]:
                for target in ["exc", "inh"]:
                    weights_dummy = self.J[source].next(len(result_matrix[(source, target)]))
                    result_matrix[(source, target)] = [[pre, post, weights_dummy[n], self.apply_delay_calib(d, par.delay_rng)] for n, (pre, post, w, d) in enumerate(result_matrix[(source, target)])]
            source = "ext"
            target = "all"
            weights_dummy = self.J[source].next(len(result_matrix[(source, target)]))
            result_matrix[(source, target)] = [[pre, post, weights_dummy[n], self.apply_delay_calib(d, par.delay_rng)] for n, (pre, post, w, d) in enumerate(result_matrix[(source, target)])]
        else:
            raise RuntimeError(f"Unknown connector used: {par.connector}")

        self.connectors = {}
        for source in ["exc", "inh"]:
            for target in ["exc", "inh"]:
                self.connectors[(source, target)] = generate_connector_from_matrix(result_matrix[(source, target)], self.simulator)

        self.connectors[("ext", "all")] = generate_connector_from_matrix(result_matrix[("ext", "all")], self.simulator)

    def _generate_connections(self, mapping):
        """
        Generates the internal and external connections of the network.

        :param mapping: The name of the mapping to be used during hardware operation.
        """
        self._generate_connectors(mapping)
        # Initialize network projections
        self.projections = {}
        for source in ["exc", "inh"]:
            for target in ["exc", "inh"]:
                receptor = "excitatory" if source == "exc" else "inhibitory"
                connector = self.connectors[(source, target)]
                self.projections[(source, target)] = []
                if self.simulator == 'nest':
                    # Use dummy Synapse as weight and delay will be set in connector matrix
                    self.projections[(source, target)].append(pynn.Projection(self.pops[source], self.pops[target], connector, pynn.StaticSynapse(), receptor_type=receptor))
                else:
                    for c in connector:
                        self.projections[(source, target)].append(pynn.Projection(self.pops[source], self.pops[target], c, target=receptor))
        logger.info("Network projections initialized")

        # Initialize external projections
        connector = self.connectors[("ext", "all")]
        self.projections[("ext", "all")] = []
        if self.simulator == 'nest':
            self.projections[("ext", "all")].append(pynn.Projection(self.pops["ext"], self.network, connector, pynn.StaticSynapse(), receptor_type="excitatory"))
        else:
            for c in connector:
                self.projections[("ext", "all")].append(pynn.Projection(self.pops["ext"], self.network, c, target="excitatory"))
        logger.info("External projections initialized")

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

    def save_results(self, g, eta, mapping="all"):
        """
        Transforms the result data of the current execution to the Neo format and
        appends it to the results of the experiment object.

        :param g: The relative inhibitory weight used in the experiment.
        :param eta: The strength of the external stimulation used in the experiment.
        :param mapping: The name of the mapping applied during hardware operation.
        """
        logger.debug("Store results")

        # save spikes
        for key, pop in self.pops.items():
            if key == "ext":
                # We do not want to record external spikes
                continue
            # Limit evaluation to relevant neurons to reduce amount of stored data
            size = pop.size if mapping == "all" else 30
            if self.simulator == "nest":
                data = pop[:size].get_data(clear=True).segments[-1]
            else:
                data = transform_spikes_to_neo(key, size, pop.getSpikes(), par.simulation_time)
            # Add annotations
            data.annotate(population_size=pop.size, sim_time=par.simulation_time, g=g, eta=eta)
            self.results[key].segments.append(data)
        logger.debug("Finished storing results")

    def get_mean_params(self, params):
        """
        Helper to extract the mean value from randomly distributed neuron parameters.

        :param params: A dictionary containing randomly distributed neuron parameters.
        :return: A dictionary with the mean values of the neuron parameters.
        """
        return {k: v.parameters["mu"] for k, v in params.items()}

    def generate_matrix(self, target_size, n_connection, source_size, conn_num, network_rng, source):
        """
        Generates a connectivity matrix for connections between two populations.

        :param target_size: The number of neurons in the target population.
        :param n_connection: The total number of connections to be implemented.
        :param source_size: The number of neurons in the source population.
        :param conn_num: The number of connections per target neuron.
        :param network_rng: The random number generator used for sampling connections.
        :param source: The name of the source population.
        :return: A connectivity matrix defining the connections between the populations.
        """
        matrix = np.zeros((4, n_connection), dtype=float)
        for target_id in range(target_size):
            # connection matrix [(neuron_pop1,neuron_pop2,weight,delay),(...)]
            # With autapses
            start = target_id * conn_num
            stop = (target_id + 1) * conn_num
            matrix[0][start:stop] = network_rng.randint(0, source_size, conn_num)
            matrix[1][start:stop] = np.repeat(target_id, conn_num)
            matrix[2][start:stop] = self.J[source].next(conn_num)
            matrix[3][start:stop] = par.delay_dist.next(conn_num)

        matrix = matrix.T
        # pyhmf expects int values for neuron numbers.
        # Using dtype = int, the weights get rounded to zero and no connection is established
        matrix = [[int(a), int(b), c, d] for a, b, c, d in matrix]
        return matrix

    def get_matrix(self, rng):
        """
        Generates a connectivity matrix for each connection between populations in the network.

        :param rng: The random number generator used for sampling connections.
        :return: A dictionary of connectivity matrices, where each matrix defines the connections between two populations.
        """
        result_matrix = {}
        for source, conn_num in zip(["exc", "inh"], [par.Cex, par.Cin]):
            source_size = self.pops[source].size
            for target in ["exc", "inh"]:
                target_size = self.pops[target].size
                n_connection = int(target_size * conn_num)
                matrix = self.generate_matrix(target_size, n_connection, source_size, conn_num, rng, source)
                result_matrix[(source, target)] = matrix
        # Add ext
        target_size = self.network.size
        source_size = self.pops["ext"].size
        source = "ext"
        target = "all"
        if par.ext_pool:
            conn_num = par.Cext
            n_connection = int(target_size * conn_num)
            matrix = self.generate_matrix(target_size, n_connection, source_size, conn_num, rng, source)
        else:
            # Implement OneToOne connection
            assert (source_size == target_size)
            conn_num = source_size
            matrix = np.zeros((4, conn_num), dtype=float)
            matrix[0] = np.arange(conn_num)
            matrix[1] = np.arange(conn_num)
            matrix[2] = self.J[source].next(conn_num)
            matrix[3] = par.delay_dist.next(conn_num)
            matrix = matrix.T
            matrix = [[int(a), int(b), c, d] for a, b, c, d in matrix]
        result_matrix[(source, target)] = matrix
        return result_matrix

    def apply_delay_calib(self, distance, delay_rng):
        """
        Translates a physical distance on the wafer into a corresponding delay value.

        :param distance: The distance between the neurons on the wafer.
        :param delay_rng: The random number generator used for sampling delays.
        :return: Corresponding delay value.
        """
        delay_calib = (0.0397, 0.6444)
        delay = delay_calib[0] * distance + delay_calib[1]
        # Apply random deviations with std of 0.1 ms
        delay = delay_rng.normal(delay, 0.1)
        return delay

    def modify_weights(self, last_g, g):
        """
        Adapts the weights of all populations according to a new relative inhibitory weight.

        :param last_g: The relative inhibitory weight used in the last experiment execution.
        :param g: The relative inhibitory weight to be used in the next experiment execution.
        """
        factor = g / last_g
        for key, proj in self.projections.items():
            if key[0] == "inh":
                # Excitatory connections are kept fixed
                for elem in proj:
                    if self.simulator == 'nest':
                        weights = elem.get('weight', format='array') * factor
                    else:
                        weights = elem.getWeights() * factor
                    # This call is very slow in the pynn.nest version used in simulation
                    # -> this function should only be during the hardware run
                    elem.setWeights(weights)

    def poissonspiketimes(self, rate, duration):
        """
        Generates Poisson distributed spike times.

        :param rate: The firing rate (in Hz) of the spiketrain.
        :param duration: The duration (in ms) for which the spiketrain is generated.
        :return: An array of spike times for the generated spiketrain.
        """
        # Seeds are not implemented by elephant
        return np.array(spike_train_generation.homogeneous_poisson_process(rate=rate * Hz, t_start=0.0 * ms, t_stop=duration * ms))

    def reconfigure_ext_spikes_hw(self, runtime, eta):
        """
        Modifies external spike times used on the hardware based on a new external spike rate.

        :param runtime: The experiment duration in milliseconds.
        :param eta: The new external spike rate.
        """
        nu_ext = par.nu_thresh * eta
        if par.ext_pool:
            nu_ext /= par.Cext
        logger.info(f"nu_ext: {nu_ext}")
        for n in self.pops["ext"]:
            spikes = self.poissonspiketimes(nu_ext, par.simulation_time)
            runtime.results().spike_times.set(n, spikes)

    def reconfigure_ext_spikes_sim(self, eta):
        """
        Modifies external spike times used in the simulation based on a new external spike rate.

        :param eta: The new external spike rate.
        """
        nu_ext = par.nu_thresh * eta
        if par.ext_pool:
            nu_ext /= par.Cext
        logger.info(f"nu_ext: {nu_ext}")
        for n in self.pops["ext"]:
            n.rate = nu_ext


