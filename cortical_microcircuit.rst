Downscaled cortical mircrocircuit on BrainScales-1
==================================================

In this notebook, a 10% version of the cortical microcircuit as described in Potjans and Diesmann (`2014 <https://doi.org/10.1093/cercor/bhs358>`_) is emulated on the BrainScaleS-1 system in Heidelberg.
Simultaneously, the same network is simulated on the Jülich HPC cluster using the software simulator NEST [`Gewaltig and Diesmann (2007) <https://doi.org/10.4249/scholarpedia.1430>`_].

.. raw:: html

    <div class="floating-box">
    <center>
    <img src="_static/cortical_microcircuit/column_structure.png" title="Network structure of the cortical microcircuit model." style="width:15%; margin-right: 10em">
    <img src="_static/wmod.png" title="BrainScaleS-1 wafer module." style="width:15%">
    </center>
    </div>

The cortical microcircuit model consists of four excitatory and four inhibitory populations of LIF neurons. The connectivity between these populations follows a predefined map, as described in the original publication. Each population receives a fixed number of excitatory external inputs, modeled as Poisson-distributed stimuli with a spike rate of 8 Hz. In total, the network comprises approximately 80,000 neurons and 300 million synapses. Its large scale and biologically plausible recurrent connectivity make it a widely used benchmark in computational neuroscience. Implementing this model on the BrainScaleS-1 hardware demonstrates the hardware's capabilities and how it can complement conventional software simulations.

BrainScaleS-1 is an analog wafer-scale neuromorphic system featuring over 200,000 analog circuits that emulate AdEx neuron model behavior and more than 43 million synapses for connectivity. Its analog nature enables emulation speeds 10,000 times faster than biological real-time. However, the physical modeling approach using analog circuits imposes limitations on model parametrization and size. To address these limitations, the cortical microcircuit model must be adapted before emulation to align with the constraints of the BrainScaleS-1 hardware. This adaptation process can be incorporated into a co-execution approach, where network behavior is analyzed using both conventional HPC simulations and hardware emulation. Once a suitable model is developed, the system’s significant speedup offers a distinct advantage for iterative experiments or long-duration emulations.

In this demo, a pre-adapted model is used, enabling execution on both the hardware and the software simulator. The adaptations involve uniformly reducing the neuron count and in-degree while preserving the original connectivity probability. To compensate for the reduced input resulting from this downscaling, synaptic weights are linearly increased. However, even with this adjustment, hardware constraints necessitate omitting some synaptic connections, leading to partial synapse loss, which is also integrated into the model. As a result, the adapted model includes 7,712 neurons and 2,373,933 internal synapses. Moreover, the external input is replaced with an external current implemented by an increased leak potential. Current-based synapses are converted to conductance-based synapses, with recalculated weights to preserve the post-synaptic potential amplitude measured from the expected mean membrane potential. Finally, synaptic time constants are increased, and normally distributed variations are introduced across all neuron parameters based on hardware calibration data. For more details on the required adaptations, refer to Schmidt (`2024 <https://doi.org/10.11588/heidok.00034446>`_).

.. code-block:: ipython3

    # Initialize the collab and some bookkeeping variables
    import nmpi
    import time
    import ebrains_drive
    from _static.helper.get_repo_info import find_repository_info_from_drive_directory_path

    nmpi_client = nmpi.Client()
    current_dir = !pwd
    repo_info = find_repository_info_from_drive_directory_path(current_dir[0], clb_oauth.get_token())

    state_nest = "not_started"
    state_bss1 = "not_started"
    bss1_job_id = None
    nest_job = None

.. code-block:: ipython3

     # Request access to the Jülich HPC cluster. This requires a user account at JURECA with matching E-Mail address.
     # If only the BrainScaleS-1 system should be used, this step can also be skipped.
     import pyunicore.client as unicore_client
     from pyunicore.credentials import create_credential
     import pyunicore.helpers as helpers

     credential = create_credential(token=clb_oauth.get_token())
     registry_url = 'https://unicore.fz-juelich.de/HBP/rest/registries/default_registry'
     registry = unicore_client.Registry(credential,registry_url)
     jureca_client = registry.site('JURECA')


In the following cell, the network parameters can be configured. The experiment code itself is stored in the `_static/cortical_microcircuit/experiment_code` folder and will be integrated later for experiment execution. This tutorial uses a pre-mapped network description tailored to the topology of a BrainScaleS-1 wafer, which is loaded during hardware execution. Consequently, network modifications are limited to neuron parameters and synaptic weight values.

In general, the BrainScaleS-1 operating system can automatically generate new network topologies [`Müller (2022) <https://doi.org/10.1016/j.neucom.2022.05.081>`_]. However, the size of the 10% cortical microcircuit model presents significant challenges for deployment on a single wafer. Due to limited on-chip routing possibilities between neurons and synapses and the use of non-optimal mapping algorithms it is not feasible to physically represent every synapse from the original model, requiring the omission of some synapses in the final representation. The provided pre-mapped network description represents an already optimized solution.

For the NEST simulation, a corresponding network description can be loaded, preserving the same network topology and synaptic delays as those expected on the BrainScaleS-1 system.

A visualization of a BrainScaleS-1 wafer and the network topology mapped to the wafer is provided below. The wafer is divided into 384 rectangular chips, each containing 512 neuron circuits. In the mapping overview, the number of neurons used per chip is indicated by shades of blue, with darker shades representing more neurons. Colored lines represent synaptic connections between the chips, while the regions corresponding to individual populations of the model are outlined with colored borders.

.. raw:: html

    <div class="floating-box">
    <center>
    <img src="_static/wafer.jpg" title="BrainScaleS-1 wafer." style="width:30%; margin-right: 10em">
    <img src="_static/cortical_microcircuit/topology_cortical_microcircuit.png" title="Mapping result of the cortical microcircuit model." style="width:30%">
    </center>
    </div>

.. code-block:: ipython3

    %%writefile ./_static/cortical_microcircuit/experiment_code/parameters.py

    ### Parameter definition of the cortical microcircuit ###
    class par:
        ### Simulator specific parameters ###
        # Simulation paramters
        simulation_time = 10000.0 # ms
        timestep = 0.1  # ms, Simulation timestep
        # Delays are disregarded on the BrainScaleS-1 system as they depend solely on the physical distance on the chip.
        # Similarly, if routing results are loaded, delays are also ignored during the NEST simulation.
        excitatory_delay = 1.5  # ms
        excitatory_delay_std = 0.5 * excitatory_delay  # ms
        inhibitory_delay = 0.8  # ms
        inhibitory_delay_std = 0.5 * inhibitory_delay  # ms
        # Scaling parameters of the model
        # The number of neurons as well as the number of synapses is scaled to 10% of the original size.
        # Since in this tutorial the compensation for different model sizes is not included,
        # changing the model size will lead to different network behavior and will conflict with the
        # provided mapping results for the BrainScaleS-1 system.
        scale = 0.1
        k_scale = 0.1
        # parallelizing
        parallel_safe = True
        threads = 48

        if simulator == "nest":
            synaptic_weight_std = 0.1 # Fraction of finally used weight
        elif simulator == "brainscales":
            synaptic_weight_std = 0.  # Deviation on hardware is already given by parameter variations


        ### Network parameters ###
        # Given in biological values (are automatically translated into corresponding hardware parameters)
        neuron_parameters = {'cm': 0.25,  # nF
                             'e_rev_E': 50.0, # mV
                             'e_rev_I': -150, # mV
                             'tau_m': 10.0, # ms
                             'i_offset': 0.0, #nA
                             'tau_refrac': 2.0, #ms
                             'tau_syn_E': 2.2, #ms
                             'tau_syn_I': 2.2, #ms
                             'v_reset': -65.0, #mV
                             'v_rest': -65.0, #mV
                             'v_thresh': -50.0 #mV
                             }

        # Choose which connector is used
        # "fixedtotalnumber": This option rebuilds the PyNN network and is always used for executions on the BrainScaleS-1 system.
        # This ensures that the entire network structure aligns with the supplied mapping results.
        # "load_routing_results": This option loads the topology of a pre-routed network on the BrainScaleS-1 hardware.
        # It achieves a comparable configuration during simulation and emulation, maintaining similar delay values
        # and omitting the same synapse connections as in the hardware implementation.
        internal_connector = "load_routing_results"

        # Path to routing results
        # required if "load_routing_results" is used
        routing_results_path = "./_static/cortical_microcircuit/implemented_routes.pickle"


        # Additional weight factors
        # Allows to rescale all excitatory and all inhibitory weights
        # The neuron configuration on the BrainScaleS system is constrained by the physical capabilities of its circuits.
        # Due to adaptations to the original network model, descriptions with very small excitatory weights
        # yield results similar to the original network behavior.
        # These weights represent the lower limit of what the system can implement.
        # However, noise observed in membrane recordings makes precise weight calibration challenging in this regime.
        # It is estimated that an excitatory weight factor of 1 on the BrainScaleS system corresponds approximately
        # to a "real" weight factor of 0.1.
        if simulator == "nest":
            exc = 0.1
        elif simulator == "brainscales":
            exc = 1.
        inh = 1.

        # Seeds for different RNGs
        kernel_seed = 123456
        weight_seed = 234567
        init_seed = 345678
        delay_seed = 24639
        var_seed = 6627

        # Names of populations
        label = ["23e", "23i", "4e", "4i", "5e", "5i", "6e", "6i"]

        # Total amount of neurons
        num_neurons = {
            '23e': 20683,
            '23i': 5834,
            '4e': 21915,
            '4i': 5479,
            '5e': 4850,
            '5i': 1065,
            '6e': 14395,
            '6i': 2948
        }

        # Probabilities for >=1 connection between neurons in the given populations.
        # The first index is for the target population; the second for the source population
        #             2/3e      2/3i    4e      4i      5e      5i      6e      6i
        conn_probs = [[0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0., 0.0076, 0.],
                      [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0., 0.0042, 0.],
                      [0.0077, 0.0059, 0.0497, 0.135, 0.0067, 0.0003, 0.0453, 0.],
                      [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0., 0.1057, 0.],
                      [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.],
                      [0.0548, 0.0269, 0.0257, 0.0022, 0.06, 0.3158, 0.0086, 0.],
                      [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
                      [0.0364, 0.001, 0.0034, 0.0005, 0.0277, 0.008, 0.0658, 0.1443]]

        # Resting potentials of each population
        # Values are already adjusted to compensate for the missing external inputs
        v_rest_new = {
            '23e': -42.52,
            '23i': -43.93,
            '4e': -35.5,
            '4i': -38.31,
            '5e': -36.9,
            '5i': -38.31,
            '6e': -24.26,
            '6i': -35.5
        }

        # Adjusted weights for the downscaled model with conductance based synapses, optimized for the settings on BSS-1
        weights = {
                'e' : {'23': {'e': 0.00303139,
                              'i': 0.00321473},
                        '4': {'e': 0.00334374,
                              'i': 0.00325535},
                        '5': {'e': 0.00330689,
                              'i': 0.00336596},
                        '6': {'e': 0.00341609,
                              'i': 0.00342977}},
                'i' : {'23': {'e': 0.02265111,
                              'i': 0.02112101},
                        '4': {'e': 0.02024846,
                              'i': 0.02083072},
                        '5': {'e': 0.0204834 ,
                              'i': 0.02011174},
                        '6': {'e': 0.01981637,
                              'i': 0.01973876}},
                }

        ### hardware specific parameters ###

        # Path to mapping results (folder must contain `marocco_results.xml.gz` and marocco_wafer.xml.gz)
        mapping_path = "/wang/data/commissioning/BSS-1/rackplace/30/mappings/wafer_30_column_smallcap_slow"

        calib_path = "/wang/data/commissioning/BSS-1/rackplace/30/calibration/2022-09-06-1_small_cap_slow"
        defects_path = "/wang/data/commissioning/BSS-1/rackplace/30/derived_plus_calib_blacklisting/2022-09-06-1_small_cap_slow_plus_driver_plus_switchram_and_sending"
        neuron_size = 8
        max_hw_voltage = 1.3
        min_hw_voltage = 0.4
        max_bio_voltage = neuron_parameters["e_rev_E"]
        min_bio_voltage = neuron_parameters["e_rev_I"]
        slow = True
        bigcap = False

        # For the BrainScaleS system, the network is recorded a second time
        # after waiting for "wait_time" in wall clock time.
        # With the system's speedup factor of 10000, this enables the study
        # of network behavior over extended biological timescales.
        record_second_sample = True
        wait_time = 8.64 # s, which corresponds to 1 day of biological time

        ### simulation specific parameters ###

        # Dictionary containing the parameters to be varied between individual neurons,
        # along with the relative standard deviation (expressed as a percentage of the mean value)
        # used for the Gaussian distribution (where 0 indicates no variation).
        # The values are derived from calibration results.
        variation = {# cm already affected by variations of tau_m
                     'e_rev_E': 11.1,
                     'e_rev_I': 1.6,
                     'tau_m': 8.0,
                     'tau_refrac': 1.5,
                     'tau_syn_E': 0.6,
                     'tau_syn_I': 0.4,
                     'v_rest': 2.,
                     'v_thresh': 1.1,
                     'v_reset': 1.6}

        # Dictionary containing the boundary values for the varied parameters.
        # The values for reset and threshold will be changed in the experiment to avoid reset > threshold.
        variation_boundaries = {'cm': (0, np.inf),
                                'e_rev_E': (-np.inf, np.inf),
                                'e_rev_I': (-np.inf, np.inf),
                                'tau_m': (3, np.inf),
                                'tau_refrac': (0, np.inf),
                                'tau_syn_E': (1.8, 4),
                                'tau_syn_I': (1.9, 6),
                                'v_rest': (-np.inf, np.inf),
                                #'v_thresh': (neuron_parameters['v_reset'], np.inf),
                                'v_thresh': (-np.inf, np.inf),
                                'v_reset': (-np.inf, 0.9 * neuron_parameters['v_thresh'])}

        # delay calibration parameters. Extracted from delay measurments on wafer 30.
        # Used during NEST simulations if routing results are loaded using internal_connector="load_routing_results"
        delay_calib = (0.0397, 0.6444)

In the next two cells, the experiment description is first sent to the BrainScaleS-1 system in Heidelberg for emulation and then to the HPC cluster in Jülich for the NEST simulation.

.. code-block:: ipython3

    # Merge results to form one file that can be executed at the sites
    # This is a workaround since only a single file can be sent via the nmpi runner
    !cd ./_static/cortical_microcircuit/experiment_code && cat imports.py parameters.py ../../network_helper/execution_helpers.py ../../network_helper/experiment_helpers.py experiment_description.py execute.py > run.py


    # Send job to Heidelberg for emulation
    wafer = 30
    collab_id = repo_info.name_in_the_url
    hw_config = {'WAFER_MODULE': wafer, "SOFTWARE_VERSION": "nmpm_software/column_changes", "CORES": "48", "PARTITION": "batch"}

    bss1_job_id = nmpi_client.submit_job(source="./_static/cortical_microcircuit/experiment_code/run.py",
                          platform=nmpi.BRAINSCALES,
                          collab_id=collab_id,
                          config=hw_config,
                          command="run.py brainscales",
                          wait=False)
    state_bss1 = "running"
    print("Job id is " + str(bss1_job_id))

In the following cell, the network will be simulated on the Jülich HPC cluster using the NEST software simulator.
To proceed with this step, the appropriate software environment needs to be loaded onto the cluster.
Please download the latest container image from https://openproject.bioai.eu/containers/ and store it on the Jülich HPC cluster.
Afterward, specify the path to the image using the `container_path` variable.
If you do not wish to run the simulation, you can simply skip this cell.

.. code-block:: ipython3

    # Merge results to form one file that can be executed at the sites
    !cd ./_static/cortical_microcircuit/experiment_code && cat imports.py parameters.py ../../network_helper/execution_helpers.py ../../network_helper/experiment_helpers.py experiment_description.py execute.py > run.py

    container_path = None
    if container_path is None:
        raise RuntimeError("No container image has been provided. Please download the latest image to the Jülich HPC cluster and specify its location using the container_path variable.")

    resources = helpers.jobs.Resources(queue="dc-cpu", cpus_per_node=8, nodes=1)
    job = helpers.jobs.Description(
        executable=f"singularity exec --app wafer {container_path} python ./_static/cortical_microcircuit/experiment_code/run.py",
        arguments=["nest"],
        resources=resources
    )

    nest_job = jureca_client.new_job(job.to_dict(), inputs = ["./_static/cortical_microcircuit/experiment_code/run.py", "./_static/cortical_microcircuit/implemented_routes.pickle"])
    state_nest = "running"
    print("NEST job submitted")

The next cell polls the two sites and waits for the experiment results.

.. code-block:: ipython3

    from _static.helper.get_data import handle_results
    # wait for both jobs to finish and copy the results to this collab
    state_bss1, state_nest, result_dir_bss1, result_dir_nest = handle_results(state_bss1, state_nest, nmpi_client, bss1_job_id, nest_job, "cortical_microcircuit")


Having received the results, the rate distribution as well as the irregularity distribution of the neurons of each population are evaluated and visualized in the next two cells.
Moreover, the spike times of a subset of neurons is depicted for all populations.
Although slight differences are observed between simulation and emulation due to additional hardware effects not accounted for in the simulated model, the overall activity remains similar in both cases.

.. code-block:: ipython3

    from _static.cortical_microcircuit.helper.plot_results import plot

    # Only consider spikes after this time to allow the network behavior to settle
    measure_spikes_from_time = 1000.0
    # For spiketime plot
    # percentage of neurons per population that are displayed
    frac_neurons = 0.1 
    # time window of plot
    t_min = 3000
    t_max = 3500

    !mkdir -p ./plots/cortical_microcircuit

    plot(result_dir_bss1, result_dir_nest, measure_spikes_from_time, frac_neurons, t_min, t_max)

.. code-block:: ipython3

    from IPython.display import HTML

    HTML(open("./_static/helper/plot_format.css", "r").read() +
    """
        <div class="row">
          <div class="column">
            <figure>
                <img src="plots/cortical_microcircuit/Rate_distribution_0.png"; title="Rates">
                <figcaption>Fig.1 Rates</figcaption>
            </figure>
           </div>
          <div class="column">
            <figure>
                <img src="plots/cortical_microcircuit/Irregularity_distribution_0.png"; title="Irregularity">
                <figcaption>Fig.2 Irregularity</figcaption>
            </figure>
          </div>
        </div>

        <div class="row">
          <div class="column">
            <figure>
                <img src="plots/cortical_microcircuit/BSS-1_spiketimes_0.png"; title="BSS-1 spiketimes">
                <figcaption>Fig.3 BSS-1 spiketimes</figcaption>
            </figure>
           </div>
          <div class="column">
            <figure>
                <img src="plots/cortical_microcircuit/NEST_spiketimes_0.png"; title="NEST spiketimes">
                <figcaption>Fig.4 NEST spiketimes</figcaption>
            </figure>
          </div>
        </div>
    """)

If the setting "record_second_sample" is enabled, the network behavior is recorded a second time on the BrainScaleS-1 system after a fixed waiting period, specified by the parameter "wait_time".
Due to the speedup factor of 10,000 on BrainScaleS-1, waiting for 8.64 seconds corresponds to observing the behavior of the downscaled cortical microcircuit after over one day of biological time.
The results of this second measurement can be visualized in the next cell.
Although the network behavior is expected and observed to remain stable, the hardware enables investigations over extended periods, such as one year of biological time. This can be achieved in less than one hour of wall-clock time, an achievement currently beyond the practical capabilities of conventional computing resources.

.. code-block:: ipython3

    from IPython.display import HTML

    HTML(open("./_static/helper/plot_format.css", "r").read() +
    """
        <h1 style="text-align:center; font-size:1.5em">BSS-1 emulation results after 1 day of biological time</h1>
        <div class="row">
          <div class="column">
            <figure>
                <img src="plots/cortical_microcircuit/Rate_distribution_1.png"; title="Rates">
                <figcaption>Fig.1 Rates</figcaption>
            </figure>
           </div>
          <div class="column">
            <figure>
                <img src="plots/cortical_microcircuit/Irregularity_distribution_1.png"; title="Irregularity">
                <figcaption>Fig.2 Irregularity</figcaption>
            </figure>
          </div>
        </div>

        <div class="row">
          <figure>
            <center>
              <img src=plots/cortical_microcircuit/BSS-1_spiketimes_1.png style="width:50%">
              <figcaption>Fig.3 BSS-1 spiketimes</figcaption>
            </center>
          </figure>
        </div>
    """)
