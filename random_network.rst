Balanced Random Network on BrainScales-1
========================================

In this notebook, the balanced random network as described in Brunel (`2000 <https://doi.org/10.1023/A:1008925309027>`_) is emulated on the BrainScaleS-1 system in Heidelberg. Simultaneously, the same network is simulated on the Jülich HPC cluster using the software simulator Nest [`Gewaltig and Diesmann (2007) <https://doi.org/10.4249/scholarpedia.1430>`_].

.. raw:: html

    <div class="floating-box">
    <center>
    <img src="_static/random_network/brunel_structure.png" title="Network structure of the balanced random network model." style="width:15%; margin-right: 10em">
    <img src="_static/wmod.png" title="BrainScaleS-1 wafer module." style="width:20%">
    </center>
    </div>

The balanced random network consists of two populations of leaky integrate-and-fire (LIF) neurons: one excitatory and one inhibitory. Connections between these populations are randomly drawn with a fixed connection probability of 0.1, ensuring that each neuron receives an equal number of incoming synapses. Additionally, each neuron is driven by Poisson-distributed spike trains from external excitatory sources.
This model is used to study the firing characteristics of interconnected LIF neurons under varying external input spike rates and different inhibitory-to-excitatory weight ratios.
Since the model is based on theoretical assumptions, it is scalable and serves as an excellent starting point for investigating the properties of the BrainScaleS-1 hardware.

BrainScaleS-1 is an analog wafer-scale neuromorphic system featuring over 200,000 analog circuits that emulate AdEx neuron model behavior and more than 43 million synapses for connectivity.
Its analog nature enables emulation speeds 10,000 times faster than biological real-time. However, the physical modeling approach using analog circuits imposes limitations on model parametrization and size.
Consequently, the model description of the balanced random network must be adapted before emulation to align with the constraints of the BrainScaleS-1 hardware.
This adaptation process can be incorporated into a co-execution approach, where network behavior is analyzed using both conventional HPC simulations and hardware emulation.
Once a suitable model is developed, the system's significant speedup offers a distinct advantage for iterative experiments or long-duration emulations.

In this demo, a pre-adapted model is used, enabling execution on both the hardware and the software simulator.
The adaptations involve uniformly reducing the neuron count and in-degree while preserving the original connectivity probability. To compensate for the reduced input resulting from this downscaling, synaptic weights are linearly increased. However, even with this adjustment, hardware constraints necessitate omitting some synaptic connections, leading to partial synapse loss, which is also integrated into the model. As a result, the adapted model includes 2,083 neurons and 690,157 internal synapses.
Moreover, the external input is replaced by 2,083 Poisson sources, with each neuron sampling 200 connections. Current-based synapses are converted to conductance-based synapses, with recalculated weights to preserve the post-synaptic potential amplitude measured from the expected mean membrane potential.
Finally, synaptic time constants are increased, and normally distributed variations are introduced across all neuron parameters based on hardware calibration data.
For more details on the required adaptations, refer to Schmidt (`2024 <https://doi.org/10.11588/heidok.00034446>`_).

.. code-block:: ipython3

    # Initialize the collab and some bookkeeping variables
    import nmpi
    import time
    import ebrains_drive
    from _static.helper.get_repo_info import find_repository_info_from_drive_directory_path

    nmpi_client = nmpi.Client()
    current_dir  =!pwd
    repo_info = find_repository_info_from_drive_directory_path(current_dir[0], clb_oauth.get_token())

    state_nest = "not_started"
    state_bss1 = "not_started"
    bss1_job_id = None
    nest_job = None

.. code-block:: ipython3

     # Request access to the Jülich HPC cluster. This requires a user account at JURECA with matching E-Mail address. If only the BrainScaleS-1 system should be used, this step can also be skipped.
     import pyunicore.client as unicore_client
     from pyunicore.credentials import create_credential
     import pyunicore.helpers as helpers

     credential = create_credential(token=clb_oauth.get_token())
     registry_url = 'https://unicore.fz-juelich.de/HBP/rest/registries/default_registry'
     registry = unicore_client.Registry(credential,registry_url)
     jureca_client = registry.site('JURECA')

In the following cell, the network parameters can be configured. The experiment code itself is stored in the `_static/random_network/experiment_code` folder and will be integrated later for experiment execution. This tutorial uses a pre-mapped network description tailored to the topology of a BrainScaleS-1 wafer, which is loaded during hardware execution. Consequently, network modifications are limited to neuron parameters, synaptic weight values, and external firing rates.

In general, the BrainScaleS-1 operating system can automatically generate new network topologies [`Müller (2022) <https://doi.org/10.1016/j.neucom.2022.05.081>`_]. However, the size of the model presents significant challenges for deployment on a single wafer. Due to limited on-chip routing possibilities between neurons and synapses and the use of non-optimal mapping algorithms it is not feasible to physically represent every synapse from the original model, requiring the omission of some synapses in the final representation. The provided pre-mapped network description represents an already optimized solution.

For the NEST simulation, a corresponding network description can be loaded, preserving the same network topology and synaptic delays as those expected on the BrainScaleS-1 system.

At low inhibitory weights, the high firing rates result in the off-chip bandwidth being insufficient to capture the spikes from all neurons. To address this limitation in network configurations with elevated spike rates, an alternative mapping topology is implemented. In this approach, 30 neurons are isolated from the rest of the network and allocated to individual chips. This configuration ensures that the full bandwidth of these chips is available for spike recording, supporting firing rates of up to 2.5 kHz. As a result, the evaluation of network behavior in the high-firing regime is restricted to these 30 neurons in both emulation and simulation. However, the remaining neurons are still emulated on the hardware, as the internal communication within the system is sufficiently fast to support their activity.

Visualizations of the mapping results are provided below. The left panel illustrates the configuration with separated neurons, while the right panel shows the configuration without separation.
The wafer consists of 384 rectangular chips, each containing 512 neuron circuits. The number of neurons utilized per chip is indicated by shades of blue, with darker shades representing higher neuron densities. Colored lines illustrate the synaptic connections between chips.

.. raw:: html

    <div class="floating-box">
    <img src="_static/random_network/topology_random_network_30.png" title="recording 30 neurons" style="width:40%">
    <img src="_static/random_network/topology_random_network_all.png" title="recording all neurons" style="width:40%">

.. code-block:: ipython3

    %%writefile ./_static/random_network/experiment_code/parameters.py

    ### Parameter definition of the balanced random network ###
    class par():
        '''
        This class handles the general information about the Network.
        '''
        # Sweep parameters
        # External firing rates defined in relation to v_thres as described in the original publication.
        # The provided mapping is optimized for a maximum external firing frequency of 3.2.
        # Using higher frequencies would necessitate a remapping of the network, which is beyond the scope of this tutorial.
        etas = [0.8, 1.6, 2.4, 3.2]
        # Relative inhibitory weights
        # The hardware supports values only within the range of 1 to 10.
        gs = [1, 4, 8, 10]

        # Neuron parameters
        # Parameter variations are added in the nest simulation to reproduce the hardware behavior
        variation = True if simulator == "nest" else False # bool to turn on and off variation
        v_thresh = 20.0
        variation_switch = 1 if variation else 0
        variation_rng = NumpyRNG(200, parallel_safe=True)
        neuron_parameters = {'tau_m': RD('normal_clipped', low=0.0, high=np.inf, mu=10.0, sigma=2. * variation_switch, rng=variation_rng),
                             'tau_syn_E': RD('normal_clipped', low=0.0, high=np.inf, mu=3.0, sigma=0.4 * variation_switch, rng=variation_rng),
                             'tau_syn_I': RD('normal_clipped', low=0.0, high=np.inf, mu=3.0, sigma=0.2 * variation_switch, rng=variation_rng),
                             'tau_refrac': RD('normal_clipped', low=0.0, high=np.inf, mu=2.0, sigma=1. * variation_switch, rng=variation_rng),
                             'cm': RD('normal_clipped', low=0.0, high=np.inf, mu=1, sigma=0 * variation_switch, rng=variation_rng),
                             'v_rest': RD('normal_clipped', low=-np.inf, high=np.inf, mu=0.0, sigma=1.96 * variation_switch, rng=variation_rng),
                             'v_reset': RD('normal_clipped', low=-np.inf, high=0.9 * v_thresh, mu=10.0, sigma=4.2 * variation_switch, rng=variation_rng),
                             'i_offset': RD('normal', mu=0.0, sigma=0 * variation_switch, rng=variation_rng),
                             'v_thresh': RD('normal_clipped', low=-np.inf, high=np.inf, mu=v_thresh, sigma=1.12 * variation_switch, rng=variation_rng),
                             'e_rev_E': RD('normal_clipped', low=-np.inf, high=np.inf, mu=140.0, sigma=14 * variation_switch, rng=variation_rng),
                             'e_rev_I': RD('normal_clipped', low=-np.inf, high=np.inf, mu=-140, sigma=1.96 * variation_switch, rng=variation_rng)}

        # General settings
        simulation_time = 10000  # runtime of the simulation (ms)

        # Simulation settings
        threads = 48
        timestep = 0.1

        ###################################################################################
        # The values below either define the network structure or are hardware-specific.  #
        # As this tutorial uses a pre-mapped network, these values should not be modified!#
        ###################################################################################

        # Neurons
        neurons = 12500  # number of neurons in the original network
        scale = 6.  # scaling factor
        relex = 0.8  # relative share of excitatory neurons
        relin = 0.2  # relative number of inhibitory neurons
        Neff = int(neurons / scale)
        Nex = int(Neff * relex)
        Nin = int(Neff - Nex)

        logger.info(f'The Network is going to have {Neff} neurons ({Nex} excitatory, {Nin} inhibitory)')

        # Connections
        delay = 1.5  # delay of connections (ms) (Fixed by physical distance on chip -> will be replaced by measured values when loading results)
        epsilon = 0.1  # epsilon * neurons = input synapses per neuron
        Cex = int(epsilon * Nex)
        Cin = int(epsilon * Nin)
        # Parameters for external connections
        Cext = 200  # number of external connections.
        ext_pop_factor = 1.
        Next = int(Neff * ext_pop_factor)
        logger.info(f'Each neuron is going to have {Cex} excitatory, {Cin} inhibitory and {Cext} external (from {Next} sources) connections')

        # General settings
        kernel_seed = 100
        ext_pool = True  # User pool for external neurons to reduce total number of external neurons
        # For the hardware execution we have to rebuild the network.
        # However, we can still load previous mapping results by setting the mapping_path variable to a valid mapping result
        connector = "load" if simulator == "nest" else "matrix"
        mapping_path = "/wang/data/commissioning/BSS-1/rackplace/30/mappings/wafer_30_random_network/"
        # Path to routing results of the mapping, which can be loaded during the NEST simulation
        routing_results_path = "./_static/random_network/"

        # Hardware settings
        network_seed = 1234
        neuron_size = 8
        calib_path = "/wang/data/commissioning/BSS-1/rackplace/30/calibration/2023-02-22_big_cap_normal_model_random"
        defects_path = "/wang/data/commissioning/BSS-1/rackplace/30/derived_plus_calib_blacklisting/2023-03-13_big_cap_normal_model_random"
        max_hw_voltage = 1.3
        min_hw_voltage = 0.45
        max_bio_voltage = neuron_parameters["e_rev_E"].parameters["mu"]
        min_bio_voltage = neuron_parameters["e_rev_I"].parameters["mu"]
        bigcap = True
        slow = False
        mapping_g = 8.0 # g value used in mapped results. All following values are set in realation to this value
        orig_weight = 1.2 # Additional weight factor

        # Connection Parameters
        # Calculation of external frequency
        thresh = neuron_parameters["v_thresh"].parameters["mu"]  # mV
        rest = neuron_parameters["v_rest"].parameters["mu"]  # mV
        tau_m = neuron_parameters["tau_m"].parameters["mu"]
        dV = 0.1 * scale

        nu_thresh = (thresh - rest) / (dV * tau_m) * 1000  # scale to include unit

        delay_rng = NumpyRNG(300, parallel_safe=True)
        delay_dist = RD('normal_clipped', low=0.3, high=np.inf, mu=delay, sigma=0.2 * variation_switch, rng=delay_rng)

        # Parameters for network connections
        # Calculate weight. Brunel uses dU = 0.1 mV
        # Using an exponential kernel with tau_syn << tau_m we can use
        # w = (dU*Cm)/tau_syn
        # 1.7 manual parameter from simulations to adjust for tau syn ! << tau_m
        tau_syn = neuron_parameters["tau_syn_E"].parameters["mu"]
        cm = neuron_parameters["cm"].parameters["mu"]  # nF
        w = orig_weight * (dV * cm) / tau_syn  # nA
        weight_rng = NumpyRNG(400, parallel_safe=True)
        # assuming V_mean to be 0 V we can calculate the conductance (in ns) with
        e_rev_E = neuron_parameters["e_rev_E"].parameters["mu"]  # mV
        e_rev_I = neuron_parameters["e_rev_I"].parameters["mu"]  # mV
        V_mean = neuron_parameters["v_rest"].parameters["mu"]  # mV
        exc_conductance = w / (e_rev_E - V_mean)
        inh_conductance = -1 * w / (e_rev_I - V_mean)


In the next two cells, the experiment description is first sent to the BrainScaleS-1 system in Heidelberg for emulation and then to the HPC cluster in Jülich for the NEST simulation.

.. code-block:: ipython3

    # Merge results to form one file that can be executed at the sites
    # This is a workaround since only a single file can be sent via the nmpi runner
    !cd ./_static/random_network/experiment_code && cat imports.py parameters.py ../../network_helper/execution_helpers.py ../../network_helper/experiment_helpers.py experiment_description.py execute.py > run.py

    # Send job to Heidelberg for emulation
    wafer = 30
    collab_id = repo_info.name_in_the_url
    hw_config = {'WAFER_MODULE': wafer, "SOFTWARE_VERSION": "nmpm_software/column_changes", "CORES": "48", "PARTITION": "batch"}

    bss1_job_id = nmpi_client.submit_job(source="./_static/random_network/experiment_code/run.py",
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
    !cd ./_static/random_network/experiment_code && cat imports.py parameters.py ../../network_helper/execution_helpers.py ../../network_helper/experiment_helpers.py experiment_description.py execute.py > run.py

    container_path = None
    if container_path is None:
        raise RuntimeError("No container image has been provided. Please download the latest image to the Jülich HPC cluster and specify its location using the container_path variable.")

    resources = helpers.jobs.Resources(queue="dc-cpu", cpus_per_node=48, nodes=1)
    job = helpers.jobs.Description(
        executable=f"singularity exec --app wafer {container_path} python ./_static/random_network/experiment_code/run.py",
        arguments=["nest"],
        resources=resources
    )

    nest_job = jureca_client.new_job(job.to_dict(), inputs = ["./_static/random_network/experiment_code/run.py", "./_static/random_network/implemented_routes_30.pickle", "./_static/random_network/implemented_routes_all.pickle"])
    state_nest = "running"
    print("Nest job submitted")

The next cell polls the two sites and waits for the experiment results.

.. code-block:: ipython3

    from _static.helper.get_data import handle_results
    # wait for both jobs to finish and copy the results to this collab
    state_bss1, state_nest, result_dir_bss1, result_dir_nest = handle_results(state_bss1, state_nest, nmpi_client, bss1_job_id, nest_job, "random_network")

Once the results are obtained, the spike times and the spike-time histogram of the first 30 neurons are visualized in the following cells. Additionally, the mean firing rate of the neurons is displayed across different regimes.

Similar to the original model, distinct firing regimes are observed. However, deviations between simulation and emulation occur at firing rates above 50 Hz due to hardware-specific effects, such as saturating synaptic input circuits, which are not accounted for in the adapted model.

These results showcase the hardware’s ability to internally route spikes at firing rates of up to 250 Hz. By isolating 30 neurons for evaluation, reliable spike injection and readout are achieved even in high-firing regimes, despite the substantial speed-up factor of 10,000. Furthermore, the results highlight the hardware's reconfigurability. While the initial network setup takes approximately one minute, subsequent adjustments, such as modifying inputs or reprogramming weights, are completed within seconds. This allows for efficient, iterative experiments once a suitable model is identified in simulation.

.. code-block:: ipython3

    from _static.random_network.helper.plot_results import plot

    !mkdir -p ./plots/random_network

    plot(result_dir_bss1, result_dir_nest)

.. code-block:: ipython3

    from IPython.display import HTML

    # which parameters to show
    first_plot = {"g":1, "eta":0.8}
    second_plot = {"g":8, "eta":0.8}

    HTML(open("./_static/helper/plot_format.css", "r").read() +
    f"""
        <div class="row">
          <div class="column">
            <figure>
              <img src="plots/random_network/BSS-1/g_{first_plot["g"]}_eta_{first_plot["eta"]}.png"; title="g={first_plot["g"]}, eta={first_plot["eta"]}">
              <figcaption>Fig.1 BSS-1 g={first_plot["g"]}, eta={first_plot["eta"]}</figcaption>
            </figure>
          </div>
          <div class="column">
            <figure>
              <img src="plots/random_network/BSS-1/g_{second_plot["g"]}_eta_{second_plot["eta"]}.png"; title="g={second_plot["g"]}, eta={second_plot["eta"]}">
              <figcaption>Fig.2 BSS-1 g={second_plot["g"]}, eta={second_plot["eta"]}</figcaption>
            </figure>
          </div>
        </div>

        <div class="row">
          <div class="column">
            <figure>
              <img src="plots/random_network/NEST/g_{first_plot["g"]}_eta_{first_plot["eta"]}.png"; title="g={first_plot["g"]}, eta={first_plot["eta"]}">
              <figcaption>Fig.3 NEST g={first_plot["g"]}, eta={first_plot["eta"]}</figcaption>
            </figure>
          </div>
          <div class="column">
            <figure>
              <img src="plots/random_network/NEST/g_{second_plot["g"]}_eta_{second_plot["eta"]}.png"; title="g={second_plot["g"]}, eta={second_plot["eta"]}">
              <figcaption>Fig.4 NEST g={second_plot["g"]}, eta={second_plot["eta"]}</figcaption>
            </figure>
          </div>
        </div>

        <h1 style="text-align:center; font-size:1.5em">Mean neuron firing rates</h1>
        <div class="row">
          <div class="column">
            <figure>
              <img src="plots/random_network/BSS-1/neuron_rate.png"; title="BSS-1"; style="left; margin-right:10px;">
              <figcaption>Fig.5 BSS-1</figcaption>
            </figure>
          </div>
          <div class="column">
            <figure>
              <img src="plots/random_network/NEST/neuron_rate.png"; title="NEST"; style="left;">
              <figcaption>Fig.6 NEST</figcaption>
            </figure>
          </div>
        </div>
    """)
