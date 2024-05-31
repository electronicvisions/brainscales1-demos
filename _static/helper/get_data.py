from urllib.request import urlretrieve
import os
import time


def evaluate_folder(path):
    """
    Increments the index of a filename until a unique, non-existing filename is found.

    :param path: The base file path where the results will be stored.
    :return: A new file path with an incremented index that does not already exist.
    """
    filecounter = 0
    filename = path
    while (os.path.exists(filename)):
        filecounter += 1
        filename = f"{path}_{filecounter}"
    return filename


def bss1_downloader(job, local_dir, experiment_name):
    """
    Downloads output data files generated by a specified job on BrainScaleS to a local directory.

    :param job: A dictionary containing the full job description, as returned by `nmpi_client.get_job()`.
    :param local_dir: The path to the local directory where results are stored.
    :param experiment_name: The name of the experiment associated with the job.
    :return: The file path to the directory where the results of the job are stored.
    """
    datalist = job["output_data"]["files"]
    directory = evaluate_folder(os.path.join(local_dir, experiment_name))
    os.makedirs(directory)
    for dataitem in datalist:
        url = dataitem["url"]
        if url.endswith((".h5", ".out")):
            filename = dataitem["url"].split("/")[-1]
            filename = "log.out" if filename.endswith(".out") else filename
            local_path = os.path.join(directory, filename)
            urlretrieve(url, local_path)
    return directory


def nest_downloader(job, local_dir, experiment_name):
    """
    Downloads output data files generated by a specified job on the HPC cluster to a local directory.

    :param job: An instance of `pyunicore.client.Job` containing the job description.
    :param local_dir: The path to the local directory where results are stored.
    :param experiment_name: The name of the experiment associated with the job.
    :return: The file path to the directory where the results of the job are stored.
    """
    work_dir = job.working_dir
    directory = evaluate_folder(os.path.join(local_dir, experiment_name))
    os.makedirs(directory)
    work_dir.stat("/stdout").download(f"{directory}/log.out")
    work_dir.stat("/stderr").download(f"{directory}/log.err")
    for name, file in work_dir.listdir(f"results/{experiment_name}").items():
        n = name.split("/")[-1]
        file.download(directory + "/" + n)
    return directory


def handle_results(state_bss1, state_nest, nmpi_client, bss1_job_id, nest_job, experiment_name):
    """
    Poll the two sites and wait for job results.
    When the results are available, download them to a local directory.

    :param state_bss1: The current status of the job on BrainScaleS.
    :param state_nest: The current status of the job on the HPC cluster.
    :param nmpi_client: The NMPI client used for executing the job on BrainScaleS.
    :param bss1_job_id: The ID of the BrainScaleS job.
    :param nest_job: The job object for the HPC cluster (e.g., a `jureca_client` job).
    :param experiment_name: The name of the experiment associated with these jobs.
    :return: A tuple containing the final states of both jobs and the paths to the result directories.
    """
    # wait for both jobs to finish and copy the results to this collab
    print("Waiting for results")
    while state_bss1 == "running" or state_nest == "running":
        time.sleep(5)
        # check for bss1
        if state_bss1 == "running":
            state_bss1 = nmpi_client.job_status(bss1_job_id)
            if state_bss1 == "finished":
                print("BrainScaleS job finished")
            elif state_bss1 == "error":
                print("BrainScaleS job failed")
        # check for nest
        if state_nest == "running":
            if not nest_job.is_running():
                state_nest = "finished"
                print("Nest job finished")

    # Handle BrainScaleS results
    if state_bss1 == "error":
        print("BrainScaleS job showed error:")
        bss1_job = nmpi_client.get_job(bss1_job_id, with_log=True)
        print(bss1_job['log'])
        # Download error log
        bss1_downloader(bss1_job, "./results/brainscales", experiment_name)
    elif state_bss1 == "finished":
        bss1_job = nmpi_client.get_job(bss1_job_id, with_log=True)
        if bss1_job['log'] is not None:
            print(bss1_job['log'])
        result_dir_bss1 = bss1_downloader(bss1_job, "./results/brainscales", experiment_name)
        print("Stored BrainScaleS results to: ", result_dir_bss1)
        state_bss1 = "finished_and_received"
    elif state_bss1 == "finished_and_received":
        print("No new job started on BrainScaleS, using old results")
    elif state_bss1 == "not_started":
        print("No job started on BrainScaleS")
        result_dir_bss1 = None
    else:
        print(f"Unknown state: {state_bss1}")

    # Handle Nest results
    if state_nest == "finished":
        result_dir_nest = nest_downloader(nest_job, "./results/nest", experiment_name)
        print("Stored Nest results to: ", result_dir_nest)
        state_nest = "finished_and_received"
    elif state_nest == "not_started":
        print("No simulation started")
        result_dir_nest = None
    elif state_nest == "finished_and_received":
        print("No new Nest job started, using old results")
    else:
        print(f"Unknown state: {state_nest}")

    return (state_bss1, state_nest, result_dir_bss1, result_dir_nest)