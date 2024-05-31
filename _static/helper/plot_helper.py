import h5py
from neo import SpikeTrain, Segment, Block
from quantities import ms


def load_hdf5(file):
    """
    Loads results from a custom HDF5 format.

    :param file: The path to the results file.
    :return: The loaded results as a Neo `Block` object.
    """
    with h5py.File(file, 'r') as hf:
        block_group = hf["block"]
        pop_name = block_group["name"][()]
        block = Block(name=pop_name)
        segments_group = block_group["segments"]
        # Keep ordering of original data
        for n_seg in range(len(segments_group)):
            segment_group = segments_group[str(n_seg)]
            segment = Segment(description=pop_name)
            segment_annotations = {key: val[()] for (key, val) in segment_group["annotations"].items()}
            spiketrain_group = segment_group["spiketrains"]
            for n_sp in range(len(spiketrain_group)):
                spiketrain = spiketrain_group[str(n_sp)]
                spikes = spiketrain["spiketrain"][()]
                sptrain_annotations = {key: val[()] for (key, val) in spiketrain.items() if key != "spiketrain"}
                sptr = SpikeTrain(spikes * ms, t_start=0 * ms, t_stop=segment_annotations["sim_time"] * ms)
                sptr.annotations = sptrain_annotations
                segment.spiketrains.append(sptr)
            segment.annotations = segment_annotations
            block.segments.append(segment)
    return block
