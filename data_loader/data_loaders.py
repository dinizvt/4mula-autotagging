import pescador
import numpy as np
from pathlib import Path
import os
import utils


def get_short_rep(audio_repr_path, x, y, frames_num):
    fp = np.memmap(audio_repr_path, dtype="float16", mode="r", shape=(frames_num, y))
    audio_rep = np.zeros([x, y])
    audio_rep[:frames_num, :] = np.array(fp)
    del fp

    return audio_rep


def read_mmap(audio_repr_path, x, y, frames_num, single_patch=False, offset=0):
    if frames_num < x:
        audio_repr = get_short_rep(audio_repr_path, x, y, frames_num)
    else:
        read_x = x if single_patch else frames_num
        fp = np.memmap(
            audio_repr_path, dtype="float16", mode="r", shape=(read_x, y), offset=offset
        )
        audio_repr = np.array(fp)
        del fp
    return audio_repr


def data_generator(id, audio_repr_path, gt, pack):
    shape, data_dir, sampling, param_sampling = pack
    audio_repr_path = Path(data_dir, "4mula__dat", audio_repr_path)

    try:
        floats_num = os.path.getsize(audio_repr_path) // 2  # each float16 has 2 bytes
        frames_num = floats_num // shape[1]

        # let's deliver some data!
        if sampling == "random":
            for i in range(0, param_sampling):
                # we use a uniform distribution to get a relative random offset depending
                # exclusively in the seed number and not in the number of frames.
                # This way for two feature types with different number of frames the
                # sampler will select roughly the same chunks of the audio.
                random_uniform = np.random.random()
                random_frame_offset = int(
                    round(random_uniform * (frames_num - shape[0]))
                )

                # idx * bands * bytes per float
                offset = random_frame_offset * shape[1] * 2
                representation = read_mmap(
                    audio_repr_path,
                    shape[0],
                    shape[1],
                    frames_num,
                    single_patch=True,
                    offset=offset,
                )

                # flatten the temporal axis
                # representation = representation.flatten()

                yield {
                    "X": representation,
                    "Y": gt,
                    "ID": id,
                }

        elif sampling == "overlap_sampling":
            audio_rep = read_mmap(
                audio_repr_path,
                shape[0],
                shape[1],
                frames_num,
            )
            last_frame = int(audio_rep.shape[0]) - int(shape[0]) + 1
            for time_stamp in range(0, last_frame, param_sampling):
                representation = audio_rep[time_stamp : time_stamp + shape[0], :]
                representation = representation.flatten()
                yield {
                    "X": representation,
                    "Y": gt,
                    "ID": id,
                }
    except FileNotFoundError:
        print('"{}" not found'.format(audio_repr_path))


class StreamerLEN(pescador.ZMQStreamer):
    def __init__(
        self,
        streamer,
        min_port=49152,
        max_port=65535,
        max_tries=100,
        copy=False,
        timeout=5,
        data_len=None,
        batch_size=None,
    ):
        super().__init__(streamer, min_port, max_port, max_tries, copy, timeout)
        self.batch_size = batch_size
        self.data_len = data_len

    def __len__(self):
        return self.data_len


def FourMuLaDataloader(
    data_dir, batch_size, val_batch_size, x_size, y_size, sampling, param_sampling
):
    data_dir = Path(data_dir)
    file_index = data_dir / "index_repr.tsv"
    [audio_repr_paths, id2audio_repr_path] = utils.load_id2path(file_index)

    # load training data
    file_ground_truth_train = data_dir / "gt_train_0.csv"
    ids_train, id2gt_train = utils.load_id2gt(file_ground_truth_train)
    # load validation data
    file_ground_truth_val = data_dir / "gt_val_0.csv"
    ids_val, id2gt_val = utils.load_id2gt(file_ground_truth_val)

    train_pack = ((x_size, y_size), data_dir, sampling, param_sampling)
    train_streams = [
        pescador.Streamer(
            data_generator, id, id2audio_repr_path[id], id2gt_train[id], train_pack
        )
        for id in ids_train
    ]
    train_mux_stream = pescador.StochasticMux(
        train_streams, n_active=batch_size * 2, rate=None, mode="exhaustive"
    )
    train_batch_streamer = pescador.Streamer(
        pescador.buffer_stream,
        train_mux_stream,
        buffer_size=batch_size,
        partial=True,
    )
    n_batches_train = int(np.ceil((len(train_streams)) / batch_size)) * param_sampling
    train_batch_streamer = StreamerLEN(
        train_batch_streamer, data_len=n_batches_train, batch_size=batch_size
    )

    val_pack = ((x_size, y_size), data_dir, "overlap_sampling", x_size)
    val_streams = [
        pescador.Streamer(
            data_generator, id, id2audio_repr_path[id], id2gt_val[id], val_pack
        )
        for id in ids_val
    ]
    val_mux_stream = pescador.ChainMux(val_streams, mode="exhaustive")
    val_batch_streamer = pescador.Streamer(
        pescador.buffer_stream,
        val_mux_stream,
        buffer_size=val_batch_size,
        partial=True,
    )
    n_batches_val = int(np.ceil((len(val_streams)) / batch_size)) * x_size
    val_batch_streamer = StreamerLEN(
        val_batch_streamer, n_batches_val, batch_size=batch_size
    )
    return train_batch_streamer, val_batch_streamer
