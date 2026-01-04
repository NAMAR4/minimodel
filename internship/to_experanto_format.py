import os
import torch
import numpy as np
from minimodel import data
import yaml
import json
import hashlib
import base64
import argparse


def main():
    # args parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse_id", type=int)
    args = parser.parse_args()
    mouse_id = args.mouse_id

    device = torch.device('cuda')

    data_path = '../data'
    output_path = './data_experanto'
    np.random.seed(1)
    # load images
    img = data.load_images(data_path, mouse_id, file=data.img_file_name[mouse_id])      #TODO ? : image wird normalisiert, gecropt und gecastet zu float32. Will man das so?

    # load neurons
    fname = '%s_nat60k_%s.npz'%(data.db[mouse_id]['mname'], data.db[mouse_id]['datexp'])
    spks, istim_train, istim_test, xpos, ypos, spks_rep_all = data.load_neurons(file_path = os.path.join(data_path, fname), mouse_id = mouse_id)
    n_stim, n_max_neurons = spks.shape


    # split train and validation set
    itrain, ival = data.split_train_val(istim_train, train_frac=0.9)

    ineur = np.arange(0, n_max_neurons)
    spks_train = spks[itrain][:,ineur]
    spks_val = spks[ival][:,ineur]

    print(type(istim_train), ' istim_train: ', istim_train.shape, istim_train.min(), istim_train.max())
    print(type(itrain), ' itrain: ', itrain.shape, itrain.min(), itrain.max())
    print(type(ival), ' ival: ', ival.shape, ival.min(), ival.max())
    print(type(spks_train), ' spks_train: ', spks_train.shape, spks_train.min(), spks_train.max())
    print(type(spks_val), ' spks_val: ', spks_val.shape, spks_val.min(), spks_val.max())
    print(type(spks_rep_all), ' spks_rep_all: ', spks_rep_all.shape)
    print('Example value from spks_rep_all: ', spks_rep_all[0].shape)
    print()

    img_train = img[istim_train][itrain]
    img_val = img[istim_train][ival]
    img_test = img[istim_test]

    print(type(img_train), ' img_train: ', img_train.shape, img_train.min(), img_train.max())
    print(type(img_val), ' img_val: ', img_val.shape, img_val.min(), img_val.max())
    print(type(img_test), ' img_test: ', img_test.shape, img_test.min(), img_test.max())

    input_Ly, input_Lx = img_train.shape[-2:]
    print("input_Ly: ", input_Ly, " | input_Lx: ", input_Lx)
    # Initialize lists to store reshaped spks_test and img_test
    spks_test_list = []
    img_test_list = []
    image_id_test_list = []

    # Iterate over each stimulus in the test set
    for i, spks in enumerate(spks_rep_all):
        nrep, nneurons = spks.shape
        spks_test_list.append(spks)
        
        # Reshape the corresponding img_test
        img_rep = img_test[i][None, :, :].repeat(nrep, axis=0)
        img_test_list.append(img_rep)
        
        # Repeat the image ID for each repeat
        image_id_test_list.append(np.repeat(istim_test[i], nrep))

    # Concatenate the reshaped test data
    spks_test = np.concatenate(spks_test_list, axis=0)
    img_test = np.concatenate(img_test_list, axis=0)
    image_id_test = np.concatenate(image_id_test_list, axis=0)

    print('spks_test: ', spks_test.shape, spks_test.min(), spks_test.max())
    print('img_test: ', img_test.shape, img_test.min(), img_test.max())
    print('image_id_test: ', image_id_test.shape, image_id_test.min(), image_id_test.max())

    # Concatenate train, val, and test sets
    spk_all = np.concatenate([spks_train, spks_val, spks_test], axis=0)
    img_all = np.concatenate([img_train, img_val, img_test], axis=0)
    image_id_all = np.concatenate([istim_train[itrain], istim_train[ival], image_id_test], axis=0)

    # setting up the tiers
    NT, NN = spk_all.shape  # NT = number of trials, NN = number of neurons
    ntrain, nval, ntest = len(spks_train), len(spks_val), len(spks_test)
    itrain = np.arange(ntrain)
    ival = np.arange(ntrain, ntrain + nval)
    itest = np.arange(ntrain + nval, ntrain + nval + ntest)

    tiers = np.zeros(NT, object)
    tiers[itrain] = 'train'
    tiers[ival] = 'validation'
    tiers[itest] = 'test'

    print(type(spk_all), ' spk_all: ', spk_all.shape, spk_all.min(), spk_all.max())
    print(type(img_all), ' img_all: ', img_all.shape, img_all.min(), img_all.max())
    print(type(image_id_all), ' image_id_all: ', image_id_all.shape, image_id_all.min(), image_id_all.max())
    print("image_id_all samples: ", image_id_all[:20])
    print((np.mean(spk_all, axis=0)).shape)
    # make folders
    output_folder = f'./nat30k_{data.mouse_names[mouse_id]}_{data.exp_date[mouse_id]}_experanto'
    output_root = os.path.join(output_path, output_folder)
    os.makedirs(data_path, exist_ok=True)

    # set up data folder
    img_path = os.path.join(output_root, 'screen')
    spk_path = os.path.join(output_root, 'responses')
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(spk_path, exist_ok=True)

    # set up screen
    screen_data_path = os.path.join(img_path, 'data')
    screen_meta_path = os.path.join(img_path, 'meta')
    os.makedirs(screen_data_path, exist_ok=True)
    os.makedirs(screen_meta_path, exist_ok=True)
    # compute hashes
    hashes = []
    base_hasher = hashlib.blake2b(digest_size=16)

    for b in image_id_all.view(np.uint8).reshape(-1, 8):
        h = base_hasher.copy()
        h.update(b)
        hashes.append(base64.b64encode(h.digest()).decode()[:20])

    # I dont know in which order the images were presented. There is no temporal connection! Order is made up.
    # Should still work, since they already interpolated the data.

    # save screen

    # save images and meta data for images
    all_meta_data = {}
    image_size = [int(input_Ly), int(input_Lx)]

    for i in range(img_all.shape[0]):
        img_savepath = os.path.join(screen_data_path, f'{i:06d}.npy')
        meta_savepath = os.path.join(screen_meta_path, f'{i:06d}.yml')


        data_ = {
            "condition_hash": hashes[i],
            "first_frame_idx": i,
            "image_class": "natural_greyscale_image_from_minimodel",
            "image_id": int(image_id_all[i]),
            "image_size": image_size,
            "modality": "image",
            "num_frames": 1,
            "pre_blank_period": 0.0667,         # 66.7 ms
            "presentation_time": 0.0667,
            "stim_type": "stimulus.Frame",
            "tier": str(tiers[i]),
            "trial_idx": i,
        }
        all_meta_data[f'{i:06d}'] = data_

        # saving
        timg = img_all[i]
        np.save(img_savepath, timg[np.newaxis, ...])
        
        with open(meta_savepath, "w") as f:
            yaml.safe_dump(data_, f)



    # save scree mean
    means_path = os.path.join(screen_meta_path, 'means.npy')
    mean = np.mean(img_train)
    np.save(means_path, mean[np.newaxis, ...])
    print("saved screen mean as: ", mean)

    # save screen std
    stds_path = os.path.join(screen_meta_path, 'stds.npy')
    std = np.std(img_train)
    np.save(stds_path, std[np.newaxis, ...])
    print("svaed screen std as: ", std)


    # save combined_meta.json
    json_path = os.path.join(img_path, 'combined_meta.json')
    with open(json_path, "w") as f:
        json.dump(all_meta_data, f, indent=2)

    # save screen meta
    data_ = {"modality": "screen"}
    meta_savepath = os.path.join(img_path, 'meta.yml')
    with open(meta_savepath, "w") as f:
        yaml.safe_dump(data_, f)


    # save timestamps.npy 
    dt = 0.0667 + 0.0667        # pre_blank period + presentation time
    timestamps = np.arange(img_all.shape[0]) * dt

    timestamps_path = os.path.join(img_path, 'timestamps.npy')
    np.save(timestamps_path, timestamps)
    # timestamps works for both spikes and images since the data is already resampled / interpolated to fit one to one
    # --- save responses ---
    # save motor coordinates
    responses_meta = os.path.join(spk_path, "meta")
    os.makedirs(responses_meta, exist_ok=True)
    coordinates_path = os.path.join(responses_meta, "cell_motor_coordinates.npy")
    coords = np.zeros((NN, 3))  # z is set to zero 
    coords[:, 0] = xpos
    coords[:, 1] = ypos
    np.save(coordinates_path, coords)

    # save unit ids
    unit_ids_path = os.path.join(responses_meta, 'unit_ids.npy')
    unit_ids = np.arange(NN)
    np.save(unit_ids_path, unit_ids)

    # save spike means
    means_path = os.path.join(responses_meta, 'means.npy')
    means = np.mean(spk_all, axis=0)
    means = means[np.newaxis, ...]
    np.save(means_path, means)
    print("saved spike means | type: ", type(means), " shape: ", means.shape, " min: " , means.min(), " max: ", means.max() )

    # save spike stds
    stds_path = os.path.join(responses_meta, 'stds.npy')
    stds = np.std(spk_all, axis=0)
    stds = stds[np.newaxis, ...]
    np.save(stds_path, stds)
    print("saved spike std | type: ", type(stds), " shape: ", stds.shape, " min: " , stds.min(), " max: ", stds.max() )

    # save phase_shifts (already corrected so 0)
    shift_path = os.path.join(responses_meta, 'phase_shifts.npy')
    shifts = np.zeros(NN)
    np.save(shift_path, shifts)


    # response meta
    data_ = {
        "dtype": str(spk_all.dtype),
        "end_time": float((timestamps[-1] + dt)),      # ending is behind the last timestamp
        "is_mem_mapped": "true",
        "modality": "sequence",
        "n_signals": NN,
        "n_timestamps": NT ,                            # timestamps.shape[0] = NT
        "neruon_properties":
            {
                "cell_motor_coordinates": "meta/cell_motor_coordinates.npy",
                "fields": "",
                "phase_shifts": "",
                "unit_ids": "meta/unit_ids.npy"
            },
        "original_sampling_rate": 30,           # 30 Hz
        "phase_shift_per_signal": "false",
        "resampling_method": "",
        "resampling_timestamp": "",     
        "sampling_rate": round(1/dt, 10),                   # rate = 7.496251874 = 1/dt = 1/0.1334 
        "start_time": float(timestamps[0]),        
        "target_sampling_rate": 7.5
    }

    meta_savepath = os.path.join(spk_path, 'meta.yml')
    with open(meta_savepath, "w") as f:
        yaml.safe_dump(data_, f)


    # save spikes
    responses_data = os.path.join(spk_path, "data.mem")
    mm = np.memmap(
        responses_data,
        dtype=spk_all.dtype,
        mode="w+",
        shape=spk_all.shape         # n_timestamps * n_signals
    )

    mm[:] = spk_all[:]   # copy data
    mm.flush()
    del mm

    # sanity check
    mm = np.memmap(responses_data, dtype=spk_all.dtype, mode="r", shape=spk_all.shape)
    print("data.mem probably saved correctly: ", np.allclose(mm[:100], spk_all[:100], atol=1.e-10))
    del mm

    # --- save global meta file ---

    # 
    data_ = {
        "config": {
            "include_eye": False,
            "include_treadmill": False,
            "resize_stimuli": False,
            "screen_h_w": image_size,
            "tracking_method": 2,
            "mask_type": "soma",
            "trace_source": "Activity",
            "stack_coordinates": False,
            "include_astrocytes": False,
            "astrocyte_channel": 1,
            "astrocyte_video_h_w": None,
            "neuro_trace_segmentation_method": 6,
            "neuro_trace_spike_method": 6,
            "neuro_channel": 1,
            "include_anatomy": False,
            "sample_dtype": str(spk_all.dtype),             # here spikes?
            "screen_dtype": str(img_all.dtype),             # here image? Right now float32 not uint8
            "interleave_value": 128,
            "n_trials_to_export": None,
            "train_size": 0.8,
            "val_size": 0.1,
            "test_size": 0.1,
            "add_blank_at_the_end": False,
            "pupil_lowpass_fr": 2
        },
        "scan_key": {
            "animal_id": "29515",
            "session": 10,
            "scan_idx": 12
        }
    }


    json_path = os.path.join(output_root, 'meta.json')
    with open(json_path, "w") as f:
        json.dump(data_, f, indent=2)

 

if __name__ == "__main__":
    main()