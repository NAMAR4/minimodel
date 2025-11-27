import os
import numpy as np
import torch
import argparse
from minimodel import data
from minimodel import model_builder
from minimodel import model_trainer
from minimodel import metrics



def main():
    # args parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse_id", type=int)
    args = parser.parse_args()


    # setup
    device = torch.device('cuda')
    mouse_id = args.mouse_id
    data_path = './data'
    weight_path = './checkpoints_trained'
    np.random.seed(1)

    # load images
    img = data.load_images(data_path, mouse_id, file=data.img_file_name[mouse_id])

    # load neurons
    fname = '%s_nat60k_%s.npz'%(data.db[mouse_id]['mname'], data.db[mouse_id]['datexp'])
    spks, istim_train, istim_test, xpos, ypos, spks_rep_all = data.load_neurons(file_path = os.path.join(data_path, fname), mouse_id = mouse_id)
    n_stim, n_neurons = spks.shape

    # split train and validation set
    itrain, ival = data.split_train_val(istim_train, train_frac=0.9)

    # normalize data
    spks, spks_rep_all = data.normalize_spks(spks, spks_rep_all, itrain)


    ineur = np.arange(0, n_neurons) #np.arange(0, n_neurons, 5)
    spks_train = torch.from_numpy(spks[itrain][:,ineur]).to(device)
    spks_val = torch.from_numpy(spks[ival][:,ineur]).to(device)

    print('spks_train: ', spks_train.shape, spks_train.min(), spks_train.max())
    print('spks_val: ', spks_val.shape, spks_val.min(), spks_val.max())

    img_train = torch.from_numpy(img[istim_train][itrain]).to(device).unsqueeze(1) # change :130 to 25:100 
    img_val = torch.from_numpy(img[istim_train][ival]).to(device).unsqueeze(1)
    img_test = torch.from_numpy(img[istim_test]).to(device).unsqueeze(1)

    print('img_train: ', img_train.shape, img_train.min(), img_train.max())
    print('img_val: ', img_val.shape, img_val.min(), img_val.max())
    print('img_test: ', img_test.shape, img_test.min(), img_test.max())

    input_Ly, input_Lx = img_train.shape[-2:]

    print("Hello World")
    print(f" mouse_id: {mouse_id}")
    
if __name__ == "__main__":
    main()