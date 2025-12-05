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
    data_path = '../data'
    weight_path = './checkpoints_trained'
    os.makedirs(weight_path, exist_ok=True)
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


    seed = 1
    feve_nlayers = []
    for nlayers in range(1, 5):
        # Building Model

        nconv1 = 192
        nconv2 = 192
        model, in_channels = model_builder.build_model(NN=len(ineur), n_layers=nlayers, n_conv=nconv1, n_conv_mid=nconv2)
        model_name = model_builder.create_model_name(data.mouse_names[mouse_id], data.exp_date[mouse_id], n_layers=nlayers, in_channels=in_channels, seed=seed)
        
        model_path = os.path.join(weight_path, model_name)
        print('model path: ', model_path)
        model = model.to(device)


        # Training the model
        print(device)
        if not os.path.exists(model_path):
            best_state_dict = model_trainer.train(model, spks_train, spks_val, img_train, img_val, device=device)
            torch.save(best_state_dict, model_path)
            print('saved model', model_path)
        model.load_state_dict(torch.load(model_path))
        print('loaded model', model_path)

        # test model
        test_pred = model_trainer.test_epoch(model, img_test)
        print('test_pred: ', test_pred.shape, test_pred.min(), test_pred.max())


        test_fev, test_feve = metrics.feve(spks_rep_all, test_pred)
        print('FEVE (test, all): ', np.mean(test_feve))

        threshold = 0.15
        print(f'filtering neurons with FEV > {threshold}')
        valid_idxes = np.where(test_fev > threshold)[0]
        print(f'valid neurons: {len(valid_idxes)} / {len(test_fev)}')
        print(f'FEVE (test, FEV>0.15): {np.mean(test_feve[test_fev > threshold])}')


if __name__ == "__main__":
    main()