import torch
import numpy as np
from dataset_class_nn import npz_class
from neural_net_class import AE_cnn6

saved_model= AE_cnn6()
saved_model.load_state_dict(
    torch.load(
        '/pbs/home/l/lkarda/EMRI_compression/neural_nets/model_states/AE_CNN_20240430_120553_4'
    )
)


loss_function= torch.nn.MSELoss()
folder= '30000_samples'
dir_path= '/sps/lisaf/lkarda/H_matrices_td/'+folder+'/tfm_singles/'
vali_path= dir_path+'tfm_singles_validation'

dataset= npz_class(dirpath= dir_path)
vali_set= npz_class(dirpath= vali_path)
batch_size= 1
num_workers= 1


vali_loader = torch.utils.data.DataLoader(
    vali_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

for i, batch in enumerate(vali_loader):
    output= saved_model(batch.float())
    loss= loss_function(output, batch.float())
    print(loss)
    
    if i == 1:
        np.savez_compressed(
            '/pbs/home/l/lkarda/EMRI_compression/ae_1month_3.npz',
            AE=output.detach().numpy()
        )
        break

