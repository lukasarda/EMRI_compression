import numpy as np
import torch

from neural_net_class import AE_cnn

auto_enc= AE_cnn()
loss_function= torch.nn.MSELoss()
optimizer = torch.optim.SGD(auto_enc.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)


file_name= 'tfm_1y'
tfm= np.load('/pbs/home/l/lkarda/EMRI_compression/'+file_name+'.npz')['ltft']


# combi= np.array([tfm[tfm.files[0]], tfm[tfm.files[1]]])
# print(combi.shape)
tfm_input= np.expand_dims(tfm, axis=0)
input_tensor= torch.from_numpy(tfm_input)

print(input_tensor.shape)


output= auto_enc(input_tensor.float())
print(output.shape)

# np.savez_compressed('/pbs/home/l/lkarda/EMRI_compression/enc_arr.npz', AE=output.detach().numpy())



# loss= loss_function(output, input_tensor.float())

# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

# print(loss)