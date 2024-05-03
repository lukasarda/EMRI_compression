import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from dataset_class_nn import npz_class
from neural_net_class import AE_cnn4

auto_enc= AE_cnn4()
loss_function= torch.nn.MSELoss()
optimizer = torch.optim.SGD(auto_enc.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)


folder= '5020_samples'
dir_path= '/sps/lisaf/lkarda/H_matrices_td/'+folder+'/tfm_singles/'
vali_path= dir_path+'tfm_singles_validation'

dataset= npz_class(dirpath= dir_path)
vali_set= npz_class(dirpath= vali_path)
batch_size= 20
num_workers= 4

print('# of samples:', len(dataset))
print('Batchsize:', batch_size)
print('# of batches:', int(len(dataset)/batch_size))

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

vali_loader = torch.utils.data.DataLoader(
    vali_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)



def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, batch in enumerate(train_loader):

        output= auto_enc(batch.float())
        loss= loss_function(output, batch.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        running_loss+= loss.item()
        if i % 5 == 4:
            last_loss = running_loss/5
            print('batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
epoch_number= 0
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

epochs= 5

for epoch in range(epochs):
    print('Epoch {}:'.format(epoch_number + 1))

    auto_enc.train()
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0

    auto_enc.eval()
    with torch.no_grad():
        for i, vdata in enumerate(vali_loader):
            voutputs = auto_enc(vdata.float())
            vloss = loss_function(voutputs, vdata.float())
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    model_path = 'AE_CNN_{}_{}'.format(timestamp, epoch_number)
    torch.save(auto_enc.state_dict(), model_path)

    epoch_number += 1