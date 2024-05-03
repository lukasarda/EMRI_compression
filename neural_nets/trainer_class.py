import torch
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, val_loader, loss_function, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.losses = {'training': [], 'validation': []}

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss= 0.
        batch_loss= 0.
        loss_list= []

        for i, batch in enumerate(self.train_loader):
            batch = batch.float().to(self.device)
            self.optimizer.zero_grad()
            output = self.model(batch)
            loss = self.loss_function(output, batch)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            batch_loss += loss.item()

            if i % 5 == 4:
                avg_batch_loss = batch_loss / 5
                print('batch {} loss: {}'.format(i + 1, avg_batch_loss))
                loss_list.append(avg_batch_loss)
                batch_loss = 0.

        avg_loss = running_loss / len(self.train_loader)  # Compute average loss for the entire epoch
        print('Training Loss:', avg_loss)
        self.losses['training'].append(loss_list)

    def validate(self, epoch):
        self.model.eval()
        running_vloss = 0.0

        with torch.no_grad():
            for i, vdata in enumerate(self.val_loader):
                vdata = vdata.float().to(self.device)
                voutputs = self.model(vdata)
                vloss = self.loss_function(voutputs, vdata)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / len(self.val_loader)
        print('Validation Loss:', avg_vloss)
        self.losses['validation'].append(avg_vloss)
