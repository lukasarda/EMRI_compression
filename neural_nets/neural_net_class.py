import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 16, 5)

        self.fc1 = nn.Linear(302000, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    


# Creating a PyTorch class
# 28*28 ==> 9 ==> 28*28
class AE_fcn(torch.nn.Module):
	def __init__(self):
		super().__init__()
		
		# Building an linear encoder with Linear
		# layer followed by Relu activation function
		# 784 ==> 9
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(512 * 618, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 36),
			torch.nn.ReLU(),
			torch.nn.Linear(36, 18),
			torch.nn.ReLU(),
			torch.nn.Linear(18, 9)
		)
		
		# Building an linear decoder with Linear
		# layer followed by Relu activation function
		# The Sigmoid activation function
		# outputs the value between 0 and 1
		# 9 ==> 784
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(9, 18),
			torch.nn.ReLU(),
			torch.nn.Linear(18, 36),
			torch.nn.ReLU(),
			torch.nn.Linear(36, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 512 * 618),
			torch.nn.Sigmoid()
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded



class AE_cnn_fcc(torch.nn.Module):
    def __init__(self):
        super().__init__()


        self.encoder_CNN = torch.nn.Sequential(
                nn.Conv2d(1, 32, 5),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(32, 16, 5),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2)
        )

        self.encoder_FCN = torch.nn.Sequential(
            torch.nn.Linear(302000, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
        )
        
        self.decoder_FCN = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 302000),
        )

        self.decoder_CNN = torch.nn.Sequential(
            nn.ConvTranspose2d(16, 32, 5),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 5),
            nn.ReLU(True),
            nn.Sigmoid()                  
        )

    def forward(self, x):
        x= self.encoder_CNN(x)
        x= x.view(1, -1)

        x= self.encoder_FCN(x)
        x= self.decoder_FCN(x)

        x= x.view(16, 125, 151)
        x= self.decoder_CNN(x)


        return x
    

class AE_cnn(torch.nn.Module):
    def __init__(self):
        super().__init__()


        self.encoder_CNN = torch.nn.Sequential(
                nn.Conv2d(2, 64, 10),
                nn.ReLU(True),
        )


        self.decoder_CNN = torch.nn.Sequential(
            nn.ConvTranspose2d(64, 2, 10),
            nn.ReLU(True),
            nn.Sigmoid()                  
        )


    def forward(self, x):
        x= self.encoder_CNN(x)
        x= self.decoder_CNN(x)

        return x


class AE_cnn2(torch.nn.Module):
    def __init__(self):
        super().__init__()


        self.encoder_CNN = torch.nn.Sequential(
            nn.Conv2d(2, 64, 10),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 10),
            nn.ReLU(True)
        )


        self.decoder_CNN = torch.nn.Sequential(
            nn.ConvTranspose2d(32, 64, 10),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 2, 10),
            nn.ReLU(True),
            nn.Sigmoid()                  
        )


    def forward(self, x):
        x= self.encoder_CNN(x)
        x= self.decoder_CNN(x)

        return x
    

class AE_cnn3(torch.nn.Module):
    def __init__(self):
        super().__init__()


        self.encoder_CNN = torch.nn.Sequential(
                nn.Conv2d(2, 128, 10),
                nn.ReLU(True),
        )


        self.decoder_CNN = torch.nn.Sequential(
            nn.ConvTranspose2d(128, 2, 10),
            nn.ReLU(True),
            nn.Sigmoid()                  
        )


    def forward(self, x):
        x= self.encoder_CNN(x)
        x= self.decoder_CNN(x)

        return x
    

class AE_cnn4(torch.nn.Module):
    def __init__(self):
        super().__init__()


        self.encoder_CNN = torch.nn.Sequential(
                nn.Conv2d(2, 256, 20),
                nn.ReLU(True),
                nn.Conv2d(256, 128, 10),
                nn.ReLU(True),
                nn.Conv2d(128, 64, 10),
                nn.ReLU(True),
        )


        self.decoder_CNN = torch.nn.Sequential(
            nn.ConvTranspose2d(64, 128, 10),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 256, 10),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 2, 20),
            nn.ReLU(True),
            nn.Sigmoid()                  
        )


    def forward(self, x):
        x= self.encoder_CNN(x)
        x= self.decoder_CNN(x)

        return x
    

class AE_cnn5(torch.nn.Module):
    def __init__(self):
        super().__init__()


        self.encoder_conv = torch.nn.Sequential(
            nn.Conv2d(2, 64, 10),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 10),
            nn.ReLU(True),
        )

        self.encoder_linear = nn.Sequential(
            nn.Linear(9484800, 128),
            nn.ReLU(),
        )
        
        self.decoder_linear = nn.Sequential(
            nn.Linear(128, 9484800),
            nn.ReLU(),
        )

        self.decoder_conv = torch.nn.Sequential(
            nn.ConvTranspose2d(32, 64, 10),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 2, 10),
            nn.ReLU(True),
            nn.Sigmoid()                  
        )


    def forward(self, x):
        x = self.encoder_conv(x)
        shape = x.shape
        x = x.view(x.size(0), -1) # Reshape/flatten to use fully connected layers
        x = self.encoder_linear(x)
        
        x = self.decoder_linear(x)
        x = x.view(x.size(0), shape[1], shape[2], shape[3])  # Reshape to match the shape before flattening
        x = self.decoder_conv(x)
        return x


class AE_cnn6(torch.nn.Module):
    def __init__(self):
        super().__init__()


        self.encoder_conv = torch.nn.Sequential(
            nn.Conv2d(2, 64, 10),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 10),
            nn.ReLU(True),
        )


        self.decoder_conv = torch.nn.Sequential(
            nn.ConvTranspose2d(32, 64, 10),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 2, 10),
            nn.ReLU(True),                
        )


    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.decoder_conv(x)
        return x