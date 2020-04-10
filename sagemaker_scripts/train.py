import argparse
import json
import os
import torch
import logging
import sys
import sagemaker_containers
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

vgg19_b = torchvision.models.vgg19_bn(pretrained=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

from model import VGG19

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG19(model_info['output_dim'])

    # Loading the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model

# Gets training data in batches from S3
def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")
    num_workers = 0

    image_transformer = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_data = datasets.ImageFolder(training_dir, transform=image_transformer)
    train_loader = torch.utils.data.DataLoader(train_data, 
                                           batch_size=batch_size, 
                                           num_workers=num_workers, 
                                           shuffle=True)

    return train_loader


def train(model, train_loader, epochs, criterion, optimizer, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    criterion    - The loss function used for training. 
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    
    for param in vgg19_b.features.parameters():
        param.requires_grad=False
        
    for epoch in range(1, epochs + 1):
        model.train() # Making sure that the model is in training mode.

        total_loss = 0

        for batch, (data, label) in enumerate(train_loader):
            # getting the data
            batch_x = data.to(device)
            batch_y = label.to(device)

            optimizer.zero_grad()

            # get predictions from model
            y_pred = model(batch_x)
            
            # perform backprop
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()
            
            if batch % 20 == 19:
                logger.info("Epoch {} : Batch {} : Train Batch Loss: {}".format(epoch,batch+1, total_loss/20))
                total_loss = 0.0

if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 20)')
    parser.add_argument('--epochs', type=int, default=4, metavar='N',
                        help='number of epochs to train (default: 4)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    parser.add_argument('--output_dim', type=int, default=3, metavar='O', 
                        help = 'output dimension (default: 3)')
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    model = VGG19(args.output_dim).to(device)

#     Defining an optimizer and loss function for training
    criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(vgg19_b.classifier.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Trains the model (given line of code, which calls the above training function)
    train(model, train_loader, args.epochs, criterion, optimizer, device)

    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'output_dim': args.output_dim,
        }
        torch.save(model_info, f)
        
    

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)