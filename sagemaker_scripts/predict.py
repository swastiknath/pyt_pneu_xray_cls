import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# import model from model.py, by name
from model import VGG19

# default content type is numpy array
NP_CONTENT_TYPE = 'application/x-npy'


# Provided model load function
def model_fn(model_dir):

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

    # Load the store model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Prep for testing
    model.to(device).eval()

    print("Done loading model.")
    return model


# Provided input data loading
def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == NP_CONTENT_TYPE:
        image = Image.open(serialized_input_data)
        return image
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

# Provided output data handling
def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    if accept == NP_CONTENT_TYPE:
        stream = BytesIO()
        np.save(stream, prediction_output)
        return stream.getvalue(), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


# Provided predict function
def predict_fn(input_data, model):
    print('Predicting class labels for the input data...')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    # Process input_data so that it is ready to be sent to our model.
    image_transformer = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])
    data = image_transformer(data).to(device)
    pred = predictor.predict(data.unsqueeze_(0))
    pred = pred.argmax()
    return pred