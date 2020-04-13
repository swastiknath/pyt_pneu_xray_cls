import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
# import model from model.py, by name
from model_resnet import Resnet 

NP_CONTENT_TYPE = 'application/x-image'
ACCEPT_TYPE='application/json'

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
    model = Resnet(model_info['output_dim'])

    # Load the store model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Prep for testing
    model.to(device).eval()

    print("Done loading model.")
    return model


def input_fn(request_body, content_type):
    if content_type == 'application/x-image':
        img_array = np.array(Image.open(io.BytesIO(request_body)))
        img_transformer = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize((224,224)), 
                                              transforms.ToTensor()])
        img_tens = img_transformer(img_array)
        return img_tens
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

# Provided output data handling
def output_fn(prediction_output, accept=ACCEPT_TYPE):
    print('Serializing the generated output.')
    return json.dumps({'result':prediction_output['output'].tolist()}), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


# Provided predict function
def predict_fn(input_data, model):
    print('Predicting class labels for the input data...')
    result = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    # Process input_data so that it is ready to be sent to our model.
    pred = model(input_data.to(device).unsqueeze(0))
    pred = pred.numpy()
    result['output'] = pred
    return result