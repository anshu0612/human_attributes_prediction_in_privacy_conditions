import os
import torch
import pickle
from torchvision import models
from functools import partial


def _load_and_save_places_model():
    '''
    Code borrowed from https://github.com/Tandon-A/emotic/blob/master/prepare_models.py
    '''

    backbone_model = 'resnet18'
    places_model_fetch_url = "http://places2.csail.mit.edu/models_places365/{}_places365.pth.tar".format(
        backbone_model)

    os.makedirs('places')
    os.system(
        'wget {} -O ./places/{}_places365.pth.tar'.format(places_model_fetch_url, backbone_model))

    model_path = './places'

    model_file = os.path.join(
        model_path, '%s_places365.pth.tar' % backbone_model)
    save_file = os.path.join(
        model_path, '%s_places365_py36.pth.tar' % backbone_model)

    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1#")
    model = torch.load(model_file, map_location=lambda storage,
                       loc: storage, pickle_module=pickle)
    torch.save(model, save_file)

    # create the network architecture
    context_model = models.__dict__[backbone_model](num_classes=365)

    # model trained on GPU could be deployed in CPU machine like this!
    checkpoint = torch.load(
        save_file, map_location=lambda storage, loc: storage)
    # the data parallel layer will add 'module' before each layer name
    state_dict = {str.replace(k, 'module.', ''): v for k,
                  v in checkpoint['state_dict'].items()}
    context_model.load_state_dict(state_dict)
    context_model.eval()

    torch.save(context_model, os.path.join(model_path, 'res_context' + '.pth'))

    print('completed preparing context model')


def get_places_model():

    if not os.path.exists('places'):
        _load_and_save_places_model()

    model_path_places = './places'
    model_context = torch.load(
        os.path.join(model_path_places, 'res_context.pth'))

    return model_context
