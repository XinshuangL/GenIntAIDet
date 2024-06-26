from .clip_models import CLIPModel

def get_model(name):
    return CLIPModel(name[5:]).cuda()
