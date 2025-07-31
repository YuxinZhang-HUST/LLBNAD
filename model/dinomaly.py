from .dinomaly_utils.model import Dinomaly
from model import MODEL


@MODEL.register_module
def dinomaly(pretrained=False, **kwargs):
    model = Dinomaly(**kwargs)
    return model
