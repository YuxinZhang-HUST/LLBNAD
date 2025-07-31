from argparse import Namespace


class cfg_model_dinomaly(Namespace):

	def __init__(self):
		Namespace.__init__(self)
		self.model = Namespace()

		self.model.name = 'dinomaly'
		self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True,
								 encoder_name='dinov2reg_vit_base_14')