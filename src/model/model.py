from pdb import set_trace as bp

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from .modules.feature_embedder import FeatureEmbedder
from .modules.temporal_aggregator import *
from process_data.utils import load_yaml
from process_data.data_config import DATA_SOURCES

class MoSE(torch.nn.Module):
	def __init__(self):
		super(MoSE, self).__init__();

		## MoSE
		self.mose_conv_in = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0);
		self.mose_conv_out = torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0);
		self.mose_relu = torch.nn.ReLU();
		
	def forward(self, x):
		w = torch.nn.AdaptiveAvgPool2d(1)(x);
		w = w.permute(0, 2, 1, 3);
		w = self.mose_conv_in(w);
		w = self.mose_relu(w);
		w = self.mose_conv_out(w);
		w = torch.nn.functional.softmax(w, dim=2);
		w = w.permute(0, 2, 1, 3);
		y = torch.mul(w, x);
		
		return y;

class SarModel(BaseModel):
	def __init__(self,
				 num_cat_pkl_path="", emb_feat_dim=32, hidden_size=128, hidden_size_coeff=6, dropout=0.3,
				 max_len=512,
				 temporal_aggregator_type="TemporalTransformerAggregator", temporal_aggregator_args={},
				 ):
		super().__init__()
		num_cat_dict = load_yaml(num_cat_pkl_path)
		self.max_len = max_len
		self.hidden_size = hidden_size
		self.feature_embedders = nn.ModuleList([
			FeatureEmbedder(num_cat_dict, ds, emb_feat_dim, hidden_size, hidden_size_coeff, dropout) 
			for ds in DATA_SOURCES
		])
		
		self.temporal_aggregator = eval(
			f"{temporal_aggregator_type}")(**temporal_aggregator_args)
		
		self.parms = nn.Sequential(nn.Linear(temporal_aggregator_args["hidden_size"], 16), nn.GELU(), nn.Linear(16, 4096), nn.GELU());
		self.mose = MoSE();
		self.gap = nn.AdaptiveAvgPool2d(1);
		self.classifier = nn.Sequential(
			nn.Linear(temporal_aggregator_args["hidden_size"], 16),
			nn.GELU(), 
			nn.Linear(16, 1), 
			nn.Sigmoid()
		)

	def forward(self, batch_idxs, seq_idxs, xs):
		"""
		batch_idxs: batch_idxs of 5 data_source, shape(5, len(given_data_source))
		seq_idxs: seq_idxs of 5 data_source, shape(5, len(given_data_source))
		xs: a list of data of given data_source, shape(5, len(given_data_source), given_data_source_features)
		"""

		## embed feature
		batch_size = int(max([max(b) for b in batch_idxs])) + 1
		xs = [self.feature_embedders[i](x) for i, x in enumerate(xs)]
		# create zero embedding map
		_x = torch.zeros((batch_size, self.max_len, self.hidden_size)).to(xs[0].get_device()).float()
		mask = torch.zeros((batch_size, self.max_len)).to(xs[0].get_device()).long()
		# put features to the right position and set the mask
		for bi, si, x in zip(batch_idxs, seq_idxs, xs):
			_x[bi, si] = x
			mask[bi, si] = 1

		# run temporal model
		x = self.temporal_aggregator(_x, mask)
		y = self.parms(x)
		y = torch.reshape(y, (y.size()[0], int(y.size()[1]/16), 4, 4));
		y = self.mose(y);
		y = torch.flatten(self.gap(y), start_dim=1);
		x = self.classifier(x + y).squeeze(-1)
		return x


if __name__ == "__main__":
	pass
	# model = SelfAttenNN('../data/preprocessed/v1/column_config_generated.yml')
	# data = torch.zeros((32, 200, 52))
	# print(model(data))
