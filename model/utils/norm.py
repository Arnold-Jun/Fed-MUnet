import torch
import math

def model_norm(model_1, model_2):
	squared_sum = 0
	for name, layer in model_1.named_parameters():
		squared_sum += torch.sum(torch.pow(layer.data.cuda() - model_2.state_dict()[name].data, 2))
	return math.sqrt(squared_sum)