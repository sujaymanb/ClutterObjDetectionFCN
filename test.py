import torch
import torch.nn as nn
import torch.utils.data as data
from model import SuctionNet
from dataset import SuctionData
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

if torch.cuda.is_available():
	print("using cuda")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class Test:
	def __init__(self, params):
		self.params = params
		self.run_name = params['run_name']

		# transform input and output
		if params['transform']:
			self.transform = transforms.Resize((params['width'],params['height']),
								interpolation=Image.NEAREST)
			self.untransform = transform = transforms.Resize((params['actual_h'],params['actual_w']),
								interpolation=Image.NEAREST)
		else:
			self.transform = None

		# load test dataset
		self.dataset = SuctionData(params['root_dir'],mode='test',transform=self.transform)
		self.dataloader = data.DataLoader(self.dataset, batch_size=1)
		
		# load saved model
		self.model = SuctionNet(params['height'],params['width'])
		self.model = torch.load('output/{}/weights/model_{}.pt'.format(params['run_name'],params['load_ep']))
		self.model.eval()
		self.sigmoid = nn.Sigmoid()

		self.use_depth = params['use_depth']
		self.overlay = params['overlay']

		if not os.path.exists(os.path.join(params['root_dir'],'output',params['run_name'],'test')):
			os.mkdir(os.path.join(params['root_dir'],'output',params['run_name'],'test'))


	def save_img(self,it,rgb,probs):
		"""
		save the test heatmaps
		args:
			it: index
			rgb: color image (to overlay)
			probs: network output probabilities
		"""
		root = self.params['root_dir']
		print(probs.shape)
		n,_,h,w = probs.shape

		for i in range(n):
			prob = np.squeeze(probs[i])
			rgb_img = np.squeeze(rgb[i])/255.0
			rgb_img = np.transpose(rgb_img, [1, 2, 0])
			colormap = plt.get_cmap('jet')
			heatmap = colormap(prob)[:,:,:3]
			print(rgb_img.shape, heatmap.shape)

			if self.overlay:
				img = (rgb_img * 0.5) + (heatmap * 0.5)
			else:
				img = prob

			#plt.imshow(overlay)
			#plt.show()
			Image.fromarray((img * 255).astype(np.uint8)).save(
							os.path.join(root,'output/{}/test/{}.png'.format(self.run_name,int(it))))


	def test(self):
		"""
		generate output for test data
		"""
		total_acc = 0.0
		it = 0.0
		for data in self.dataloader:
			it += 1
			
			rgb, depth = data['color'], data['depth']
			rgb = rgb.to(device)
			if self.use_depth:
				depth = depth.to(device)
			
			if self.use_depth:
				outputs = self.model(rgb, depth)
			else:
				outputs = self.model(rgb)

			if self.transform:
				outputs = self.untransform(outputs)
				rgb = self.untransform(rgb)

			probs = self.sigmoid(outputs).cpu().detach().numpy()
			self.save_img(it,rgb.cpu().detach().numpy(),probs)


if __name__ == '__main__':
	params = {'root_dir': os.path.dirname(os.path.realpath(__file__)),
			  'run_name': 'rgb_depth_32_1e-4',
			  'load_ep': 1499,
			  'height': 480,
			  'width': 640,
			  'actual_h': 480,
			  'actual_w': 640,
			  'threshold': 0.5,
			  'use_depth': True,
			  'transform': False,
			  'overlay': False
			  }

	test = Test(params)
	test.test()