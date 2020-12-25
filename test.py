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

if torch.cuda.is_available():
	print("using cuda")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class Test:
	def __init__(self, params):
		self.params = params
		self.run_name = params['run_name']
		
		self.dataset = SuctionData(params['root_dir'],mode='test',transform=params['transform'])
		self.dataloader = data.DataLoader(self.dataset, batch_size=1)
		self.model = SuctionNet(params['height'],params['width'])
		self.model = torch.load('output/{}/weights/model_{}.pt'.format(params['run_name'],params['load_ep']))
		self.model.eval()
		self.sigmoid = nn.Sigmoid()

		self.use_depth = params['use_depth']

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
			overlay = (rgb_img * 0.5) + (heatmap * 0.5)

			#plt.imshow(overlay)
			#plt.show()
			Image.fromarray((overlay * 255).astype(np.uint8)).save(
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
			probs = self.sigmoid(outputs).cpu().detach().numpy()

			self.save_img(it,rgb.cpu().detach().numpy(),probs)


if __name__ == '__main__':

	transform = transforms.Resize((200,200),interpolation=Image.NEAREST)
	params = {'root_dir': os.path.dirname(os.path.realpath(__file__)),
			  'run_name': 'rgb_depth_64_1e-3',
			  'load_ep': 879,
			  'height': 480,
			  'width': 640,
			  'threshold': 0.5,
			  'use_depth': True,
			  'transform': transform
			  }

	test = Test(params)
	test.test()