import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import SubsetRandomSampler
from model import SuctionNet, SuctionNetRGB
from dataset import SuctionData
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image

if torch.cuda.is_available():
	print("using cuda")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Train:
	def __init__(self, params):
		"""
		args:
			params: parameters passed in as dict
		"""
		self.params = params

		# resize the image or not
		if params['transform']:
			transform = transforms.Resize((params['height'],params['width']),
											interpolation=Image.NEAREST)
		else:
			transform = None

		# experiment output folder name
		self.run_name = params['run_name']

		# load dataset and split into test and validation
		self.dataset = SuctionData(params['root_dir'],transform=transform)
		data_size = len(self.dataset)
		indices = list(range(data_size))
		split = int(np.floor(params['val_split'] * data_size))
		np.random.shuffle(indices)
		train_ind, val_ind = indices[split:], indices[:split]
		train_sampler = SubsetRandomSampler(train_ind)
		val_sampler = SubsetRandomSampler(val_ind)
		train_loader = data.DataLoader(self.dataset, 
									batch_size=params['batch_size'],
									sampler=train_sampler)
		val_loader = data.DataLoader(self.dataset, 
									batch_size=params['batch_size'],
									sampler=val_sampler)
		self.dataloader = {'train': train_loader,
							'val': val_loader}
		
		# init model (use depth input or not)
		self.use_depth = params['use_depth']
		if self.use_depth:
			self.model = SuctionNet(params['height'],params['width']).to(device)
		else:
			self.model = SuctionNetRGB(params['height'],params['width']).to(device)
		self.opt = torch.optim.Adam(self.model.parameters(),
									lr=params['lr'])
		self.lossfn = nn.BCEWithLogitsLoss()
		self.sigmoid = nn.Sigmoid()
		
		self.num_epochs = params['num_epochs']
		self.create_dirs()

		# logs
		self.tr_logs = {'it': [], 
						'loss': [],
						'acc': []}
		self.val_logs = {'it': [], 
						'loss': [],
						'acc': []}


	def create_dirs(self):
		"""
		create directories to save model checkpts and logs
		"""
		root = self.params['root_dir']

		if not os.path.exists(os.path.join(root,'output')):
			os.mkdir(os.path.join(root,'output'))
		elif os.path.exists(os.path.join(root,'output/%s' % self.run_name)):
			import shutil
			shutil.rmtree(os.path.join(root,'output/%s' % self.run_name))

		os.mkdir(os.path.join(root,'output/%s' % self.run_name))
		os.mkdir(os.path.join(root,'output/%s/weights' % self.run_name))
		os.mkdir(os.path.join(root,'output/%s/logs' % self.run_name))
		with open(os.path.join(root,'output/%s/params.txt' % self.run_name), 'w') as f:
			f.write(json.dumps(self.params))


	def disp_img(self,img):
		"""
		save/display debug image
		"""
		img = np.squeeze(img)
		plt.clf()
		plt.imshow(img, cmap='jet', interpolation='nearest')
		plt.colorbar(cmap='jet')
		plt.savefig("debug.png")


	def calc_acc(self, label, probs):
		"""
		TODO: 
		Calculate accuracy:
		Args:
			label: gt values
			probs: probabilites output
		"""
		pred = np.zeros_like(probs)
		pred[probs > self.params['threshold']] = 1.
		total = (pred == 0.).sum() + (pred == 1.).sum()
		acc = np.sum(pred == label)/total

		return acc


	def log_data(self, it, phase, loss, acc):
		""" 
		save the logs for validation or training 
		Args:
			it: iteration
			phase: training or validation
			loss: loss value
			acc: accuracy value
		"""
		root = self.params['root_dir']
		# if train phase
		if phase == 'train':
			self.tr_logs['it'].append(it)
			self.tr_logs['loss'].append(loss)
			self.tr_logs['acc'].append(acc)
			np.savez(os.path.join(root,'output/%s/logs/train.npz' % self.run_name),
					it=self.tr_logs['it'], loss=self.tr_logs['loss'], acc=self.tr_logs['acc'])

		# if test phase
		if phase == 'val':
			self.val_logs['it'].append(it)
			self.val_logs['loss'].append(loss)
			self.val_logs['acc'].append(acc)
			np.savez(os.path.join(root,'output/%s/logs/val.npz' % self.run_name),
				it=self.val_logs['it'], loss=self.val_logs['loss'], acc=self.val_logs['acc'])

		print('It: {} {} Loss: {} Acc: {}'.format(it, phase, loss, acc))


	def validate(self, it):
		""" 
		Validate on validation set 
		args:
			it: current training iteration
		"""
		self.model.eval()

		val_loss = 0.0
		val_acc = 0.0
		n = 0
		for data in self.dataloader['val']:
			n += 1
			rgb, depth, label = data['color'], data['depth'], data['label']
			rgb = rgb.to(device)
			if self.use_depth:
				depth = depth.to(device)
			label = label.to(device)

			with torch.set_grad_enabled(False):
				if self.use_depth:
					outputs = self.model(rgb, depth)
				else:
					outputs = self.model(rgb)
				probs = self.sigmoid(outputs).cpu().detach().numpy()
				loss = self.lossfn(outputs, label)
				val_loss += loss.item()
				val_acc += self.calc_acc(label, probs)

		val_loss /= n
		val_acc /= n

		self.log_data(it, 'val', val_loss, val_acc)


	def train(self):
		""" 
		Run training on training set 
		"""
		it = 0
		for ep in range(self.num_epochs):
			self.model.train()
				
			# iterate trhough training batches
			for data in self.dataloader['train']:
				it += 1

				rgb, depth, label = data['color'], data['depth'], data['label']
				rgb = rgb.to(device)
				if self.use_depth:
					depth = depth.to(device)
				label = label.to(device)
				#print("label",label.shape)

				self.opt.zero_grad()

				with torch.set_grad_enabled(True):
					if self.use_depth:
						outputs = self.model(rgb, depth)
					else:
						outputs = self.model(rgb)
					probs = self.sigmoid(outputs).cpu().detach().numpy()
					acc = self.calc_acc(label, probs)
					#print("output",outputs.shape)
					loss = self.lossfn(outputs, label)
					loss.backward()
					self.opt.step()

				# debug display
				if self.params['disp']:
					with torch.no_grad():
						self.disp_img(probs[0])

				# logging
				self.log_data(it, 'train', loss.item(), acc)

			if (ep+1)%self.params['save_freq'] == 0:
				torch.save(self.model, 'output/{}/weights/model_{}.pt'.format(self.params['run_name'],ep))

			# do validation end of each epoch
			self.validate(it)


if __name__ == '__main__':
	params = {'root_dir': os.path.dirname(os.path.realpath(__file__)),
			  'run_name': 'rgb_depth_64_3e-4',
			  'batch_size':100,
			  'val_split': 0.2,
			  'height': 200,
			  'width': 200,
			  'lr': 3e-4,
			  'num_epochs': 10000,
			  'save_freq': 20,
			  'disp': True,
			  'threshold': 0.5,
			  'use_depth': True,
			  'transform': True}

	trainer = Train(params)
	trainer.train()