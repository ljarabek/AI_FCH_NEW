import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from constants import *
from torch.utils.tensorboard import SummaryWriter
from data.dataset import PET_CT_Dataset, master_list, m_list_settings
from torch.utils.data import DataLoader
from resnets.densenet import densenet121
from resnets.resnet import resnet10, resnet50
from datetime import datetime
from time import time
import torch.nn.functional as F
from data.sampling import sample_by_label
import json
import random
import pickle
from argparse import ArgumentParser
from models.unet import UNet3D
from models.unet3d.model import ResidualUNet3D
# models.unet.U Net_alternative ALTERNATIVE HAS SIGMOID ACTIVATION!!!

from models import unet_128i
from models.my_models import MyModel
import numpy as np
from multi_slice_viewer.multi_slice_viewer import seg_viewer
from multi_slice_viewer.slice_viewer import multi_slice_viewer


# model_used = resnet10(num_classes=5, activation="softmax")
# model_used = MyModel


class Run():
    def __init__(self, dct):
        # SAMPLE FOR VALIDATION AND TEST SETS
        self.dct = dct
        self.val_length = dct['val_length']  # , self.test_length = 20, 0
        self.batch_size = dct['batch_size']
        self.classifications_file = dct['classifications_file']
        self.lr = dct['learning_rate']
        print("all %s" % len(master_list))
        sample = random.sample(range(0, len(master_list)), k=self.val_length)  # + self.test_length
        # sample  = sample_by_label(master_list, val_size=self.val_length, n_min=2)
        # self.test_list = [e for i, e in enumerate(master_list) if i in sample[self.val_length:]]
        self.val_list = [e for i, e in enumerate(master_list) if i in sample]
        self.train_list = [e for i, e in enumerate(master_list) if i not in sample]

        print("train length: %s \t val length: %s \t test length: " % (
            len(self.train_list), len(self.val_list)))  # , len(self.te)))

        self.train_dataset = PET_CT_Dataset(self.train_list)
        self.val_dataset = PET_CT_Dataset(self.val_list)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True,
                                       drop_last=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False,
                                     drop_last=False)
        if dct['log_dir'] is None:
            self.writer = SummaryWriter(log_dir="run_test/%s" % datetime.now().strftime(
                "%m%d%Y_%H:%M:%S"))  # TODO: dodaj tle notr folder z enkodiranim časom...
        else:
            self.writer = SummaryWriter(log_dir=dct['log_dir'])
        self.log_dir = self.writer.log_dir
        print("Writing to logdir %s" % self.log_dir)
        with open(os.path.join(self.log_dir, "settings.json"), "w") as f:
            json.dump(self.dct, f)

        self.modeln = dct['model_name']
        self.model = self._init_model(self.modeln)  # TODO self._init_model(model_name=self.modeln)
        self.model = self.model.to(device)

        self.loss_ce = nn.MSELoss()
        self.loss = self.INN_loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)  # weight_decay=5e-3, momentum=0.9)

        self.global_step = 0
        self.val_top_loss = 1e5
        self.train_top_loss = 1e5
        self.sig = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def INN_loss(self, otpt, real):  # 3 channel output - low, mid, max;; NCHWD format
        # real format NHWD (C=1)!
        version = self.dct['loss_version']
        tightness = self.dct['tightness']
        mean_loss_weight = self.dct['mean_loss_weight']
        base_loss_scalar = self.dct['base_loss_scalar']
        use_mid = self.dct['use_recon_loss']
        low = torch.clone(otpt[:, 0])
        if use_mid:
            mid = torch.clone(otpt[:, 1])
            mid = torch.unsqueeze(mid, dim=1).to(device)
        high = torch.clone(otpt[:, 2])
        zero = torch.zeros_like(real).to(device)

        if version == 'v3':
            outsiders_high = torch.max(real - high, other=zero)
            outsiders_low = torch.max(low - real, zero)
            loss = torch.pow(outsiders_high, exponent=2) + torch.pow(outsiders_low, exponent=2)
            # loss = self.tanh(loss).clone()
            print("Be4 tightness loss")
            print(loss.mean())
            # t_loss = torch.abs(high - low)

            # Selects values where
            loss = loss.mean()
            t_loss_high = torch.where(outsiders_high == 0, high, zero)
            t_loss_low = torch.where(outsiders_low == 0, low, zero)
            t_loss_ = t_loss_high - t_loss_low

            loss += (tightness * t_loss_).mean()
            print("after tightness loss")
            print(loss.mean())
            loss = loss.mean()
        elif version == 'v2':
            loss = torch.pow(torch.max(real - high, other=zero).to(device), exponent=2).to(device) + \
                   torch.pow(torch.max(low - real, zero), exponent=2)
            loss = self.tanh(loss).clone()  # torch.clone(loss)) TODO: OMG TO DELA!!
            loss *= base_loss_scalar
            print("Be4 tightness loss")
            print(loss.mean())
            loss += tightness * torch.abs(high - low)  # 0.01
            print("after tightness loss")
            print(loss.mean())
            print("")
            if use_mid:
                loss += mean_loss_weight * self.loss_ce(mid.double(), real.double())
            # print(loss.mean())
            loss = loss.mean()  # prej mean...
        elif version == "v1":
            loss = torch.pow(torch.max(real - high, other=zero).to(device), exponent=2).to(device) + \
                   torch.pow(torch.max(low - real, zero), exponent=2)
            # loss = self.tanh(loss).clone()  # torch.clone(loss)) TODO: OMG TO DELA!!
            loss *= base_loss_scalar
            print("Be4 tightness loss")
            print(loss.mean())
            loss += tightness * (high - low)
            print("after tightness loss")
            print(loss.mean())
            print("")
            if use_mid:
                loss += mean_loss_weight * self.loss_ce(mid.double(), real.double())
            # print(loss.mean())
            loss = loss.mean()  # prej mean...
        else:
            loss = None
            raise NotImplementedError("implement loss!")
        # print(loss)
        return loss

    def _init_model(self, model_name):
        if model_name.lower() == "mymodel":
            return MyModel(num_classes=5)
        if model_name.lower() == 'resnet10':
            return resnet10(num_classes=5, activation="softmax")
        if model_name.lower() == 'interval_nn':
            return UNet3D(in_channel=2, n_classes=6)
        if model_name.lower() == "leone___":
            return unet_128i.Simply3DUnet(num_in_channels=1, num_out_channels=3, depth=3, init_feature_size=32, bn=True)
        if model_name.lower() == "resunet3d":
            return ResidualUNet3D(1, 3, final_sigmoid=False, f_maps=64, is_segmentation=False,  # fmaps default 64
                                  num_levels=3)  # 3 output maps if use_mid is true
        else:
            return None

    def forward(self, *inputs):
        ct, pet, merged, label, entry = inputs
        inp = torch.Tensor(ct.float())
        inp = inp.to(device)  # no schema error!!
        label = label.to(device)
        target = torch.Tensor(pet.float()).to(device)
        self.model = self.model.to(device)
        otpt = self.model(inp)
        loss = self.loss(otpt.double(), target.double())
        return loss, otpt

    def epoch_train(self):
        self.model = self.model.train()
        epoch_loss = 0
        for ct, pet, merged, label, _ in self.train_loader:
            # if random.randint(0, 10) <= 7:
            #    continue  # skip step in 70%
            self.optimizer.zero_grad()
            loss, otpt = self.forward(ct, pet, merged, label, _)
            loss.backward()
            self.optimizer.step()
            print(loss)
            epoch_loss += loss.sum().detach().cpu()
        if epoch_loss == 0:
            print("no sampling train")
            epoch_loss = 1e9
        epoch_loss /= len(self.train_list)
        self.writer.add_scalar("train_loss", epoch_loss, global_step=self.global_step)
        return epoch_loss

    def epoch_val(self):
        self.model = self.model.eval()
        epoch_loss = 0
        log_txt = ""
        for ct, pet, merged, label, _ in self.val_loader:
            # if random.randint(0, 10) <= 7:
            #    continue
            with torch.no_grad():
                loss, otpt = self.forward(ct, pet, merged, label, _)
            epoch_loss += loss.sum().detach().cpu()
            log_txt += f'truth: \t{str(label.detach().cpu().numpy())} output: \t{str(otpt.detach().cpu().numpy())}\n'
        if epoch_loss == 0:
            print("no sampling on val")
            epoch_loss = 1e9
        epoch_loss /= len(self.val_list)
        # self.writer.add_text("val_", text_string=log_txt, global_step=self.global_step)
        self.writer.add_scalar("val_loss", epoch_loss, global_step=self.global_step)
        return epoch_loss

    def evaluate_classification(self):
        self.model = torch.load(os.path.join(self.writer.log_dir, "best_val.pth"))
        label_list = m_list_settings['encoding'][1]
        try:
            with open(self.classifications_file, "rb") as f:
                classifications = pickle.load(f)
        except:
            classifications = dict()
            classifications['val_loss'] = list()  # se itak požene na koncu, ko je že zoptimiziran..
            classifications['model_version'] = list()
            classifications['truth'] = dict()
            classifications['pred'] = dict()
            classifications['CT_dirs'] = list()
            classifications['PET_dirs'] = list()
            for l in label_list:
                classifications['truth'][l] = list()
                classifications['pred'][l] = list()

        self.model = self.model.eval()
        val_loss = 0
        for ct, pet, merged, label, entry in self.val_loader:
            loss, otpt = self.forward(ct, pet, merged, label, entry)
            val_loss += loss.sum().detach().cpu()
            otpt = otpt.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            for b in range(otpt.shape[0]):  # for example in batch
                classifications['model_version'].append(self.writer.log_dir)
                classifications['CT_dirs'].append(entry['CT_dir'])
                classifications['PET_dirs'].append(entry['PET_dir'])
                for il, l in enumerate(label_list):
                    classifications['truth'][l].append(label[b, il])
                    classifications['pred'][l].append(otpt[b, il])
        classifications['val_loss'].append(val_loss)

        with open(self.classifications_file, "wb") as f:
            pickle.dump(classifications, f)

    def train(self, no_epochs=10):
        for i in range(no_epochs):
            t0 = time()
            self.global_step += 1
            tr = self.epoch_train()
            val = self.epoch_val()
            self.writer.add_scalars(main_tag="losses", tag_scalar_dict={'train_loss': tr, "val_loss": val},
                                    global_step=self.global_step)
            if val < self.val_top_loss:
                torch.save(self.model, os.path.join(self.writer.log_dir, "best_val.pth"))
                self.val_top_loss = val
                print("saved_top_model_val")
            if tr < self.train_top_loss:
                torch.save(self.model, os.path.join(self.writer.log_dir, "best_tr.pth"))
                self.train_top_loss = tr
                print("saved_top_model_tr")

            print(f"STEP: {i} TRAINLOSS: {tr} VALLOSS {val} dt {time() - t0}")
        self.writer.close()


if __name__ == "__main__":
    cross_validation_fold = 12
    dct = {
        'model_name': "resunet3d",
        'log_dir': None,
        'batch_size': 2,
        'val_length': 4,
        'healthy_only': True,
        'learning_rate': 3e-2,
        'classifications_file': "sample.pkl",

        'loss_version': 'v3',  # LOSS PARAMETERS
        'tightness': 0.01,
        'mean_loss_weight': 0.1,
        'base_loss_scalar': 1,
        'use_recon_loss': False

    }
    if dct['healthy_only']:
        master_list = [d for d in master_list if
                       d['histo_lokacija'] == "healthy"]

    run = Run(dct)
    run.train(100)
