import torchio as tio
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]
import torch
import torch.nn as nn
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_rng_state = 42
torch.seed = 42
random.seed(42)

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from resnets.densenet import densenet121
from resnets import resnet, resnext, wide_resnet, pre_act_resnet, densenet
from resnets.resnet import resnet10, resnet101

from datetime import datetime
from time import time
import torch.nn.functional as F
import json
import pickle
import numpy as np
from multi_slice_viewer.multi_slice_viewer import seg_viewer
from multi_slice_viewer.slice_viewer import multi_slice_viewer

from data_new.master_list import slist


class Run():
    def __init__(self, modeln="MyModel", val_length=10, batch_size=2, classifications_file="classifications.pkl",
                 learning_rate=3e-2):

        # SAMPLE FOR VALIDATION AND TEST SETS
        self.val_length = val_length  # , self.test_length = 20, 0
        self.batch_size = batch_size
        self.classifications_file = classifications_file
        self.lr = learning_rate
        self.transform = self._init_transform()

        n_train = 90
        self.ltrain = random.sample(slist, n_train)
        self.lval = [x for x in slist if x not in self.ltrain]

        # self.ltrain = slist[1:5]
        # self.lval = slist[1:5]

        print(len(self.ltrain))
        print(len(self.lval))

        self.train_loader = self._init_loader(self.ltrain, None) #self.transform
        self.val_loader = self._init_loader(self.lval, None)
        # self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)


        self.writer = SummaryWriter()
        self.model = self._init_model(resnet101)  # resnet10
        self.model = self.model.to(device)
        self.act_fn = torch.nn.Sigmoid().to(device)

        # self.loss_ce = nn.BCELoss()
        self.loss_ce = nn.BCEWithLogitsLoss()  # NIMAMO SIGMOIDA! naj bi se BCE loss uporabljal z sigmoidom, ne pa z softmax!!
        self.loss_ce = self.loss_ce.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-5)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, verbose=True)


        # momentum=0.9)  # works better?
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)  # weight_decay=5e-3, momentum=0.9)
        self.global_step = 0
        self.val_top_loss = 1e5
        self.train_top_loss = 1e5

    def __(self):
        for sample in self.train_loader:
            CT, PET = sample['CT'][tio.DATA], sample['PET'][tio.DATA]
            seg_viewer(CT[0, 0], PET[0, 0])
            break
        return

    def _init_transform(self):
        rescale = tio.RescaleIntensity(percentiles=(3, 97))  # (0.5, 99.5)
        spatial = tio.OneOf({
            tio.RandomAffine(): 0.2,
            tio.RandomElasticDeformation(): 0.1,
        },
            p=0.75,
        )
        transforms = [rescale, spatial]
        transform = tio.Compose(transforms)
        return transform

    def _init_loader(self, subject_list, transform):
        subjects_dataset = tio.SubjectsDataset(subject_list, transform=transform)
        # sampler = tio.data.UniformSampler(patch_size=(160, 160, 45))
        # queue_len = 20
        # patches_queue = tio.data.Queue(
        #     subjects_dataset,
        #     queue_len,
        #     samples_per_volume=10,  # how many patches from each vol
        #     sampler=sampler,
        #     num_workers=4
        # )
        loader = DataLoader(subjects_dataset, batch_size=4)#patches_queue
        return loader

    @staticmethod
    def _init_model(model):
        return model(num_classes=5)

    # TODO: implement forward function
    def forward(self, sample, train=True):
        tng = torch.no_grad()
        if train:
            self.optimizer.zero_grad()
        else:
            tng.__enter__()
        CT, PET = sample['CT'][tio.DATA], sample['PET'][tio.DATA]
        CT = CT.to(device).float()
        PET = PET.to(device).float()
        inp = torch.cat([CT, PET], dim=1)
        inp = inp.to(device)
        label = sample['label'].float().to(device)
        otpt = self.model(inp)
        loss = self.loss_ce(otpt, label)
        if train:
            loss.backward()
            self.optimizer.step()
        else:
            tng.__exit__()
        return loss

    def epoch_train(self):
        self.model = self.model.train()
        epoch_loss = 0
        c = 0
        for sample in self.train_loader:
            loss = self.forward(sample, True)
            epoch_loss += loss.sum().detach().cpu().numpy()
            c += 1
            # print(f"train loss {loss}")
        epoch_loss /= c
        self.writer.add_scalar("train_loss", epoch_loss, global_step=self.global_step)
        return epoch_loss

    def epoch_val(self):
        self.model = self.model.eval() # it breaks the model WHEN TRAINING LOOP WITH FEW ITERATIONS!!!!!!
        epoch_loss = 0
        c = 0
        for sample in self.val_loader:  #
            loss = self.forward(sample, True)
            epoch_loss += loss.sum().detach().cpu().numpy()
            c += 1
            # print(f"val loss {loss}")
        epoch_loss /= c

        self.writer.add_scalar("val_loss", epoch_loss, global_step=self.global_step)
        return epoch_loss

    def train(self, no_epochs=10):

        for i in range(no_epochs):
            t0 = time()
            self.global_step += 1
            tr = self.epoch_train()
            val = self.epoch_val()
            self.scheduler.step(val)
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

            # MAKE NEW LOADER AFTER EACH EPOCH:
            self.train_loader = self._init_loader(self.ltrain, None)
            self.val_loader = self._init_loader(self.lval, None)  # (0.5, 99.5)

        self.writer.close()



from pprint import pprint


def learning_rate_search(run_obj: Run, steps=100) -> list:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(run_obj.optimizer, milestones=list(range(1, 1000)), gamma=0.5)
    lr_list = list()
    for i in range(steps):
        for param_group in run_obj.optimizer.param_groups:
            lr = param_group["lr"]
        loss = run_obj.epoch_train()
        scheduler.step()
        lr_list.append((lr, loss))
        print((lr, loss))
    return lr_list


if __name__ == '__main__':
    r = Run(learning_rate=0.001)
    r.train(100)
