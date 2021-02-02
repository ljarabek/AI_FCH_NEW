import torch
import os
import pickle
from tqdm import tqdm
from INN_PRETRAIN import *


def gen_output(run, device_=device,
               model_file="/media/leon/2tbssd/PRESERNOVA/AI_FCH_NEW/run_test/abs_healthy/best_val.pth",
               output_folder="/media/leon/2tbssd/PRESERNOVA/AI_FCH_NEW/run_test/abs_healthy/pickles"):
    # run = torch.load("/media/leon/2tbssd/PRESERNOVA/AI_FCH/runs_unceirtanty_model/test_run.pth")
    # "/media/leon/2tbssd/PRESERNOVA/AI_FCH_NEW/run_test/ADAM_better/pickles/"
    run.model = torch.load(model_file)

    pickles_dir = output_folder
    os.makedirs(pickles_dir, exist_ok=False)

    ## This makes pickles containing CTs, PETS, as well as INN outputs.

    for ct, pet, merged, label, _ in tqdm(run.val_loader):
        wanted_info = ['ime', 'priimek', 'histo_lokacija', 'histologija', 'SGD/MGD', 'CT_dir', 'PET_dir']
        inp = torch.Tensor(ct.float())
        inp = inp.to(device_)
        target = torch.Tensor(pet.float()).to(device_)
        otpt = run.model(inp)
        loss = run.loss_ce(otpt, target)

        for i in range(2):  # RANGE(BATCH_SIZE!!)TODO: !!
            pkl = dict()
            lst = [_[info][i] for info in wanted_info]

            pkl['info'] = lst
            pkl['CT'] = ct[i, 0].detach().cpu().numpy()
            pkl['PET'] = pet[i, 0].detach().cpu().numpy()
            pkl['output'] = otpt[i].detach().cpu().numpy()
            unique_id = str(hash(str(lst)))[1:]
            with open(os.path.join(pickles_dir, unique_id), "wb") as f:
                pickle.dump(pkl, f)

    for ct, pet, merged, label, _ in tqdm(run.train_loader):
        wanted_info = ['ime', 'priimek', 'histo_lokacija', 'histologija', 'SGD/MGD', 'CT_dir', 'PET_dir']
        inp = torch.Tensor(ct.float())
        inp = inp.to(device_)
        target = torch.Tensor(pet.float()).to(device_)
        otpt = run.model(inp)
        loss = run.loss_ce(otpt, target)

        for i in range(2):  # RANGE(BATCH_SIZE!!)TODO: !!
            pkl = dict()
            lst = [_[info][i] for info in wanted_info]

            pkl['info'] = lst
            pkl['CT'] = ct[i, 0].detach().cpu().numpy()
            pkl['PET'] = pet[i, 0].detach().cpu().numpy()
            pkl['output'] = otpt[i].detach().cpu().numpy()
            unique_id = str(hash(str(lst)))[1:]
            with open(os.path.join(pickles_dir, unique_id), "wb") as f:
                pickle.dump(pkl, f)

    return


if __name__ == "__main__":
    #for i in range(1,4):
        model_dir = "/media/leon/2tbssd/PRESERNOVA/AI_FCH_NEW/run_test/v5_%s"%i
        #print(model_dir)
        dct_dir = os.path.join(model_dir, "settings.json")
        with open(dct_dir, "r") as f:
            dct = json.load(f)
        r = Run(dct=dct)
        with torch.no_grad():
            gen_output(r, model_file=model_dir + "/best_val.pth", output_folder=model_dir + "/pickles")
