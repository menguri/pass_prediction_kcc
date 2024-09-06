import argparse
import json
import os
import time
import ast

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from src.dataset import IntendedReceiverDataset
# from src.model import Model
#from src.new_model import DualTransformer
from src.receiver_new_model import DualTransformer
from src.utils import calc_trace_dist, get_params_str, num_trainable_params
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Modified from https://github.com/ezhan94/multiagent-programmatic-supervision/blob/master/train.py


# Helper functions
def printlog(line):
    print(line)
    with open(save_path + "/log.txt", "a") as file:
        file.write(line + "\n")

def loss_str(losses: dict):
    ret = ""
    for metric, value in losses.items():
        if not metric.startswith("type"):
            ret += " {}: {:.4f} |".format(metric, value)
    return ret[:-2]


# For one epoch
def run_epoch(
    model: nn.DataParallel,
    optimizer: torch.optim.Adam,
    mode,
    print_every=50,
    epoch=0,
):
    log_interval = 20
    total_loss = 0
    # torch.autograd.set_detect_anomaly(True)

    if mode == "train":
        loader = train_loader
        model.train()
    elif mode == "valid":
        loader = valid_loader
        model.eval()
    elif mode == "success_test":
        loader = success_test_loader
        model.eval()
    # receiver의 end-location은 실패한 패스에 대해서 정답정보가 없으므로 사용불가능
    # elif mode == "unsuccess_test":
    #     loader = unsuccess_test_loader
    #     model.eval()
    else:
        print("mode error : ", mode)
        exit()

    loss_dict = {"mse_loss": 0}

    for batch_idx, data in enumerate(loader):
        passer_input = data[0].to(default_device)
        teammate_input = data[1].to(default_device)
        opponent_input = data[2].to(default_device)
        ball_input = data[3].to(default_device)

        team_poss_input = data[4].to(default_device)
        event_player_input = data[5].to(default_device)

        passer_mask_list = data[6].to(default_device)
        target = data[7].to(default_device)
        target_loc = data[8].to(default_device)

        input = [
            passer_input,
            teammate_input,
            opponent_input,
            ball_input,
            team_poss_input,
            event_player_input,
            passer_mask_list,
        ]

        # *input : *은 unpakcing연산자로 각 요소(sprinter_input,teammate_input...)를 개별적인 인자로 전달
        if mode == "train":
            out = model(*input)
        else:
            with torch.no_grad():
                out = model(*input)

        # CrossEntropyLoss에는 log_softmax 연산이 내부에서 수행됩니다.
        # 따라서 CrossEntropyLoss의 input으로는 model의 원본 output으로 사용해도 된다(softmax취할 필요가 없다는 뜻)
        # batch size(32)개의 예측값-정답값에 대한 각각의 loss값을 가지는데, nn.CrossEntropyLoss는 batch sioze개의 loss값들을 모두 더해서 batch size로 나눠줌
        # reduction = mean(default) or sum :  평균으로 할 시 서로 다른 batch를 가질 때, 문제가 발생함
        #loss = nn.CrossEntropyLoss()(out, target)

        #학습이 끝난 후 사용함
        loss = nn.MSELoss(reduction = "sum")(out, target_loc)  # MSE loss

        #특정 batch에서 backwaring할 때 사용함
        batch_loss = loss / len(data)

        loss_dict["mse_loss"] += loss.item()

        if mode == "train":
            optimizer.zero_grad()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.module.parameters(), clip)
            optimizer.step()

        total_loss += batch_loss.item()
        # batch별 loss변화율 파악
        if (batch_idx + 1) % log_interval == 0:
            file_name = f"loss/{mode}_loss"

            writer.add_scalar(file_name, total_loss / log_interval, batch_idx + epoch * len(loader))

            total_loss = 0

    # run_epoch에서 batch단위로 수행된 모든 loss,accuracy를 평균화함
    # 해당 epoch에서 수행된 loss,accuracy가 출력됨
    for key, value in loss_dict.items():
        loss_dict[key] = value / len(loader.dataset)

    return loss_dict

# Main starts here
parser = argparse.ArgumentParser()

parser.add_argument("-t", "--trial", type=int, required=True)
parser.add_argument("--model", type=str, required=True, default="transformer")

parser.add_argument("--n_epochs", type=int, required=False, default=200, help="num epochs")
parser.add_argument("--batch_size", type=int, required=False, default=32, help="batch size")
parser.add_argument(
    "--start_lr",
    type=float,
    required=False,
    default=0.0001,
    help="starting learning rate",
)
parser.add_argument("--min_lr", type=float, required=False, default=0.0001, help="minimum learning rate")
parser.add_argument("--clip", type=int, required=False, default=10, help="gradient clipping")
parser.add_argument(
    "--print_every_batch",
    type=int,
    required=False,
    default=50,
    help="periodically print performance",
)
parser.add_argument(
    "--save_every_epoch",
    type=int,
    required=False,
    default=10,
    help="periodically save model",
)

parser.add_argument(
    "--pretrain_time",
    type=int,
    required=False,
    default=0,
    help="num epochs to train macro policy",
)

parser.add_argument(
    "--pretrain", type=int, required=False, default=0, help="success pass pretrained model"
)

parser.add_argument("--seed", type=int, required=False, default=128, help="PyTorch random seed")

parser.add_argument("--add_info", action="store_true", default=False, help="use dist_with_ball and angle_with_ball")

parser.add_argument("--cuda", action="store_true", default=False, help="use GPU")
parser.add_argument(
    "--cont",
    action="store_true",
    default=False,
    help="continue training previous best model",
)
parser.add_argument("--best_loss", type=float, required=False, default=0, help="best loss")

args, _ = parser.parse_known_args()

now = datetime.now()
schedule = f"model={args.model}, trial={args.trial}"
name = f"runs/now={schedule}"
writer = SummaryWriter(name)

if __name__ == "__main__":
    args.cuda = torch.cuda.is_available()
    default_device = "cuda:0"

    # Parameters to save
    params = {
        "model": args.model,
        "batch_size": args.batch_size,
        "start_lr": args.start_lr,
        "min_lr": args.min_lr,
        "seed": args.seed,
        "add_info": args.add_info,
        "cuda": args.cuda,
        "pretrain" : args.pretrain,
    }

    # Hyperparameters
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    clip = args.clip
    print_every = args.print_every_batch
    save_every = args.save_every_epoch
    pretrain_time = args.pretrain_time
    best_loss = args.best_loss
    
    # Set manual seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load model
    model = nn.DataParallel(DualTransformer(params, parser).to(default_device))

    # Update params with model parameters
    params = model.module.params
    params["total_params"] = num_trainable_params(model)

    # Create save path and saving parameters
    save_path = "saved/{:03d}".format(args.trial)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "/model")
    with open(f"{save_path}/params.json", "w") as f:
        json.dump(params, f, indent=4)
    
    # Continue a pretrained model experiment
    if args.pretrain != 0:
        printlog("Pretrained Model loading {}".format(args.pretrain))
        pretrain_path = "saved/{:03d}".format(args.pretrain)
        state_dict = torch.load(
            "{}/model/{}_state_dict_best.pt".format(pretrain_path, args.model)
        )

        # pretrained_dict = state_dict
        # model_dict = model.module.state_dict()
        
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}

        # model_dict.update(pretrained_dict)
        # model.module.load_state_dict(model_dict)
        
        model.module.load_state_dict(state_dict)
    
    # for name, param in model.named_parameters():
    #     if "output_fc_xy" in name:
    #         print(name)
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    
    printlog(f"{args.trial} {args.model}")
    printlog(model.module.params_str)
    printlog("n_params {:,}".format(params["total_params"]))
    printlog("############################################################")

    print()
    print("Generating success pass datasets...")

    dataset = pd.read_csv("../data/EPV-data/all-match.csv")
    dataset = dataset[dataset["eventName"] == "Pass"]
    dataset["freeze_frame"] = dataset["freeze_frame"].apply(ast.literal_eval)
    dataset["Baseline Intended-Receiver"] = dataset["Baseline Intended-Receiver"].apply(ast.literal_eval)

    dataset = dataset[
        (dataset["eventName"] == "Pass") & (dataset["accurate"] == 1) & (dataset["start_frame"] < dataset["end_frame"])
    ]

    test_dataset = pd.read_csv(f"./saved/022/success_test_dataset.csv")
    train_dataset = dataset[~dataset["event_id"].isin(test_dataset["event_id"])]

    train_dataset, valid_dataset = train_test_split(train_dataset, test_size=0.1)

    print(
        "before IntendedReceiverDataset : ",
        train_dataset.shape,
        valid_dataset.shape,
        test_dataset.shape,
    )
    
    train_dataset = IntendedReceiverDataset(train_dataset, flip_pitch=False, augment=False)
    valid_dataset = IntendedReceiverDataset(valid_dataset)
    test_dataset = IntendedReceiverDataset(test_dataset)

    print(
        "after IntendedReceiverDataset : ",
        train_dataset.passer_inputs.shape,
        valid_dataset.passer_inputs.shape,
        test_dataset.passer_inputs.shape,
    )

    nw = len(model.device_ids) * 4
    # valid, test는 평가용이므로 shuffle할 필요가 없음

    #sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        #sampler=sampler,
        num_workers=nw,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )
    success_test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )

    # Train loop
    epochs_since_best = 0
    lr = max(args.start_lr, args.min_lr)

    for e in range(n_epochs):
        epoch = e + 1

        # Set a custom learning rate schedule
        if epochs_since_best == 20 and lr > args.min_lr:
            # Load previous best model
            path = "{}/model/{}_state_dict_best.pt".format(save_path, args.model)
            state_dict = torch.load(path)

            # Decrease learning rate
            lr = max(lr * 0.5, args.min_lr)
            printlog("########## lr {} ##########".format(lr))
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        # Remove parameters with requires_grad=False (https://github.com/pytorch/pytorch/issues/679)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)

        start_time = time.time()
        printlog("\nEpoch {:d}".format(epoch))

        train_losses = run_epoch(model, optimizer, mode="train", print_every=print_every, epoch=e)
        printlog("Train:\t" + loss_str(train_losses))
        
        valid_losses = run_epoch(model, optimizer, mode="valid")
        printlog("Valid:\t" + loss_str(valid_losses))

        test_losses = run_epoch(model, optimizer, mode="success_test")
        printlog("Success Test:\t" + loss_str(test_losses))

        epoch_time = time.time() - start_time
        printlog("Time:\t {:.2f}s".format(epoch_time))

        # test_loss = sum([value for key, value in test_losses.items() if key.endswith("loss")])
        train_loss = train_losses["mse_loss"]
        valid_loss = valid_losses["mse_loss"]

        # Best model on test set
        if best_loss == 0 or valid_loss < best_loss:
            best_loss = valid_loss
            epochs_since_best = 0

            path = "{}/model/{}_state_dict_best.pt".format(save_path, args.model)
            if epoch <= pretrain_time:
                path = "{}/model/{}_state_dict_best_pretrain.pt".format(save_path, args.model)
            torch.save(model.module.state_dict(), path)
            printlog("########### Best Loss ###########")

        # Periodically save model
        if epoch % save_every == 0:
            path = "{}/model/{}_state_dict_{}.pt".format(save_path, args.model, epoch)
            torch.save(model.module.state_dict(), path)
            printlog("########## Saved Model ##########")

    printlog("Best Valid Loss: {:.4f}".format(best_loss))