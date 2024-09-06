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
from src.my_model import DualTransformer
from src.utils import calc_class_acc, get_params_str, num_trainable_params
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
    train=False,
    print_every=50,
    epoch=0,
):
    log_interval = 20
    total_loss = 0
    # torch.autograd.set_detect_anomaly(True)
    model.train() if train else model.eval()

    loader = train_loader if train else valid_loader
    loss_dict = {"ce_loss": [], "accuracy": []}

    for batch_idx, data in enumerate(loader):
        passer_input = data[0].to(default_device)
        teammate_input = data[1].to(default_device)
        opponent_input = data[2].to(default_device)
        ball_input = data[3].to(default_device)

        team_poss_input = data[4].to(default_device)
        event_player_input = data[5].to(default_device)

        input = [
            passer_input,
            teammate_input,
            opponent_input,
            ball_input,
            team_poss_input,
            event_player_input,
        ]

        passer_mask_list = data[6].to(default_device)
        target = data[7].to(default_device)

        # *input : *은 unpakcing연산자로 각 요소(sprinter_input,teammate_input...)를 개별적인 인자로 전달
        if train:
            out = model(*input)
        else:
            with torch.no_grad():
                out = model(*input)

        #out : (batch, teammate) : 각 batch별 마지막 시점의 passer과 같은 팀원 11명 중 한명을 classifier하는 문제
        batch_size, _ = out.shape
        mask = torch.zeros_like(out)
        batch_indices = np.arange(batch_size)
        mask[batch_indices, passer_mask_list] = -np.inf
        out += mask 

        # CrossEntropyLoss에는 log_softmax 연산이 내부에서 수행됩니다.
        # 따라서 CrossEntropyLoss의 input으로는 model의 원본 output으로 사용해도 된다(softmax취할 필요가 없다는 뜻)
        loss = nn.CrossEntropyLoss()(out, target)

        loss_dict["ce_loss"] += [loss.item()]
        loss_dict["accuracy"] += [calc_class_acc(out, target)]

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.module.parameters(), clip)
            optimizer.step()

        total_loss += loss.item()
        # batch별 loss변화율 파악
        if (batch_idx + 1) % log_interval == 0:
            file_name = file_name = "loss/train_loss" if train else "loss/valid_loss"

            writer.add_scalar(
                file_name, total_loss / log_interval, batch_idx + epoch * len(loader)
            )

            total_loss = 0

    # run_epoch에서 batch단위로 수행된 모든 loss,accuracy를 평균화함
    # 해당 epoch에서 수행된 loss,accuracy가 출력됨
    for key, value in loss_dict.items():
        loss_dict[key] = np.mean(value)

    return loss_dict


# Main starts here
parser = argparse.ArgumentParser()

parser.add_argument("-t", "--trial", type=int, required=True)
parser.add_argument("--model", type=str, required=True, default="transformer")

parser.add_argument(
    "--n_epochs", type=int, required=False, default=200, help="num epochs"
)
parser.add_argument(
    "--batch_size", type=int, required=False, default=32, help="batch size"
)
parser.add_argument(
    "--start_lr",
    type=float,
    required=False,
    default=0.0001,
    help="starting learning rate",
)
parser.add_argument(
    "--min_lr", type=float, required=False, default=0.0001, help="minimum learning rate"
)
parser.add_argument(
    "--clip", type=int, required=False, default=10, help="gradient clipping"
)
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
    "--seed", type=int, required=False, default=128, help="PyTorch random seed"
)

parser.add_argument(  
    "--add_info", action="store_true", default=False, help="use dist_with_ball and angle_with_ball"
)

parser.add_argument("--cuda", action="store_true", default=False, help="use GPU")
parser.add_argument(
    "--cont",
    action="store_true",
    default=False,
    help="continue training previous best model",
)
parser.add_argument(
    "--best_loss", type=float, required=False, default=0, help="best loss"
)
parser.add_argument(
    "--pretrain", type=int, required=False, default=0, help="success pass pretrained model"
)

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
        model.module.load_state_dict(state_dict)
    else:
        print("select pretrained model path")
        exit()

    # Freeze all layers except for the output layer
    for name, param in model.named_parameters():
        if "output_fc" not in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    printlog(f"{args.trial} {args.model}")
    printlog(model.module.params_str)
    printlog("n_params {:,}".format(params["total_params"]))
    printlog("############################################################")

    print()
    print("Generating unsuccess pass datasets...")

    dataset = pd.read_csv('../data/EPV-data/labeling-all-match.csv')
    dataset = dataset[(dataset["eventName"] == "Pass") & (dataset["accurate"] == 0) & (dataset['start_frame'] < dataset['end_frame'])]
    dataset = dataset[dataset['no pass'].isna()]
    dataset = dataset[dataset['True Intended-receiver'].notna()]
    dataset = dataset.drop(columns=['to'])
    dataset = dataset.rename(columns={'True Intended-receiver':'to'})

    #실패한 패스의 절반(367)개 정도만 사전학습에 사용할 예정
    #train : valid : test = 164 : 19 : 184
    #실패한 패스 일부만 사용하는게 목적        
    train_dataset, test_dataset  = train_test_split(dataset, test_size=0.5)
    train_dataset, valid_dataset = train_test_split(train_dataset , test_size=0.1)
    test_dataset.to_csv(f"./{save_path}/unsuccess_test_dataset.csv", index=False)

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

    sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        #sampler = sampler,
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
    test_loader = DataLoader(
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
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr
        )

        start_time = time.time()
        printlog("\nEpoch {:d}".format(epoch))

        train_losses = run_epoch(
            model, optimizer, train=True, print_every=print_every, epoch=e
        )
        printlog("Train:\t" + loss_str(train_losses))

        valid_losses = run_epoch(model, optimizer, train=False)
        printlog("Valid:\t" + loss_str(valid_losses))

        epoch_time = time.time() - start_time
        printlog("Time:\t {:.2f}s".format(epoch_time))

        # test_loss = sum([value for key, value in test_losses.items() if key.endswith("loss")])
        train_loss = train_losses["ce_loss"]
        valid_loss = valid_losses["ce_loss"]
        valid_accuracy = valid_losses["accuracy"]

        # Best model on test set
        if best_loss == 0 or valid_loss < best_loss:
            best_loss = valid_loss
            best_accuracy = valid_accuracy
            epochs_since_best = 0

            path = "{}/model/{}_state_dict_best.pt".format(save_path, args.model)
            if epoch <= pretrain_time:
                path = "{}/model/{}_state_dict_best_pretrain.pt".format(
                    save_path, args.model
                )
            torch.save(model.module.state_dict(), path)
            printlog("########### Best Loss ###########")

        # Periodically save model
        if epoch % save_every == 0:
            path = "{}/model/{}_state_dict_{}.pt".format(save_path, args.model, epoch)
            torch.save(model.module.state_dict(), path)
            printlog("########## Saved Model ##########")

    printlog(
        "Best Valid Loss: {:.4f} | Accuracy : {:.4f}".format(best_loss, best_accuracy)
    )
