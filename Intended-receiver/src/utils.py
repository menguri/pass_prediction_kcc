import torch

def num_trainable_params(model):
    total = 0
    for p in model.parameters():
        count = 1
        for s in p.size():
            count *= s
        total += count
    return total


def parse_model_params(model_args, params, parser):
    if parser is None:
        return params

    for arg in model_args:
        if arg.startswith("n_") or arg.endswith("_dim"):
            parser.add_argument("--" + arg, type=int, required=True)
        elif arg == "dropout":
            parser.add_argument("--" + arg, type=float, default=0)
        else:
            parser.add_argument("--" + arg, action="store_true", default=False)
    args, _ = parser.parse_known_args()

    for arg in model_args:
        params[arg] = getattr(args, arg)

    return params


def get_params_str(model_args, params):
    ret = ""
    for arg in model_args:
        if arg in params:
            ret += " {} {} |".format(arg, params[arg])
    return ret[1:-2]


def calc_speed(xy):
    x = xy[:, :, 0]
    y = xy[:, :, 1]
    vx = torch.diff(x, prepend=x[:, [0]]) / 0.1
    vy = torch.diff(y, prepend=y[:, [0]]) / 0.1
    speed = torch.sqrt(vx**2 + vy**2 + torch.tensor(1e-6).to(xy.device))
    return torch.stack([x, y, speed], -1)


def calc_real_loss(pred_xy, input, n_features=6, eps=torch.tensor(1e-6), aggfunc="mean"):
    eps = eps.to(pred_xy.device)
    if len(pred_xy.shape) == 2:
        pred_xy = pred_xy.unsqueeze(0).clone()
    if len(input.shape) == 2:
        input = input.unsqueeze(0).clone()

    # Calculate the angle between two consecutive velocity vectors
    # We skip the division by time difference, which is eventually reduced
    vels = pred_xy.diff(dim=1)
    speeds = torch.linalg.norm(vels, dim=-1)
    cos_num = torch.sum(vels[:, :-1] * vels[:, 1:], dim=-1) + eps
    cos_denom = speeds[:, :-1] * speeds[:, 1:] + eps
    cosines = torch.clamp(cos_num / cos_denom, -1 + eps, 1 - eps)
    angles = torch.acos(cosines)

    # Compute the distance between the ball and the nearest player
    pred_xy = torch.unsqueeze(pred_xy, dim=2)
    player_x = input[:, :, 0 : n_features * 22 : n_features]
    player_y = input[:, :, 1 : n_features * 22 : n_features]
    player_xy = torch.stack([player_x, player_y], dim=-1)
    ball_dists = torch.linalg.norm(pred_xy - player_xy, dim=-1)
    nearest_dists = torch.min(ball_dists, dim=-1).values[:, 1:-1]

    # Either course angle must be close to 0 or the ball must be close to a player
    if aggfunc == "mean":
        return (torch.tanh(angles) * nearest_dists).mean()
    else:  # if aggfunc == "sum"
        return (torch.tanh(angles) * nearest_dists).sum()


def calc_trace_dist(pred_xy, target_xy, aggfunc="mean"):
    if aggfunc == "mean":
        return torch.norm(pred_xy - target_xy, dim=-1).mean().item()
    else:  # if aggfunc == "sum":
        return torch.norm(pred_xy - target_xy, dim=-1).sum().item()


def calc_class_acc(pred_poss, target_poss, aggfunc="mean"):
    if aggfunc == "mean":
        return (torch.argmax(pred_poss, dim=1) == target_poss).float().mean().item()
    else:  # if aggfunc == "sum":
        return (torch.argmax(pred_poss, dim=1) == target_poss).float().sum().item()
