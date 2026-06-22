import torch
import torch.nn as nn

from .features import FEATURE_DIM
from .paths import ensure_parent_dir

ACCEPT, REGEN, ESCALATE = 0, 1, 2
ACTION_NAMES = ["accept", "regen", "escalate"]
N_ACTIONS = 3


class GatingPolicy(nn.Module):
    """Tiny MLP mapping q-free features -> {accept, regenerate, escalate}."""

    def __init__(self, dim=FEATURE_DIM, hidden=64, n_actions=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def act_greedy(self, feats):
        return int(self.net(feats).argmax(dim=-1))

    @torch.no_grad()
    def act_sample(self, feats):
        probs = torch.softmax(self.net(feats), dim=-1)
        return int(torch.multinomial(probs, 1))


def save_policy(policy, path):
    ensure_parent_dir(path)
    torch.save(policy.state_dict(), path)


def load_policy(path, device="cpu"):
    policy = GatingPolicy().to(device)
    policy.load_state_dict(torch.load(path, map_location=device))
    policy.eval()
    return policy
