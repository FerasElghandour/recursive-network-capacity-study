"""Train the study's three tree models on generated inputs, without dataset downloads.

This demonstrates optimization and topology; it is not a held-out benchmark.
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from run_grid import Tree, counts


def run(epochs=40, seed=0):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(seed + 123)
    x = torch.randn(256, 16, generator=generator)
    y = (x[:, 0] * x[:, 1] + .5 * x[:, 4] - .3 * x[:, 10] > 0).long()
    results = []
    for kind in ("balanced", "random", "skewed"):
        torch.manual_seed(seed)
        model = Tree((2,) * 8, 8, 2, kind, seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        initial = F.cross_entropy(model(x), y).item()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            final = F.cross_entropy(model(x), y).item()
            accuracy = (model(x).argmax(1) == y).float().mean().item()
        results.append({
            "topology": kind, "parameters": counts(model)["params"],
            "max_merge_depth": max(model.depths),
            "initial_loss": round(initial, 6), "final_loss": round(final, 6),
            "training_accuracy": round(accuracy, 6),
        })
    return {"scope": "synthetic training demonstration; no held-out evaluation",
            "samples": 256, "feature_groups": 8, "features": 16,
            "epochs": epochs, "seed": seed, "device": "cpu", "models": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.epochs, args.seed), indent=2))
