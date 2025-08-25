#!/usr/bin/env python3
import os
import sys
import torch


from tictacshoot.pytorch.NNet import NNetWrapper as NNWrapper
from tictacshoot.CustomTicTacToeGame import CustomTicTacToeGame as Game

# --- User settings ---
CHECKPOINT_FOLDER = '/pretrained_models/tictacshoot/pytorch/'
CHECKPOINT_FILE   = 'best.pth.tar' 
ONNX_OUT          = 'tictacshoot.onnx'
OPSET             = 17                   # 17+ is a good modern choice
USE_DYNAMO        = True                 # Set False to force TorchScript path


def build_model():
    """Constructs wrapper+game, loads weights, and returns the bare nn.Module and game."""
    game = Game()
    wrapper = NNWrapper(game)

    # Optional: load your trained weights
    ckpt_path = os.path.join(CHECKPOINT_FOLDER, CHECKPOINT_FILE)
    if os.path.isfile(ckpt_path):
        wrapper.load_checkpoint(CHECKPOINT_FOLDER, CHECKPOINT_FILE)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"No checkpoint found at {ckpt_path}. Exporting random-initialized weights.")

    model = wrapper.nnet
    model.eval()  # IMPORTANT: ensures dropout/batchnorm behave deterministically
    return model, game


def make_dummy_input(game):
    # Expecting a board tensor shaped (C, H, W); batch first when exporting
    C, H, W = game.getInitBoard().shape
    dummy = torch.zeros(1, C, H, W, dtype=torch.float32)
    return dummy


def export_with_dynamo(model, dummy):
    torch.onnx.export(
        model,
        (dummy,),
        ONNX_OUT,
        input_names=["input"],
        output_names=["policy_log_probs", "value"],
        dynamic_axes={
            "input": {0: "batch"},
            "policy_log_probs": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=True,           # <- TorchDynamo-based exporter (PyTorch 2.1+)
    )


def export_with_torchscript(model, dummy):
    # TorchScript-based exporter (works on older PyTorch or if Dynamo has issues)
    traced = torch.jit.trace(model, (dummy,))
    torch.onnx.export(
        traced,
        (dummy,),
        ONNX_OUT,
        input_names=["input"],
        output_names=["policy_log_probs", "value"],
        dynamic_axes={
            "input": {0: "batch"},
            "policy_log_probs": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )


if __name__ == "__main__":
    model, game = build_model()
    dummy = make_dummy_input(game)

    if USE_DYNAMO:
        print("Exporting with TorchDynamo-based exporter...")
        export_with_dynamo(model, dummy)
    else:
        print("Exporting with TorchScript-based exporter...")
        export_with_torchscript(model, dummy)

    print(f"Saved ONNX model to: {ONNX_OUT}")