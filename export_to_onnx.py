import os
import sys
import torch


from tictacshoot.pytorch.NNet import NNetWrapper as NNWrapper
from tictacshoot.CustomTicTacToeGame import CustomTicTacToeGame as Game

# --- User settings ---
CHECKPOINT_FOLDER = './pretrained_models/tictacshoot/pytorch'  # match your run output
CHECKPOINT_FILE   = 'best.pth.tar'
ONNX_OUT          = 'tictactoe.onnx'
OPSET             = 17


def build_model():
    """Constructs wrapper+game, loads weights, and returns the bare nn.Module and game."""
    game = Game()
    wrapper = NNWrapper(game)

    ckpt_path = os.path.join(CHECKPOINT_FOLDER, CHECKPOINT_FILE)
    if os.path.isfile(ckpt_path):
        wrapper.load_checkpoint(CHECKPOINT_FOLDER, CHECKPOINT_FILE)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"No checkpoint found at {ckpt_path}. Exporting random-initialized weights.")

    model = wrapper.nnet
    model.eval()
    return model, game


def make_dummy_input(game):
    # Expecting a board tensor shaped (C, H, W); batch first when exporting
    C, H, W = game.getInitBoard().shape
    dummy = torch.zeros(1, C, H, W, dtype=torch.float32)
    return dummy


def export_with_dynamo_if_available(model, dummy):
    """Use torch.onnx.dynamo_export when present; otherwise return False."""
    if hasattr(torch.onnx, 'dynamo_export'):
        print('Using torch.onnx.dynamo_export ...')
        # Optional dynamic shapes flag (batch dim stays flexible)
        try:
            from torch.onnx import ExportOptions
            ep = torch.onnx.dynamo_export(model, dummy, export_options=ExportOptions(dynamic_shapes=True))
        except Exception:
            # Fallback without options if older signature
            ep = torch.onnx.dynamo_export(model, dummy)
        ep.save(ONNX_OUT)
        return True
    return False


def export_with_classic_export(model, dummy):
    """Classic exporter (TorchScript-style tracing under the hood)."""
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
    )


if __name__ == "__main__":
    model, game = build_model()
    dummy = make_dummy_input(game)

    used_dynamo = export_with_dynamo_if_available(model, dummy)
    if not used_dynamo:
        print('torch.onnx.dynamo_export not found; using classic torch.onnx.export ...')
        export_with_classic_export(model, dummy)

    print(f"Saved ONNX model to: {ONNX_OUT}")