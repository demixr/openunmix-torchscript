import torch
import os
from torch.utils.mobile_optimizer import optimize_for_mobile

from openunmix import utils
from openunmix import model
from openunmix.model import OpenUnmix

target_urls_umxhq = {
    "bass": "https://zenodo.org/api/files/1c8f83c5-33a5-4f59-b109-721fdd234875/bass-8d85a5bd.pth",
    "drums": "https://zenodo.org/api/files/1c8f83c5-33a5-4f59-b109-721fdd234875/drums-9619578f.pth",
    "other": "https://zenodo.org/api/files/1c8f83c5-33a5-4f59-b109-721fdd234875/other-b52fbbf7.pth",
    "vocals": "https://zenodo.org/api/files/1c8f83c5-33a5-4f59-b109-721fdd234875/vocals-b62c91ce.pth",
}

target_urls_umxl = {
    "bass": "https://zenodo.org/api/files/f8209c3e-ba60-48cf-8e79-71ae65beca61/bass-2ca1ce51.pth",
    "drums": "https://zenodo.org/api/files/f8209c3e-ba60-48cf-8e79-71ae65beca61/drums-69e0ebd4.pth",
    "other": "https://zenodo.org/api/files/f8209c3e-ba60-48cf-8e79-71ae65beca61/other-c8c5b3e6.pth",
    "vocals": "https://zenodo.org/api/files/f8209c3e-ba60-48cf-8e79-71ae65beca61/vocals-bccbd9aa.pth",
}


def get_umx_models(
    target_urls, hidden_size=512, targets=None, device="cpu", pretrained=True
):
    """Download openunmix pretrained models

    Args:
        target_urls: dict with the link to download the model for bass, drums, other and vocals
        hidden_size: size for bottleneck layer
        targets: list of stems
        device: the device on which the model will be used
        pretrained: boolean for pretrained weights

    Returns:
        target_models: list with all the models
    """
    if targets is None:
        targets = ["vocals", "drums", "bass", "other"]

    # determine the maximum bin count for a 16khz bandwidth model
    max_bin = int(utils.bandwidth_to_max_bin(rate=44100.0, n_fft=4096, bandwidth=16000))

    target_models = {}
    for target in targets:
        # load open unmix model
        target_unmix = OpenUnmix(
            nb_bins=4096 // 2 + 1,
            nb_channels=2,
            hidden_size=hidden_size,
            max_bin=max_bin,
        )

        # enable centering of stft to minimize reconstruction error
        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(
                target_urls[target], map_location=device
            )
            target_unmix.load_state_dict(state_dict, strict=False)
            target_unmix.eval()

        target_unmix.to(device)
        target_models[target] = target_unmix
    return target_models


def create_separator(target_models, device="cpu"):
    """Create separator class which contains all models

    Args:
        target_models: list of all models
        device: the device on which the model will be used

    Returns:
        separator: separator class which contains all models
    """
    separator = (
        model.Separator(
            target_models=target_models,
            niter=1,
            residual=False,
            n_fft=4096,
            n_hop=1024,
            nb_channels=2,
            sample_rate=44100.0,
            filterbank="asteroid",
        )
        .eval()
        .to(device)
    )

    return separator

def quantize_model(model):
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model, inplace=True)

    torch.quantization.convert(model, inplace=True)

def create_script(model_name, separator):
    """Create the torchscript model from a separator

    Args:
        model_name: name of the torchscript file to create
        separator: separator class which contains all models
    """
    jit_script = torch.jit.script(separator)
    torchscript_model_opti = optimize_for_mobile(jit_script)
    torchscript_model_opti._save_for_lite_interpreter(f"dist/{model_name}.ptl")


def main():
    device = "cpu"

    separator_umxhq = create_separator(get_umx_models(target_urls_umxhq), device=device)
    separator_umxl = create_separator(
        get_umx_models(target_urls_umxl, hidden_size=1024), device=device
    )

    if not os.path.exists("dist"):
        os.mkdir("dist")

    quantize_model(separator_umxhq)
    quantize_model(separator_umxl)

    create_script("umxhq", separator_umxhq)
    create_script("umxl", separator_umxl)


if __name__ == "__main__":
    main()
