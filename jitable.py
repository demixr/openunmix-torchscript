import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from openunmix import utils
from openunmix import model
from openunmix.model import OpenUnmix

def umxl_spec(targets=None, device="cpu", pretrained=True):

    # set urls for weights
    target_urls = {
        "bass": "https://zenodo.org/api/files/f8209c3e-ba60-48cf-8e79-71ae65beca61/bass-2ca1ce51.pth",
        "drums": "https://zenodo.org/api/files/f8209c3e-ba60-48cf-8e79-71ae65beca61/drums-69e0ebd4.pth",
        "other": "https://zenodo.org/api/files/f8209c3e-ba60-48cf-8e79-71ae65beca61/other-c8c5b3e6.pth",
        "vocals": "https://zenodo.org/api/files/f8209c3e-ba60-48cf-8e79-71ae65beca61/vocals-bccbd9aa.pth",
    }

    if targets is None:
        targets = ["vocals", "drums", "bass", "other"]

    # determine the maximum bin count for a 16khz bandwidth model
    max_bin = int(utils.bandwidth_to_max_bin(rate=44100.0, n_fft=4096, bandwidth=16000))

    target_models = {}
    for target in targets:
        # load open unmix model
        target_unmix = OpenUnmix(
            nb_bins=4096 // 2 + 1, nb_channels=2, hidden_size=1024, max_bin=max_bin
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

def main():
    niter = 1
    device = 'cpu'

    target_models = umxl_spec(device=device, pretrained=True)

    separator = (
        model.Separator(
            target_models=target_models,
            niter=niter,
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

    ok = torch.jit.script(separator)
    torchscript_model_opti = optimize_for_mobile(ok)
    torchscript_model_opti._save_for_lite_interpreter("openunmix.ptl")

if __name__ == "__main__":
    main()
