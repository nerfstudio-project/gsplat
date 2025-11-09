import torch

from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
import argparse
import json


def main(load_config: Path, output_dir: Path):

    _, pipeline, _, _ = eval_setup(load_config, test_mode="inference")
    splatfacto_model = pipeline.model

    means3d = splatfacto_model.means.cpu().detach().numpy()
    scales3d = torch.exp(splatfacto_model.scales).cpu().detach().numpy()
    quats = (
        (splatfacto_model.quats / splatfacto_model.quats.norm(dim=-1, keepdim=True))
        .cpu()
        .detach()
        .numpy()
    )

    colors_features = torch.cat(
        (
            splatfacto_model.features_dc[:, None, :],
            splatfacto_model.features_rest,
        ),
        dim=1,
    )
    colors = torch.sigmoid(colors_features[:, 0, :]).cpu().detach().numpy()
    opacities = torch.sigmoid(splatfacto_model.opacities).cpu().detach().numpy()

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "embedding.json", "w") as f:
        linear_bias = (
            (
                splatfacto_model.mlp_head.weight[:, 3:]
                @ splatfacto_model.embedding_appearance.mean()
                + splatfacto_model.mlp_head.bias
            )
            .cpu()
            .detach()
            .numpy()
            .tolist()
        )
        linear_weight = splatfacto_model.mlp_head.weight[:, :3].cpu().detach().numpy().reshape(-1).tolist()
        json.dump({"linearWeight": linear_weight, "linearBias": linear_bias}, f)

    pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(output_dir / "dataparser_transforms.json")
    means3d.astype("float32").tofile(output_dir / "means3d.bin")
    scales3d.astype("float32").tofile(output_dir / "scales3d.bin")
    quats.astype("float32").tofile(output_dir / "quats.bin")
    colors.astype("float32").tofile(output_dir / "colors.bin")
    opacities.astype("float32").tofile(output_dir / "opacities.bin")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--load_config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    args = parser.parse_args().__dict__

    main(**args)