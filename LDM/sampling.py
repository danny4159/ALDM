import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import nibabel as nib
import pytorch_lightning as pl
from monai.inferers import sliding_window_inference

from omegaconf import OmegaConf
from pytorch_lightning.trainer import Trainer

from ldm.util import instantiate_from_config

# ensure local modules (e.g., taming) are on path when running as a script
sys.path.append(os.getcwd())


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="ddim steps",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="output directory for saved predictions (overrides config outdir)",
    )
    return parser



def get_affine(src_path, affine_override=None):
    # preserve orientation/spacing; prefer provided affine (e.g., from MetaTensor)
    if affine_override is not None:
        return affine_override
    return nib.load(src_path).affine

def save_nii(img, name, path, src_path, affine_override=None):
    img = np.transpose(img, (1, 2, 3, 0))

    affine = get_affine(src_path, affine_override)
    nifti_img = nib.Nifti1Image(img, affine)
    nib.save(nifti_img, f"{path}/{name}.nii.gz")

def maybe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

if __name__ == "__main__":
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    if opt.outdir:
        config["outdir"] = opt.outdir

    # model
    model = instantiate_from_config(config.model)
    model.eval()
    model.to("cuda")

    sw_cfg = config.get("inference", {}).get("sliding_window", {})
    sw_enabled = sw_cfg.get("enabled", False)

    # when sliding window is on, avoid random cropping so full volumes are kept
    if sw_enabled:
        # enforce batch_size=1 to avoid stacking volumes of different shapes
        try:
            config.data.params.batch_size = 1
        except Exception:
            pass
        for split in ("validation", "test"):
            try:
                ds_cfg = config.data.params[split]
                ds_params = ds_cfg.get("params", ds_cfg)
                ds_params["apply_rand_crop"] = False
                if sw_cfg.get("disable_foreground_crop", True):
                    ds_params["apply_foreground_crop"] = False
                # keep original volume size by skipping pad/crop if desired
                if sw_cfg.get("disable_spatial_pad", True):
                    ds_params["spatial_size"] = None
            except Exception:
                pass

    try:
        test_params = config.data.params.test.get("params", config.data.params.test)
        print(f"[sampling] sw_enabled={sw_enabled}, test spatial_size={test_params.get('spatial_size', None)}, "
              f"apply_rand_crop={test_params.get('apply_rand_crop', None)}, "
              f"apply_foreground_crop={test_params.get('apply_foreground_crop', None)}")
    except Exception:
        pass

    # data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    test_loader = data.test_dataloader()
    # respect outdir override if provided via CLI (merged into config)
    outdir = config.get("outdir", None)
    if outdir is None:
        outdir = "results/" + opt.base[0].split("/")[-1].split(".")[0]
    root_path = outdir
    first_log = True
    for batch in test_loader:
        src_path = batch["path"][0]
        tgt_path = batch.get("target_path", [src_path])[0]
        subject = batch["subject_id"][0]
        print("processing...", subject)

        sub_path = os.path.join(root_path, subject)
        maybe_mkdir(sub_path)

        if sw_enabled:
            roi_size = tuple(sw_cfg.get("roi_size", config.data.params.test.spatial_size))
            sw_batch_size = sw_cfg.get("batch_size", 1)
            overlap = float(sw_cfg.get("overlap", 0.25))
            mode = sw_cfg.get("mode", "gaussian")
            padding_mode = sw_cfg.get("padding_mode", "constant")
            cval = sw_cfg.get("cval", sw_cfg.get("constant_values", 0.0))
            y_cls = batch["target_class"].to(model.device)

            def _predict(x):
                x = x.to(model.device)
                bs = x.shape[0]
                patch_batch = {
                    "source": x,
                    "target": x,  # dummy placeholder; target is not used during sampling
                    "target_class": y_cls[:bs],
                }
                log = model.log_images(patch_batch, N=bs, sample=True, ddim_steps=opt.ddim_steps)
                return log["samples_x0_quantized"]

            sw_kwargs = dict(
                inputs=batch["source"],
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=_predict,
                overlap=overlap,
                mode=mode,
                sw_device=model.device,
                device=model.device,
                padding_mode=padding_mode,
            )
            if padding_mode == "constant":
                sw_kwargs["cval"] = cval
            recon = sliding_window_inference(**sw_kwargs)[0].detach().cpu().numpy()
        else:
            log = model.log_images(batch, ddim_steps=opt.ddim_steps)
            recon = log["samples_x0_quantized"][0].detach().cpu().numpy()
        # save pred, source, target with proper orientation
        src_affine = getattr(batch["source"][0], "affine", None)
        tgt_affine = getattr(batch["target"][0], "affine", None)
        save_nii(recon, "pred", sub_path, tgt_path, affine_override=tgt_affine)
        save_nii(batch["source"][0].detach().cpu().numpy(), "source", sub_path, src_path, affine_override=src_affine)
        save_nii(batch["target"][0].detach().cpu().numpy(), "target", sub_path, tgt_path, affine_override=tgt_affine)
        if first_log:
            print(f"[sampling] first batch shapes: source {tuple(batch['source'].shape)}, target {tuple(batch['target'].shape)}, recon {recon.shape}")
            first_log = False
