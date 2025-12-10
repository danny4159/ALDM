import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.losses.lpips import LPIPS
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 use_actnorm=False, disc_conditional=False,
                 disc_ndf=32, disc_loss="hinge", num_classes=1,
                 perceptual_weight=0.0, cycle_weight=0.0):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.num_classes = num_classes
        self.perceptual_weight = perceptual_weight
        self.cycle_weight = cycle_weight
        self.perceptual_loss = LPIPS().eval() if perceptual_weight > 0 else None
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf,
                                                 out_ch=num_classes
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self._nan_reported = False

    def _check_finite(self, tensor_dict, split, optimizer_idx):
        if self._nan_reported:
            return
        bad = []
        for name, tensor in tensor_dict.items():
            if tensor is None:
                continue
            t = tensor.detach()
            finite_mask = torch.isfinite(t)
            if not finite_mask.all():
                nan_count = (~finite_mask).sum().item()
                if finite_mask.any():
                    finite_vals = t[finite_mask]
                    tmin = finite_vals.min().item()
                    tmax = finite_vals.max().item()
                else:
                    tmin = float("nan")
                    tmax = float("nan")
                bad.append(f"{name}(nan_count={nan_count}, min={tmin}, max={tmax})")
        if bad:
            print(f"[NaN-Guard] split={split} opt_idx={optimizer_idx} non-finite components: {', '.join(bad)}")
            self._nan_reported = True

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, label=None, split="train",
                cycle_recon=None, cycle_target=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_loss is not None:
            perceptual_device = next(self.perceptual_loss.parameters()).device
            if perceptual_device != inputs.device:
                self.perceptual_loss = self.perceptual_loss.to(inputs.device)
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor(0.0, device=inputs.device)

        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        batch_size = inputs.shape[0]
        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if label is None:
                logits_fake = self.discriminator(reconstructions.contiguous())
                g_loss = -torch.mean(logits_fake)
            else:
                # multi-class classification
                logits_fake = self.discriminator(reconstructions.contiguous())
                targets = label.clone()
                logits_reshaped = logits_fake.view(batch_size, self.num_classes, -1).mean(dim=2)
                g_loss = F.cross_entropy(logits_reshaped, targets)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            cycle_loss = torch.tensor(0.0, device=inputs.device)
            if self.cycle_weight > 0 and cycle_recon is not None:
                target_for_cycle = cycle_target if cycle_target is not None else inputs
                cycle_loss = torch.abs(target_for_cycle.contiguous() - cycle_recon.contiguous()).mean()

            loss = (
                nll_loss
                + disc_factor * g_loss
                + self.codebook_weight * codebook_loss.mean()
                + self.cycle_weight * cycle_loss
            )
            self._check_finite(
                {
                    "rec_loss": rec_loss,
                    "p_loss": p_loss,
                    "nll_loss": nll_loss,
                    "g_loss": g_loss,
                    "codebook_loss": codebook_loss.mean(),
                    "cycle_loss": cycle_loss,
                    "total_loss": loss,
                },
                split=split,
                optimizer_idx=optimizer_idx,
            )

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   }
            if self.codebook_weight > 0:
                log["{}/quant_loss".format(split)] = codebook_loss.detach().mean()
            if self.perceptual_weight > 0:
                log["{}/p_loss".format(split)] = p_loss.detach().mean()
            if self.cycle_weight > 0:
                log["{}/cycle_loss".format(split)] = cycle_loss.detach().mean()
            if disc_factor > 0:
                log["{}/disc_factor".format(split)] = torch.tensor(disc_factor)
                log["{}/g_loss".format(split)] = g_loss.detach().mean()
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if label is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())

                disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
                d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            else:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_real_reshaped = logits_real.view(batch_size, self.num_classes, -1).mean(dim=2)
                real_targets = label.clone()
                real_loss = F.cross_entropy(logits_real_reshaped, real_targets)
                
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
                logits_fake_reshaped = logits_fake.view(batch_size, self.num_classes, -1).mean(dim=2)
                fake_targets = torch.zeros(batch_size, dtype=torch.long, device=inputs.device)  # fake class (0)
                fake_loss = F.cross_entropy(logits_fake_reshaped, fake_targets)
                
                disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
                d_loss = disc_factor * 0.5 * (real_loss + fake_loss)

            self._check_finite(
                {
                    "disc_loss": d_loss,
                    "logits_real": logits_real,
                    "logits_fake": logits_fake,
                },
                split=split,
                optimizer_idx=optimizer_idx,
            )

            if disc_factor > 0:
                log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                       "{}/logits_real".format(split): logits_real.detach().mean(),
                       "{}/logits_fake".format(split): logits_fake.detach().mean()
                       }
                return d_loss, log
            # if disc_factor == 0, skip logging/optimization
            return d_loss, {}
