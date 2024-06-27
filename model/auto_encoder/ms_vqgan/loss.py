import torch
import torch.nn as nn
import torch.nn.functional as F

from model.auto_encoder.ms_vqgan.discriminator import NLayerDiscriminator


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


class VQLossWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=0,
        use_actnorm=False,
        disc_conditional=False,
        disc_ndf=64,
        disc_loss="hinge",
        rec_loss="bce",
        occ_loss="bce",
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.rec_loss_type = rec_loss
        assert self.rec_loss_type in ["bce", "l1"]

        self.perceptual_weight = perceptual_weight
        # if self.perceptual_weight > 0:
        #     self.perceptual_loss = LPIPS().eval()

        self.discriminator = None
        if disc_weight > 0:
            self.discriminator = NLayerDiscriminator(
                input_nc=disc_in_channels,
                n_layers=disc_num_layers,
                use_actnorm=use_actnorm,
                ndf=disc_ndf,
            ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")

        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        if occ_loss == "l1":
            self.use_occ_l1 = True
        else:
            self.use_occ_l1 = False

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    @staticmethod
    def cal_rec_loss(target, rec, loss_type, mask=None):
        if mask is not None:
            rec = rec[mask]
            target = target[mask]
        if loss_type == "bce":
            bce_loss = F.binary_cross_entropy_with_logits(
                rec.contiguous(), target.contiguous()
            )
            return bce_loss
        elif loss_type == "l1":
            rec_loss = torch.abs(target.contiguous() - rec.contiguous())
            return torch.mean(rec_loss)
        else:
            raise NotImplementedError

    @staticmethod
    def rescale_volume(volume, output_shape):
        # Reshape volume to (batch_size * chn, 1, H, W, L)
        volume = volume.view(-1, 1, volume.size(2), volume.size(3), volume.size(4))

        # Perform interpolation
        output_volume = F.interpolate(
            volume, size=output_shape, mode="trilinear", align_corners=False
        )

        # Reshape back to original shape
        output_volume = output_volume.view(
            -1, volume.size(1), output_shape[0], output_shape[1], output_shape[2]
        )

        return output_volume

    @staticmethod
    def cal_occ_acc(target, rec, group, mask=None):
        if mask is not None:
            target = target[mask]
            rec = rec[mask]
        occ_score = torch.sigmoid(rec)
        thres = [0.5, 0.7, 0.9]
        rec_bin = torch.zeros_like(rec)
        pr_dict = {}
        for i in thres:
            occ_mask = occ_score > i
            rec_bin[occ_mask] = 1.0
            rec_bin[~occ_mask] = 0.0
            rec_bool = rec_bin.to(torch.bool)
            target_bool = target.to(torch.bool)
            TP = (rec_bool & target_bool).to(torch.int).sum()
            TN = ((~rec_bool) & (~target_bool)).to(torch.int).sum()
            FP = ((rec_bool) & (~target_bool)).to(torch.int).sum()
            FN = ((~rec_bool) & target_bool).to(torch.int).sum()
            Precision = TP / (TP + FP)
            Recall = TP / (TP + FN)
            key_p = "PR/{}/thres{}/P".format(group, i)
            key_r = "PR/{}/thres{}/R".format(group, i)
            if not torch.isnan(Precision):
                pr_dict[key_p] = Precision
            if not torch.isnan(Recall):
                pr_dict[key_r] = Recall
        return pr_dict

    def forward(
        self,
        codebook_loss,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        split="train",
    ):

        downsample_factor = 4
        # rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        tsdf = reconstructions["tsdf"]
        if "occ_l1" in reconstructions:
            occ_lv1 = reconstructions["occ_l1"]
            downsample_factor = tsdf.shape[-1] // occ_lv1.shape[-1]

        occ_lv2 = reconstructions["occ_l2"]
        tsdf_gt = inputs
        occ_lv2_gt = (tsdf_gt.abs() < 1.0).to(torch.float)
        if "occ_l1" in reconstructions:
            occ_lv1_gt = torch.nn.functional.max_pool3d(
                occ_lv2_gt, downsample_factor, stride=downsample_factor
            )

        with torch.no_grad():
            upsampler = torch.nn.ConvTranspose3d(
                1, 1, downsample_factor, downsample_factor
            )
            upsampler.weight *= 0
            upsampler.weight += 1
            upsampler.bias *= 0
            occ_upsampler = upsampler.to(tsdf_gt.device)

        tsdf_rec_mask = tsdf_gt.abs() < 1.0
        tsdf_rec_loss = self.cal_rec_loss(tsdf_gt, tsdf, "l1", mask=tsdf_rec_mask)

        if not self.use_occ_l1:
            if "occ_l1" in reconstructions:
                occ_lv1_rec_loss = self.cal_rec_loss(
                    occ_lv1_gt, occ_lv1, "bce", mask=None
                )
            occ_lv2_mask = occ_upsampler(occ_lv1_gt).to(torch.bool)
            occ_lv2_rec_loss = self.cal_rec_loss(
                occ_lv2_gt, occ_lv2, "bce", mask=occ_lv2_mask
            )
        else:
            occ_lv2_gt_soft = 1.0 - tsdf_gt.abs()
            occ_lv2_gt_soft[occ_lv2_gt_soft < 0] = 0.0
            origin_shape = list(occ_lv2_gt_soft.shape)[2:]
            new_shape = [k // downsample_factor for k in origin_shape]
            occ_lv1_gt_soft = self.rescale_volume(occ_lv2_gt_soft, new_shape)

            occ_lv1_rec_loss = self.cal_rec_loss(occ_lv1_gt_soft, occ_lv1, "l1")
            occ_lv2_rec_loss = self.cal_rec_loss(occ_lv2_gt_soft, occ_lv2, "l1")
        if "occ_l1" in reconstructions:
            occ_lv1_log = self.cal_occ_acc(occ_lv1_gt, occ_lv1, "occlv1")

        occ_lv2_log = self.cal_occ_acc(occ_lv2_gt, occ_lv2, "occlv2", mask=occ_lv2_mask)

        rec_loss = tsdf_rec_loss + occ_lv2_rec_loss
        if "occ_l1" in reconstructions:
            rec_loss += occ_lv1_rec_loss

        p_loss = torch.tensor([0.0])
        nll_loss = rec_loss

        # now the GAN part
        if self.discriminator_weight > 0 and self.discriminator is not None:
            if optimizer_idx == 0:
                # generator update
                if cond is None:
                    assert not self.disc_conditional
                    logits_fake = self.discriminator(
                        reconstructions["tsdf"].contiguous()
                    )
                else:
                    assert self.disc_conditional
                    logits_fake = self.discriminator(
                        torch.cat((reconstructions["tsdf"].contiguous(), cond), dim=1)
                    )
                g_loss = -torch.mean(logits_fake)

                try:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)

                disc_factor = adopt_weight(
                    self.disc_factor,
                    global_step,
                    threshold=self.discriminator_iter_start,
                )
                loss = (
                    nll_loss
                    + d_weight * disc_factor * g_loss
                    + self.codebook_weight * codebook_loss.mean()
                )

                log = {
                    "{}/total_loss".format(split): loss.clone().detach().mean(),
                    "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                    "{}/nll_loss".format(split): nll_loss.detach().mean(),
                    "{}/rec_loss".format(split): rec_loss.detach().mean(),
                    "{}/tsdf_rec_loss".format(split): tsdf_rec_loss.detach().mean(),
                    "{}/occ_lv2_rec_loss".format(
                        split
                    ): occ_lv2_rec_loss.detach().mean(),
                    "{}/p_loss".format(split): p_loss.detach().mean(),
                    "{}/d_weight".format(split): d_weight.detach(),
                    "{}/disc_factor".format(split): torch.tensor(disc_factor),
                    "{}/g_loss".format(split): g_loss.detach().mean(),
                }
                # print(occ_lv1_log)
                # print(occ_lv2_log)
                if "occ_l1" in reconstructions:
                    log.update(
                        {
                            "{}/occ_lv1_rec_loss".format(
                                split
                            ): occ_lv1_rec_loss.detach().mean()
                        }
                    )
                if "occ_l1" in reconstructions:
                    log.update(occ_lv1_log)
                log.update(occ_lv2_log)
                return loss, log

            if optimizer_idx == 1:
                # second pass for discriminator update
                if cond is None:
                    logits_real = self.discriminator(inputs.contiguous().detach())
                    logits_fake = self.discriminator(
                        reconstructions["tsdf"].contiguous().detach()
                    )
                else:
                    logits_real = self.discriminator(
                        torch.cat((inputs.contiguous().detach(), cond), dim=1)
                    )
                    logits_fake = self.discriminator(
                        torch.cat(
                            (reconstructions["tsdf"].contiguous().detach(), cond), dim=1
                        )
                    )

                disc_factor = adopt_weight(
                    self.disc_factor,
                    global_step,
                    threshold=self.discriminator_iter_start,
                )
                d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

                log = {
                    "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                    "{}/logits_real".format(split): logits_real.detach().mean(),
                    "{}/logits_fake".format(split): logits_fake.detach().mean(),
                }
                return d_loss, log
        else:
            assert optimizer_idx == 0
            loss = nll_loss + self.codebook_weight * codebook_loss.mean()
            return loss, None
