import inspect
import torch
import importlib
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
import os
import torch.nn as nn
from lightning.utils.utils import get_text_logger
import ipdb
from lightning.model.framediff import score_network
from lightning.data.framediff import se3_diffuser
class framediff_Lightning_Model(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters()
        self.model_conf = conf.model
        self.frame_conf = conf.frame
        self.diffuser = se3_diffuser.SE3Diffuser(self.frame_conf)
        self.model = score_network.ScoreNetwork(self.model_conf, self.diffuser)
        pass



    def add_noise(self, batch):
        t = (1 - self.sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + self.sampling_eps
        sigma, dsigma = self.noise(t)
        perturbed_batch = self.graph.sample_transition(batch, sigma[:, None])
        return perturbed_batch, sigma, dsigma

    def forward(self, batch, cond):
        perturbed_batch, sigma, dsigma = self.add_noise(batch)
        log_score = self.model(perturbed_batch, cond, sigma)
        loss = self.graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
        try:
            assert torch.isfinite(loss).all()
        except:
            ipdb.set_trace()
        loss = (dsigma[:, None] * loss).sum(dim=-1).mean()
        # try:
        #     assert torch.isfinite(loss).all()
        # except:
        #     ipdb.set_trace()
        return loss

    def training_step(self, batch, batch_idx, **kwargs):
        vqid, seq = batch['vqid'], batch['seq']
        loss = self(batch=seq, cond=vqid)

        self.train_loss = loss.item()
        self.log("global_step", self.global_step, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        vqid, seq = batch['vqid'], batch['seq']
        loss = self(batch=seq, cond=vqid)
        # self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        # sample_seq = self.sample(cond=vqid, batch_size=vqid.shape[0], steps=20, sample='analytic', device='cuda' if torch.cuda.is_available() else 'cpu')
        # acc = cal_acc_token(sample_seq, seq)
        # self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        # log_dict = {'val_loss': loss, 'val_acc': acc}
        # self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

        # output = results['output']
        # acc = (output.argmax(dim=-1) == batch['label']).float().mean()

        # self.log_dict({"val_acc": acc})
        # return self.log_dict
        # return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')

    def get_schedular(self, optimizer, lr_scheduler='onecycle'):
        if lr_scheduler == 'step':
            scheduler = lrs.StepLR(optimizer,
                                   step_size=self.hparams.lr_decay_steps,
                                   gamma=self.hparams.lr_decay_rate)
        elif lr_scheduler == 'cosine':
            scheduler = lrs.CosineAnnealingLR(optimizer,
                                              T_max=self.hparams.lr_decay_steps,
                                              eta_min=self.hparams.lr_decay_min_lr)
        elif lr_scheduler == 'onecycle':
            scheduler = lrs.OneCycleLR(optimizer, max_lr=self.hparams.lr, steps_per_epoch=self.hparams.steps_per_epoch,
                                       epochs=self.hparams.epoch, three_phase=False)

        elif lr_scheduler == 'LambdaLR':
            def lr_lambda(current_step):
                warmup_steps = self.hparams.warmup_steps
                if current_step < self.hparams.warmup_steps:
                    # Linearly increase learning rate
                    return float(current_step) / float(max(1, warmup_steps))
                else:
                    # After warmup, apply other schedule, e.g., constant
                    return 1.0

            scheduler = lrs.LambdaLR(optimizer, lr_lambda=lr_lambda)
            # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)

        else:
            raise ValueError('Invalid lr_scheduler type!')

        return scheduler

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999),
                                      weight_decay=weight_decay)

        schecular = self.get_schedular(optimizer, self.hparams.lr_scheduler)

        return [optimizer], [{"scheduler": schecular, "interval": "step"}]

    def lr_scheduler_step(self, *args, **kwargs):
        scheduler = self.lr_schedulers()
        scheduler.step()

    def configure_loss(self):
        # self.loss_function = nn.NLLLoss()
        # self.loss_function = self.graph.score_entropy()
        pass



    def sample(self, cond, batch_size, steps, sample, device, eps=1e-5, denoise=True):
        # self.eval()
        self.predictor = get_predictor(sample)(self.graph, self.noise)
        self.projector = lambda x: x
        self.denoiser = Denoiser(self.graph, self.noise)
        self.sampling_score_fn = mutils.get_score_fn(self.model, train=False, sampling=True)

        batch_dims = (batch_size, self.hparams.block_size)  # sample shape
        x = self.graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = self.projector(x)
            x = self.predictor.update_fn(self.sampling_score_fn, x, cond, t, dt)

        if denoise:
            # denoising step
            x = self.projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = self.denoiser.update_fn(self.sampling_score_fn, x, cond, t)

        return x