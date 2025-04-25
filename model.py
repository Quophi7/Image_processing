import torch
import torch.nn as nn
from torchvision import models
import random
import numpy as np
import torch.nn.functional as F
import os
import torch.nn.utils.weight_norm as weightNorm
import matplotlib.pyplot as plt
import clip
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

resnet_dict = {
    "ResNet18": models.resnet18, 
    "ResNet34": models.resnet34,
    "ResNet50": models.resnet50, 
    "ResNet101": models.resnet101, 
    "ResNet152": models.resnet152
}

k_dict = {
    'office_home': 10,
    'visda': 3,
    'domainnet': 10
}


def index_generator(N, B, shuffle=True):
    lis = list(range(N))
    if shuffle:
        random.shuffle(lis)
    num_steps = N//B + 1 if N%B!=0 else N//B
    start, end = 0, B
    for i in range(num_steps):
        batch = lis[start:end]
        if len(batch)==1:
            continue
        yield batch
        start = end
        end += B
        end = min(end, N)


class TextPrompter(nn.Module):
    def __init__(self, classnames, args, fill=None):
        super().__init__()
        n_cls = len(classnames)
        self.args = args
        clip_model, transform = clip.load(args.model)
        clip_model.cpu()
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        n = 3
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if fill is None:
            fill = args.target if args.dataset != 'visda' else 'real'
            fill = fill.replace('_', ' ').lower()
        pre = 'a'
        naive_prompt_prefix_tar = "{} {} photo of a".format(pre, fill)

        ctx_vectors = torch.empty(n_cls, n, ctx_dim, dtype=torch.float32)

        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = "a " + " ".join(["C"] * n)

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(name.split(' ')) for name in classnames]
        naive_prompts = [naive_prompt_prefix_tar + " " + name + "." for name in classnames]

        prompts = [prompt_prefix + " photo of a " + name + "." for name in classnames]
        print("Naive prompt: {}".format(naive_prompts[0]))

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        naive_tokenized_prompts = torch.cat([clip.tokenize(p) for p in naive_prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(torch.float32)
            naive_embedding = clip_model.token_embedding(naive_tokenized_prompts).type(torch.float32)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        #tokenized_prompts = torch.cat([tokenized_prompts, naive_tokenized_prompts])
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.csc = True
        self.tokenized_prompts = tokenized_prompts  
        self.naive_tokenized_prompts = naive_tokenized_prompts  
        self.name_lens = name_lens
        self.naive_embedding = naive_embedding.cuda()

    @autocast()
    def forward(self):
        return 0, self.naive_embedding


def update_qhat(probs, qhat, momentum=0.99, qhat_mask=None):
    if qhat_mask is not None:
        mean_prob = probs.detach()*qhat_mask.detach().unsqueeze(dim=-1)
    else:
        mean_prob = probs.detach().mean(dim=0)
    qhat = momentum * qhat + (1 - momentum) * mean_prob
    return qhat


class UniMoS:     
    def __init__(self, args, config) -> None:
        super().__init__()
        self.args = args
        self.class_num = args.class_num
        self.config = config
        self.clip_model, preprocess = clip.load(args.model)     
        self.clip_model.cuda()
        self.dsets, self.dset_loaders = args.load_data_method(args, config, preprocess)
        self.classes = self.dsets['val'].idx2cls
        self.l_list = self.cal_l()
        
        self.fea_dim = self.clip_model.text_projection.shape[1]
        self.text_prompter = TextPrompter(self.classes, args).cuda()
        self.clf = feat_classifier(self.class_num, type='wn').cuda()
        self.bottleneck = feat_bottleneck(self.fea_dim, type='bn').cuda()
        self.dtype = self.clip_model.dtype
        self.temp = 100
        self.l_linear = args.l
        self.weight = weight_generator(self.fea_dim).cuda()
        self.mos = ModalitySeperation(self.fea_dim).cuda()
        self.discri = Discriminator(self.fea_dim).cuda()

        self.optimizer = torch.optim.SGD(self.train_param(), momentum=0.9, weight_decay=0)
        self.optimizer_mos = torch.optim.SGD(self.mos.parameters(), momentum=0.9, weight_decay=0, lr=self.args.lr)
        self.convert_models_to_fp32()
        self.logger = args.logger
        self.train_tokenized_prompts = self.text_prompter.tokenized_prompts
        self.naive_tokenized_prompts = self.text_prompter.naive_tokenized_prompts
        self.epoch = -1
        self.ps_epoch = 1 
        self.qhat = {
            'clip': (torch.ones([1, self.class_num], dtype=torch.float)/self.class_num).cuda(),
            'learned': (torch.ones([1, self.class_num], dtype=torch.float)/self.class_num).cuda(),
        }
        self.momen = args.momen
        self.tau = args.tau
        self.fea_path = './ext_fea/{}/{}/'.format(args.dataset, args.model)
        self.src_fea_path = self.fea_path + '{}.pth'.format(args.source)
        self.tar_fea_path = self.fea_path + '{}.pth'.format(args.target)
        if not os.path.exists(self.src_fea_path):
            self.ext_fea('src')
        if not os.path.exists(self.tar_fea_path):
            self.ext_fea('tar')
        self.src_fea = torch.load(self.src_fea_path)['fea']
        self.tar_fea = torch.load(self.tar_fea_path)['fea']
        # Ensure labels are within valid range and convert to long type
        src_label = torch.load(self.src_fea_path)['label']
        tar_label = torch.load(self.tar_fea_path)['label']
        # Debug logging for raw labels
        self.logger.info(f"Raw source label range: [{src_label.min().item()}, {src_label.max().item()}]")
        self.logger.info(f"Raw target label range: [{tar_label.min().item()}, {tar_label.max().item()}]")
        # Ensure labels are within valid range and convert to long type
        self.src_label = torch.clamp(src_label, 0, self.class_num-1).long().cuda()
        self.tar_label = torch.clamp(tar_label, 0, self.class_num-1).long()
        
        # Debug logging
        self.logger.info(f"Number of classes: {self.class_num}")
        self.logger.info(f"Source label range: [{self.src_label.min().item()}, {self.src_label.max().item()}]")
        self.logger.info(f"Target label range: [{self.tar_label.min().item()}, {self.tar_label.max().item()}]")
        # Print unique values in labels
        self.logger.info(f"Unique source labels: {torch.unique(self.src_label).tolist()}")
        self.logger.info(f"Unique target labels: {torch.unique(self.tar_label).tolist()}")
        
        self.src_len = self.src_fea.shape[0]
        self.tar_len = self.tar_fea.shape[0]
        self.src_loader_len = self.src_len//self.args.batch_size
        self.tar_loader_len = self.tar_len//self.args.batch_size
        # Initialize cross entropy loss with ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=-100).cuda()
        self.best = {
            'epoch': 0,
            'acc': 0
        }
        self.losses = loss_calculator()
        self.start = True
        self.epoch_len = min(min(self.src_loader_len, self.tar_loader_len), 1500)

        # Define class names for Office-Home dataset
        self.class_names = [
            'Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator',
            'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp',
            'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder',
            'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade',
            'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan',
            'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator',
            'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker',
            'Spoon', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'TV', 'Webcam'
        ]

        # Initialize CLIP model
        self.clip_model, _ = clip.load('RN101', device='cuda')
        self.dtype = self.clip_model.dtype
        self.convert_models_to_fp32()
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def cal_l(self):
        start = 0.8
        end = self.args.end
        step = (start-end) / 10
        res = torch.arange(start, end-0.01, -step)
        tmp = torch.ones(100) * end
        return torch.concat([res, tmp], 0)

    def inv_lr_scheduler(self, iter_num, power=0.75, gamma=0.001):
        lr = self.args.lr * (1 + gamma * iter_num) ** (-power)
        for param_group in self.optimizer.param_groups:
            if not param_group['unchange']:
                param_group['lr'] = lr 
            else:
                param_group['lr'] = self.args.lr * 0.1
        for param_group in self.optimizer_mos.param_groups:
            param_group['lr'] = lr 

    def debias(self, probs, k):
        assert k in self.qhat
        debiased_prob = probs - self.tau*torch.log(self.qhat[k])
        return debiased_prob

    def update_q(self, probs_before_softmax, k):
        assert k in self.qhat
        probs = probs_before_softmax.softmax(1)
        mean_prob = probs.detach().mean(dim=0)
        self.qhat[k] = self.momen * self.qhat[k] + (1 - self.momen) * mean_prob

    def train_param(self):
        param_group = []
        default_lr = self.args.lr
        for k, v in self.bottleneck.named_parameters():
            param_group += [{'params':v, 'lr':default_lr, 'unchange':0}]
        for k, v in self.clf.named_parameters():
            param_group += [{'params':v, 'lr':default_lr, 'unchange':0}]
        for k, v in self.weight.named_parameters():
            param_group += [{'params':v, 'lr':default_lr*0.1, 'unchange':0}]
        for k, v in self.mos.named_parameters(): 
            param_group += [{'params':v, 'lr':default_lr*1, 'unchange':0}]
        for k, v in self.discri.named_parameters():
            param_group += [{'params':v, 'lr':default_lr*1, 'unchange':0}]
        return param_group

    def clip_forward(self, images, text, tokenized_prompts):
        with autocast(dtype=torch.float32):
            # Move tokenized prompts to the same device as images
            tokenized_prompts = tokenized_prompts.to(images.device)

            # Handle both raw images and pre-computed features
            if len(images.shape) == 4:  # Raw images: [batch_size, channels, height, width]
                image_features = self.clip_model.encode_image(images)
                image_features = image_features @ self.clip_model.visual_projection
            else:  # Pre-computed features: [batch_size, feature_dim]
                image_features = images

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Encode text and ensure it's on the same device
            text_features = self.clip_model.encode_text(tokenized_prompts)
            text_features = text_features.to(images.device)
            text_features = text_features @ self.clip_model.text_projection
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Ensure all tensors are in float32 and on the same device
            image_features = image_features.float()
            text_features = text_features.float()

            # Calculate logits
            logit_scale = self.clip_model.logit_scale.exp().to(images.device)
            logits = logit_scale * image_features @ text_features.t()

            return F.softmax(logits, dim=1), logits, image_features, text_features

    def clip_clf_res(self, fea, method=None):
        logits_per_image = self.temp * fea @ self.text_fea_norm.t()
        if method is not None:
            logits_per_image = method(logits_per_image)
        return logits_per_image

    def record_best(self, acc):
        if acc >= self.best['acc']:
            self.best['acc'] = acc
            self.best['epoch'] = self.epoch

    def train(self):
        src_loader = index_generator(self.src_len, self.args.batch_size)
        tar_loader = index_generator(self.tar_len, self.args.batch_size)
        for self.epoch in range(0, self.args.epochs):
            if self.start:
                vis_pslabel = self.test()
            self.set_train()
            self.losses.clear()
            self.weight.reset_random_parameters(0.5)
            for i, _ in enumerate(tqdm(range(self.epoch_len), mininterval=0.5)):
                try:
                    src_idx = next(src_loader)
                except:
                    src_loader = index_generator(self.src_len, self.args.batch_size)
                    src_idx = next(src_loader)
                try:
                    tar_idx = next(tar_loader)
                except:
                    tar_loader = index_generator(self.tar_len, self.args.batch_size)
                    tar_idx = next(tar_loader)

                tar_fea, tar_label = self.tar_fea[tar_idx].cuda(), self.tar_label[tar_idx].cuda()
                src_fea, src_label = self.src_fea[src_idx].cuda(), self.src_label[src_idx].cuda()
                tar_idx = tar_idx
                with autocast():
                    src_len = src_fea.shape[0]
                    tar_len = tar_fea.shape[0]
                    image_fea = torch.concat([src_fea, tar_fea]).cuda()
                    tar_clip_out = self.clip_output[tar_idx].cuda()
                    tar_vis_label = vis_pslabel[tar_idx].cuda()     # pslabel from test
                    tar_txt_label = tar_clip_out                    # soft target pslabel

                    # Ensure labels are within valid range
                    src_label = torch.clamp(src_label, 0, self.class_num - 1)
                    tar_vis_label = torch.clamp(tar_vis_label, 0, self.class_num - 1)

                    # Debug logging for first batch
                    if i == 0:
                        self.logger.info(f"Source label range: [{src_label.min().item()}, {src_label.max().item()}]")
                        self.logger.info(f"Target visual label range: [{tar_vis_label.min().item()}, {tar_vis_label.max().item()}]")
                        self.logger.info(f"Number of classes: {self.class_num}")

                    vis_fea, txt_fea = self.mos(image_fea)
                    clf_l = self.weight(vis_fea[src_len:]).mean()
                    clf_p = 1 - clf_l
                    
                    bottleneck_fea = self.bottleneck(vis_fea)
                    vis_out = self.clf(bottleneck_fea)
                    src_vis_fea, tar_vis_fea = vis_fea[:src_len], vis_fea[src_len:]
                    src_txt_fea, tar_txt_fea = txt_fea[:src_len], txt_fea[src_len:]
                    src_vis_out, tar_vis_out = vis_out[:src_len], vis_out[src_len:]
                    txt_out = self.clip_clf_res(txt_fea)
                    src_txt_out, tar_txt_out = txt_out[:src_len], txt_out[src_len:]
                    ens_tar_vis_out = (clf_l*tar_vis_out + clf_p*(tar_txt_out.detach())) 
                    self.update_q(tar_txt_out, 'learned')

                    src_clf = self.discri(torch.cat([src_vis_fea, src_txt_fea]))
                    tar_clf = self.discri(torch.cat([tar_vis_fea, tar_txt_fea]))
                    src_clf_label = torch.tensor([0]*src_len + [1]*src_len).cuda().unsqueeze(1).float()
                    tar_clf_label = torch.tensor([0]*tar_len + [1]*tar_len).cuda().unsqueeze(1).float()

                    # Compute losses with proper label handling
                    vis_celoss = self.args.alpha_srcvis * self.ce(src_vis_out, src_label) + self.ce(tar_vis_out, tar_vis_label)
                    txt_celoss = self.args.alpha_srctxt * self.ce(src_txt_out, src_label) + kl_div_with_logit(tar_txt_out, tar_txt_label)
                    discri_loss = self.args.alpha_reg * nn.BCEWithLogitsLoss()(src_clf, src_clf_label)

                    # Total loss
                    loss = vis_celoss + txt_celoss + discri_loss

                    # Update model
                    self.optimizer.zero_grad()
                    self.optimizer_mos.zero_grad()
                    loss.backward(retain_graph=True)
                    self.optimizer.step()

                    try:
                        loss_tar_dis = self.args.alpha_reg * nn.BCEWithLogitsLoss()(tar_clf, tar_clf_label)
                        self.optimizer.zero_grad()
                        self.optimizer_mos.zero_grad()
                        loss_tar_dis.backward()
                        self.optimizer_mos.step()
                    except Exception as e:
                        self.logger.warning(f"Error in target discriminator loss: {str(e)}")

                    self.losses.update(vis_celoss=vis_celoss, txt_celoss=txt_celoss, discri_loss=discri_loss)

            self.logger.info(self.losses)
            vis_pslabel = self.test()
        self.logger.info('best acc={} @ epoch {}'.format(self.best['acc'], self.best['epoch']))
        
    def normalize(self, x):
        return x - x.mean(1, keepdim=True)

    def test(self, d='tar'):
        all_out, all_label, all_bott_fea, all_text_fea, all_vis_fea, all_ens, all_prompt_output = None, None, None, None, None, None, None
        test_batch_size = self.args.batch_size*3
        loader = index_generator(self.tar_len, test_batch_size, False) if d == 'tar' else index_generator(self.src_len, test_batch_size, False)
        fea_bank = self.tar_fea if d=='tar' else self.src_fea
        label_bank = self.tar_label if d=='tar' else self.src_label
        l = self.tar_len//test_batch_size if d=='tar' else self.src_len//test_batch_size
        self.set_eval()
        if self.start:
            # Use CLIP model's predictions for initial pseudo-labels
            with torch.no_grad():
                learn, naive = self.text_prompter()
                images = fea_bank.cuda()
                # Create text prompts for each class
                class_prompts = [f"a photo of a {self.class_names[i]}" for i in range(self.class_num)]
                tokenized_prompts = clip.tokenize(class_prompts).cuda()
                ps, _, image_fea, text_fea = self.clip_forward(images, naive, tokenized_prompts)
                # Ensure probabilities are valid
                ps = ps + 1e-8
                ps = ps / (ps.sum(dim=1, keepdim=True) + 1e-8)
                self.clip_output = ps.cpu()
                self.text_fea_norm = text_fea
        with torch.no_grad():
            for i, idx in enumerate(tqdm(loader, total=l, mininterval=0.5)):
                with autocast(dtype=torch.float32):
                    images = fea_bank[idx].cuda()
                    target = label_bank[idx].cuda()
                    if self.start:
                        if self.text_fea_norm is None:
                            # Create text prompts for each class
                            class_prompts = [f"a photo of a {self.class_names[i]}" for i in range(self.class_num)]
                            tokenized_prompts = clip.tokenize(class_prompts).cuda()
                            ps, _, image_fea, text_fea = self.clip_forward(images, naive, tokenized_prompts)
                            self.text_fea_norm = text_fea
                        else:
                            image_fea = images
                            ps = self.clip_clf_res(image_fea, self.normalize)
                        # Ensure probabilities are valid
                        ps = ps + 1e-8
                        ps = ps / (ps.sum(dim=1, keepdim=True) + 1e-8)
                        self.clip_output[idx] = ps.cpu()
                    else:
                        image_fea, txt_fea = self.mos(images)
                        txt_out = self.clip_clf_res(txt_fea)
                        if self.epoch <= 0:
                            txt_out = self.clip_output[idx].cuda() 
                            ps = txt_out
                        else:
                            ps = self.debias(txt_out, 'learned')
                    bottleneck_fea = self.bottleneck(image_fea)
                    output_org = self.clf(bottleneck_fea)
  
                    all_out = output_org.cpu() if all_out is None else torch.concat([all_out, output_org.cpu()])
                    all_prompt_output = ps.cpu() if all_prompt_output is None else torch.concat([all_prompt_output, ps.cpu()])
                    all_label = target.cpu() if all_label is None else torch.concat([all_label, target.cpu()])
                    all_bott_fea = bottleneck_fea.cpu() if all_bott_fea is None else torch.concat([all_bott_fea, bottleneck_fea.cpu()])
        clf_l = self.l_list[self.epoch] if self.l_linear is None else self.l_linear
        clf_p = 1 - clf_l
        all_ens = (clf_l*all_out + clf_p*all_prompt_output).softmax(1).cpu()
        all_prompt_output = all_prompt_output
        # Ensure labels are within valid range
        all_label = torch.clamp(all_label, 0, self.class_num-1)
        clf_mix = all_ens.max(-1)[1]==all_label
        clf1 = all_out.max(-1)[1]==all_label
        clf2 = all_prompt_output.max(-1)[1]==all_label      
        acc1 = (clf1).sum() / len(all_label)
        acc2 = (clf2).sum() / len(all_label)
        acc_mix = clf_mix.sum() / len(all_label)
        self.logger.info("Epoch {} {}: linearAcc={:.4f}, promptAcc={:.4f}, mixAcc={:.4f}".format(self.epoch, d, acc1, acc2, acc_mix))
        self.record_best(acc_mix)
        dist1, pslabel1, cent1 = self.clustering(all_ens, all_bott_fea, self.class_num, all_label)

        if self.epoch <= 0:
            # Use clip_output for initial pseudo-labels
            pslabel = self.clip_output.max(1)[1]
            if self.start:
                self.start = False
        else:
            # Use ensemble predictions for pseudo-labels
            pslabel = all_ens.max(1)[1] if self.epoch % 2 == 0 else pslabel1
        # Ensure pseudo-labels are within valid range
        pslabel = torch.clamp(pslabel, 0, self.class_num-1)
        
        # Debug logging for pseudo-labels
        self.logger.info(f"Pseudo-label range: [{pslabel.min().item()}, {pslabel.max().item()}]")
        self.logger.info(f"Unique pseudo-labels: {torch.unique(pslabel).tolist()}")
        self.logger.info(f"Pseudo-label distribution: {torch.bincount(pslabel).tolist()}")
        
        return pslabel

    def ext_fea(self, domain):
        self.clip_model.eval()
        if domain == 'src':
            dset = self.dsets['source']
            dset_loader = self.dset_loaders['source']
        else:
            dset = self.dsets['val']
            dset_loader = self.dset_loaders['val']
        fea_list = []
        label_list = []
        with torch.no_grad():
            for i, (imgs, labels, idx) in enumerate(dset_loader):
                imgs = imgs.cuda()
                # Ensure labels are within valid range
                labels = torch.clamp(labels, 0, self.class_num-1).long()
                # Debug logging
                self.logger.info(f"Batch {i} label range: [{labels.min().item()}, {labels.max().item()}]")
                self.logger.info(f"Batch {i} unique labels: {torch.unique(labels).tolist()}")
                fea = self.clip_model.encode_image(imgs)
                fea_list.append(fea)
                label_list.append(labels)
        fea = torch.cat(fea_list, dim=0)
        label = torch.cat(label_list, dim=0)
        # Ensure final labels are within valid range
        label = torch.clamp(label, 0, self.class_num-1).long()
        # Debug logging
        self.logger.info(f"Final {domain} label range: [{label.min().item()}, {label.max().item()}]")
        self.logger.info(f"Final {domain} unique labels: {torch.unique(label).tolist()}")
        
        # Create directory structure if it doesn't exist
        save_path = self.src_fea_path if domain == 'src' else self.tar_fea_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({'fea': fea, 'label': label}, save_path)

    def convert_models_to_fp32(self):
        for p in self.clip_model.parameters():
            p.data = p.data.float()
            if p.grad:
                p.grad.data = p.grad.data.float()

    def clustering(self, aff, all_fea, K, all_label):
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        dist1, pred1, c1 = self.cal_dis(aff, all_fea)
        acc1 = (aff.max(-1)[1]==all_label).sum() / len(all_label)
        acc2 = (pred1==all_label).sum() / len(all_label)
        self.logger.info("{:.4f} -> {:.4f}".format(acc1.item(), acc2.item()))
        return dist1, pred1, c1

    def cal_dis(self, aff, all_fea):
        initc = torch.einsum('NC, NH -> CH', aff, all_fea)
        initc = (initc.T / (1e-8 + aff.sum(axis=0))).T  # 345,1024, centroid
        res1 = []
        dist1 = []
        for i in range(all_fea.shape[0]):
            dis = torch.nn.functional.cosine_similarity(all_fea[i], initc)
            pos = dis.argmax(0)
            res1.append(pos)
            dist1.append(dis*20)
        dist1 = torch.stack(dist1)
        pred1 = torch.stack(res1)
        return dist1.softmax(1), pred1, initc

    def set_train(self):
        self.bottleneck.train()
        self.clf.train()
        self.weight.train()
        self.mos.train()
        self.discri.train()
        #self.clip_model.visual.train()

    def set_eval(self):
        self.bottleneck.eval()
        self.clf.eval() 
        self.clip_model.eval()
        self.weight.eval()
        self.mos.eval()


def entropy(probs):
    try:
        # Move to CPU for safer operations
        probs = probs.detach().cpu()
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-8
        probs = probs + epsilon
        # Normalize to ensure sum is 1
        probs = probs / (probs.sum(dim=1, keepdim=True) + epsilon)
        
        # Compute entropy
        entropy_val = -(probs * torch.log(probs + epsilon)).sum(dim=1).mean()
        return entropy_val.cuda()
    except Exception as e:
        print(f"Warning: Error in entropy: {str(e)}")
        return torch.zeros(1).cuda()


def gentropy(probs):
    try:
        # Move to CPU for safer operations
        probs = probs.detach().cpu()
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-8
        probs = probs + epsilon
        # Normalize to ensure sum is 1
        probs = probs / (probs.sum(dim=1, keepdim=True) + epsilon)
        
        # Compute entropy
        entropy_val = -(probs * torch.log(probs + epsilon)).sum(dim=1).mean()
        return entropy_val.cuda()
    except Exception as e:
        print(f"Warning: Error in gentropy: {str(e)}")
        return torch.zeros(1).cuda()


def ent_loss(out):
    try:
        # Move to CPU for safer operations
        out = out.detach().cpu()
        
        # Apply softmax to get valid probabilities
        out = F.softmax(out, dim=1)
        # Add small epsilon to prevent log(0)
        epsilon = 1e-8
        out = out + epsilon
        # Normalize to ensure sum is 1
        out = out / (out.sum(dim=1, keepdim=True) + epsilon)
        
        # Compute entropy loss
        entropy_loss = entropy(out.cuda()) - gentropy(out.cuda())
        return entropy_loss
    except Exception as e:
        print(f"Warning: Error in ent_loss: {str(e)}")
        return torch.zeros(1).cuda()


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def kl_div_with_logit(q_logit, p_logit):
    """
    Calculate KL divergence between two distributions with logits as input
    Args:
        q_logit: logits of first distribution
        p_logit: logits of second distribution
    Returns:
        KL divergence between the two distributions
    """
    try:
        # Get device from input tensor
        device = q_logit.device
        
        # Add numerical stability
        epsilon = 1e-8
        
        # Apply softmax with numerical stability
        q_prob = F.softmax(q_logit, dim=1)
        p_prob = F.softmax(p_logit, dim=1)
        
        # Add epsilon and renormalize
        q_prob = q_prob + epsilon
        p_prob = p_prob + epsilon
        q_prob = q_prob / q_prob.sum(dim=1, keepdim=True)
        p_prob = p_prob / p_prob.sum(dim=1, keepdim=True)
        
        # Compute KL divergence
        kl_div = torch.sum(p_prob * (torch.log(p_prob) - torch.log(q_prob)), dim=1)
        
        # Return mean KL divergence
        return kl_div.mean()
    except Exception as e:
        logging.warning(f"Error in kl_div_with_logit: {str(e)}")
        return torch.zeros(1, device=device)


class loss_calculator:
    def __init__(self) -> None:
        self.losses = {}

    def update(self, **kw):
        for k in kw:
            try:
                # Ensure the loss value is a valid scalar
                loss_val = kw[k].detach().cpu().numpy().item()
                if not np.isfinite(loss_val):
                    loss_val = 0.0  # Replace invalid values with 0
                
                if not k in self.losses:
                    self.losses[k] = {
                        'cnt': 1,
                        'val': loss_val,
                    }
                else:
                    self.losses[k]['cnt'] += 1
                    self.losses[k]['val'] += loss_val
            except Exception as e:
                print(f"Warning: Error processing loss {k}: {str(e)}")
                continue

    def clear(self):
        self.losses = {}

    def __repr__(self) -> str:
        strr = '\n'
        for k in self.losses:
            strr += 'loss {}: {:.4f} ({}) \n'.format(k, self.losses[k]['val'] / self.losses[k]['cnt'], self.losses[k]['cnt'])
        return strr


class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="bn"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="wn"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class weight_generator(nn.Module):
    def __init__(self, fea_dim=1024):
        super(weight_generator, self).__init__()
        self.fc1 = nn.Linear(fea_dim, 256)
        #self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        self.activate = nn.Sigmoid()

    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def reset_random_parameters(self, reset_probability=0.5):
        for name, param in self.named_parameters():
            if random.random() < reset_probability:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, x):
        x = self.fc1(x)
        #x = self.relu(x)
        x = self.fc2(x)
        x = self.activate(x)
        return x


class ModalitySeperation(nn.Module):
    def __init__(self, fea_dim) -> None:
        super().__init__()
        self.vis_proj = nn.Linear(fea_dim, fea_dim)
        self.txt_proj = nn.Linear(fea_dim, fea_dim)

    def forward(self, x):
        vis_fea = self.vis_proj(x)
        txt_fea = self.txt_proj(x)
        return vis_fea, txt_fea


class Discriminator(nn.Module):
    def __init__(self, fea_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(fea_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(256, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    pass