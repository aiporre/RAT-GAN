from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.config import cfg, cfg_from_file

#from datasets import TextDataset
#from datasets import prepare_data
#for flower dataset, please use the fllowing dataset files
from datasets_flower import TextDataset
from datasets_flower import prepare_data
from DAMSM import RNN_ENCODER,CustomLSTM

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from model import NetG,NetD
import torchvision.utils as vutils

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

import multiprocessing
multiprocessing.set_start_method('spawn', True)

UPDATE_INTERVAL = 200
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def make_batch_one(data):
    img, caption, captions_len, class_id, key = data
    img = [torch.unsqueeze(img[0], dim=0)]
    key = tuple([key])
    caption = torch.tensor([caption])
    captions_len = torch.tensor([captions_len])
    class_id = torch.tensor([class_id])
    return img, caption, captions_len, class_id, key



def predict_one(text, text_encoder, netG, dataset, device):
    model_dir = '/tmp' # cfg.TRAIN.NET_G
    split_dir = 'generated'
    # Build and load the generator
    # for coco wrap netG with DataParallel because it's trained on two 3090
    #    netG = nn.DataParallel(netG).cuda()
    # get device and load model in correct map_location
    if torch.cuda.is_available():
        netG.load_state_dict(torch.load('../models/%s/netG_500.pth'%(cfg.CONFIG_NAME)))
    else:
        netG.load_state_dict(torch.load('../models/%s/netG_500.pth'%(cfg.CONFIG_NAME), map_location=torch.device('cpu')))

    netG.eval()
    batch_size = 1 # cfg.TRAIN.BATCH_SIZE
    s_tmp = model_dir
    save_dir = '%s/%s' % (s_tmp, split_dir)
    mkdir_p(save_dir)

    if text is None or isinstance(text, int):
        # generate a random text from the dataset length
        if isinstance(text, int):
            data_index = text % len(dataset)
        else:
            data_index = random.randint(0, len(dataset))
        data = dataset[data_index]
        data = make_batch_one(data)
        img, cap, cap_len, cls_id, key = prepare_data(data)
        text = dataset.caption2text(cap.flatten().detach().cpu().numpy().tolist())
        hidden = text_encoder.init_hidden(batch_size)
        # words_embs: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_emb, sent_emb = text_encoder(cap, cap_len, hidden)
        words_emb, sent_emb = words_emb.detach(), sent_emb.detach()
        with torch.no_grad():
            noise = torch.randn(batch_size, 100)
            noise = noise.to(device)
            netG.lstm.init_hidden(noise)
            fake_img = netG(noise, sent_emb)
        s_tmp = '%s/%s' % (save_dir, key[0])
        folder = s_tmp[:s_tmp.rfind('/')]
        if not os.path.isdir(folder):
            print('Make a new folder: ', folder)
            mkdir_p(folder)
        im = fake_img[0].data.cpu().numpy()
        # [-1, 1] --> [0, 255]
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)
        fullpath = '%s.png' % (s_tmp)
        im.save(fullpath)
        print('out1:', fullpath)
        print('out2:', text)
        print('out3:', dataset.class_id_inverted[cls_id[0]])
        print('out4:', key)



    # cnt = 0
    # for i in range(10):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
    #     for step, data in enumerate(dataloader, 0):
    #         imags, captions, cap_lens, class_ids, keys = prepare_data(data)
    #         cnt += batch_size
    #         if step % 100 == 0:
    #             print('step: ', step)
    #         # if step > 50:
    #         #     break
    #         hidden = text_encoder.init_hidden(batch_size)
    #         # words_embs: batch_size x nef x seq_len
    #         # sent_emb: batch_size x nef
    #         words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
    #         words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
    #         #######################################################
    #         # (2) Generate fake images
    #         ######################################################
    #         with torch.no_grad():
    #             noise = torch.randn(batch_size, 100)
    #             noise=noise.to(device)
    #             netG.lstm.init_hidden(noise)
    #
    #             fake_imgs = netG(noise,sent_emb)
    #         for j in range(batch_size):
    #             s_tmp = '%s/single/%s' % (save_dir, keys[j])
    #             folder = s_tmp[:s_tmp.rfind('/')]
    #             if not os.path.isdir(folder):
    #                 print('Make a new folder: ', folder)
    #                 mkdir_p(folder)
    #             im = fake_imgs[j].data.cpu().numpy()
    #             # [-1, 1] --> [0, 255]
    #             im = (im + 1.0) * 127.5
    #             im = im.astype(np.uint8)
    #             im = np.transpose(im, (1, 2, 0))
    #             im = Image.fromarray(im)
    #             fullpath = '%s_%3d.png' % (s_tmp,i)
    #             im.save(fullpath)



# def train(dataloader,netG,netD,text_encoder,optimizerG,optimizerD,state_epoch,batch_size,device):
#     mkdir_p('../models/%s' % (cfg.CONFIG_NAME))
#
#     for epoch in range(state_epoch+1, cfg.TRAIN.MAX_EPOCH+1):
#         torch.cuda.empty_cache()
#
#         for step, data in enumerate(dataloader, 0):
#             #torch.cuda.empty_cache()
#
#             imags, captions, cap_lens, class_ids, keys = prepare_data(data)
#             hidden = text_encoder.init_hidden(batch_size)
#             # words_embs: batch_size x nef x seq_len
#             # sent_emb: batch_size x nef
#             words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
#             words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

            # imgs=imags[0].to(device)
            # real_features = netD(imgs)
            # output = netD.COND_DNET(real_features,sent_emb)
            # errD_real = torch.nn.ReLU()(1.0 - output).mean()

            # output = netD.COND_DNET(real_features[:(batch_size - 1)], sent_emb[1:batch_size])
            # errD_mismatch = torch.nn.ReLU()(1.0 + output).mean()

            # # synthesize fake images
            #
            # noise = torch.randn(batch_size, 100)
            # noise=noise.to(device)
            # netG.lstm.init_hidden(noise)
            #
            # fake = netG(noise,sent_emb)
            #
            # # G does not need update with D
            # fake_features = netD(fake.detach())

            # errD_fake = netD.COND_DNET(fake_features,sent_emb)
            # errD_fake = torch.nn.ReLU()(1.0 + errD_fake).mean()

            # errD = errD_real + (errD_fake + errD_mismatch)/2.0
            # optimizerD.zero_grad()
            # optimizerG.zero_grad()
            # errD.backward()
            # optimizerD.step()

            # #MA-GP
            # interpolated = (imgs.data).requires_grad_()
            # sent_inter = (sent_emb.data).requires_grad_()
            # features = netD(interpolated)
            # out = netD.COND_DNET(features,sent_inter)
            # grads = torch.autograd.grad(outputs=out,
            #                         inputs=(interpolated,sent_inter),
            #                         grad_outputs=torch.ones(out.size()).cuda(),
            #                         retain_graph=True,
            #                         create_graph=True,
            #                         only_inputs=True)
            # grad0 = grads[0].view(grads[0].size(0), -1)
            # grad1 = grads[1].view(grads[1].size(0), -1)
            # grad = torch.cat((grad0,grad1),dim=1)
            # grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            # d_loss_gp = torch.mean((grad_l2norm) ** 6)
            # d_loss = 2.0 * d_loss_gp
            # optimizerD.zero_grad()
            # optimizerG.zero_grad()
            # d_loss.backward()
            # optimizerD.step()
            #
            # # update G
            # features = netD(fake)
            # output = netD.COND_DNET(features,sent_emb)
            # errG = - output.mean()
            # optimizerG.zero_grad()
            # optimizerD.zero_grad()
            # errG.backward()
            # optimizerG.step()

            # print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
            #     % (epoch, cfg.TRAIN.MAX_EPOCH, step, len(dataloader), errD.item(), errG.item()))

        # vutils.save_image(fake.data,
        #                 '%s/fake_samples_epoch_%03d.png' % ('../imgs', epoch),
        #                 normalize=True)

        # if epoch%10==0:
        #     torch.save(netG.state_dict(), '../models/%s/netG_%03d.pth' % (cfg.CONFIG_NAME, epoch))
        #     torch.save(netD.state_dict(), '../models/%s/netD_%03d.pth' % (cfg.CONFIG_NAME, epoch))

    # return count


def merge_from_file_cfg(cfg_file):
    if cfg_file is not None:
        cfg_from_file(cfg_file)

def build_model(gpu_id=-1, data_dir='', manualSeed=100):
    if gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = gpu_id

    if data_dir != '':
        cfg.DATA_DIR = data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        manualSeed = 100
    elif manualSeed is None:
        manualSeed = 100
        # manualSeed = random.randint(1, 10000)
    print("seed now is : ", manualSeed)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, 'test',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    print(dataset.n_words, dataset.embeddings_num)
    print('----> the dataset is:  ', dataset)
    assert dataset, "dataset has zero elements, check the captions.pkl file and the data directory"
    # # validation data #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm = CustomLSTM(256, 256)

    netG = NetG(cfg.TRAIN.NF, 100, lstm).to(device)
    netD = NetD(cfg.TRAIN.NF).to(device)

    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    if torch.cuda.is_available():
        text_encoder.cuda()

    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()
    return text_encoder, netG, dataset, device

if __name__ == "__main__":
    args = parse_args()
    merge_from_file_cfg(args.cfg_file)
    text_encoder, netG, dataset, device = build_model(gpu_id=args.gpu_id, data_dir=args.data_dir, manualSeed=args.manualSeed)
    ##########################################################################
    # predict one image from text
    text = 1 # "this flower is white with many petals"
    predict_one(text, text_encoder, netG, dataset, device)  # generate images for the whole valid dataset