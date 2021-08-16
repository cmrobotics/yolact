from data import COCODetection, get_label_map, MEANS, COLORS
from yolact.yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools
import random
from torch import nn
from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image
import operator, functools
from dataclasses import dataclass
import re
import matplotlib.pyplot as plt
import cv2

import faiss


@dataclass
class ImageData:
    x: float
    y: float
    rotation_z: float
    rotation_w: float
    side: str

def get_image_data(image_file):
    regular_expression = "IMAGE_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)_-serena\-camera\-(right|left)\-color\-image_raw_frame_\d+.png"
    result = re.search(regular_expression, image_file)
    if result:
        data = result.group(1).split()
        return ImageData(float(result.group(1).split()[0]) / 1000.0,
                        float(result.group(2).split()[0]) / 1000.0,
                        float(result.group(3).split()[0]) / 1000.0,
                        float(result.group(4).split()[0]) / 1000.0,
                        result.group(5).split())
    return None

class NearestNeighbourIndex():

    EPSILLON = 0.00001

    def __init__(self, output_folder,
            embedding_sizes=[
                [256, 69, 69],
                [256, 35, 35],
                [256, 18, 18],
                [256, 9, 9],
                [256, 5, 5]
            ],
            embedding_enabled=[ False, False, True, True, True ]):
        self.output_folder = output_folder
        self.embedding_sizes = embedding_sizes
        self.embedding_enabled = embedding_enabled
        self.embedding_flat_sizes = self.get_flat_sizes()
        self.number_layers = len([e for e in self.embedding_enabled if e])

    def create_indices(self, number_of_trees=20):
        indices = []
        for layer_index, embedding_size in enumerate(self.embedding_flat_sizes):
            if self.embedding_enabled[layer_index]:
                quantizer = faiss.IndexFlatL2(embedding_size)
                bytes_per_vector = 8
                bits_per_byte = 8
                nlist = 100
                index = faiss.IndexIVFPQ(quantizer, embedding_size, nlist, bytes_per_vector, bits_per_byte)
                index.cp.min_points_per_centroid = 5   # quiet warning
                index.quantizer_trains_alone = 2
                indices.append(index)
        return indices

    def create_pcas(self):
        pcas = []
        for layer_index, embedding_size in enumerate(self.embedding_flat_sizes):
            if self.embedding_enabled[layer_index]:
                pca = faiss.PCAMatrix (embedding_size, embedding_size // 4)
                pcas.append(index)
        return pcas

    def get_indices_file_paths(self):
        file_paths = []
        for layer_index, _ in enumerate(range(self.number_layers)):
            file_paths.append(f'{self.output_folder}/embeddings_{layer_index}.faiss')
        return file_paths

    def save_indices(self, indices):
        file_paths = self.get_indices_file_paths()
        for layer_index, index in enumerate(indices):
            faiss.write_index(index, file_paths[layer_index])

    def indices_created(self):
        file_paths =self.get_indices_file_paths()
        for path in file_paths:
            if not os.path.exists(path):
                return False
        return True

    def get_embeddings(self, indices, net:Yolact, path:str, sample_index_offset):
        frame = torch.from_numpy(cv2.imread(path)).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds, embeddings = net(batch)
        embeddings = [embedding for layer_index, embedding in enumerate(embeddings)
            if self.embedding_enabled[layer_index]]
        return embeddings

    def normalize_embeddings(self, embeddings_batch_in_layer, means, stds):
        embeddings_batch_in_layer = embeddings_batch_in_layer.view(embeddings_batch_in_layer.size()[0],
            embeddings_batch_in_layer.size()[1],
            -1)
        embeddings_batch_in_layer = (embeddings_batch_in_layer - means) / (stds + NearestNeighbourIndex.EPSILLON)
        embeddings_batch_in_layer = embeddings_batch_in_layer.view(embeddings_batch_in_layer.size()[0], -1)
        return embeddings_batch_in_layer

    def add_to_index(self, embeddings_batch, indices, means, stds):
        embeddings_count = len(embeddings_batch)
        for layer_index, embeddings_batch_in_layer in enumerate(embeddings_batch):
            index = indices[layer_index]
            if len(means) < embeddings_count or len(stds) < embeddings_count:
                mean = StyleExtractor.mean(embeddings_batch_in_layer)
                std  = StyleExtractor.standard_deviation(embeddings_batch_in_layer, mean)
                embeddings_batch_in_layer = self.normalize_embeddings(embeddings_batch_in_layer, mean, std)
                embeddings_batch_in_layer = embeddings_batch_in_layer.cpu().numpy()
                index.train(embeddings_batch_in_layer)
                index.add(embeddings_batch_in_layer)
                means.append(mean)
                stds.append(std)
            else:
                embeddings_batch_in_layer = self.normalize_embeddings(embeddings_batch_in_layer,
                    means[layer_index], stds[layer_index])
                embeddings_batch_in_layer = embeddings_batch_in_layer.cpu().numpy()
                index.add(embeddings_batch_in_layer)
        return means, stds

    def create_content_embedding_indices(self, input_folder, indices, batch_size=1000):
        print('**** START EMBEDDING INDICES GENERATION ****')
        sample_index_offset = 0
        embeddings_batch = None
        means, stds = [], []
        with open(f'{self.output_folder}/images.txt', 'w') as images_file:
            for file_path in Path(input_folder).glob('*'):
                path = str(file_path)
                name = os.path.basename(path)
                image_data = get_image_data(name)
                print(f"{sample_index_offset} - Processing: {image_data}")
                images_file.write(f"{name}\n")
                name = '.'.join(name.split('.')[:-1]) + '.png'
                out_path = os.path.join(self.output_folder, name)
                embeddings = self.get_embeddings(indices, net, path, sample_index_offset)
                if embeddings_batch is not None:
                    embeddings_batch = [torch.cat([embedding_batch, embeddings[layer_index]], 0) \
                        for layer_index, embedding_batch in enumerate(embeddings_batch)]
                else:
                    embeddings_batch = embeddings
                if sample_index_offset > 0 and sample_index_offset % ( batch_size - 1 ) == 0:
                    means, stds = self.add_to_index(embeddings_batch, indices, means, stds)
                    embeddings_batch = None
                sample_index_offset = sample_index_offset + 1
            if embeddings_batch is not None:
                means, stds = self.add_to_index(embeddings_batch, indices, means, stds)
        self.save_indices(indices)
        print('**** END EMBEDDING INDICES GENERATION ****')
        return means, stds

    def get_flat_sizes(self):
        embedding_flat_sizes = []
        for layer_index, embedding_size in enumerate(self.embedding_sizes):
            embedding_flat_sizes.append(functools.reduce(operator.mul, embedding_size, 1))
        return embedding_flat_sizes

    def load_indices(self):
        indices = []
        file_paths = self.get_indices_file_paths()
        for layer_index, file_path in enumerate(file_paths):
            indices.append(faiss.read_index(file_path))
        return indices

    def load_statistics(self):
        indices = []
        means = torch.load(f"{self.output_folder}/means.torch")
        print([print(mean.size()) for mean in means], "means")
        stds = torch.load(f"{self.output_folder}/stds.torch")
        return (means, stds)

    def index_images(self, net:Yolact, input_folder:str):
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
        if not self.indices_created():
            indices = self.create_indices(self.embedding_flat_sizes)
            means, stds = self.create_content_embedding_indices(input_folder, indices)
            torch.save(means, f"{self.output_folder}/means.torch")
            torch.save(stds,  f"{self.output_folder}/stds.torch")
        indices = self.load_indices()

class StyleExtractor(nn.Module):

    def __init__(self):
        super(StyleExtractor, self).__init__()

    def mean(embeddings):
        embeddings = embeddings.view(embeddings.size()[0], embeddings.size()[1], -1)
        means = embeddings.std(0)
        means = means.unsqueeze(0)
        return means

    def standard_deviation(embeddings, mean):
        embeddings = embeddings.view(embeddings.size()[0], embeddings.size()[1], -1)
        embeddings = embeddings - mean
        embeddings = embeddings.permute(0, 2, 1).contiguous()
        embeddings = embeddings.view(-1, embeddings.size()[2])
        stds = embeddings.std(0)
        stds = stds.unsqueeze(0).unsqueeze(2)
        return stds

    def style(embeddings, means, stds):
        embeddings = embeddings.view(embeddings.size()[0], embeddings.size()[1], -1)
        embeddings = embeddings - means
        embeddings = embeddings / ( stds + NearestNeighbourIndex.EPSILLON )
        style = torch.bmm(embeddings, embeddings.transpose(1, 2))
        style = style / embeddings.size(2)
        return style


from multiprocessing.pool import ThreadPool
from queue import Queue

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False, display_fps=False,
                        emulate_playback=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True

    if args.seed is not None:
        random.seed(args.seed)

def evaluate(net:Yolact, dataset, train_mode=False):
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug
    nn_index = NearestNeighbourIndex('indices')
    if args.images is not None:
        nn_index.index_images(net, args.images)
        return
if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.detect:
        cfg.eval_mask_branch = False

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        dataset = None

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()

        evaluate(net, dataset)