import os, glob, time, argparse, datetime, shutil, re
from functools import cmp_to_key
import fitz, cv2
import numpy as np
from sklearn.cluster import DBSCAN
import networkx as nx
from PIL import Image
from modules.layoutlmv3.model_init import Layoutlmv3_Predictor
from modules.ocr_utils import Painter, enlarge_rects, boxes_to_rects, points_in_rects
from modules.ocr_utils import rects_inte_rects, rects_union_rects, rects_area
from modules.ocr_utils import split_interval_excluding, get_cropped_image, extend_lists, clean_text
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import unimernet
from unimernet.common.config import Config as unimernet_Config
from surya.model.ordering.processor import load_processor as surya_load_processor
from surya.model.ordering.model import load_model as surya_load_model
from surya.ordering import batch_ordering
from rapid_table import RapidTable, VisTable
from rapidocr_paddle import RapidOCR
import utils
import threading
import logging
import fasttext
import numpy
import langid
import traceback
#修改pillow resize 图片分辨率限制
Image.MAX_IMAGE_PIXELS=2147483648
language_dict={"chinese":'ch'}
def layout_model_init(weight, config_file):
    model = Layoutlmv3_Predictor(weight, config_file)
    return model


def mfd_model_init(weight):
    mfd_model = YOLO(weight)
    return mfd_model


def mfr_model_init(weight_dir, device='cpu'):
    args = argparse.Namespace(cfg_path="configs/unimernet_config.yaml", options=None)
    cfg = unimernet_Config(args)
    cfg.config.model.pretrained = os.path.join(weight_dir, "unimernet_base.pth")
    cfg.config.model.model_config.model_name = weight_dir
    cfg.config.model.tokenizer_config.path = weight_dir
    task = unimernet.tasks.setup_task(cfg)
    model = task.build_model(cfg)
    model = model.to(device)
    vis_processor = unimernet.processors.load_processor('formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)
    return model, vis_processor


class MathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # if not pil image, then convert to pil image
        if isinstance(self.image_paths[idx], str):
            raw_image = Image.open(self.image_paths[idx])
        else:
            raw_image = self.image_paths[idx]
        if self.transform:
            image = self.transform(raw_image)
        return image


def pt_lang_map(lang):
    latin_lang = [
        'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
        'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
        'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
        'sw', 'tl', 'tr', 'uz', 'vi', 'french', 'german'
    ]
    arabic_lang = ['ar', 'fa', 'ug', 'ur']
    cyrillic_lang = [
        'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
        'dar', 'inh', 'che', 'lbe', 'lez', 'tab'
    ]
    devanagari_lang = [
        'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
        'sa', 'bgc'
    ]
    if lang in latin_lang:
        lang = "latin"
    elif lang in arabic_lang:
        lang = "arabic"
    elif lang in cyrillic_lang:
        lang = "cyrillic"
    elif lang in devanagari_lang:
        lang = "devanagari" 
    return lang
    

def pt_model_init(weight_dir, lang, use_cuda):
    lang_to_path = {
          'ch': ['ch_PP-OCRv3_det_infer', 'ch_PP-OCRv3_rec_slim_infer', 'ppocr_keys_v1'], 
          'en': ['en_PP-OCRv3_det_infer', 'en_PP-OCRv3_rec_infer', 'en_dict'], 
          'korean': ['Multilingual_PP-OCRv3_det_infer', 'korean_PP-OCRv4_rec_infer','korean_dict'],  
          'japan': ['Multilingual_PP-OCRv3_det_infer', 'japan_PP-OCRv4_rec_infer','japan_dict'], 
          'chinese_cht': ['Multilingual_PP-OCRv3_det_infer', 'japan_PP-OCRv4_rec_infer','chinese_cht_dict'], 
          'ta': ['Multilingual_PP-OCRv3_det_infer', 'ta_PP-OCRv4_rec_infer','ta_dict'],
          'te': ['Multilingual_PP-OCRv3_det_infer', 'te_PP-OCRv4_rec_infer','te_dict'],
          'ka': ['Multilingual_PP-OCRv3_det_infer', 'kannada_PP-OCRv4_rec_infer','ka_dict'],
          'latin': ['en_PP-OCRv3_det_infer', 'latin_PP-OCRv3_rec_infer','latin_dict'],
          'arabic': ['Multilingual_PP-OCRv3_det_infer', 'arabic_PP-OCRv4_rec_infer','arabic_dict'],
          'cyrillic': ['Multilingual_PP-OCRv3_det_infer', 'cyrillic_PP-OCRv3_rec_infer','cyrillic_dict'],
          'devanagari': ['Multilingual_PP-OCRv3_det_infer', 'devanagari_PP-OCRv4_rec_infer','devanagari_dict'],     
        }
    det_model_path = os.path.join(weight_dir, lang_to_path[lang][0])
    rec_model_path = os.path.join(weight_dir, lang_to_path[lang][1])
    rec_keys_path = os.path.join(os.path.join(weight_dir, 'dict'), lang_to_path[lang][2]+'.txt')
    pt_model = RapidOCR(config_path='configs/rapidocr_config.yaml', det_use_cuda=use_cuda, rec_use_cuda=use_cuda, 
                        det_model_path=det_model_path, rec_model_path=rec_model_path, rec_keys_path=rec_keys_path)
    return pt_model


def mfd(mfd_model, img, img_size=1888, conf_thres=0.25, iou_thres=0.45):
    mfd_res = mfd_model.predict(img, imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=True)[0] 
    mf_dets = [] 
    for xyxy, conf, cla in zip(mfd_res.boxes.xyxy.cpu(), mfd_res.boxes.conf.cpu(), mfd_res.boxes.cls.cpu()): 
        xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy] 
        mf_item = { 
            'category_id': 13 + int(cla.item()),
            'poly': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
            'score': round(float(conf.item()), 2),
            }
        mf_dets.append(mf_item)
    mf_dets = np.array(mf_dets, object)
    mf_boxes = np.array([mf_det['poly'] for mf_det in mf_dets], int).reshape([-1, 8])
    mf_rects = boxes_to_rects(mf_boxes).reshape([-1, 4])
    mf_points = np.transpose(mf_boxes.reshape([-1, 4, 2]), axes=(1, 0, 2))
    mf_scores = np.array([mf_det['score'] for mf_det in mf_dets])
    mf_cats = np.array([mf_det['category_id'] for mf_det in mf_dets])
    return mf_dets, mf_boxes, mf_rects, mf_points, mf_scores, mf_cats


def mf_refine(mf_dets, mf_boxes, mf_rects, mf_points, mf_scores, mf_cats):
    if not len(mf_dets):
        return mf_dets, mf_boxes, mf_rects, mf_points, mf_scores, mf_cats
    mf_inte_x_min, mf_inte_y_min, mf_inte_x_max, mf_inte_y_max, \
        mf_inte_area, mf_inte_valid = rects_inte_rects(mf_rects, mf_rects, return_valid=True)
    mf_rects_area = rects_area(mf_rects)
    mask_mf_sub = (mf_inte_area / mf_rects_area.reshape([-1, 1])) > 0.8
    mask_mf_dup = mask_mf_sub * mask_mf_sub.T
    mask_mf_sub = mask_mf_sub * (~mask_mf_dup)
    mask_mf_dup = mask_mf_dup * (((mf_scores[:, np.newaxis] - mf_scores) < 0) +\
        (((mf_scores[:, np.newaxis] - mf_scores) == 0) * (np.arange(len(mf_dets))[:, np.newaxis] - np.arange(len(mf_dets)) > 0)))
    inds_mf = ~np.any(mask_mf_dup + mask_mf_sub, axis=1)
    mf_dets = mf_dets[inds_mf]
    mf_boxes = mf_boxes[inds_mf]
    mf_rects = mf_rects[inds_mf]
    mf_points = mf_points[:, inds_mf, :]
    mf_scores = mf_scores[inds_mf]
    mf_cats = mf_cats[inds_mf]
    
    mf_union_rects_xmin, mf_union_rects_ymin, mf_union_rects_xmax, mf_union_rects_ymax, \
        mf_union_rects_area = rects_union_rects(mf_rects, mf_rects)
    mf_union_height = mf_union_rects_ymax - mf_union_rects_ymin
    mf_inte_x_min, mf_inte_y_min, mf_inte_x_max, mf_inte_y_max, \
        mf_inte_area, mf_inte_valid = rects_inte_rects(mf_rects, mf_rects, return_valid=False)
    mf_apart_width = np.where(mf_inte_x_min > mf_inte_x_max, mf_inte_x_min - mf_inte_x_max, 0)
    mf_inte_height = np.where(mf_inte_y_min < mf_inte_y_max, mf_inte_y_max - mf_inte_y_min, 0)
    mf_height = mf_rects[:, 3] - mf_rects[:, 1]
    mf_height = np.minimum(mf_height[:, np.newaxis], mf_height)
    mf_union_mask1 = np.ones([len(mf_cats), len(mf_cats)], bool)
    mf_union_mask1[(mf_cats==14)[:, np.newaxis] & (mf_cats==14).T] = False
    mf_union_mask1[mf_cats==14, mf_cats==14] = True
    mf_union_mask1 = mf_union_mask1 * ((mf_inte_height / mf_union_height) > 0.6) * (mf_apart_width < (mf_height * 0.1))
    mf_union_mask2 = (np.where(mf_inte_valid, mf_inte_area, 0) / mf_union_rects_area) > 0.6
    mf_union_mask = mf_union_mask1 + mf_union_mask2
    inline_union_mf_edges = list(zip(*list(np.where(mf_union_mask))))
    g = nx.Graph()
    g.add_edges_from(inline_union_mf_edges) # pass pairs here
    g_comps = [list(a) for a in list(nx.connected_components(g))] # get merged pairs here
    mf_boxes_merged = []
    mf_scores_merged = []
    mf_cats_merged = [] 
    for g_comp in g_comps:
        mf_cats_merged.append(np.max(mf_cats[g_comp]))
        mf_scores_merged.append(np.min(mf_scores[g_comp]))
        if len(g_comp) > 1:
            mf_rect = [np.min(mf_rects[g_comp][:, 0]), np.min(mf_rects[g_comp][:, 1]), np.max(mf_rects[g_comp][:, 2]), np.max(mf_rects[g_comp][:, 3])]
            mf_box = [mf_rect[0], mf_rect[1], mf_rect[2], mf_rect[1], mf_rect[2], mf_rect[3], mf_rect[0], mf_rect[3]]
            mf_boxes_merged.append(mf_box)      
        else:
            mf_boxes_merged.append(mf_boxes[g_comp[0]])
    mf_boxes = np.array(mf_boxes_merged)
    mf_scores = np.array(mf_scores_merged)
    mf_cats = np.array(mf_cats_merged)
    mf_rects = boxes_to_rects(mf_boxes)
    mf_points = np.transpose(mf_boxes.reshape([-1, 4, 2]), axes=(1, 0, 2))
    return mf_dets, mf_boxes, mf_rects, mf_points, mf_scores, mf_cats


def layout_det(layout_model, img, min_side_len=800): 
    w,h = img.size
    factor = np.clip(min_side_len / h if h < w else min_side_len / w, 0, 1)
    print(f"img size: {h} {w} {factor}")
    img_resized = img.resize((round(w*factor), round(h*factor)))
    #img_resized = cv2.resize(img, (round(w*factor), round(h*factor)))
    img_resized = cv2.cvtColor(numpy.asarray(img_resized),cv2.COLOR_RGB2BGR)

    layout_dets = layout_model(img_resized, ignore_catids=[])['layout_dets'] 
    for det in layout_dets:
        det['poly'] = [c / factor for c in det['poly']]
    return layout_dets


def layout_filter(layout_dets, mf_dets):
    mf_dets_iso = [
        {'category_id': det['category_id'],
         'poly': det['poly'],
         'score': 1,
         } for i, det in enumerate(mf_dets) if det['category_id'] == 14]
    # layout_dets = [det for det in layout_dets if det['category_id'] != 8]
    layout_dets.extend(mf_dets_iso)
    layout_dets = np.array(layout_dets, object)
    layout_boxes = np.array([layout_det['poly'] for layout_det in layout_dets]).reshape([-1, 8])
    layout_rects = boxes_to_rects(layout_boxes)
    layout_points = np.transpose(layout_boxes.reshape([-1, 4, 2]), axes=(1, 0, 2))
    layout_scores = np.array([det['score'] for det in layout_dets])
    layout_cats = np.array([det['category_id'] for det in layout_dets])
    return layout_dets, layout_boxes, layout_rects, layout_points, layout_scores, layout_cats
    
    
def layout_refine(layout_dets, layout_boxes, layout_rects, layout_points, layout_scores, layout_cats): 
    layout_inte_x_min, layout_inte_y_min, layout_inte_x_max, layout_inte_y_max, \
        layout_inte_area, layout_inte_valid = rects_inte_rects(layout_rects, layout_rects, return_valid=True)
    layout_rects_area = rects_area(layout_rects)
    mask_layout_sub = (layout_inte_area / layout_rects_area.reshape([-1, 1])) > 0.8
    mask_layout_dup = mask_layout_sub * mask_layout_sub.T
    mask_layout_sub = mask_layout_sub * (~mask_layout_dup)
    mask_layout_sub_fig = np.ones([len(layout_dets), len(layout_dets)], bool)
    fig_inds = (layout_cats==3).nonzero()[0]
    mask_layout_sub_fig[:, fig_inds] = False
    mask_layout_sub_fig[fig_inds, :] = False
    mask_layout_sub_fig[np.ix_(fig_inds, fig_inds)] = True
    mask_layout_dup = mask_layout_dup * (((layout_scores[:, np.newaxis] - layout_scores) < 0) +\
        (((layout_scores[:, np.newaxis] - layout_scores) == 0) * (np.arange(len(layout_dets))[:, np.newaxis] - np.arange(len(layout_dets)) > 0)))
    inds_layout = ~np.any(mask_layout_dup + mask_layout_sub * mask_layout_sub_fig, axis=1)
    layout_dets = layout_dets[inds_layout]
    layout_boxes = layout_boxes[inds_layout]
    layout_rects = layout_rects[inds_layout]
    layout_scores = layout_scores[inds_layout]
    layout_cats = layout_cats[inds_layout]
    layout_points = layout_points[:, inds_layout, :]
    mask_layout_sub = mask_layout_sub[np.ix_(inds_layout, inds_layout)]
    for det in layout_dets:
        det['sub'] = []
    sub, sup = np.where(mask_layout_sub)
    for j in range(len(sup)):
        layout_dets[sup[j]]['sub'].append(sub[j])
    for det in layout_dets:
        det['sub'] = np.array(det['sub'])
    return layout_dets, layout_boxes, layout_rects, layout_points, layout_scores, layout_cats


def layout_order(surya_model, surya_processor, pil_img, layout_dets, layout_boxes, layout_rects, layout_points, layout_scores, layout_cats):
    layout_order = np.argsort([res['position'] for res in batch_ordering([pil_img], [layout_rects.tolist()], surya_model, surya_processor)[0].dict()['bboxes']]) 
    layout_dets = layout_dets[layout_order] 
    layout_boxes = layout_boxes[layout_order] 
    layout_rects = layout_rects[layout_order] 
    layout_points = layout_points[:, layout_order, :]
    layout_scores = layout_scores[layout_order] 
    layout_cats = layout_cats[layout_order]   
    return layout_dets, layout_boxes, layout_rects, layout_points, layout_scores, layout_cats


def ptd(pt_model, img): 
    pt_boxes = pt_model.text_det(img)[0].reshape([-1, 8]) 
    pt_rects = boxes_to_rects(pt_boxes)
    return pt_boxes, pt_rects


def pt_split(pt_boxes, pt_rects, mf_rects): 
    mf_inte_pt_xmin, mf_inte_pt_ymin, mf_inte_pt_xmax, mf_inte_pt_ymax, mf_inte_pt_area, mask_mf_inte_pt = \
        rects_inte_rects(mf_rects, pt_rects)   
    mf_height, pt_height = mf_rects[:, 3] - mf_rects[:, 1], pt_rects[:, 3] - pt_rects[:, 1]
    mask_mf_break_pt = ((mf_inte_pt_ymax - mf_inte_pt_ymin) / np.minimum(mf_height[:, np.newaxis], pt_height)) > 0.5
    mf_break_inds, pt_break_inds = np.where(mask_mf_break_pt)
    pt_boxes_splited = np.expand_dims(pt_boxes, 1).tolist()
    pt_segs = [np.array([0], dtype=int)] * len(pt_boxes)
    mf_segs = [np.array([], dtype=int)] * len(pt_boxes)
    for pt_ind in np.unique(pt_break_inds):
        pt_x = pt_rects[pt_ind][[0, 2]]
        pt_y = pt_rects[pt_ind][[1, 3]]
        mf_inds = mf_break_inds[pt_break_inds==pt_ind]
        mf_x = mf_rects[mf_inds][:, [0, 2]]
        pt_x_splited = split_interval_excluding(pt_x, mf_x)
        pt_rect = np.hstack([pt_x_splited[:, 0].reshape(-1, 1), np.repeat(pt_y[0], len(pt_x_splited)).reshape([-1, 1]),\
                              pt_x_splited[:, 1].reshape(-1, 1), np.repeat(pt_y[1], len(pt_x_splited)).reshape([-1, 1])])
        pt_box = np.array([pt_rect[:, 0], pt_rect[:, 1], pt_rect[:, 2], pt_rect[:, 1], pt_rect[:, 2], pt_rect[:, 3], pt_rect[:, 0], pt_rect[:, 3]]).transpose().tolist()
        pt_boxes_splited[pt_ind] = pt_box
        pt_segs[pt_ind] = np.arange(len(pt_box))
        mf_segs[pt_ind] = mf_inds
    pt_segs = [pt_segs[j] + np.append([0], np.cumsum([len(l) for l in pt_segs])[:-1])[j] for j in range(len(pt_segs))]  
    pt_boxes = np.array(extend_lists(pt_boxes_splited), int).reshape([-1, 8])
    pt_points = np.transpose(pt_boxes.reshape([-1, 4, 2]), axes=(1, 0, 2))
    pt_rects = boxes_to_rects(pt_boxes)
    seg_ptmf = [np.hstack([pt_segs[j], mf_segs[j]+len(pt_rects)]) for j in range(len(pt_segs))]
    mask_ptmf_seg = np.zeros([len(pt_rects)+len(mf_rects), len(pt_rects)+len(mf_rects)], bool)
    for j in range(len(seg_ptmf)):
        mask_ptmf_seg[np.ix_(seg_ptmf[j], seg_ptmf[j])] = True
    return pt_boxes, pt_rects, pt_points, mask_ptmf_seg


def pt_att_layout(pt_rects, pt_points, layout_rects):
    pt_rects_core = enlarge_rects(pt_rects)
    pt_inte_layout_xmin, pt_inte_layout_ymin, pt_inte_layout_xmax, pt_inte_layout_ymax, pt_inte_layout_area, mask_pt_inte_layout = \
        rects_inte_rects(pt_rects_core, layout_rects)
    pt_rects_area = rects_area(pt_rects_core) 
    pt_inte_layout_ratio = pt_inte_layout_area / pt_rects_area.reshape([-1, 1]) 
    mask_pt_att_layout = pt_inte_layout_ratio > 0.7
    return mask_pt_att_layout


def mf_att_layout(mf_rects, mf_points, layout_rects):
    mf_rects_core = enlarge_rects(mf_rects)
    mf_inte_layout_xmin, mf_inte_layout_ymin, mf_inte_layout_xmax, mf_inte_layout_ymax, mf_inte_layout_area, mask_mf_inte_layout = \
        rects_inte_rects(mf_rects_core, layout_rects)
    mf_rects_area = rects_area(mf_rects_core)

    mf_inte_layout_ratio = mf_inte_layout_area / mf_rects_area.reshape([-1, 1])
    mask_mf_att_layout = mf_inte_layout_ratio > 0.7
    return mask_mf_att_layout


def ptr(pt_model, img, pt_rects, pt_points): 
    rects = pt_rects.copy()
    points = pt_points.copy()
    # cropped_img_width = np.max([np.linalg.norm(pt_points[0] - pt_points[1], axis=1), np.linalg.norm(pt_points[2] - pt_points[3], axis=1)], axis=0).astype(int) 
    cropped_img_height = np.max([np.linalg.norm(pt_points[0] - pt_points[3], axis=1), np.linalg.norm(pt_points[1] - pt_points[2], axis=1)], axis=0).astype(int) 
    cropped_img_factor = np.clip(48/cropped_img_height, 0, 1) 
    valid = np.full(len(pt_rects), True)
    cropped_imgs = []
    for i in range(len(rects)): 
        xmin, ymin, xmax, ymax = rects[i]
        xmin, ymin = np.floor([xmin, ymin]).astype(int)
        xmax, ymax = np.ceil([xmax, ymax]).astype(int)
        pts = points[:, i, :]
        pts[:, 0] = pts[:, 0] - xmin
        pts[:, 1] = pts[:, 1] - ymin
        new_size = [round((xmax-xmin+1)*cropped_img_factor[i]), round((ymax-ymin+1)*cropped_img_factor[i])]
        if not all(new_size):
            valid[i] = False
        else:
            temp_img = cv2.resize(img[ymin:(ymax+1), xmin:(xmax+1)], new_size)
            pts = pts * cropped_img_factor[i]
            pts[:, 0] = np.floor(pts[:, 0])
            pts[:, 1] = np.ceil(pts[:, 1])
            cropped_img = get_cropped_image(temp_img, pts.astype(np.float32))
            h, w = cropped_img.shape[0:2]
            if h > 48:
                cropped_img = cv2.resize(cropped_img, (round(w*48/h), 48))
            cropped_imgs.append(cropped_img)
    texts, scores = tuple(map(list, list(zip(*pt_model.text_rec(cropped_imgs)[0])))) if len(cropped_imgs) else ([], [])
    texts = list(map(lambda x: x.replace('$', '\$'), texts))
    ptr_texts = np.full(len(pt_rects), '', object)
    ptr_texts[valid] = texts
    ptr_scores = np.full(len(pt_rects), 0, np.float32)
    ptr_scores[valid] = scores
    return ptr_texts, ptr_scores
 
    
def mfr(mfr_model, mfr_transform, device, img, mf_rects, mf_points, mf_cats): 
    rects = mf_rects.copy()
    points = mf_points.copy()
    # cropped_img_width = np.max([np.linalg.norm(mf_points[0] - mf_points[1], axis=1), np.linalg.norm(mf_points[2] - mf_points[3], axis=1)], axis=0).astype(int) 
    cropped_img_height = np.max([np.linalg.norm(mf_points[0] - mf_points[3], axis=1), np.linalg.norm(mf_points[1] - mf_points[2], axis=1)], axis=0).astype(int) 
    cropped_img_factor = np.clip(48/cropped_img_height, 0, 1) 
    #cropped_img_factor = np.full(cropped_img_height.shape, 1)
    cropped_img_factor = np.where(mf_cats == 13, cropped_img_factor, 1)
    valid = np.full(len(mf_rects), True)
    cropped_imgs = []
    for i in range(len(mf_rects)): 
        xmin, ymin, xmax, ymax = rects[i]
        xmin, ymin = np.floor([xmin, ymin]).astype(int)
        xmax, ymax = np.ceil([xmax, ymax]).astype(int)
        pts = points[:, i, :]
        pts[:, 0] = pts[:, 0] - xmin
        pts[:, 1] = pts[:, 1] - ymin
        new_size = [round((xmax-xmin+1)*cropped_img_factor[i]), round((ymax-ymin+1)*cropped_img_factor[i])]
        if not all(new_size):
            valid[i] = False
        else:
            temp_img = cv2.resize(img[ymin:(ymax+1), xmin:(xmax+1)], new_size)
            pts = pts * cropped_img_factor[i]
            pts[:, 0] = np.floor(pts[:, 0])
            pts[:, 1] = np.ceil(pts[:, 1])
            cropped_img = get_cropped_image(temp_img, pts.astype(np.float32))
            h, w = cropped_img.shape[0:2]
            if h > 48:
                cropped_img = cv2.resize(cropped_img, (round(w*48/h), 48))
            cropped_imgs.append(cropped_img)
    cropped_imgs = [Image.fromarray(img) for img in cropped_imgs]
    mfr_dataset = MathDataset(cropped_imgs, transform=mfr_transform)
    mfr_dataloader = DataLoader(mfr_dataset, batch_size=128, num_workers=0)
    texts = [] 
    for imgs in mfr_dataloader: 
        imgs = imgs.to(device)
        texts.extend(mfr_model.generate({'image': imgs})['pred_str'])
    texts = list(map(lambda x: x.replace('$', '\$'), texts))
    texts = ['${}$'.format(text) if mf_cats[i]==13 else '$${}$$'.format(text) for i, text in enumerate(texts)]
    mfr_texts = np.full(len(mf_rects), '', object)
    mfr_texts[valid] = texts
    return mfr_texts


def ptmf(pt_boxes, pt_rects, pt_points, ptr_texts, ptr_scores, mask_pt_att_layout, mf_boxes, mf_rects, mf_points, mfr_texts, mask_mf_att_layout): 
    ptmf_boxes = np.vstack([pt_boxes, mf_boxes])
    ptmf_rects = np.vstack([pt_rects, mf_rects])
    ptmf_points = np.concatenate([pt_points, mf_points], axis=1)
    ptrmfr_texts = np.hstack([ptr_texts, mfr_texts]).astype(object) 
    ptrmfr_scores = np.hstack([ptr_scores, np.ones(len(mfr_texts))])    
    mask_ptmf_att_layout = np.concatenate([mask_pt_att_layout, mask_mf_att_layout], axis=0)
    return ptmf_boxes, ptmf_rects, ptmf_points, ptrmfr_texts, ptrmfr_scores, mask_ptmf_att_layout
  
    
def layout_ptmf(layout_dets, ptmf_boxes, ptmf_rects, ptrmfr_texts, ptrmfr_scores, mask_ptmf_att_layout, mask_ptmf_seg):
    ptmf_inte_x_min, ptmf_inte_y_min, ptmf_inte_x_max, ptmf_inte_y_max, \
        ptmf_inte_area, ptmf_inte_valid = rects_inte_rects(ptmf_rects, ptmf_rects, return_valid=False) 
    ptmf_inte_height = ptmf_inte_y_max - ptmf_inte_y_min
    # ptmf_inte_width = ptmf_inte_x_max - ptmf_inte_x_min
    ptmf_height = ptmf_rects[:, 3] - ptmf_rects[:, 1] 
    ptmf_min_height = np.minimum(ptmf_height[:, np.newaxis], ptmf_height)
    mask_ptmf_line = (ptmf_inte_height / ptmf_min_height) > 0.5 
    ptmf_order = np.where(mask_ptmf_line, ptmf_rects[:, 2][:, np.newaxis] - ptmf_rects[:, 2], ptmf_rects[:, 3][:, np.newaxis] - ptmf_rects[:, 3]) 
    ch_pattern = r'\u4e00-\u9fa5'
    ja_pattern = r'\u3040-\u309f\u30a0-\u30ff\u31f0-\u31ff'
    chpu_pattern = r'\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b'
    enpu_pattern = r'\.\,\?\!\(\)\[\]\{\}\<\>\:\;\'\"\\\/\-\=\+\*\&\^\%\$\#\@\|\~\`\_\s'
    nu_pattern = r'0-9〇一二三四五六七八九十百千万'
    hy_pattern = r'\S-'
    mask_ptmf_p1 = np.array(list(map(lambda x: bool(re.search(r'^[{}{}{}]'.format(ch_pattern, ja_pattern, chpu_pattern), x)), ptrmfr_texts)))
    mask_ptmf_p2 = np.array(list(map(lambda x: bool(re.search(r'([{}{}{}]|{})$'.format(ch_pattern, ja_pattern, chpu_pattern, hy_pattern), x)), ptrmfr_texts)))
    mask_ptmf_p3 = np.array(list(map(lambda x: bool(re.search(r'[{}{}{}]$'.format(ch_pattern, ja_pattern, chpu_pattern), x)), ptrmfr_texts)))
    mask_ptmf_p4 = np.array(list(map(lambda x: bool(re.search(r'[{}]'.format(nu_pattern), x[-3:]))\
                                     | bool(re.fullmatch(r'[{}{}{}]*'.format(chpu_pattern, enpu_pattern, nu_pattern), x)), ptrmfr_texts)))
    for i_layout, det in enumerate(layout_dets):
        ptmf_ids = mask_ptmf_att_layout[:, i_layout].nonzero()[0]
        # ptmf_id_to_ind = {ptmf_id: ptmf_ind for ptmf_ind, ptmf_id in enumerate(ptmf_ids)}
        layout_ptmf_order = sorted(range(len(ptmf_ids)), key=lambda i: cmp_to_key(lambda x, y: ptmf_order[x, y])(ptmf_ids[i]))
        ptmf_ids = ptmf_ids[layout_ptmf_order]
        if len(ptmf_ids):
            ptmf_is_seg_end = ~mask_ptmf_seg[ptmf_ids[:-1], ptmf_ids[1:]]
            ptmf_is_seg_end = np.hstack([ptmf_is_seg_end, [True]])
            ptmf_is_line_end = ~mask_ptmf_line[ptmf_ids[:-1], ptmf_ids[1:]]
            ptmf_is_line_end = np.hstack([ptmf_is_line_end, [True]])
            ptmf_is_new_line = ~mask_ptmf_line[ptmf_ids[1:], ptmf_ids[:-1]]
            ptmf_is_new_line = np.hstack([[True], ptmf_is_new_line])
            line_ptmf = np.split(ptmf_ids, np.where(ptmf_is_new_line)[0][1:])
        
            line_is_new_para = np.zeros(len(line_ptmf)+1, bool)
            ls_ptmf_ids = [line[0] for line in line_ptmf]
            le_ptmf_ids = [line[-1] for line in line_ptmf]
            eps = np.mean(ptmf_height[ptmf_ids])*0.5
            ls_x = np.mean(ptmf_boxes[:, [0, 6]][ls_ptmf_ids], axis=1).reshape([-1, 1])
            ls_labels = DBSCAN(eps=eps, min_samples=2).fit(ls_x).labels_
            le_x = np.mean(ptmf_boxes[:, [2, 4]][le_ptmf_ids], axis=1)
            le_x_diff = np.abs(le_x[:, np.newaxis] - le_x)
            le_num = np.sum((np.abs(le_x_diff) < eps).astype(int), axis=1)
            le_ind = np.argmax(np.where(le_num>5, le_x, 0))
            if max(ls_labels)==1 and min(ls_labels)==0:
                x0, x1 = le_x[(ls_labels[1:]==0).nonzero()[0]], le_x[(ls_labels[1:]==1).nonzero()[0]]
                m0, m1 = np.mean(x0), np.mean(x1)
                v0, v1 = np.mean(np.abs(x0 - m0)), np.mean(np.abs(x1 - m1))
                new_para_label = np.argmin([m0, m1])
                if [m0, m1][1-new_para_label] - [m0, m1][new_para_label] > 5 * [v0, v1][1-new_para_label]:
                    line_is_new_para[(ls_labels==new_para_label).nonzero()[0]] = True
                if [m0, m1][1-new_para_label] - le_x[-1] > 5 * [v0, v1][1-new_para_label]:
                    line_is_new_para[-1] = True
            elif all(ls_labels==0):
                if le_num[le_ind] > 5:
                    line_is_new_para[(le_x_diff[le_ind] > 15 * eps).nonzero()[0] + 1] = True
            le_is_idx = mask_ptmf_p4[le_ptmf_ids] * (le_x_diff[le_ind] < 3 * eps)
            if np.sum(le_is_idx) / len(line_ptmf) > 0.8:
                line_is_new_para[(le_x_diff[le_ind] < 3 * eps).nonzero()[0] + 1] = True
            ptmf_is_new_para = ptmf_is_new_line.copy()
            ptmf_is_new_para[ptmf_is_new_line] = line_is_new_para[:-1]
        else: 
            ptmf_is_seg_end = ptmf_is_line_end = ptmf_is_new_line = ptmf_is_new_para = np.array([])
            line_is_new_para = np.array([False])
        det['ptmf_boxes'] = ptmf_boxes[ptmf_ids]
        det['ptmf_rects'] = ptmf_rects[ptmf_ids]
        det['ptrmfr_texts'] = ptrmfr_texts[ptmf_ids]
        det['content'] = list(map(lambda x: re.sub(r'(?<! ) (?! )', '  ', x), det['ptrmfr_texts']))
        det['content'] = np.where(ptmf_is_seg_end*(1-ptmf_is_line_end), list(map(lambda x: x+'  ', det['content'])), det['content'])       
        det['content'] = np.where((1-ptmf_is_seg_end)*(1-ptmf_is_line_end)*(1-mask_ptmf_p3[ptmf_ids]), list(map(lambda x: x+' ', det['content'])), det['content'])
        det['content'] = np.where(ptmf_is_line_end*(1-mask_ptmf_p2[ptmf_ids]), list(map(lambda x: x+' ', det['content'])), det['content'])
        det['content'] = np.where(1-mask_ptmf_p1[ptmf_ids], list(map(lambda x: ' '+x, det['content'])), det['content'])
        det['content'] = np.where(ptmf_is_new_para, list(map(lambda x: '\n\n'+x, det['content'])),  det['content'])
        if line_is_new_para[-1]:
            det['content'][-1] = det['content'][-1]+'\n\n'
        det['content'] = ''.join(det['content'])
    return


def layout_figure(pil_img, layout_dets, layout_rects, output_dir, upload_dir, figure_offset, page_id):
    fig_dets = [(layout_ind, det) for layout_ind, det in enumerate(layout_dets) if det['category_id'] == 3]
    figcap_dets = [(layout_ind, det) for layout_ind, det in enumerate(layout_dets) if det['category_id'] == 4 or 
                   (det['category_id'] == 1 and bool(re.search('^[\s]*fig', det['content'].lower())))]
    fig_inds, fig_dets = map(list, list(zip(*fig_dets))) if len(fig_dets) else ([], [])
    figcap_inds, figcap_dets = map(list, list(zip(*figcap_dets))) if len(figcap_dets) else ([], [])
    figcap_adj_num = [2 if det['category_id'] == 4 else 1 for det in figcap_dets]
    mask_fig_adj_cap = (np.abs(np.array(fig_inds)[:, np.newaxis] - np.array(figcap_inds))) \
        - np.array([np.sum(np.sign(fig_dets[i]['sub']-fig_inds[i]) + np.sign(fig_dets[i]['sub']-figcap_inds[j]) \
        == 0) for i in range(len(fig_dets)) for j in range(len(figcap_dets))]).reshape(len(fig_dets), len(figcap_dets)) \
        <= np.array(figcap_adj_num)[np.newaxis, :]
    fig_rects, figcap_rects = layout_rects[fig_inds], layout_rects[figcap_inds]
    fig_inte_cap_x_min, fig_inte_cap_y_min, fig_inte_cap_x_max, fig_inte_cap_y_max, \
        fig_inte_cap_area, fig_inte_cap_valid = rects_inte_rects(fig_rects, figcap_rects, return_valid=False)
    fig_width, fig_height = fig_rects[:, 2] - fig_rects[:, 0], fig_rects[:, 3] - fig_rects[:, 1]
    figcap_width, figcap_height = figcap_rects[:, 2] - figcap_rects[:, 0], figcap_rects[:, 3] - figcap_rects[:, 1]
    mask_fig_inte_cap_x = ((fig_inte_cap_x_max - fig_inte_cap_x_min) / np.minimum(fig_width[:, np.newaxis], figcap_width)) > 0.7
    mask_fig_inte_cap_y = ((fig_inte_cap_y_max - fig_inte_cap_y_min) / np.minimum(fig_height[:, np.newaxis], figcap_height)) > 0.7
    dist_fig_cap = np.full(mask_fig_inte_cap_x.shape, np.inf)
    dist_fig_cap = np.where(mask_fig_inte_cap_x, fig_inte_cap_y_min - fig_inte_cap_y_max, dist_fig_cap)
    dist_fig_cap = np.where(mask_fig_inte_cap_y, fig_inte_cap_x_min - fig_inte_cap_x_max, dist_fig_cap)
    dist_fig_cap = np.where(mask_fig_inte_cap_x * mask_fig_inte_cap_y, 0, dist_fig_cap)
    dist_fig_cap[~mask_fig_adj_cap] = np.inf
    if all(dist_fig_cap.shape): 
        inds_cap_att_fig = np.argmin(dist_fig_cap, axis=0)  
        inds_cap_att_fig = np.where(dist_fig_cap[inds_cap_att_fig, np.arange(dist_fig_cap.shape[1])] < np.inf, inds_cap_att_fig, None)
    else:
        inds_cap_att_fig = np.full(dist_fig_cap.shape[1], None)      
    valid = inds_cap_att_fig != None
    figcap_dets = [det for i_figcap, det in enumerate(figcap_dets) if valid[i_figcap]]
    inds_cap_att_fig, figcap_rects = inds_cap_att_fig[valid], figcap_rects[valid]
    figcap_width, figcap_height = figcap_width[valid], figcap_height[valid]
    
    figures = []
    for i_fig, det in enumerate(fig_dets):
        xmin, ymin, xmax, ymax = np.round(fig_rects[i_fig]).astype(int)
        cropped_img = pil_img.crop([xmin, ymin, xmax, ymax])
        url = 'figures/{}-{}-whxyp_{}_{}_{}_{}_{}.jpg'.format(
            datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"),
            figure_offset + i_fig + 1,
            round(fig_width[i_fig]/pil_img.width, 4),
            round(fig_height[i_fig]/pil_img.height, 4), 
            round(fig_rects[i_fig, 0]/pil_img.width, 4), 
            round(fig_rects[i_fig, 1]/pil_img.height, 4), 
            int(page_id),
            )                                                    
        cropped_img.save(os.path.join(output_dir, url))
        det['url'] = url   
        #utils.upload_file(os.path.join(upload_dir, url), os.path.join(output_dir, url))
        figures.append({'location': [[xmin, ymin, xmax, ymax], int(page_id)],
                        'oss_path': os.path.join(upload_dir, url),
                        'figure_id': figure_offset + i_fig + 1,
                        'caption': []
                        })  
    for i_figcap, det in enumerate(figcap_dets):
        url = fig_dets[inds_cap_att_fig[i_figcap]]['url'].split('-whxyp')[0]
        url = '{}-whxyp_{}_{}_{}_{}_{}.txt'.format(
            url, 
            round(figcap_width[i_figcap]/pil_img.width, 4),
            round(figcap_height[i_figcap]/pil_img.height, 4),
            round(figcap_rects[i_figcap, 0]/pil_img.width, 4), 
            round(figcap_rects[i_figcap, 1]/pil_img.height, 4),
            int(page_id))      
        content = clean_text(det['content'])                                       
        with open(os.path.join(output_dir, url), 'w', encoding='utf-8') as f:
            f.write(content)
        # utils.upload_file(os.path.join(upload_dir, url), os.path.join(output_dir, url))
        figures[inds_cap_att_fig[i_figcap]]['caption'].append(content)
    return figures
    

def layout_table(table_model, img, layout_dets, layout_rects, output_dir, upload_dir, table_offset, page_id):
    tab_dets = [(layout_ind, det) for layout_ind, det in enumerate(layout_dets) if det['category_id'] == 5]
    tabcap_dets = [(layout_ind, det) for layout_ind, det in enumerate(layout_dets) if det['category_id'] in {6, 7} or 
                   (det['category_id'] == 1 and bool(re.search('^[\s]*table', det['content'].lower())))]
    tab_inds, tab_dets = map(list, list(zip(*tab_dets))) if len(tab_dets) else ([], [])
    tabcap_inds, tabcap_dets = map(list, list(zip(*tabcap_dets))) if len(tabcap_dets) else ([], [])
    tabcap_adj_num = [2 if det['category_id'] in {6, 7} else 1 for det in tabcap_dets]
    mask_tab_adj_cap = (np.abs(np.array(tab_inds)[:, np.newaxis] - np.array(tabcap_inds))) <= np.array(tabcap_adj_num)[np.newaxis, :]
    tab_rects, tabcap_rects = layout_rects[tab_inds], layout_rects[tabcap_inds]
    tab_inte_cap_x_min, tab_inte_cap_y_min, tab_inte_cap_x_max, tab_inte_cap_y_max, \
        tab_inte_cap_area, tab_inte_cap_valid = rects_inte_rects(tab_rects, tabcap_rects, return_valid=False)
    tab_width, tab_height = tab_rects[:, 2] - tab_rects[:, 0], tab_rects[:, 3] - tab_rects[:, 1]
    tabcap_width, tabcap_height = tabcap_rects[:, 2] - tabcap_rects[:, 0], tabcap_rects[:, 3] - tabcap_rects[:, 1]
    mask_tab_inte_cap_x = ((tab_inte_cap_x_max - tab_inte_cap_x_min) / np.minimum(tab_width[:, np.newaxis], tabcap_width)) > 0.7
    mask_tab_inte_cap_y = ((tab_inte_cap_y_max - tab_inte_cap_y_min) / np.minimum(tab_height[:, np.newaxis], tabcap_height)) > 0.7
    dist_tab_cap = np.full(mask_tab_inte_cap_x.shape, np.inf)
    dist_tab_cap = np.where(mask_tab_inte_cap_x, tab_inte_cap_y_min - tab_inte_cap_y_max, dist_tab_cap)
    dist_tab_cap = np.where(mask_tab_inte_cap_y, tab_inte_cap_x_min - tab_inte_cap_x_max, dist_tab_cap)
    dist_tab_cap = np.where(mask_tab_inte_cap_x * mask_tab_inte_cap_y, 0, dist_tab_cap)
    dist_tab_cap[~mask_tab_adj_cap] = np.inf
    if all(dist_tab_cap.shape): 
        inds_cap_att_tab = np.argmin(dist_tab_cap, axis=0)  
        inds_cap_att_tab = np.where(dist_tab_cap[inds_cap_att_tab, np.arange(dist_tab_cap.shape[1])] < np.inf, inds_cap_att_tab, None)
    else:
        inds_cap_att_tab = np.full(dist_tab_cap.shape[1], None)      
    valid = inds_cap_att_tab != None
    tabcap_dets = [det for i_tabcap, det in enumerate(tabcap_dets) if valid[i_tabcap]]
    inds_cap_att_tab, tabcap_rects = inds_cap_att_tab[valid], tabcap_rects[valid]
    tabcap_width, tabcap_height = tabcap_width[valid], tabcap_height[valid]

    tabcap_dets = [det for i_tabcap, det in enumerate(tabcap_dets) if inds_cap_att_tab[i_tabcap] != None]
    tables = []
    for i_tab, det in enumerate(tab_dets): 
        xmin, ymin, xmax, ymax = np.round(tab_rects[i_tab]).astype(int)
        cropped_img = img[ymin:ymax, xmin:xmax]
        tab_ptmf_boxes = det['ptmf_boxes'].copy()
        tab_ptmf_boxes[:, [0, 2, 4, 6]] -= xmin
        tab_ptmf_boxes[:, [1, 3, 5, 7]] -= ymin
        tab_ptmf_result = list(map(list, list(zip(*[tab_ptmf_boxes.reshape([-1, 4, 2]).tolist(), det['ptrmfr_texts'], np.ones(len(det['ptmf_boxes']), np.float32)]))))
        tab_html_str, tab_cell_bboxes, elapsed = table_model(cropped_img, tab_ptmf_result)
        url = 'tables/{}-{}-whxyp_{}_{}_{}_{}_{}.html'.format(
            datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"),
            table_offset + i_tab + 1,
            round(tab_width[i_tab]/img.shape[1], 4),
            round(tab_height[i_tab]/img.shape[0], 4), 
            round(tab_rects[i_tab, 0]/img.shape[1], 4), 
            round(tab_rects[i_tab, 1]/img.shape[0], 4),
            int(page_id))                                    
        with open(os.path.join(output_dir, url), 'w', encoding='utf-8') as f:
            f.write(tab_html_str)
        cv2.imwrite(os.path.join(output_dir, url.replace('html', 'jpg')), cropped_img)
        det['url'] = url 
        det['html'] = tab_html_str
        #utils.upload_file(os.path.join(upload_dir, url), os.path.join(output_dir, url))
        tables.append({'location': [[xmin, ymin, xmax, ymax], int(page_id)],
                        'oss_path': os.path.join(upload_dir, url),
                        'table_id': table_offset + i_tab + 1,
                        'caption': []
                        })  
    for i_tabcap, det in enumerate(tabcap_dets):
        url = tab_dets[inds_cap_att_tab[i_tabcap]]['url'].split('-whxyp')[0]
        url = '{}-whxyp_{}_{}_{}_{}_{}.txt'.format(
            url, 
            round(tabcap_width[i_tabcap]/img.shape[1], 4),
            round(tabcap_height[i_tabcap]/img.shape[0], 4),
            round(tabcap_rects[i_tabcap, 0]/img.shape[1], 4),
            round(tabcap_rects[i_tabcap, 1]/img.shape[0], 4),
            int(page_id))  
        content = clean_text(det['content'])                                             
        with open(os.path.join(output_dir, url), 'w', encoding='utf-8') as f:
            f.write(content)
        # utils.upload_file(os.path.join(upload_dir, url), os.path.join(output_dir, url))
        tables[inds_cap_att_tab[i_tabcap]]['caption'].append(content)
    return tables


def layout_content(layout_dets, output_dir, page_id, cat2names): 
    content = ''
    for layout_ind, det in enumerate(layout_dets):
        if cat2names[det['category_id']] in ["title"]:    
            content += '\n\n#  {}\n\n'.format(det['content'])
        elif cat2names[det['category_id']] in ["plain_text"]:
            content += '\n\n{}\n\n'.format(det['content'])
        elif cat2names[det['category_id']] in ["isolate_formula", "isolated_formula", "figure_caption", "table_caption", "formula_caption", "table_footnote"]:
            content += '\n\n{}\n\n'.format(det['content'])
        elif cat2names[det['category_id']] in ["figure"]:
            content += '\n\n![]({})\n\n'.format(det['url'])
        elif cat2names[det['category_id']] in ["table"]:
            content += '\n\n![]({})\n\n'.format(det['url'])
            #content += '{}\n\n'.format(det['html'])
    with open(os.path.join(output_dir, 'debug/{}.md'.format(page_id)), 'w', encoding='utf-8') as f:
        f.write(clean_text(content))
    return content


def debug(pil_img, layout_dets, layout_rects, pt_rects, mf_rects, output_dir, page_id, cat2names):
    s = np.where(pil_img.width > pil_img.height, 3000/pil_img.width, 3000/pil_img.height)
    font_size = max(np.mean(pt_rects[:, 3]*s - pt_rects[:, 1]*s), 10) if len(pt_rects) else 40
    width = round(max(font_size/15, 3))
    painter = Painter()
    painter.create('temp', pil_img.resize([round(pil_img.width*s), round(pil_img.height*s)]))
    for i, det in enumerate(layout_dets): 
        painter.rectangle('temp', layout_rects[i]*s, 'blue', width)
        painter.text('temp', [layout_rects[i][0]*s, layout_rects[i][1]*s], str(i), 'blue', font_size=font_size*1.5)
        for j, rect in enumerate(det['ptmf_rects']*s):
            painter.rectangle('temp', rect, 'red', width)
            painter.text('temp', [rect[2], rect[1]], '{}'.format(j), 'red', font_size=font_size/1.5)
    painter.save('temp', os.path.join(output_dir, 'debug/{}_a.jpg'.format(page_id)))
    
    painter.create('temp', pil_img.resize([round(pil_img.width*s), round(pil_img.height*s)]))
    for i, rect in enumerate(layout_rects*s): 
        painter.rectangle('temp', rect, 'blue', width)
        painter.text('temp', [rect[0], rect[1]], cat2names[layout_dets[i]['category_id']], 'blue', font_size=font_size*1.5)
    for i, rect in enumerate(pt_rects*s): 
        painter.rectangle('temp', rect, 'red', width)
        painter.text('temp', [rect[2], rect[1]], str(i), 'red', font_size=font_size/1.5)
    for i, rect in enumerate(mf_rects*s): 
        painter.rectangle('temp', rect, 'green', width)
        painter.text('temp', [rect[2], rect[1]], str(i), 'green', font_size=font_size/1.5)
    painter.save('temp', os.path.join(output_dir, 'debug/{}_b.jpg'.format(page_id))) 


class paddleOCRMix():
    def __init__(self):
        self.table_model = None
        self.pt_models = None
        self.use_cuda = None
        self.device = None
        self.cat2names = None
        self.surya_processor = None
        self.surya_model = None
        self.layout_model = None
        self.mfr_model = None
        self.mfr_transform = None
        self.mfd_model = None
        self.dpi = None

    def model_init(self):
        thread_id = threading.current_thread().ident
        time_start = time.time()
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.use_cuda = self.device == 'cuda'
            self.dpi = 600

            self.layout_model = layout_model_init(weight='models/Layout/model_final.pth',
                                                  config_file='configs/layoutlmv3_base_inference.yaml')
            self.surya_model = surya_load_model(checkpoint='models/vikp/surya_order/', device=self.device)
            self.surya_processor = surya_load_processor(checkpoint='models/vikp/surya_order/')
            self.mfd_model = mfd_model_init('models/MFD/weights.pt')
            self.mfr_model, mfr_vis_processors = mfr_model_init('models/MFR/UniMERNet', device=self.device)
            self.mfr_transform = transforms.Compose([mfr_vis_processors, ])
            self.pt_models = {lang: pt_model_init('models/PaddleOCR', lang, self.use_cuda) for lang in ['ch', 'en', 'latin', 'cyrillic']}
            self.table_model = RapidTable(model_path='models/RapidTable/ch_ppstructure_mobile_v2_SLANet.onnx')
            self.cat2names = ["title", "plain_text", "abandon", "figure", "figure_caption", "table", "table_caption",
                              "table_footnote",
                              "isolate_formula", "formula_caption", " ", " ", " ", "inline_formula", "isolated_formula"]
            self.subject_model = fasttext.load_model("models/subject_textclf/classifier_multi_subject.bin")
            time_end = time.time()
            logging.info(f"thread {thread_id} success init all models!Cost time {time_end - time_start} seconds.")
            print(f"thread {thread_id} success init all models!Cost time {time_end - time_start} seconds.")
        except Exception as e:
            time_end = time.time()
            logging.error(f"thread {thread_id} init model failure!Cost time {time_end - time_start} seconds.")
            print(f"thread {thread_id} init model failure!Cost time {time_end - time_start} seconds.Exception:{e}")

    def process(self, pdf_path, data, random_uuid):
        thread_id = threading.current_thread().ident

        logging.info(f"thread {thread_id} start process data {random_uuid}!")
        print(f"-----------thread {thread_id} start process data {random_uuid}!--------------")
        output_path = "./output"
        try:
            time_start = time.time()
            data_language = data.get('language',None)
            lang = data.get('language',None) if data_language is not None else "other"
            if lang.lower() == 'other':
                data_title = data.get('title',None)
                lang = langid.classify(data_title) if data_title is not None and data_title != '' else 'en'
            lang = language_dict.get(lang,lang)
            lang = pt_lang_map(lang)
            if lang not in self.pt_models:
                self.pt_models[lang] = pt_model_init('models/PaddleOCR', lang, self.use_cuda)              
            output_dir = os.path.join(output_path, os.path.basename(pdf_path).replace('.pdf', ''))
            if os.path.exists(output_dir):
                shutil.copytree(output_dir, output_dir + '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f"))
                shutil.rmtree(output_dir)
            os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'debug'), exist_ok=True)
            upload_dir = "pipeline_result/algos_version_" + str(utils.algos_version) + "/" + data[
                "version"] + "/" + str(random_uuid) + "/"
            doc = fitz.open(pdf_path)
            doc_content = ''
            doc_figures = []
            doc_tables = []
            for i_page in range(len(doc)):
                page = doc.load_page(i_page)
                pix = page.get_pixmap(matrix=fitz.Matrix(self.dpi / 72, self.dpi / 72))
                pil_img = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
                img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
                t1 = time.time()
                page_id = str(i_page + 1).zfill(len(str(len(doc))))
                with torch.cuda.amp.autocast():
                    mf_dets, mf_boxes, mf_rects, mf_points, mf_scores, mf_cats = mfd(self.mfd_model, img)
                    mfd_elapsed = time.time() - t1
                    mf_dets, mf_boxes, mf_rects, mf_points, mf_scores, mf_cats = mf_refine(mf_dets, mf_boxes, mf_rects,
                                                                                           mf_points, mf_scores,
                                                                                           mf_cats)
                    print(f"thread {thread_id} mfd_mfrefine model already finish!")
                    t1 = time.time()
                    layout_dets = layout_det(self.layout_model, pil_img)
                    layout_det_elapsed = time.time() - t1
                    layout_dets, layout_boxes, layout_rects, layout_points, layout_scores, layout_cats = layout_filter(
                        layout_dets, mf_dets)
                    layout_dets, layout_boxes, layout_rects, layout_points, layout_scores, layout_cats = layout_refine(
                        layout_dets, layout_boxes, layout_rects, layout_points, layout_scores, layout_cats)
                    t1 = time.time()
                    layout_dets, layout_boxes, layout_rects, layout_points, layout_scores, layout_cats = layout_order(
                        self.surya_model, self.surya_processor, pil_img, layout_dets, layout_boxes, layout_rects,
                        layout_points, layout_scores, layout_cats)
                    layout_order_elapsed = time.time() - t1
                    t1 = time.time()
                    pt_boxes, pt_rects = ptd(self.pt_models[lang], img)
                    ptd_elapsed = time.time() - t1
                    pt_boxes, pt_rects, pt_points, mask_ptmf_seg = pt_split(pt_boxes, pt_rects, mf_rects)
                    mask_pt_att_layout = pt_att_layout(pt_rects, pt_points, layout_rects)
                    mask_mf_att_layout = mf_att_layout(mf_rects, mf_points, layout_rects)
                    t1 = time.time()
                    ptr_texts, ptr_scores = ptr(self.pt_models[lang], img, pt_rects, pt_points)
                    ptr_elapsed = time.time() - t1
                    t1 = time.time()
                    mfr_texts = mfr(self.mfr_model, self.mfr_transform, self.device, img, mf_rects, mf_points, mf_cats)
                    mfr_elapsed = time.time() - t1
                    ptmf_boxes, ptmf_rects, ptmf_points, ptrmfr_texts, ptrmfr_scores, mask_ptmf_att_layout = \
                        ptmf(pt_boxes, pt_rects, pt_points, ptr_texts, ptr_scores, mask_pt_att_layout, mf_boxes,
                             mf_rects,
                             mf_points, mfr_texts, mask_mf_att_layout)
                    layout_ptmf(layout_dets, ptmf_boxes, ptmf_rects, ptrmfr_texts, ptrmfr_scores, mask_ptmf_att_layout, mask_ptmf_seg)

                    figure_offset = len(doc_figures)
                    figures = layout_figure(pil_img, layout_dets, layout_rects, output_dir, upload_dir, figure_offset,
                                            page_id)
                    doc_figures.extend(figures)
                    t1 = time.time()
                    table_offset = len(doc_tables)
                    tables = layout_table(self.table_model, img, layout_dets, layout_rects, output_dir, upload_dir,
                                          table_offset, page_id)
                    doc_tables.extend(tables)
                    table_elapsed = time.time() - t1
                    content = layout_content(layout_dets, output_dir, page_id, self.cat2names)
                    doc_content += content
                    # debug(pil_img, layout_dets, layout_rects, pt_rects, mf_rects, output_dir, page_id, self.cat2names) 
                print('=' * 80)
                print('page {}, layout {}, mfd {}, mfr {}, ptd {}, ptr {}, table {}, ordering {}'.format(
                    i_page, layout_det_elapsed, mfd_elapsed, mfr_elapsed, ptd_elapsed, ptr_elapsed, table_elapsed,
                    layout_order_elapsed))
                print("=" * 80)
            mmd_path = os.path.join(output_dir, '{}.mmd'.format(os.path.basename(pdf_path).replace('.pdf', '')))
            with open(mmd_path, 'w', encoding='utf-8') as f:
                f.write(clean_text(doc_content))
            logging.info(
                f"thread {thread_id} success process data {random_uuid}! Total pages:{len(doc)}, total time {time.time() - time_start} seconds!")
            print(
                f"thread {thread_id} success process data {random_uuid}! Total pages:{len(doc)}, total time {time.time() - time_start} seconds!")
            paddleocr_dict = {'figure': doc_figures, 'table': doc_tables, 'version': utils.algos_version,
                              'subject': None}
            subject = utils.get_subject(doc_content, self.subject_model)
            paddleocr_dict['subject'] = subject
            return str(paddleocr_dict), True, mmd_path, ""
        except Exception as e:
            traceback.print_exc()
            logging.error(f"thread {thread_id} process data {random_uuid} fail! Error: " + str(e))
            print(f"thread {thread_id} process data {random_uuid} fail! Error: " + str(e))
            return "", False, "", str(e)
