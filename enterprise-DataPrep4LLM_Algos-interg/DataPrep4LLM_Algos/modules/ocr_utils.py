# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 09:22:48 2024

@author: ZJ
"""
import os, cv2, re
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Painter(object):
    def __init__(self): 
        self.figs = {}
        self.works = {}
    
    def create(self, name, img):
        if isinstance(img, str):
            img = Image.open(img)
        self.figs[name] = img.copy()
        self.works[name] = ImageDraw.Draw(self.figs[name])
        
    def rectangle(self, name, box, color='green', width=1):
        if isinstance(box, np.ndarray):
            box = box.reshape(-1).tolist()
        if len(box) == 8:
            xmin, ymin, xmax, ymax = box[0], box[1], box[4], box[5]
        else: 
            xmin, ymin, xmax, ymax = box
        if xmin < xmax and ymin < ymax:
            self.works[name].rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=width)
            
    def polygon(self, name, poly, color='red', width=1):
        self.works[name].polygon(list(poly), outline=color, width=width)
        
    def text(self, name, pos, s, color='blue', font_size=5):
        x, y = pos
        font_path = 'simhei.ttf'
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
        self.works[name].text((x, y), s, fill=color, font=font)
        
    def save(self, name, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.figs[name].save(output_path)
        temp = self.figs.pop(name)
        temp = self.works.pop(name)


def enlarge_rects(rects, dw=-0.2, dh=-0.2):
    w = rects[:, 2] - rects[:, 0]
    h = rects[:, 3] - rects[:, 1]
    dw1 = w * dw / 2 
    dh1 = h * dh / 2
    enlarged_rects = np.array(rects, float)
    enlarged_rects[:, 0] -=  dw1
    enlarged_rects[:, 1] -= dh1
    enlarged_rects[:, 2] += dw1
    enlarged_rects[:, 3] += dh1
    return enlarged_rects
    

def boxes_to_rects(boxes):
    xmin = boxes[:, [0, 2, 4, 6]][np.arange(boxes.shape[0]).tolist(), np.argmin(boxes[:, [0, 2, 4, 6]], axis=1)]
    xmax = boxes[:, [0, 2, 4, 6]][np.arange(boxes.shape[0]).tolist(), np.argmax(boxes[:, [0, 2, 4, 6]], axis=1)]
    ymin = boxes[:, [1, 3, 5, 7]][np.arange(boxes.shape[0]).tolist(), np.argmin(boxes[:, [1, 3, 5, 7]], axis=1)]
    ymax = boxes[:, [1, 3, 5, 7]][np.arange(boxes.shape[0]).tolist(), np.argmax(boxes[:, [1, 3, 5, 7]], axis=1)]
    rects = np.array([xmin, ymin, xmax, ymax]).transpose()
    return rects
   
    
def points_in_rects(rects, pts):
    xmin = rects[:, 0][:, np.newaxis]
    ymin = rects[:, 1][:, np.newaxis]
    xmax = rects[:, 2][:, np.newaxis]
    ymax = rects[:, 3][:, np.newaxis]
    x = pts[:, 0][np.newaxis, :]
    y = pts[:, 1][np.newaxis, :]
    inside_x = (x >= xmin) & (x <= xmax)
    inside_y = (y >= ymin) & (y <= ymax)
    inside = inside_x & inside_y
    return inside.T 


def rects_inte_rects(rects1, rects2, return_valid=True):
    x_min = np.maximum(rects1[:, 0, np.newaxis], rects2[:, 0])
    y_min = np.maximum(rects1[:, 1, np.newaxis], rects2[:, 1])
    x_max = np.minimum(rects1[:, 2, np.newaxis], rects2[:, 2])
    y_max = np.minimum(rects1[:, 3, np.newaxis], rects2[:, 3])
    valid = (x_min < x_max) & (y_min < y_max)
    width = x_max - x_min
    height = y_max - y_min
    area = height * width
    if return_valid:
        x_min = np.where(valid, x_min, np.nan)
        y_min = np.where(valid, y_min, np.nan)
        x_max = np.where(valid, x_max, np.nan)
        y_max = np.where(valid, y_max, np.nan)
        area = np.where(valid, area, 0)
    return x_min, y_min, x_max, y_max, area, valid


def rects_area(rects):
    width = rects[..., 2] - rects[..., 0]
    height = rects[..., 3] - rects[..., 1]
    area = width * height 
    return area


def rects_union_rects(rects1, rects2):
    rects1_union_rects2_rects_xmin = np.minimum(rects1[:, 0][:, np.newaxis], rects2[:, 0][np.newaxis, :])
    rects1_union_rects2_rects_xmax = np.maximum(rects1[:, 2][:, np.newaxis], rects2[:, 2][np.newaxis, :])
    rects1_union_rects2_rects_ymin = np.minimum(rects1[:, 1][:, np.newaxis], rects2[:, 1][np.newaxis, :])
    rects1_union_rects2_rects_ymax = np.maximum(rects1[:, 3][:, np.newaxis], rects2[:, 3][np.newaxis, :])
    rects1_union_rects2_rects_area = (rects1_union_rects2_rects_xmax - rects1_union_rects2_rects_xmin) * (rects1_union_rects2_rects_ymax - rects1_union_rects2_rects_ymin)
    return rects1_union_rects2_rects_xmin, rects1_union_rects2_rects_ymin, rects1_union_rects2_rects_xmax, rects1_union_rects2_rects_ymax, \
        rects1_union_rects2_rects_area
  

def split_interval_excluding(a, intervals):
    intervals = np.array(intervals)
    intervals = intervals[np.argsort(intervals[:, 0])]
    a_start, a_end = a
    starts = intervals[:, 0]
    ends = intervals[:, 1]
    valid_starts = np.clip(starts, a_start, a_end)
    valid_ends = np.clip(ends, a_start, a_end)
    split_points = np.unique(np.hstack(([a_start], valid_starts, valid_ends, [a_end])))
    split_intervals = np.column_stack((split_points[:-1], split_points[1:]))
    mask = np.ones(len(split_intervals), dtype=bool)
    for start, end in zip(valid_starts, valid_ends):
        mask &= (split_intervals[:, 1] <= start) | (split_intervals[:, 0] >= end)
    result = split_intervals[mask] 
    return result


def get_cropped_image(img, points):
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img = np.clip(dst_img, 0.0, 255.0).astype(np.uint8)
    return dst_img


def clean_text(text):
    text = re.sub(r'(?<! ) (?! )', '', text)
    text = re.sub(r'[ ]+', ' ', text)   
    text = re.sub(r'[ ]?\n{2,}[ ]?', '\n\n', text)
    return text.strip() 


def extend_lists(ll):
    l = [e for l in ll for e in l]
    return l


def invert_index_by_value(list_of_lists):
    # 记录每个值对应的行号
    row_indices = np.repeat(np.arange(len(list_of_lists)), [len(lst) for lst in list_of_lists])
    # 展平所有的值
    all_values = np.concatenate(list_of_lists)
    
    # 对值进行排序，并同时排序行号
    sorted_order = np.argsort(all_values)
    sorted_values = all_values[sorted_order]
    sorted_row_indices = row_indices[sorted_order]
    
    # 找到每个唯一值的起始索引和计数
    unique_values, counts = np.unique(sorted_values, return_counts=True)
    
    # 计算拆分点
    split_indices = np.cumsum(counts)[:-1]
    
    # 根据拆分点拆分行号数组
    inverted = np.split(sorted_row_indices, split_indices)
    
    # 将结果转换为列表格式（如果需要）
    inverted_list = [list(group) for group in inverted]
    
    return inverted_list


def filter_cliques(cliques, node_range):
    # Convert cliques to a padded NumPy array
    max_clique_size = max(len(clique) for clique in cliques)
    cliques_padded = np.array([clique + [-1]*(max_clique_size - len(clique)) for clique in cliques])

    # Identify cliques containing all nodes in the node range
    node_range_array = np.array(node_range)
    # contains_all_nodes = np.all(np.isin(cliques_padded, node_range_array), axis=1)

    # Shrink cliques based on node range
    is_in_node_range = np.isin(cliques_padded, node_range_array)
    # cliques_shrunk = np.where(
    #     contains_all_nodes[:, np.newaxis],
    #     cliques_padded,
    #     np.where(is_in_node_range, cliques_padded, -1)
    # )
    cliques_shrunk = np.where(is_in_node_range, cliques_padded, -1)
    
    # Create the indicator matrix
    num_cliques, _ = cliques_shrunk.shape
    max_node = cliques_padded.max()
    num_nodes = max_node + 1  # Assuming node indices start at 0

    indicator_matrix = np.zeros((num_cliques, num_nodes), dtype=int)
    clique_indices = np.repeat(np.arange(num_cliques), cliques_shrunk.shape[1])
    node_indices = cliques_shrunk.flatten()
    valid_mask = node_indices != -1
    indicator_matrix[clique_indices[valid_mask], node_indices[valid_mask]] = 1

    # Identify maximal cliques
    is_subset = np.all(
        indicator_matrix[:, np.newaxis, :] <= indicator_matrix[np.newaxis, :, :],
        axis=2
    )
    is_dup = np.all(
        indicator_matrix[:, np.newaxis, :] == indicator_matrix[np.newaxis, :, :],
        axis=2
    )
    is_subset = is_subset * (~is_dup)   
    is_dup[np.triu_indices(num_cliques)] = False
    is_maximal = ~np.any(is_subset + is_dup, axis=1)
    maximal_clique_indices = np.where(is_maximal)[0]

    # Extract maximal cliques
    maximal_cliques = [clique[clique>=0].tolist() for clique in cliques_shrunk[maximal_clique_indices]]
    return maximal_cliques, maximal_clique_indices

def drop_dup(list_of_lists):
    llen = [len(l) for l in list_of_lists]
    arr = np.hstack(list_of_lists)
    arr_new = -np.ones(len(arr), int)
    idx = np.sort(np.unique(arr, return_index=True)[1])
    arr_u = arr[idx]
    arr_new[idx] = arr_u
    lil_new = [a[a>0].tolist() for a in np.split(arr_new, np.cumsum(llen)[:-1])]
    return lil_new, arr_u