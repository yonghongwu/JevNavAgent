"""Real OpenSlide observations, explicit action budgets, separate reference scoring.

Policies receive only ``observation()``. Reference methods are called exclusively
by data builders, training reward wrappers, or the offline evaluator.
"""
from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
import openslide

REQUEST = ("This lymph-node whole slide is known to contain metastatic tumor. "
           "Find ONE tumor-containing field using read_region, then submit a point "
           "in your LAST view. Any tumor focus is acceptable. The scored field is "
           "512x512 level-0 pixels centered at the point, with center inside tumor "
           "and at least 25% annotated tumor. Inspect a supporting field at "
           "downsample 4 or 1 before submitting. This is a localization task.")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def finite_point(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1000 for v in value))


def safe_untrusted_json(value):
    """Preserve invalid model outputs in ledgers without emitting JSON NaN."""
    if isinstance(value,float) and not math.isfinite(value):
        return {'invalid_nonfinite_number':repr(value)}
    if isinstance(value,dict):return {str(k):safe_untrusted_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe_untrusted_json(v) for v in value]
    return value


@lru_cache(maxsize=512)
def parse_annotations(path):
    polygons = []
    for ann in ET.parse(path).getroot().findall('.//Annotation'):
        group = ann.get('PartOfGroup')
        if ann.get('Type') == 'Dot' or group == 'None':
            continue
        if group not in ('Tumor', 'Exclusion'):
            raise ValueError(f'Unknown annotation group {group!r}')
        pts = [(float(p.get('X')), float(p.get('Y')))
               for p in ann.findall('./Coordinates/Coordinate')]
        if len(pts) < 3 or not all(math.isfinite(x) and math.isfinite(y) for x, y in pts):
            raise ValueError('Invalid polygon')
        xs, ys = zip(*pts)
        polygons.append({'group': group, 'points': pts,
                         'bbox': (min(xs), min(ys), max(xs), max(ys))})
    if not any(p['group'] == 'Tumor' for p in polygons):
        raise ValueError('No tumor polygons')
    return polygons


def label_mask(polygons, box, size):
    x0, y0, x1, y1 = box
    w, h = size
    image = Image.new('L', size, 0)
    draw = ImageDraw.Draw(image)
    for group, value in (('Tumor', 255), ('Exclusion', 0)):
        for polygon in polygons:
            b = polygon['bbox']
            if polygon['group'] != group or b[2] < x0 or b[0] > x1 or b[3] < y0 or b[1] > y1:
                continue
            pts = [((x - x0) * w / (x1 - x0), (y - y0) * h / (y1 - y0))
                   for x, y in polygon['points']]
            draw.polygon(pts, fill=value)
    return image


def point_inside(point, polygons):
    x, y = point
    def inside_polygon(p):
        if not (p['bbox'][0] <= x <= p['bbox'][2] and p['bbox'][1] <= y <= p['bbox'][3]):
            return False
        inside = False
        pts = p['points']
        for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
            if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
        return inside
    return (any(p['group'] == 'Tumor' and inside_polygon(p) for p in polygons)
            and not any(p['group'] == 'Exclusion' and inside_polygon(p) for p in polygons))


def score_point(annotation_path, dimensions, point, observed_views):
    if point is None or len(point) != 2 or not all(math.isfinite(v) for v in point):
        return {'success': False, 'localization_hit': False, 'reason': 'invalid_or_missing_submission'}
    cx, cy = point
    box = [round(cx) - 256, round(cy) - 256, round(cx) + 256, round(cy) + 256]
    polygons = parse_annotations(str(annotation_path))
    inside = point_inside(point, polygons)
    fraction = float(np.asarray(label_mask(polygons, box, (512, 512)), dtype=np.float64).mean() / 255)
    w, h = dimensions
    in_bounds = box[0] >= 0 and box[1] >= 0 and box[2] <= w and box[3] <= h
    supported = any(v['requested_downsample'] <= 4
                    and v['native_bbox'][0] <= box[0] and v['native_bbox'][1] <= box[1]
                    and v['native_bbox'][2] >= box[2] and v['native_bbox'][3] >= box[3]
                    for v in observed_views)
    hit = inside and fraction >= .25 and in_bounds
    return {'success': bool(hit and supported), 'localization_hit': bool(hit),
            'point_level0': list(point), 'field_level0_bbox': box,
            'field_tumor_fraction': fraction, 'center_inside_tumor': bool(inside),
            'observed_high_resolution_support': bool(supported), 'field_inside_slide': bool(in_bounds)}


def draw_grid(image):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for value in range(0, 1001, 200):
        x, y = round(value * (w - 1) / 1000), round(value * (h - 1) / 1000)
        draw.line((x, 0, x, h), fill=(140, 140, 140), width=1)
        draw.line((0, y, w, y), fill=(140, 140, 140), width=1)
        draw.rectangle((min(x, w-30), 0, min(x, w-30)+30, 13), fill='white')
        draw.text((min(x, w-30), 0), str(value), fill='black')
        draw.rectangle((0, min(y, h-13), 30, min(y, h-13)+13), fill='white')
        draw.text((0, min(y, h-13)), str(value), fill='black')
    return image


class WSIEnvironment:
    def __init__(self, row, output_dir=None, budget=8,coordinate_step=None):
        self.row = dict(row)
        self.budget = int(budget)
        self.coordinate_step=float(coordinate_step) if coordinate_step else None
        if self.coordinate_step and (self.coordinate_step<=0 or 1000/self.coordinate_step%1):
            raise ValueError('coordinate_step must divide1000')
        if not 1 <= self.budget <= 64:
            raise ValueError('read budget must be 1..64')
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.slide = openslide.OpenSlide(str(row['slide_path']))
        self.w, self.h = self.slide.dimensions
        self.overview = self.slide.get_thumbnail((1024, 1024)).convert('RGB')
        self.reset()

    def reset(self, seed=0):
        self.seed = int(seed)
        self.views, self.images, self.records = [], [], []
        self.decisions, self.done, self.final = 0, False, None
        self.status = 'running'
        return self.observation()

    def observation(self):
        # Do not pass row, annotation paths, source keys, or reference outcomes.
        public_views = [{k: v for k, v in view.items()
                         if k not in ('raw_image', 'image', 'raw_sha256')}
                        for view in self.views]
        request=REQUEST
        if self.coordinate_step:
            request+=f' Every read center and submit point coordinate must be a multiple of {self.coordinate_step:g} (including0 and1000).'
        return {'request': request, 'overview': self.overview,
                'images': list(self.images), 'views': public_views,
                'dimensions': [self.w, self.h],
                'last_view': public_views[-1] if public_views else None,
                'last_tool_error':self.records[-1]['error'] if self.records else None,
                'action_history':[{k:record[k] for k in ('decision','action','error','views_used')}
                                  for record in self.records],
                'coordinate_step':self.coordinate_step,
                'remaining_budget': self.budget - len(self.views),
                'decisions_remaining': self.budget + 1 - self.decisions}

    def _read(self, action):
        center, ds = action.get('center'), action.get('downsample')
        if not finite_point(center) or type(ds) not in (int, float) or ds not in (64, 16, 4, 1):
            raise ValueError('center must be global0..1000; downsample 64/16/4/1')
        self._check_grid(center)
        if len(self.views) >= self.budget:
            raise ValueError('read_budget_exhausted')
        span = 768 * ds
        cx, cy = center[0] * self.w / 1000, center[1] * self.h / 1000
        x0, y0 = round(cx - span / 2), round(cy - span / 2)
        level = self.slide.get_best_level_for_downsample(ds)
        actual = self.slide.level_downsamples[level]
        n = math.ceil(span / actual)
        rgba = self.slide.read_region((x0, y0), level, (n, n))
        image = Image.new('RGB', rgba.size, 'white')
        image.paste(rgba, mask=rgba.getchannel('A'))
        image = image.resize((768, 768), Image.Resampling.LANCZOS)
        box = [x0, y0, x0 + n * actual, y0 + n * actual]
        index = len(self.views)
        view = {'index': index, 'requested_global_center': list(center),
                'requested_downsample': ds, 'native_level': level,
                'native_level_downsample': actual, 'read_pixels': [n, n],
                'native_bbox': box,
                'global_bbox_1000': [box[0]*1000/self.w, box[1]*1000/self.h,
                                     box[2]*1000/self.w, box[3]*1000/self.h],
                'returned_pixels': [768, 768]}
        if self.output_dir:
            raw = self.output_dir / f'view_{index}_pixels.png'
            grid = self.output_dir / f'view_{index}_grid.jpg'
            image.save(raw)
            draw_grid(image).save(grid, quality=90)
            view.update(raw_image=str(raw), image=str(grid),
                        raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
        self.images.append(image)
        self.views.append(view)

    def submitted_point(self, action):
        if not self.views or not isinstance(action, dict) or not finite_point(action.get('point')):
            raise ValueError('submit requires a valid local point and a previous view')
        self._check_grid(action['point'])
        box = self.views[-1]['native_bbox']
        u, v = action['point']
        return [box[0]+u*(box[2]-box[0])/1000, box[1]+v*(box[3]-box[1])/1000]

    def _check_grid(self,point):
        if self.coordinate_step and any(abs(v/self.coordinate_step-round(v/self.coordinate_step))>1e-8 for v in point):
            raise ValueError(f'coordinates_must_be_multiples_of_{self.coordinate_step:g}')

    def step(self, action):
        if self.done:
            raise RuntimeError('episode already ended')
        self.decisions += 1
        start = time.monotonic()
        error = None
        try:
            if not isinstance(action, dict):
                raise ValueError('action must be JSON object')
            if action.get('action') == 'read_region':
                self._read(action)
            elif action.get('action') == 'submit':
                self.submitted_point(action)
                self.final, self.done, self.status = dict(action), True, 'submitted'
            else:
                raise ValueError('unknown_action')
        except ValueError as exc:
            error = str(exc)
        if self.decisions >= self.budget + 1 and not self.done:
            self.done, self.status = True, 'decision_budget_exhausted'
        record = {'decision': self.decisions, 'action': safe_untrusted_json(action), 'error': error,
                  'elapsed_seconds': time.monotonic() - start, 'views_used': len(self.views)}
        self.records.append(record)
        return self.observation(), self.done, {'status': self.status, 'error': error}

    def evaluate(self):
        point = self.submitted_point(self.final) if self.final else None
        return score_point(self.row['annotation_path'], (self.w, self.h), point, self.views)

    def view_outcome(self, view_index=-1):
        """Offline/reference-only; center qualification, not exhaustive field search."""
        v = self.views[view_index]
        b = v['native_bbox']
        point = [(b[0]+b[2])/2, (b[1]+b[3])/2]
        out = score_point(self.row['annotation_path'], (self.w, self.h), point, [v])
        out['view_tumor_fraction'] = float(np.asarray(label_mask(
            parse_annotations(str(self.row['annotation_path'])), b, (192, 192))).mean()/255)
        return out

    def close(self):
        self.slide.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
