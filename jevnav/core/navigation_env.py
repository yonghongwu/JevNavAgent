"""Annotation-blind, replayable navigation over real multi-resolution whole slides.

Only ``observation`` is a policy input. Rows, annotation files, and answers are
deliberately absent from it. Reference scoring belongs to the caller.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import openslide
from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 64 * 1024 * 1024

DOWNSAMPLES = (64, 16, 4, 1)
CANDIDATE_FEATURE_NAMES = (
    'cx_norm', 'cy_norm', 'log4_downsample_normalized', 'x0_norm', 'y0_norm',
    'x1_norm', 'y1_norm', 'thumbnail_tissue_fraction', 'relative_dx',
    'relative_dy', 'is_root', 'scale_step',
)
CANDIDATE_FEATURE_DIM = len(CANDIDATE_FEATURE_NAMES)
DEFAULT_QUESTION = (
    'Find an observed field supporting metastatic tumor in this known-positive '
    'lymph-node whole slide. Which conclusion is supported by the observed '
    'high-resolution evidence? Read supporting tissue before answering.'
)
DEFAULT_CHOICES = ['Metastatic tumor is supported by the observed field.',
                   'The observed field supports only non-tumor tissue or background.']


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'invalid_nonfinite_number': repr(value)}
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


class CandidateGenerator:
    """Uses only the displayed thumbnail and already-read view geometry.

    Sixteen spatially stratified thumbnail tissue centers are always available.
    The current view supplies a 3x3 finer grid and four same-scale translations.
    At level 0 the 3x3 grid refines at quarter-field spacing. Repeated fields are
    masked before indexing; the resulting list contains at most 29 candidates.
    """
    def __init__(self, overview, dimensions, output_size=512, max_candidates=32):
        self.dimensions = tuple(dimensions)
        self.output_size = int(output_size)
        self.max_candidates = int(max_candidates)
        pixels = np.asarray(overview.convert('RGB'), dtype=np.float32)
        maximum, minimum = pixels.max(2), pixels.min(2)
        self.tissue = ((maximum - minimum > 18) & (minimum < 235) & (maximum > 40))
        self.th, self.tw = self.tissue.shape
        self.integral = np.pad(self.tissue.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        self.roots = []
        for gy in range(4):
            for gx in range(4):
                x0, x1 = gx * self.tw // 4, (gx + 1) * self.tw // 4
                y0, y1 = gy * self.th // 4, (gy + 1) * self.th // 4
                ys, xs = np.nonzero(self.tissue[y0:y1, x0:x1])
                if len(xs):
                    # Coordinate-wise medians can lie between tissue fragments;
                    # snap to the actual tissue pixel nearest that median.
                    mx, my = np.median(xs), np.median(ys)
                    k = int(np.argmin((xs - mx) ** 2 + (ys - my) ** 2))
                    self.roots.append(((x0 + float(xs[k]) + .5) / self.tw,
                                       (y0 + float(ys[k]) + .5) / self.th))
        if not self.roots:
            self.roots = [((gx + .5) / 4, (gy + .5) / 4)
                          for gy in range(4) for gx in range(4)]

    def bbox(self, center, downsample):
        w, h = self.dimensions
        span = self.output_size * downsample
        x0 = round(center[0] * w - span / 2)
        y0 = round(center[1] * h - span / 2)
        return [x0, y0, x0 + span, y0 + span]

    @staticmethod
    def key(box, downsample):
        return (int(box[0]), int(box[1]), int(downsample))

    def tissue_fraction(self, box):
        w, h = self.dimensions
        xa = max(0, min(self.tw, math.floor(box[0] / w * self.tw)))
        xb = max(0, min(self.tw, math.ceil(box[2] / w * self.tw)))
        ya = max(0, min(self.th, math.floor(box[1] / h * self.th)))
        yb = max(0, min(self.th, math.ceil(box[3] / h * self.th)))
        if xa >= xb or ya >= yb:
            return 0.
        area = max(1., (box[2] - box[0]) / w * self.tw *
                   (box[3] - box[1]) / h * self.th)
        total = self.integral[yb, xb] - self.integral[ya, xb] - self.integral[yb, xa] + self.integral[ya, xa]
        return float(min(1., total / area))

    def generate(self, views):
        w, h = self.dimensions
        current = views[-1] if views else None
        current_key = self.key(current['bbox_level0'], current['downsample']) if current else None
        proposals = []
        if current:
            b = current['bbox_level0']
            cx, cy = (b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h
            ds = current['downsample']
            finer = max(1, ds // 4)
            spacing = self.output_size * (finer if ds > 1 else .25)
            for dy in (0, -1, 1):
                for dx in (0, -1, 1):
                    proposals.append(([cx + dx * spacing / w, cy + dy * spacing / h], finer, False))
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                proposals.append(([cx + dx * self.output_size * ds / 2 / w,
                                   cy + dy * self.output_size * ds / 2 / h], ds, False))
        proposals.extend((list(center), 64, True) for center in self.roots)
        candidates, seen = [], ({current_key} if current_key else set())
        for center, ds, is_root in proposals:
            center = [float(min(1., max(0., v))) for v in center]
            box = self.bbox(center, ds)
            key = self.key(box, ds)
            if key in seen:
                continue
            seen.add(key)
            if current:
                cb = current['bbox_level0']
                relx = (center[0] * w - (cb[0] + cb[2]) / 2) / (cb[2] - cb[0])
                rely = (center[1] * h - (cb[1] + cb[3]) / 2) / (cb[3] - cb[1])
                step = math.log(current['downsample'] / ds, 4) / 3
            else:
                relx = rely = step = 0.
            features = [*center, math.log(ds, 4) / 3,
                        box[0] / w, box[1] / h, box[2] / w, box[3] / h,
                        self.tissue_fraction(box), relx, rely, float(is_root), step]
            candidate_id = hashlib.sha256(f'{key[0]}:{key[1]}:{ds}'.encode()).hexdigest()[:16]
            candidates.append({'id': candidate_id, 'action': 'read_region', 'center': center,
                               'downsample': ds, 'bbox_level0': box, 'features': features})
            if len(candidates) >= self.max_candidates:
                break
        return candidates


class NavigationEnvironment:
    def __init__(self, row, budget=8, output_dir=None, output_size=512,
                 question=None, choices=None, slide_opener=None, seed=0):
        if type(budget) is not int or not 1 <= budget <= 64:
            raise ValueError('budget must be an integer in 1..64')
        if type(output_size) is not int or output_size < 512:
            raise ValueError('output_size must be an integer >=512 for native512 support')
        self.budget, self.output_size = budget, output_size
        self.question = question if question is not None else row.get('question', DEFAULT_QUESTION)
        self.choices = list(choices if choices is not None else row.get('choices', DEFAULT_CHOICES))
        if len(self.choices) < 2:
            raise ValueError('at least two choices are required')
        # Do not retain the source row: it may contain annotations and outcomes.
        self.slide = (slide_opener or openslide.OpenSlide)(str(row['slide_path']))
        self.dimensions = list(self.slide.dimensions)
        started = time.monotonic()
        self.overview = self.slide.get_thumbnail((1024, 1024)).convert('RGB')
        self.overview_seconds = time.monotonic() - started
        self.mpp = []
        for name in ('openslide.mpp-x', 'openslide.mpp-y'):
            try:
                value = float(self.slide.properties[name])
                self.mpp.append(value if math.isfinite(value) and value > 0 else None)
            except (KeyError, TypeError, ValueError):
                self.mpp.append(None)
        self.generator = CandidateGenerator(self.overview, self.dimensions, output_size)
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.overview.save(self.output_dir / 'overview.png')
        self.reset(seed)

    def reset(self, seed=0):
        self.seed = int(seed)
        self.images, self.views, self.records = [], [], []
        self.decisions = 0
        self.read_seconds = 0.
        self.done, self.status, self.final = False, 'running', None
        return self.observation()

    def observation(self):
        return {'question': self.question, 'choices': list(self.choices),
                'overview': self.overview.copy(), 'images': [im.copy() for im in self.images],
                'views': json.loads(json.dumps(self.views)), 'dimensions': list(self.dimensions),
                'mpp': list(self.mpp), 'remaining_budget': self.budget - len(self.views),
                'decisions_remaining': self.budget + 1 - self.decisions,
                'candidates': [] if self.done or len(self.views) >= self.budget else self.generator.generate(self.views),
                'status': self.status}

    def _read(self, candidate):
        started = time.monotonic()
        box, ds = candidate['bbox_level0'], candidate['downsample']
        level = self.slide.get_best_level_for_downsample(ds)
        actual = float(self.slide.level_downsamples[level])
        count = math.ceil(self.output_size * ds / actual)
        rgba = self.slide.read_region((box[0], box[1]), level, (count, count)).convert('RGBA')
        image = Image.new('RGB', rgba.size, 'white')
        image.paste(rgba, mask=rgba.getchannel('A'))
        # Crop the small native-level rounding excess rather than silently
        # stretching it over the declared field of view.
        extent = self.output_size * ds / actual
        image = image.transform((self.output_size, self.output_size), Image.Transform.EXTENT,
                                (0, 0, extent, extent), Image.Resampling.BICUBIC)
        view = {'index': len(self.views), 'candidate_id': candidate['id'],
                'center': list(candidate['center']), 'downsample': ds,
                'bbox_level0': list(box), 'requested_downsample': ds, 'native_bbox': list(box),
                'native_level': level, 'native_level_downsample': actual,
                'read_pixels': [count, count], 'returned_pixels': [self.output_size] * 2,
                'physical_span_um': [(box[2] - box[0]) * self.mpp[0] if self.mpp[0] else None,
                                     (box[3] - box[1]) * self.mpp[1] if self.mpp[1] else None]}
        if self.output_dir:
            image.save(self.output_dir / f'view_{len(self.views):02d}.png')
        self.images.append(image)
        self.views.append(view)
        self.read_seconds += time.monotonic() - started

    def costs(self):
        return {'reads': len(self.views), 'decisions': self.decisions,
                'overview_seconds': self.overview_seconds, 'read_seconds': self.read_seconds,
                'overview_returned_pixels': self.overview.width * self.overview.height,
                'read_returned_pixels': sum(v['returned_pixels'][0] * v['returned_pixels'][1] for v in self.views),
                'read_region_requested_pixels': sum(v['read_pixels'][0] * v['read_pixels'][1] for v in self.views)}

    def step(self, action):
        if self.done:
            raise RuntimeError('episode already ended')
        candidates = self.observation()['candidates']
        self.decisions += 1
        error = None
        try:
            if not isinstance(action, dict):
                raise ValueError('action_must_be_object')
            if action.get('type') == 'read':
                k = action.get('candidate_index')
                if len(self.views) >= self.budget:
                    raise ValueError('read_budget_exhausted')
                if type(k) is not int or not 0 <= k < len(candidates):
                    raise ValueError('invalid_candidate_index')
                self._read(candidates[k])
            elif action.get('type') == 'stop':
                k = action.get('answer_index')
                if type(k) is not int or not 0 <= k < len(self.choices):
                    raise ValueError('invalid_answer_index')
                self.final, self.done, self.status = dict(action), True, 'stopped'
            else:
                raise ValueError('unknown_action_type')
        except (ValueError, openslide.OpenSlideError, OSError) as exc:
            error = str(exc)
        if not self.done and self.decisions >= self.budget + 1:
            self.done, self.status = True, 'decision_budget_exhausted'
        self.records.append({'decision': self.decisions, 'action': _safe(action),
                             'error': error, 'views_used': len(self.views)})
        return self.observation(), self.done, {'status': self.status, 'error': error,
                                              'remaining_budget': self.budget - len(self.views)}

    def serialize_observation(self):
        """Policy-safe JSON state; image paths refer exclusively to actual reads."""
        if self.output_dir is None:
            raise ValueError('output_dir is required for serializing image paths')
        obs = self.observation()
        del obs['overview']
        obs['images'] = [str((self.output_dir / 'overview.png').resolve())] + [
            str((self.output_dir / f'view_{k:02d}.png').resolve()) for k in range(len(self.views))]
        return obs

    def replay(self, actions, seed=0):
        self.reset(seed)
        for action in actions:
            self.step(action)
        return self.observation()

    def close(self):
        self.slide.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


WSINavigationEnvironment = NavigationEnvironment
