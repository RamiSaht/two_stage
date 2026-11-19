"""Drawings of the specialization methods (Figs. 5 and 6) and of the construction of the Monte Carlo
dropout distribution (Figs. 8 and 9), written as SVG and saved as PNG."""
from html import escape

import numpy as np
import resvg_py

WIDTH = 310
FONT = "'CMU Sans Serif', 'Latin Modern Sans', sans-serif"
INK, MUTED, LINE = '#1a1a1a', '#4a4a4a', '#6a6a6a'
SHARED, GRAY, LIGHT, BOX = '#ececec', '#b4b4b4', '#e4e4e4', '#ececec'


def fmt(value):
    return f'{float(value):.2f}'.rstrip('0').rstrip('.')


def css(**properties):
    return ';'.join(f"{key.replace('_', '-')}:{value}" for key, value in properties.items())


def tint(hex_color, amount):
    rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(255 - (255 - c) * amount):02x}' for c in rgb)


class SVG:
    """82 mm wide drawing, 1 unit = 1 px of a 310 px wide page."""

    def __init__(self, name, height, markers):
        self.name, self.height, self.markers = name, height, markers
        self.layers = []

    def layer(self, label):
        self.layers.append((label, []))

    def _add(self, markup):
        self.layers[-1][1].append(markup)

    def open_group(self, group_id):
        self._add(f'<g id="{group_id}" inkscape:label="{group_id}">')

    def close_group(self):
        self._add('</g>')

    def path(self, d, fill='none', stroke=LINE, width=0.75, dash=None, arrow=None, element_id=None,
             cap='round', join='round'):
        props = dict(fill=fill, stroke=stroke, stroke_width=fmt(width), stroke_linejoin=join, stroke_linecap=cap)
        if dash:
            props['stroke_dasharray'] = dash
        if arrow:
            props['marker_end'] = f'url(#{arrow})'
        name = f'id="{element_id}" ' if element_id else ''
        self._add(f'<path {name}d="{d}" style="{css(**props)}"/>')

    def line(self, x1, y1, x2, y2, color=INK, width=0.8, dash=None, arrow=None, element_id=None):
        self.path(f'M {fmt(x1)},{fmt(y1)} L {fmt(x2)},{fmt(y2)}', stroke=color, width=width, dash=dash,
                  arrow=arrow, element_id=element_id, cap='butt', join='miter')

    def rect(self, x, y, width, height, fill=BOX, stroke='none', stroke_width=0.8, rx=0, element_id=None):
        props = dict(fill=fill, stroke=stroke, stroke_width=fmt(stroke_width))
        name = f'id="{element_id}" ' if element_id else ''
        corner = f' rx="{fmt(rx)}"' if rx else ''
        self._add(f'<rect {name}x="{fmt(x)}" y="{fmt(y)}" width="{fmt(width)}" height="{fmt(height)}"{corner} '
                  f'style="{css(**props)}"/>')

    def circle(self, x, y, r, fill=INK, stroke='none', stroke_width=0.6, element_id=None):
        name = f'id="{element_id}" ' if element_id else ''
        self._add(f'<circle {name}cx="{fmt(x)}" cy="{fmt(y)}" r="{fmt(r)}" '
                  f'style="{css(fill=fill, stroke=stroke, stroke_width=fmt(stroke_width))}"/>')

    def text(self, x, y, runs, size=10, anchor='middle', color=INK, bold=False, rotate=None, element_id=None):
        """`runs` is a string or a list of (text, italic) pieces."""
        if isinstance(runs, str):
            runs = [(runs, False)]
        spans = ''.join(f'<tspan style="font-style:{"italic" if italic else "normal"}">{escape(text)}</tspan>'
                        for text, italic in runs)
        props = dict(font_family=FONT, font_size=f'{size}px', font_weight='bold' if bold else 'normal',
                     text_anchor=anchor, fill=color, stroke='none')
        name = f'id="{element_id}" ' if element_id else ''
        turn = f' transform="rotate({rotate} {fmt(x)} {fmt(y)})"' if rotate is not None else ''
        self._add(f'<text {name}x="{fmt(x)}" y="{fmt(y)}"{turn} xml:space="preserve" style="{css(**props)}">'
                  f'{spans}</text>')

    def markup(self):
        layers = []
        for number, (label, items) in enumerate(self.layers, start=1):
            body = '\n    '.join(items)
            layers.append(f'  <g inkscape:groupmode="layer" id="layer{number}" inkscape:label="{escape(label)}">\n'
                          f'    {body}\n  </g>')
        markers = ''.join(
            f'<marker id="{name}" markerUnits="userSpaceOnUse" markerWidth="{size}" markerHeight="{size}" '
            f'viewBox="0 0 6 6" refX="{ref}" refY="3" orient="auto"><path d="{shape}" style="fill:{color};stroke:none"/>'
            f'</marker>' for name, (size, ref, shape, color) in self.markers.items())
        return ('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
                f'<svg version="1.1" width="82mm" height="{fmt(82 * self.height / WIDTH)}mm" '
                f'viewBox="0 0 {WIDTH} {fmt(self.height)}" id="{self.name}" xmlns="http://www.w3.org/2000/svg" '
                'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
                'xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd">\n'
                '  <sodipodi:namedview pagecolor="#ffffff" inkscape:document-units="mm"/>\n'
                f'  <defs>{markers}</defs>\n' + '\n'.join(layers) + '\n</svg>\n')

    def save(self, path, dpi=300):
        """PNG at `path`."""
        path = path.with_suffix('.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(resvg_py.svg_to_bytes(svg_string=self.markup(), background='white', dpi=dpi)))
        return path


# ------------------------------------------------------------------------------------------
# Figs. 5 and 6: component specialization methods
# ------------------------------------------------------------------------------------------

GROUPS = {
    'specialized': dict(title='Component-specialized', lines=['Component-', 'specialized'], color='#68a9ce'),
    'independent': dict(title='Independent models', lines=['Independent', 'models'], color='#e44848'),
    'system': dict(title='System-level', lines=['System-level'], color='#8a8a8a'),
}
SUBGROUPS = {'conditioning': 'Conditioning', 'multihead': 'Multi-head'}
METHODS = [('specialized', 'conditioning', 'Embedding', 'embed'),
           ('specialized', 'multihead', 'MLP heads', 'head_mlp'),
           ('specialized', 'multihead', 'Linear heads', 'head_linear'),
           ('independent', None, 'Separate models', 'separate'),
           ('system', None, 'Comp. ignored', 'none')]
TREE_MARKERS = {'arrow': (4.5, 5.6, 'M 0,0.6 L 6,3 L 0,5.4 Z', LINE)}


def node(fig, x, cy, width, height, lines, fill, stroke='none', element_id='node'):
    fig.open_group(element_id)
    fig.rect(x, cy - height / 2, width, height, fill=fill, stroke=stroke, rx=2.5, element_id=f'{element_id}-box')
    first = cy + 3.4 - 5.6 * (len(lines) - 1)
    for number, line in enumerate(lines):
        fig.text(x + width / 2, first + 11.2 * number, line, element_id=f'{element_id}-text{number}')
    fig.close_group()


def specialization_tree():
    """Fig. 5: the five methods, grouped by how they use the component."""
    leaf_h, top = 28, 4
    centre = [top + leaf_h / 2 + i * (leaf_h + 6) for i in range(len(METHODS))]
    root_x, root_w, group_x, group_w, sub_x, sub_w, leaf_x, leaf_w = 7, 62, 83, 58, 155, 60, 229, 74
    fig = SVG('specialization_tree', centre[-1] + leaf_h / 2 + top, TREE_MARKERS)

    def mean_of(indices):
        return sum(centre[i] for i in indices) / len(indices)

    def elbow(x1, y1, x2, y2, element_id):
        middle = (x1 + x2) / 2
        fig.path(f'M {fmt(x1)},{fmt(y1)} H {fmt(middle)} V {fmt(y2)} H {fmt(x2)}', stroke='#8a8a8a', width=0.8,
                 element_id=element_id)

    groups = {key: [i for i, m in enumerate(METHODS) if m[0] == key] for key in GROUPS}
    subgroups = {key: [i for i, m in enumerate(METHODS) if m[1] == key] for key in SUBGROUPS}
    root_y = mean_of(range(len(METHODS)))

    fig.layer('Connectors')
    for key, members in groups.items():
        elbow(root_x + root_w, root_y, group_x, mean_of(members), f'edge-root-{key}')
        for i in members:
            if METHODS[i][1] is None:
                elbow(group_x + group_w, mean_of(members), leaf_x, centre[i], f'edge-{key}-{METHODS[i][3]}')
    for key, members in subgroups.items():
        group = METHODS[members[0]][0]
        elbow(group_x + group_w, mean_of(groups[group]), sub_x, mean_of(members), f'edge-{group}-{key}')
        for i in members:
            elbow(sub_x + sub_w, mean_of(members), leaf_x, centre[i], f'edge-{key}-{METHODS[i][3]}')

    fig.layer('Root')
    node(fig, root_x, root_y, root_w, 40, ['Use of', 'component', 'information'], SHARED, element_id='root')
    fig.layer('Groups')
    for key, members in groups.items():
        spec = GROUPS[key]
        node(fig, group_x, mean_of(members), group_w, 28, spec['lines'], tint(spec['color'], 0.35),
             stroke=spec['color'], element_id=f'group-{key}')
    for key, members in subgroups.items():
        colour = GROUPS[METHODS[members[0]][0]]['color']
        node(fig, sub_x, mean_of(members), sub_w, 18, [SUBGROUPS[key]], tint(colour, 0.18), stroke=colour,
             element_id=f'subgroup-{key}')
    fig.layer('Methods')
    for i, (group, _, name, code) in enumerate(METHODS):
        node(fig, leaf_x, centre[i], leaf_w, leaf_h, [name, [(f'({code})', True)]], '#ffffff',
             stroke=GROUPS[group]['color'], element_id=f'method-{code}')
    return fig


BOX_H, GROUP_GAP, ROW_GAP, DROP, SPLIT, LAYER_SHIFT = 17, 12, 24, 24, 12, 3
COLUMNS = [('Input', 4, 64), ('Backbone', 66, 120), ('Specialization', 120, 272), ('Output', 272, 306)]
SEPARATORS = (65, 120, 272)
X = dict(input=(7, 52), backbone=(71, 46), dense=(123, 30), head=(165, 56), select=(236, 30), embedding=(123, 52),
         concat=189, dense_after_concat=(202, 30), head_after_concat=(242, 26), rul=(277, 26))
ROW_LIST = [('specialized', 'embed', 'Conditioning: embedding'),
            ('specialized', 'heads', 'Multi-head: linear or MLP heads'),
            ('independent', 'separate', 'Separate models'),
            ('system', 'none', 'No component')]


def _span(key):
    x, width = X[key]
    return x, x + width


def _box(fig, key, cy, label, fill, stroke='none', element_id='box'):
    x, width = X[key]
    node(fig, x, cy, width, BOX_H, [label], fill, stroke, element_id)


def _arrow(fig, x1, y1, x2, y2, element_id=None, lead=2.0, tail=5.5):
    if abs(y1 - y2) < 0.5:
        d = f'M {fmt(x1)},{fmt(y1)} H {fmt(x2)}'
    else:
        a, b = x1 + lead, x2 - tail
        middle = (a + b) / 2
        d = (f'M {fmt(x1)},{fmt(y1)} H {fmt(a)} C {fmt(middle)},{fmt(y1)} {fmt(middle)},{fmt(y2)} '
             f'{fmt(b)},{fmt(y2)} H {fmt(x2)}')
    fig.path(d, arrow='arrow', element_id=element_id)


def _inputs(fig, c, code, with_component=True):
    _box(fig, 'input', c, 'Window', '#ffffff', stroke=GRAY, element_id=f'{code}-window')
    if with_component:
        _box(fig, 'input', c + DROP, 'Component', '#ffffff', stroke=GRAY, element_id=f'{code}-component')


def _output(fig, c, from_x, code):
    _box(fig, 'rul', c, 'RUL', '#ffffff', stroke=INK, element_id=f'{code}-rul')
    _arrow(fig, from_x, c, X['rul'][0], c)


def _select(fig, c, outputs_y, from_x, code):
    x0, x1 = _span('select')
    _box(fig, 'select', c, 'Select', '#ffffff', stroke=INK, element_id=f'{code}-select')
    for y, entry in zip(outputs_y, (c - 4, c + 4)):
        _arrow(fig, from_x, y, x0, entry)
    middle = (x0 + x1) / 2
    fig.path(f'M {fmt(_span("input")[1])},{fmt(c + DROP)} H {fmt(middle)} V {fmt(c + BOX_H / 2)}', arrow='arrow',
             element_id=f'{code}-component-to-select')
    _output(fig, c, x1, code)


def _row_embed(fig, c, colour, code):
    _inputs(fig, c, code)
    _box(fig, 'backbone', c, 'Backbone', SHARED, element_id=f'{code}-backbone')
    _arrow(fig, _span('input')[1], c, X['backbone'][0], c)
    _box(fig, 'embedding', c + DROP, 'Embedding', tint(colour, 0.3), stroke=colour, element_id=f'{code}-embedding')
    _arrow(fig, _span('input')[1], c + DROP, X['embedding'][0], c + DROP)
    top = c - 9
    fig.open_group(f'{code}-concatenated')
    for number, (fill, stroke) in enumerate([(SHARED, '#8a8a8a')] * 3 + [(tint(colour, 0.45), colour)] * 2):
        fig.rect(X['concat'], top + number * 6, 7, 6, fill=fill, stroke=stroke, stroke_width=0.6, rx=1)
    fig.close_group()
    fig.text(X['concat'] + 3.5, top - 4, 'concatenated', color=MUTED, element_id=f'{code}-concat-label')
    _arrow(fig, _span('backbone')[1], c, X['concat'], c)
    _arrow(fig, _span('embedding')[1], c + DROP, X['concat'], top + 24)
    x0, width = X['dense_after_concat']
    node(fig, x0, c, width, BOX_H, ['Dense'], SHARED, element_id=f'{code}-dense')
    _arrow(fig, X['concat'] + 7, c, x0, c)
    hx, hw = X['head_after_concat']
    node(fig, hx, c, hw, BOX_H, ['Head'], SHARED, element_id=f'{code}-head')
    _arrow(fig, x0 + width, c, hx, c)
    _output(fig, c, hx + hw, code)


def _row_heads(fig, c, colour, code):
    _inputs(fig, c, code)
    _box(fig, 'backbone', c, 'Backbone', SHARED, element_id=f'{code}-backbone')
    _box(fig, 'dense', c, 'Dense', SHARED, element_id=f'{code}-dense')
    _arrow(fig, _span('input')[1], c, X['backbone'][0], c)
    _arrow(fig, _span('backbone')[1], c, X['dense'][0], c)
    heads_y = [c - SPLIT, c + SPLIT]
    x, width = X['head']
    for y, label in zip(heads_y, ['Transformer', 'Bearing']):
        corner = f'M {fmt(x + LAYER_SHIFT)},{fmt(y - BOX_H / 2 - LAYER_SHIFT)} h {fmt(width)} v {fmt(BOX_H)}'
        fig.rect(x + LAYER_SHIFT, y - BOX_H / 2 - LAYER_SHIFT, width, BOX_H, fill='#ffffff', stroke=colour,
                 stroke_width=0.7, rx=2.5, element_id=f'{code}-head-{label.lower()}-optional-layer')
        fig.path(corner, stroke='#ffffff', width=0.9)
        fig.path(corner, stroke=colour, width=0.7, dash='2,1.5')
        _box(fig, 'head', y, label, tint(colour, 0.3), stroke=colour, element_id=f'{code}-head-{label.lower()}')
        _arrow(fig, _span('dense')[1], c, x, y)
    _select(fig, c, heads_y, x + width + LAYER_SHIFT, code)


def _row_separate(fig, c, colour, code):
    _inputs(fig, c, code)
    rows_y = [c - SPLIT, c + SPLIT]
    fill = tint(colour, 0.3)
    for y, label in zip(rows_y, ['Transformer', 'Bearing']):
        name = label.lower()
        _box(fig, 'backbone', y, 'Backbone', fill, stroke=colour, element_id=f'{code}-{name}-backbone')
        _box(fig, 'dense', y, 'Dense', fill, stroke=colour, element_id=f'{code}-{name}-dense')
        _box(fig, 'head', y, label, fill, stroke=colour, element_id=f'{code}-{name}-head')
        _arrow(fig, _span('input')[1], c, X['backbone'][0], y)
        _arrow(fig, _span('backbone')[1], y, X['dense'][0], y)
        _arrow(fig, _span('dense')[1], y, X['head'][0], y)
    _select(fig, c, rows_y, _span('head')[1], code)


def _row_none(fig, c, colour, code):
    _inputs(fig, c, code, with_component=False)
    for key, label in (('backbone', 'Backbone'), ('dense', 'Dense'), ('head', 'Head')):
        _box(fig, key, c, label, SHARED, element_id=f'{code}-{key}')
    _arrow(fig, _span('input')[1], c, X['backbone'][0], c)
    _arrow(fig, _span('backbone')[1], c, X['dense'][0], c)
    _arrow(fig, _span('dense')[1], c, X['head'][0], c)
    _output(fig, c, _span('head')[1], code)


ROWS = {'embed': (_row_embed, 21, DROP + BOX_H / 2),
        'heads': (_row_heads, SPLIT + BOX_H / 2 + LAYER_SHIFT, DROP + BOX_H / 2),
        'separate': (_row_separate, SPLIT + BOX_H / 2, DROP + BOX_H / 2),
        'none': (_row_none, BOX_H / 2, BOX_H / 2)}


def specialization_models():
    """Fig. 6: how each model uses the input window and the component."""
    fig = SVG('specialization_models', 100, TREE_MARKERS)
    header_y, y = 11, 18
    layout = []
    for group, spec in GROUPS.items():
        top, rows = y, []
        y += 3
        for _, code, title in [row for row in ROW_LIST if row[0] == group]:
            _, above, below = ROWS[code]
            title_y = y + 9
            c = title_y + 6 + above
            rows.append((code, title, title_y, c))
            row_bottom = c + below
            y = row_bottom + ROW_GAP
        layout.append((group, spec, top, row_bottom + 3, rows))
        y = row_bottom + 5 + GROUP_GAP
    bottom = y - GROUP_GAP

    fig.layer('Group backgrounds')
    for group, spec, top, end, _ in layout:
        fig.rect(2, top, WIDTH - 4, end - top, fill=tint(spec['color'], 0.1), rx=4, element_id=f'{group}-background')
    fig.layer('Columns')
    for x in SEPARATORS:
        fig.path(f'M {fmt(x)},{fmt(header_y - 9)} V {fmt(bottom)}', stroke=GRAY, width=0.7, dash='2.5,2',
                 element_id=f'separator-{fmt(x)}')
    for label, x0, x1 in COLUMNS:
        fig.text((x0 + x1) / 2, header_y, label, element_id=f'header-{label.lower()}', bold=True)
    for group, spec, top, end, rows in layout:
        fig.layer(spec['title'])
        background = tint(spec['color'], 0.1)
        fig.rect(2, top, WIDTH - 4, end - top, fill='none', stroke=spec['color'], stroke_width=1, rx=4,
                 element_id=f'{group}-outline')
        for code, title, title_y, c in rows:
            fig.rect(6, title_y - 9, len(title) * 4.9 + 4, 12, fill=background, rx=1)
            fig.text(8, title_y, title, anchor='start', color=MUTED, element_id=f'{code}-title')
            fig.open_group(f'model-{code}')
            ROWS[code][0](fig, c, spec['color'], code)
            fig.close_group()

    fig.layer('Legend')
    legend_y = bottom + 8
    fig.rect(8, legend_y, 14, 9, fill=SHARED, rx=2.5)
    fig.text(26, legend_y + 7.8, 'Shared by both components', anchor='start', color=MUTED, element_id='legend-shared')
    for offset, group in zip((158, 165), ('specialized', 'independent')):
        colour = GROUPS[group]['color']
        fig.rect(offset, legend_y, 7, 9, fill=tint(colour, 0.3), stroke=colour, stroke_width=0.6, rx=1)
    fig.text(177, legend_y + 7.8, 'Specific to one component', anchor='start', color=MUTED,
             element_id='legend-specific')
    layer_y = legend_y + 14
    blue = GROUPS['specialized']['color']
    fig.rect(8, layer_y, 14, 9, fill='#ffffff', stroke=blue, stroke_width=0.7, rx=1)
    fig.path(f'M 8,{fmt(layer_y)} h 14 v 9 h -14 Z', stroke='#ffffff', width=0.9)
    fig.path(f'M 8,{fmt(layer_y)} h 14 v 9 h -14 Z', stroke=blue, width=0.7, dash='2,1.5')
    fig.text(26, layer_y + 7.8, 'Optional multiple layers (MLP heads)', anchor='start', color=MUTED,
             element_id='legend-layer')
    fig.height = layer_y + 13
    return fig


# ------------------------------------------------------------------------------------------
# Figs. 8 and 9: Monte Carlo dropout for one input window
# ------------------------------------------------------------------------------------------

LABEL_SIZE, SMALL_SIZE = 10.6, 9.3
NETWORK_COLORS = ['#3a87bb', '#e44848', '#7a58a8']
NETWORK_SHAPE = (3, 4, 4, 1)
MCD_MARKERS = {'arrow': (6, 5.4, 'M 0,0.5 L 6,3 L 0,5.5 Z', INK),
               'arrow-small': (4, 5.4, 'M 0,0.5 L 6,3 L 0,5.5 Z', MUTED)}


def _input_traces(fig, window, x, y, width, height, group_id='input-window'):
    """Three channels of the real input window, each in its own lane."""
    channels = np.argsort(window.std(axis=0))[-3:]
    fig.open_group(group_id)
    fig.rect(x, y, width, height)
    lane = (height - 6) / 3
    for number, channel in enumerate(channels):
        values = np.array([block.mean() for block in np.array_split(window[:, channel], min(48, len(window)))])
        span = max(values.max() - values.min(), 1e-6)
        xs = np.linspace(x + 3, x + width - 3, len(values))
        ys = y + 3 + lane * (number + 1) - 1 - (values - values.min()) / span * (lane - 2)
        fig.path('M ' + ' L '.join(f'{fmt(a)},{fmt(b)}' for a, b in zip(xs, ys)), stroke=MUTED, width=0.6,
                 element_id=f'{group_id}-channel-{number + 1}', cap='butt')
    fig.close_group()


def _network(fig, x, y, width, height, seed, group_id, colour=INK):
    """Schematic network in which about half of the hidden units are switched off."""
    rng = np.random.default_rng(seed)
    columns = np.linspace(x, x + width, len(NETWORK_SHAPE))
    positions, active = [], []
    for column, units in enumerate(NETWORK_SHAPE):
        positions.append([(columns[column], y + height * (unit + 0.5) / units) for unit in range(units)])
        on = rng.random(units) >= 0.5 if 0 < column < len(NETWORK_SHAPE) - 1 else np.ones(units, bool)
        if not on.any():
            on[rng.integers(units)] = True
        active.append(on)
    fig.open_group(group_id)
    for column in range(len(NETWORK_SHAPE) - 1):
        for a, (x1, y1) in enumerate(positions[column]):
            for b, (x2, y2) in enumerate(positions[column + 1]):
                if active[column][a] and active[column + 1][b]:
                    fig.line(x1, y1, x2, y2, color=GRAY, width=0.45)
    for column, nodes in enumerate(positions):
        for unit, (cx, cy) in enumerate(nodes):
            if active[column][unit]:
                fig.circle(cx, cy, 2.1, fill=colour)
            else:
                fig.circle(cx, cy, 1.8, fill='#ffffff', stroke=GRAY, stroke_width=0.7)
    fig.close_group()


def _pass_stack(fig, x, y, width, height, seed, group_id, colour, cards=3, shift=(5, -4)):
    fig.open_group(group_id)
    for card in reversed(range(cards)):
        cx, cy = x + card * shift[0], y + card * shift[1]
        fig.rect(cx - 3, cy - 3, width + 6, height + 6, fill='#ffffff', stroke=GRAY, stroke_width=0.6,
                 element_id=f'{group_id}-card-{card + 1}')
        _network(fig, cx, cy, width, height, seed + 10 * card, f'{group_id}-pass-{card + 1}', colour)
    fig.close_group()


def _axis_range(values, pad=1.5):
    return max(np.floor(values.min() - pad), 0.0), np.ceil(values.max() + pad)


def _scale(low, high, left, right):
    factor = (right - left) / (high - low)
    return lambda value: left + (value - low) * factor


def _ticks(low, high, step=10):
    return np.arange(np.ceil(low / step) * step, high + 1e-9, step) + 0.0


def _place_points(values, x_of, low, columns, on_top=()):
    """Positions of the points of a histogram with one-day bars, `columns` points wide."""
    values = np.asarray(values, float)
    bars = np.floor(values - low).astype(int)
    rows, cols, heights = np.zeros(len(values), int), np.zeros(len(values), int), {}
    for bar in np.unique(bars):
        members = np.flatnonzero(bars == bar)
        top_row = (len(members) - 1) // columns
        heights[int(bar)] = top_row + 1
        taken = set()
        for i in members:
            if i in on_top:
                col = min(int((values[i] - low - bar) * columns), columns - 1)
                while (top_row, col) in taken:
                    col = (col + 1) % columns
                taken.add((top_row, col))
                rows[i], cols[i] = top_row, col
        free = ((r, c) for r in range(top_row + 1) for c in range(columns) if (r, c) not in taken)
        for i, (r, c) in zip([i for i in members if i not in on_top], free):
            rows[i], cols[i] = r, c
    pitch = (x_of(low + 1) - x_of(low)) / columns
    return x_of(low + bars + (cols + 0.5) / columns), (rows + 0.5) * pitch, heights, pitch


def _outline(fig, heights, x_of, low, base, pitch, element_id):
    first, last = min(heights), max(heights)
    d = [f'M {fmt(x_of(low + first))},{fmt(base)}']
    for bar in range(first, last + 1):
        top = base - heights.get(bar, 0) * pitch
        d += [f'L {fmt(x_of(low + bar))},{fmt(top)}', f'L {fmt(x_of(low + bar + 1))},{fmt(top)}']
    d += [f'L {fmt(x_of(low + last + 1))},{fmt(base)}', 'Z']
    fig.path(' '.join(d), fill=BOX, stroke=INK, width=0.7, element_id=element_id)


def mc_dropout_single(draws, window, network=1, n_passes=100):
    """Fig. 8: every pass of one network gives one point of its distribution."""
    values = draws[network * n_passes:(network + 1) * n_passes]
    row_centres, shown = [32, 70, 108, 160], [0, 1, 2, n_passes - 1]
    labels = ['Pass 1', 'Pass 2', 'Pass 3', f'Pass {n_passes}']
    low, high = _axis_range(values)
    left, right = 118, 304
    x_of = _scale(low, high, left, right)
    columns = max(2, round((x_of(low + 1) - x_of(low)) / 4.5))
    x, lift, heights, pitch = _place_points(values, x_of, low, columns, on_top=shown)
    base = 190 + max(heights.values()) * pitch
    fig = SVG('mc_dropout_single_network', base + 62, MCD_MARKERS)
    y = base - lift

    fig.layer('Input window')
    middle = (row_centres[0] + row_centres[-1]) / 2
    fig.text(2, middle - 31, 'Same input', size=LABEL_SIZE, anchor='start')
    fig.text(2, middle - 20, 'window', size=LABEL_SIZE, anchor='start')
    _input_traces(fig, window, 2, middle - 15, 48, 30)
    fig.layer('Connectors')
    fig.line(50, middle, 60, middle, element_id='input-to-bus')
    fig.line(60, row_centres[0], 60, row_centres[-1], element_id='bus')
    for centre in row_centres:
        fig.line(60, centre, 69, centre, arrow='arrow')
    for i, centre in zip(shown, row_centres):
        fig.line(x[i], centre + 4, x[i], y[i] - pitch * 0.6, color=MUTED, width=0.6, dash='2,1.6', arrow='arrow-small',
                 element_id=f'drop-pass-{i + 1}')
        fig.line(107, centre, x[i] - 3.4, centre, width=0.7, arrow='arrow', element_id=f'output-pass-{i + 1}')
    fig.layer('Network, one row per pass')
    for number, (centre, i) in enumerate(zip(row_centres, shown)):
        _network(fig, 70, centre - 13, 36, 26, seed=31 + number, group_id=f'network-pass-{i + 1}')
        fig.text(88, centre - 16, labels[number], size=SMALL_SIZE)
    for offset in (-5, 0, 5):
        fig.circle(88, 129.5 + offset, 1.2, fill=MUTED)
    fig.layer('Distribution')
    _outline(fig, heights, x_of, low, base, pitch, 'distribution-outline')
    for offset, text in ((-28, 'Distribution of'), (-15, 'one network')):
        fig.text(2, base + offset, text, size=LABEL_SIZE, anchor='start')
    fig.text(2, base - 3, f'{values.size} points, one per pass', size=SMALL_SIZE, anchor='start', color=MUTED)
    fig.layer('Points (one per pass)')
    for i in range(values.size):
        if i not in shown:
            fig.circle(x[i], y[i], pitch * 0.42, fill=NETWORK_COLORS[network], element_id=f'pass-{i + 1:03d}')
    for i, centre in zip(shown, row_centres):
        fig.circle(x[i], y[i], pitch * 0.5, fill=INK, element_id=f'pass-{i + 1:03d}')
        fig.circle(x[i], centre, 2.8, fill=INK, element_id=f'output-point-pass-{i + 1}')
    fig.layer('Axis and summary')
    _axis(fig, x_of, base, low, high, left, right, step=5 if high - low < 20 else 10)
    _summary(fig, x_of, base + 40, values.mean(), np.quantile(values, [0.025, 0.975]))
    return fig


def _axis(fig, x_of, y, low, high, left, right, step=10):
    fig.open_group('rul-axis')
    fig.line(left, y, right, y)
    for tick in _ticks(low, high, step):
        fig.line(x_of(tick), y, x_of(tick), y + 3.5)
        fig.text(x_of(tick), y + 13.5, f'{tick:.0f}', size=SMALL_SIZE, color=MUTED)
    fig.text((left + right) / 2, y + 26, 'Predicted RUL [days]', size=LABEL_SIZE)
    fig.close_group()


def _summary(fig, x_of, y, mean_value, bounds):
    fig.open_group('mean-and-interval')
    fig.line(x_of(bounds[0]), y, x_of(bounds[1]), y, width=1)
    for end in bounds:
        fig.line(x_of(end), y - 3.5, x_of(end), y + 3.5, width=1)
    fig.circle(x_of(mean_value), y, 3, fill=INK, element_id='mean-marker')
    fig.text(x_of(mean_value), y + 14, 'Mean', size=SMALL_SIZE)
    fig.text(x_of(bounds[0]) - 6, y + 3.3, '95% interval', size=SMALL_SIZE, anchor='end')
    fig.close_group()


def mc_dropout_ensemble(draws, window, point, n_networks=3, n_passes=100):
    """Fig. 9: three networks, 100 passes each, pooled into one predictive distribution."""
    network_of = np.repeat(np.arange(n_networks), n_passes)
    interval = np.quantile(draws, [0.025, 0.975])
    fig = SVG('mc_dropout_ensemble', 289, MCD_MARKERS)
    low, high = _axis_range(draws)
    plot_left, plot_right, hist_left, hist_right, top, bottom = 38, 214, 224, 304, 95, 249
    y_of = _scale(low - 1, high, bottom, top)
    x_of = _scale(0.5, n_networks * n_passes + 0.5, plot_left, plot_right)
    centres = [x_of(m * n_passes + 0.5 + n_passes / 2) for m in range(n_networks)]
    late = np.arange((n_networks - 1) * n_passes + 70, n_networks * n_passes)
    above = late[draws[late] > point + 3]
    highlighted = int(above[0]) if above.size else int(late[np.argmax(draws[late])])

    fig.layer('Input and networks')
    _input_traces(fig, window, 2, 3, 44, 28)
    fig.text(52, 12, 'Same input window, new dropout in every pass', size=LABEL_SIZE, anchor='start')
    fig.line(46, 17, centres[-1], 17, element_id='bus')
    for m, centre in enumerate(centres):
        fig.line(centre, 17, centre, 28.5, arrow='arrow')
        _pass_stack(fig, centre - 20, 40, 30, 24, seed=71 + m, group_id=f'network-{m + 1}', colour=NETWORK_COLORS[m])
        fig.text(centre, 78, f'Network {m + 1}', size=SMALL_SIZE)
        fig.text(centre, 89, f'{n_passes} passes', size=SMALL_SIZE, color=MUTED)
    fig.text(hist_left, 30, 'One RUL value', size=LABEL_SIZE, anchor='start')
    fig.text(hist_left, 42, 'per forward pass', size=SMALL_SIZE, anchor='start', color=MUTED)
    fig.layer('Network boundaries')
    for m in range(1, n_networks):
        fig.line(x_of(m * n_passes + 0.5), top, x_of(m * n_passes + 0.5), bottom, color=LIGHT, width=0.8)
    fig.layer('Distribution')
    bars, counts = np.unique(np.floor(draws - low).astype(int), return_counts=True)
    filled = dict(zip(bars.tolist(), counts.tolist()))
    length = (hist_right - hist_left - 10) / counts.max()
    d = []
    for bar in range(bars.min(), bars.max() + 1):
        reach = hist_left + filled.get(bar, 0) * length
        d += [f'{fmt(reach)},{fmt(y_of(low + bar))}', f'{fmt(reach)},{fmt(y_of(low + bar + 1))}']
    fig.path(f'M {fmt(hist_left)},{fmt(y_of(low + bars.min()))} L ' + ' L '.join(d)
             + f' L {fmt(hist_left)},{fmt(y_of(low + bars.max() + 1))} Z', fill=BOX, stroke=INK, width=0.7,
             element_id='distribution-histogram')
    fig.text((hist_left + hist_right) / 2, top - 6, 'Pooled distribution', size=SMALL_SIZE)
    fig.layer('Points (one per pass)')
    for i, value in enumerate(draws):
        if i != highlighted:
            fig.circle(x_of(i + 1), y_of(value), 1.45, fill=NETWORK_COLORS[network_of[i]], element_id=f'pass-{i + 1:03d}')
    fig.circle(x_of(highlighted + 1), y_of(draws[highlighted]), 2.6, fill=INK, stroke='#ffffff', stroke_width=0.6,
               element_id=f'pass-{highlighted + 1:03d}')
    fig.line(hist_left + 6, 47, x_of(highlighted + 1) + 2.2, y_of(draws[highlighted]) - 2.8, color=MUTED, width=0.6,
             arrow='arrow-small', element_id='connector-one-point')
    fig.layer('Axes and summary')
    fig.line(plot_left, top, plot_left, bottom)
    fig.line(plot_left, bottom, plot_right, bottom)
    fig.line(hist_left, top, hist_left, bottom)
    for tick in _ticks(low, high):
        fig.line(plot_left - 3.5, y_of(tick), plot_left, y_of(tick))
        fig.text(plot_left - 6, y_of(tick) + 3.2, f'{tick:.0f}', size=SMALL_SIZE, anchor='end', color=MUTED)
    for tick in (1, 100, 200, 300):
        fig.line(x_of(tick), bottom, x_of(tick), bottom + 3.5)
        fig.text(x_of(tick), bottom + 13.5, f'{tick}', size=SMALL_SIZE, color=MUTED)
    fig.text((plot_left + plot_right) / 2, bottom + 27, 'Forward pass', size=LABEL_SIZE)
    fig.text(11, (top + bottom) / 2, 'Predicted RUL [days]', size=LABEL_SIZE, rotate=-90)
    fig.line(plot_left, y_of(point), hist_right, y_of(point), width=0.9, element_id='mean-line')
    fig.open_group('interval-bracket')
    fig.line(hist_right - 3, y_of(interval[0]), hist_right - 3, y_of(interval[1]), width=1)
    for end in interval:
        fig.line(hist_right - 6.5, y_of(end), hist_right + 0.5, y_of(end), width=1)
    fig.close_group()
    fig.text(hist_right, y_of(interval[1]) - 4, '95% interval', size=SMALL_SIZE, anchor='end')
    fig.text(hist_right - 8, y_of(point) + 11, 'Mean', size=SMALL_SIZE, anchor='end')
    return fig
