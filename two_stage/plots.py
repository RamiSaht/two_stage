from contextlib import contextmanager

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import config as cfg
from .evaluation import BACKBONE_NAME, CASE_FAILURES, GROUPS, display_name

INK, MUTED, RULE = '#0b0b0b', '#898781', '#d9d9d9'
BLUE, RED, GREEN, GRAY = '#68a9ce', '#e44848', '#71cd69', '#b4b4b4'
GROUP_COLOR = {'specialized': BLUE, 'independent': RED, 'system': '#8a8a8a', 'baseline': INK}
STATUS_COLOR = {0: '#ccd6dd', 1: '#68a9ce', 2: '#7fb069', 3: '#71cd69', 4: '#e44848', 5: '#b4b4b4'}
LINE_CONSTANT = (0, (4, 2))
LINE_NOMINAL = (0, (5, 1.5, 1, 1.5))
MM = 1 / 25.4
WIDTH_MM, HEIGHT_MM, LEFT_MM = 178, 215, 27

_AVAILABLE = {f.name for f in font_manager.fontManager.ttflist}
FONT = 'CMU Sans Serif' if 'CMU Sans Serif' in _AVAILABLE else 'DejaVu Sans'
BASE = {'font.family': FONT, 'xtick.labelsize': 8, 'ytick.labelsize': 8}
REPORT = {**BASE, 'font.size': 8, 'axes.linewidth': 1, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
          'lines.linewidth': 2}


@contextmanager
def style(extra=None):
    with plt.rc_context({**REPORT, **(extra or {})}):
        yield


def bold(size):
    for font in font_manager.fontManager.ttflist:
        if font.name == FONT and font.weight in (700, 'bold') and font.style == 'normal':
            return FontProperties(fname=font.fname, size=size)
    return FontProperties(family=FONT, weight='bold', size=size)


def tint(hex_color, amount):
    """The colour mixed with white."""
    rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(255 - (255 - c) * amount):02x}' for c in rgb)


def minus(value, digits=1):
    return f'{value:.{digits}f}'.replace('-', '−')


def frame(ax, width=0.6):
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(width)


def save(fig, path, dpi=300, tight=True):
    """PNG at `path`. The stage 2 figures keep their exact size (tight=False), the width of the page."""
    path = path.with_suffix('.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight' if tight else None)


# ------------------------------------------------------------------------------------------
# Data and preprocessing
# ------------------------------------------------------------------------------------------

def care_period(anomaly=False):
    """Fig. 1: training and prediction period of a time series, with the anomaly if there is one."""
    width, height, split = 236, 118 if anomaly else 80, 155
    with style():
        fig = plt.figure(figsize=(width / 72, height / 72))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_axis_off()
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        def span(start, end, y, colour):
            ax.annotate('', xy=(start, y), xytext=(end, y),
                        arrowprops=dict(arrowstyle='<->', color=colour, lw=1, mutation_scale=6, shrinkA=0, shrinkB=0))
            ax.plot([start, start], [y - 4.5, y + 4.5], color=colour, lw=1.2)
            ax.plot([end, end], [y - 4.5, y + 4.5], color=colour, lw=1.2)

        ax.add_patch(plt.Rectangle((0, 39), split, 41, color=BLUE, lw=0))
        ax.add_patch(plt.Rectangle((split + 0.5, 39), width - split - 0.5, 41, color=GRAY, lw=0))
        span(4, split - 1, 26, BLUE)
        span(split + 2, width - 2, 26, GRAY)
        ax.text(split / 2, 17, 'Training Period', color=BLUE, ha='center', va='center', fontsize=10)
        ax.text((split + width) / 2, 12, 'Prediction\nPeriod', color=GRAY, ha='center', va='center', fontsize=10,
                linespacing=1.0)
        if anomaly:
            ax.add_patch(plt.Rectangle((171, 44), 60, 31, color=RED, lw=0))
            span(171, 230, 96, RED)
            ax.text(200.5, 101, 'Anomaly\nPeriod', color=RED, ha='center', va='top', fontsize=10, linespacing=1.0)
        return fig


def rul_label():
    """Fig. 2: the generated RUL label, with illustrative values."""
    assumed, actual, pause_start, pause_end = 30.0, 38.5, 12.0, 17.0
    pause = pause_end - pause_start
    time = np.array([0.0, pause_start, pause_end, assumed])
    rul = np.array([assumed - pause, assumed - pause_end, assumed - pause_end, 0.0])
    with style():
        fig, ax = plt.subplots(figsize=(3.25, 2.1))
        ax.axvspan(pause_start, pause_end, color=GRAY, alpha=0.4, lw=0, label='Non-operation')
        ax.plot(time, rul, color=BLUE, lw=1.6, label='Generated RUL')
        ax.axvline(assumed, color=RED, ls='--', lw=1.2, label='Assumed failure point')
        ax.axvline(actual, color=INK, ls=':', lw=1.2, label='Actual failure (unknown)')
        ax.set_xlim(0, actual + 2)
        ax.set_ylim(0, rul[0] * 1.05)
        ax.annotate('', xy=(actual, rul[0] * 0.3), xytext=(assumed, rul[0] * 0.3),
                    arrowprops=dict(arrowstyle='<->', color=INK, lw=1, shrinkA=0, shrinkB=0))
        ax.text((assumed + actual) / 2, rul[0] * 0.34, 'Unknown\nlabel\nerror', ha='center', va='bottom',
                fontsize=8, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('Calendar time', fontsize=8)
        ax.set_ylabel('RUL [operating time]', fontsize=8)
        ax.legend(fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
        fig.tight_layout()
        return fig


def downtime_filter(removed, event_order):
    """Fig. 4: share of the records removed by the downtime filter, per event and status code."""
    codes = [c for c in cfg.STATUS_TEXT if c != cfg.NORMAL_STATUS and removed[f'n_{c}'].sum() > 0]
    top = removed['removed_pct'].max() * 1.15
    rc = {'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9, 'legend.fontsize': 8,
          'axes.edgecolor': '#8c8c8c', 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
          'ytick.major.width': 0.6, 'axes.unicode_minus': False, 'axes.spines.top': False,
          'axes.spines.right': False}
    with plt.rc_context({**BASE, **rc}):
        fig, axes = plt.subplots(2, 1, figsize=(3.25, 6.5))
        x = np.arange(len(event_order))
        for ax, period in zip(axes, ('training period', 'prediction period')):
            part = removed[removed['period'] == period].set_index('event').loc[event_order]
            order = [event_order[i] for i in np.argsort(part['removed_pct'].values, kind='stable')]
            part = part.loc[order]
            bottom = np.zeros(len(order))
            for code in codes:
                height = 100 * part[f'n_{code}'].values / part['records'].values
                ax.bar(x, height, width=0.66, bottom=bottom, color=STATUS_COLOR[code], edgecolor='white',
                       lw=0.4, zorder=3, label=f'{code}  {cfg.STATUS_TEXT[code]}' if ax is axes[0] else None)
                bottom += height
            ax.set_ylim(0, top)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
            ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0f}'))
            ax.set_xticks(x)
            ax.set_xticklabels([str(e) for e in order])
            ax.set_title(period.capitalize(), loc='left', pad=4)
            ax.set_axisbelow(True)
            ax.tick_params(length=2.5)
            ax.set_xlabel('Event')
            ax.set_ylabel('Records removed [%]')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=len(codes), frameon=False, loc='lower center', bbox_to_anchor=(0.5, 0.1),
                   handlelength=1.3, columnspacing=1.6, handletextpad=0.5)
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        return fig


# ------------------------------------------------------------------------------------------
# Stage 1
# ------------------------------------------------------------------------------------------

TARGET_LABEL = {'transformer': 'Transformer Cell Temp. (standardized)',
                'bearing': 'Rotor Bearing Temp. (standardized)'}


def detection(det, title, component, event_start=None, figsize=(3.25, 6)):
    """Figs. 10 and 11: measured and predicted target, and the CUSUM score against its threshold."""
    with plt.rc_context(BASE):
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        axes[0].plot(det['time_stamp'], det['y'], lw=1.5, color='black', label='Measured', alpha=0.8)
        axes[0].plot(det['time_stamp'], det['y_hat'], lw=1.5, color='tab:blue', label='NBM prediction')
        axes[0].set_ylabel(TARGET_LABEL[component], fontsize=8)
        axes[1].plot(det['time_stamp'], det['score'], lw=1.5, color='tab:red', label='CUSUM of residuals')
        axes[1].axhline(det['threshold'].iloc[0], color='gray', ls='--', lw=1.5, label='Threshold')
        axes[1].fill_between(det['time_stamp'], 0, det['score'].max(), where=det['flag'].values.astype(bool),
                             color='red', alpha=0.12, label='Above threshold')
        axes[1].set_yscale('symlog')
        axes[1].set_ylabel('CUSUM score', fontsize=8)
        if event_start is not None:
            for ax in axes:
                ax.axvline(event_start, color='green', ls='--', lw=1.5)
            axes[0].plot([], [], color='green', ls='--', label='Event start')
        axes[0].legend(fontsize=8, loc='lower right')
        axes[1].legend(fontsize=8, loc='lower right')
        fig.suptitle(title, fontsize=10)
        fig.autofmt_xdate()
        fig.tight_layout()
        axes[1].set_xticks(axes[1].get_xticks()[::2])
        return fig


def care_grid(grid):
    """CARE score of every detector setting (not in the paper)."""
    with style():
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
        for ax, (component, part) in zip(axes, grid.groupby('component')):
            table = part.pivot(index='k', columns='p', values='CARE')
            image = ax.imshow(table.values, cmap='Blues', vmin=0, vmax=1)
            for (i, j), value in np.ndenumerate(table.values):
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if value > 0.6 else INK)
            ax.set_xticks(range(len(table.columns)), [f'{p:g}' for p in table.columns])
            ax.set_yticks(range(len(table.index)), [f'{k:g}' for k in table.index])
            ax.set_xlabel('Threshold percentile p')
            ax.set_ylabel('Slack k')
            ax.set_title(f'{component.capitalize()} detector', loc='left', fontsize=9)
        fig.colorbar(image, ax=axes, label='CARE score', shrink=0.9)
        return fig


# ------------------------------------------------------------------------------------------
# Stage 2: summary figures
# ------------------------------------------------------------------------------------------

BLOCKS = [('Component-specialized (proposed)', GROUP_COLOR['specialized'], GROUPS['specialized']),
          ('Independent models (benchmark)', GROUP_COLOR['independent'], GROUPS['independent']),
          ('System-level (benchmark)', GROUP_COLOR['system'], GROUPS['system']),
          ('Baseline', INK, GROUPS['baseline'])]
SUBGROUPS = [(name, [f'{b}+{m}' for b in ('cnn', 'lstm', 'cnn_lstm')])
             for name, m in (('Embedding', 'embed'), ('Linear heads', 'head_linear'), ('MLP heads', 'head_mlp'))]
SUMMARY_PANELS = [
    ('mae', 'MAE (days)', 'lower is better', (10.0, 25.0), [10, 15, 20, 25], 'constant', True),
    ('r2', 'R$^2$', 'higher is better', (-2.0, 0.5), [-2, -1, 0], 'constant', True),
    ('crps', 'CRPS (days)', 'lower is better', (8.0, 22.0), [10, 15, 20], None, False),
    ('cov95', '95% coverage', 'nominal level 0.95', (0.0, 1.0), [0, 0.5, 1], 0.95, False),
    ('width95', 'Sharpness (days)', '95% interval width', (0.0, 35.0), [0, 10, 20, 30], None, False),
]
COMPONENT_PANELS = {
    'transformer': [
        ('mae', 'MAE (days)', 'lower is better', (0.0, 32.0), [0, 10, 20, 30], 'constant', True),
        ('r2', 'R$^2$', 'higher is better', (-2.0, 1.0), [-2, -1, 0, 1], 'constant', True),
        ('crps', 'CRPS (days)', 'lower is better', (0.0, 28.0), [0, 10, 20], None, False),
        ('cov95', '95% coverage', 'nominal level 0.95', (0.0, 1.0), [0, 0.5, 1], 0.95, False),
        ('width95', 'Sharpness (days)', '95% interval width', (0.0, 45.0), [0, 10, 20, 30, 40], None, False)],
    'bearing': [
        ('mae', 'MAE (days)', 'lower is better', (10.0, 25.0), [10, 15, 20, 25], 'constant', True),
        ('r2', 'R$^2$', 'higher is better', (-2.0, 0.5), [-2, -1, 0], 'constant', True),
        ('crps', 'CRPS (days)', 'lower is better', (8.0, 22.0), [10, 15, 20], None, False),
        ('cov95', '95% coverage', 'nominal level 0.95', (0.0, 1.0), [0, 0.5, 1], 0.95, False),
        ('width95', 'Sharpness (days)', '95% interval width', (0.0, 40.0), [0, 10, 20, 30, 40], None, False)],
}
COMPONENT_HEADING = {'transformer': 'Transformer failures (T2 in cases 5 and 6, T3 in cases 8 and 9)',
                     'bearing': 'Bearing failures (B2 in cases 5 and 8, B3 in cases 6 and 9)'}
SHADE = LinearSegmentedColormap.from_list('shade', ['#e4e4e4', '#f6f6f6', '#e4e4e4'])
DETERMINISTIC = 'Baseline is\ndeterministic'


def row_label(model):
    return 'Constant' if model == 'constant' else BACKBONE_NAME[model.split('+')[0]]


def _row_positions():
    y, positions, colours, spans = 0.0, {}, {}, []
    for title, colour, models in BLOCKS:
        y += 1.1
        first = y
        for i, model in enumerate(models):
            if i and i % 3 == 0:
                y += 0.6
            positions[model], colours[model] = y, colour
            y += 1.0
        spans.append((title, colour, first, y - 1.0))
        y += 1.0
    return positions, colours, spans


def _offset_x(ax, points):
    base = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    return base + mtransforms.ScaledTranslation(points / 72, 0, ax.figure.dpi_scale_trans)


def _offset_y(ax, points):
    base = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    return base + mtransforms.ScaledTranslation(0, -points / 72, ax.figure.dpi_scale_trans)


def _label_rows(ax, positions, colours, spans):
    ax.set_yticks(list(positions.values()))
    ax.set_yticklabels([row_label(m) for m in positions], fontsize=8)
    for label, model in zip(ax.get_yticklabels(), positions):
        label.set_color(colours[model])
    for name, models in SUBGROUPS:
        ys = [positions[m] for m in models]
        colour = colours[models[0]]
        ax.plot([0, 0], [min(ys) - 0.35, max(ys) + 0.35], color=colour, lw=0.7, transform=_offset_x(ax, -45),
                clip_on=False)
        ax.text(0, np.mean(ys), name, rotation=90, ha='center', va='center', fontsize=7.5, color=colour,
                transform=_offset_x(ax, -51), clip_on=False)
    for title, colour, first, _ in spans:
        ax.text(0, first - 1.05, title, fontproperties=bold(8.5), color=colour, ha='left', va='center',
                transform=_offset_x(ax, -(LEFT_MM - 2) / 25.4 * 72), zorder=10, clip_on=False,
                bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none'))


def _shade(ax, span, vertical=False, fontsize=7.5):
    """Shaded box where the deterministic baseline has no value."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if vertical:
        extent = (span[0], span[1], min(ylim), max(ylim))
        centre = ((span[0] + span[1]) / 2, (ylim[0] + ylim[1]) / 2)
    else:
        extent = (min(xlim), max(xlim), span[1], span[0])
        centre = ((xlim[0] + xlim[1]) / 2, (span[0] + span[1]) / 2)
    ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=SHADE, aspect='auto', extent=extent, zorder=1,
              interpolation='bilinear')
    ax.text(*centre, DETERMINISTIC, ha='center', va='center', fontsize=fontsize, color=MUTED, style='italic',
            zorder=2, linespacing=1.1, rotation=90 if vertical else 0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def summary(scores, panels=SUMMARY_PANELS, heading=None, height_mm=HEIGHT_MM):
    """Fig. 12 (and Figs. 15 and 16): every model on every metric. A dot is one case (or failure),
    the bar is the mean of the dots."""
    metrics = [p[0] for p in panels]
    positions, colours, spans = _row_positions()
    means = scores.groupby('model')[metrics].mean()
    with style():
        top_mm, bottom_mm = (17 if heading else 10), 22
        fig, axes = plt.subplots(1, len(panels), sharey=True, figsize=(WIDTH_MM * MM, height_mm * MM),
                                 gridspec_kw=dict(wspace=0.2))
        fig.subplots_adjust(left=LEFT_MM / WIDTH_MM, right=0.99, top=1 - top_mm / height_mm,
                            bottom=bottom_mm / height_mm)
        for ax, (metric, title, note, limits, ticks, reference, point_metric) in zip(axes, panels):
            for model, y in positions.items():
                if model == 'constant' and not point_metric:
                    continue
                values = scores.loc[scores['model'] == model, metric].to_numpy()
                mean, colour = means.loc[model, metric], colours[model]
                low = limits[0] if metric == 'r2' else None
                clipped = low is not None and mean < low
                shown = np.maximum(values, low) if low is not None else values
                ax.scatter(shown, np.full(len(shown), y), s=9, color=colour, alpha=0.45, linewidths=0, zorder=2,
                           clip_on=False)
                bar = low if clipped else mean
                ax.plot([bar, bar], [y - 0.36, y + 0.36], color=colour, lw=2.4, solid_capstyle='butt', zorder=3,
                        clip_on=False)
                if clipped:
                    ax.annotate(minus(mean), xy=(low, y), xytext=(4, 6), textcoords='offset points', fontsize=6.5,
                                color=colour, va='center', ha='left')
            if reference == 'constant':
                ax.axvline(means.loc['constant', metric], color=INK, lw=0.6, ls=LINE_CONSTANT, zorder=0)
            elif reference is not None:
                ax.axvline(reference, color=INK, lw=0.6, ls=LINE_NOMINAL, zorder=0)
            ax.set_xlim(*limits)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f'{t:g}'.replace('-', '−') for t in ticks])
            for _, _, _, last in spans[:-1]:
                ax.axhline(last + 1.05, color=RULE, lw=0.5, zorder=0)
            ax.set_title(title, fontsize=9.5, color=INK, loc='left')
            ax.set_xlabel(note, fontsize=7.5, color=MUTED)
            ax.tick_params(axis='x', labelsize=7.5, length=2.5, width=0.6)
            ax.tick_params(axis='y', length=0)
            frame(ax)
        axes[0].set_ylim(max(positions.values()) + 0.8, -0.7)
        _label_rows(axes[0], positions, colours, spans)
        for ax, panel in zip(axes, panels):
            if not panel[-1]:
                _shade(ax, (positions['constant'] - 0.75, positions['constant'] + 0.75), fontsize=6.5)
        handles = [Line2D([], [], marker='o', ls='', color=MUTED, alpha=0.6, ms=3.4, label='Single case'),
                   Line2D([], [], marker='|', ls='', color=MUTED, ms=10, mew=2.4, label='Mean of the four cases'),
                   Line2D([], [], color=INK, lw=0.7, ls=LINE_CONSTANT, label='Constant baseline'),
                   Line2D([], [], color=INK, lw=0.7, ls=LINE_NOMINAL, label='Nominal coverage level')]
        fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8,
                   bbox_to_anchor=(0.5 + LEFT_MM / WIDTH_MM / 2, 0.0))
        if heading:
            fig.text(2 / WIDTH_MM, 1 - 5 / height_mm, heading, fontproperties=bold(10.5), color=INK, ha='left',
                     va='center')
        return fig


def component_summary(failure_scores, component):
    """Figs. 15 and 16: the summary for the failures of one component."""
    part = failure_scores[failure_scores['component'] == component]
    return summary(part, COMPONENT_PANELS[component], COMPONENT_HEADING[component], height_mm=190)


CASE_PANELS = [('mae', 'MAE (days)', None, True), ('r2', 'R$^2$', (-2.0, 0.6), True),
               ('crps', 'CRPS (days)', None, False), ('cov95', '95% coverage', (0, 1.0), False),
               ('width95', 'Sharpness: 95% interval width (days)', None, False)]
GROUP_TITLE_SHORT = {'Component-specialized (proposed)': 'Component-specialized (proposed)',
                     'Independent models (benchmark)': 'Independent models\n(benchmark)',
                     'System-level (benchmark)': 'System-level\n(benchmark)', 'Baseline': 'Baseline'}
CONSTANT_LINE = {'transformer': LINE_CONSTANT, 'bearing': (0, (1, 1.5))}


def case_bars(failure_scores, case):
    """Figs. 13 and 17 to 19: two bars per model, one for each held-out failure of the case."""
    order = [m for _, _, models in BLOCKS for m in models]
    colours = {m: colour for _, colour, models in BLOCKS for m in models}
    x, xs, spans = 0.0, {}, []
    for title, colour, models in BLOCKS:
        first = x
        for i, model in enumerate(models):
            if i and i % 3 == 0:
                x += 0.4
            xs[model] = x
            x += 1
        spans.append((title, colour, first, x - 1))
        x += 1.0
    table = failure_scores[failure_scores['case'] == case].set_index(['model', 'event'])
    failures = CASE_FAILURES[case]
    width = 0.38
    with style():
        fig, axes = plt.subplots(len(CASE_PANELS), 1, figsize=(WIDTH_MM * MM, HEIGHT_MM * MM), sharex=True,
                                 gridspec_kw=dict(hspace=0.32))
        fig.subplots_adjust(left=0.07, right=0.99, top=0.935, bottom=38 / HEIGHT_MM)
        for ax, (metric, title, limits, point_metric) in zip(axes, CASE_PANELS):
            for model in order:
                if model == 'constant' and not point_metric:
                    continue
                colour = colours[model]
                for k, event in enumerate(failures):
                    value = table.loc[(model, event), metric]
                    shown = max(value, limits[0]) if limits else value
                    position = xs[model] + (k - 0.5) * width
                    ax.bar(position, shown, width=width, linewidth=0.6, edgecolor=colour, zorder=2,
                           color=colour if k == 0 else tint(colour, 0.35), hatch=None if k == 0 else '////')
                    if limits and value < limits[0]:
                        ax.text(position, limits[0] + 0.05, minus(value), rotation=90, fontsize=5.5, ha='center',
                                va='bottom', zorder=4, color='white' if colour == INK and k == 0 else INK)
            if point_metric:
                for event in failures:
                    ax.axhline(table.loc[('constant', event), metric], color=INK, lw=0.6, zorder=1,
                               ls=CONSTANT_LINE['transformer' if cfg.FAILURE_NAME[event][0] == 'T' else 'bearing'])
            if metric == 'cov95':
                ax.axhline(0.95, color=INK, lw=0.6, ls=LINE_NOMINAL, zorder=1)
            if metric == 'r2':
                ax.axhline(0, color=INK, lw=0.5)
            ax.set_xlim(min(xs.values()) - 0.8, max(xs.values()) + 0.8)
            if limits:
                ax.set_ylim(*limits)
            ax.tick_params(labelsize=7.5, length=2.5, width=0.6)
            ax.tick_params(axis='x', length=0)
            ax.set_title(title, fontsize=9.5, color=INK, loc='left')
            frame(ax)
            if not point_metric:
                _shade(ax, (xs['constant'] - 0.6, xs['constant'] + 0.6), vertical=True, fontsize=6)
        ax = axes[-1]
        ax.set_xticks(list(xs.values()))
        ax.set_xticklabels([row_label(m) for m in xs], rotation=90, fontsize=7.5)
        for label, model in zip(ax.get_xticklabels(), xs):
            label.set_color(colours[model])
        for name, models in SUBGROUPS:
            positions = [xs[m] for m in models]
            colour = colours[models[0]]
            ax.plot([min(positions) - 0.35, max(positions) + 0.35], [0, 0], color=colour, lw=0.7,
                    transform=_offset_y(ax, 46), clip_on=False)
            ax.text(np.mean(positions), 0, name, ha='center', va='top', fontsize=7.5, color=colour,
                    transform=_offset_y(ax, 50), clip_on=False)
        for title, colour, first, last in spans:
            ax.text((first + last) / 2, 0, GROUP_TITLE_SHORT[title], ha='center', va='top', fontproperties=bold(8),
                    color=colour, transform=_offset_y(ax, 66), clip_on=False, linespacing=1.1)
        t_event, b_event = failures
        t_name, b_name = cfg.FAILURE_NAME[t_event], cfg.FAILURE_NAME[b_event]
        handles = [Patch(facecolor='#9a9a9a', edgecolor='#555', label=f'{t_name} (transformer, event {t_event})'),
                   Patch(facecolor=tint('#9a9a9a', 0.35), edgecolor='#555', hatch='////',
                         label=f'{b_name} (bearing, event {b_event})'),
                   Line2D([], [], color=INK, lw=0.7, ls=CONSTANT_LINE['transformer'], label=f'Constant on {t_name}'),
                   Line2D([], [], color=INK, lw=0.7, ls=CONSTANT_LINE['bearing'], label=f'Constant on {b_name}'),
                   Line2D([], [], color=INK, lw=0.7, ls=LINE_NOMINAL, label='Nominal coverage level')]
        fig.legend(handles=handles, loc='upper right', ncol=3, frameon=False, fontsize=7.5,
                   bbox_to_anchor=(0.99, 0.999))
        fig.text(0.07, 0.982, f'Case {case}: held-out {t_name} and {b_name}', fontproperties=bold(9.5), color=INK)
        return fig


# ------------------------------------------------------------------------------------------
# Stage 2: Case 5 and the Constant baseline
# ------------------------------------------------------------------------------------------

def case_predictions(arrays, scores, chosen, case=5):
    """Fig. 14: the predictions of the selected models on the two held-out failures."""
    failures = CASE_FAILURES[case]
    order = [chosen['specialized'], chosen['independent'], chosen['system']]
    colour = {chosen[g]: GROUP_COLOR[g] for g in ('specialized', 'independent', 'system')}
    with style():
        fig, axes = plt.subplots(len(order), len(failures), sharex='col', sharey='col',
                                 figsize=(WIDTH_MM * MM, 125 * MM))
        fig.subplots_adjust(left=0.125, right=0.99, top=0.93, bottom=0.17, hspace=0.32, wspace=0.14)
        for col, event in enumerate(failures):
            name = cfg.FAILURE_NAME[event]
            component = 'transformer' if name[0] == 'T' else 'bearing'
            start = min(arrays[f'{m}|{event}|t'].min() for m in order + ['constant'])

            def days(t):
                return (t - start) / 86400.0

            constant_x = days(arrays[f'constant|{event}|t'])
            constant = float(np.mean(arrays[f'constant|{event}|mean']))
            top = max(max(arrays[f'{m}|{event}|mean'].max(), arrays[f'{m}|{event}|y'].max()) for m in order) * 1.12
            for row, model in enumerate(order):
                ax, key, c = axes[row, col], f'{model}|{event}', colour[model]
                x, pct = days(arrays[f'{key}|t']), arrays[f'{key}|pct']
                ax.fill_between(x, pct[0], pct[4], color=tint(c, 0.25), lw=0, zorder=1)
                ax.fill_between(x, pct[1], pct[3], color=tint(c, 0.5), lw=0, zorder=2)
                ax.plot(x, arrays[f'{key}|mean'], color=c, lw=1.2, zorder=4)
                ax.plot(x, arrays[f'{key}|y'], color=INK, lw=1.0, ls=(0, (4, 2)), zorder=5)
                ax.plot([constant_x.min(), constant_x.max()], [constant, constant], color=INK, lw=0.8,
                        ls=(0, (1, 1.5)), zorder=3)
                score = scores[(scores['model'] == model) & (scores['event'] == event)].iloc[0]
                ax.text(0.98, 0.94, f'MAE {score.mae:.1f} d\nCRPS {score.crps:.1f} d\nR$^2$ {minus(score.r2, 2)}',
                        transform=ax.transAxes, ha='right', va='top', fontsize=6.8, color=INK, linespacing=1.15,
                        zorder=6, bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.8', lw=0.8, alpha=0.8))
                if row == 0:
                    ax.set_title(f'Case {case}: {component.capitalize()} failure ({name})', loc='left',
                                 fontsize=9.5, color=INK, pad=6)
                if col == 0:
                    ax.set_ylabel('RUL (days)', fontsize=8)
                    words = display_name(model).rsplit(' ', 2)
                    ax.text(-0.19, 0.5, f'{words[0]}\n{" ".join(words[1:])}', transform=ax.transAxes, rotation=90,
                            ha='center', va='center', fontproperties=bold(7.5), color=c, linespacing=1.1)
                if row == len(order) - 1:
                    ax.set_xlabel('Time in the prediction period (days)', fontsize=7.5, color=MUTED)
                ax.set_ylim(0, top)
                ax.tick_params(labelsize=7.5, length=2.5, width=0.6)
                frame(ax)
        handles = [Patch(facecolor=tint('#8a8a8a', 0.5), label='50% interval'),
                   Patch(facecolor=tint('#8a8a8a', 0.25), label='95% interval'),
                   Line2D([], [], color='#8a8a8a', lw=1.2, label='Mean prediction'),
                   Line2D([], [], color=INK, lw=1.0, ls=(0, (4, 2)), label='Generated RUL'),
                   Line2D([], [], color=INK, lw=0.8, ls=(0, (1, 1.5)), label='Constant baseline')]
        fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=7.5,
                   bbox_to_anchor=(0.53, 0.0))
        return fig


def constant_baseline(table, constant, case=5):
    """Fig. 7: the RUL labels of the training failures against operating time, and the constant."""
    colour = {'transformer': '#1a1a1a', 'bearing': '#8a8a8a'}
    const_line, const_text = '#71cd69', '#3f9a38'

    def merged(values, names):
        groups = []
        for value, name in sorted(zip(values, names)):
            if groups and value - groups[-1][0][-1] < 1.0:
                groups[-1][0].append(value)
                groups[-1][1].append(name)
            else:
                groups.append(([value], [name]))
        return [(np.mean(v), ', '.join(sorted(n))) for v, n in groups]

    with style({'font.size': 7.5}):
        fig, ax = plt.subplots(figsize=(85 * MM, 62 * MM))
        fig.subplots_adjust(left=0.12, right=0.97, top=0.9, bottom=0.36)
        for _, row in table.iterrows():
            c = colour[row.component]
            ax.plot([0, row.max_label - row.min_label], [row.max_label, row.min_label], color=c, lw=1.4, zorder=3)
            ax.scatter([0], [row.max_label], s=26, color=c, zorder=5, clip_on=False)
            ax.scatter([row.max_label - row.min_label], [row.min_label], s=26, facecolor='white', edgecolor=c,
                       lw=1.3, zorder=5, clip_on=False)
        for component, c in colour.items():
            part = table[table.component == component]
            for value, _ in merged(part.max_label, part.name):
                ax.text(1.3, value + 1.5, f'{value:.0f} d', fontsize=7, color=c, ha='left', va='bottom')
            for value, names in merged(part.max_label - part.min_label, part.name):
                ax.text(value, 3.5, names, fontsize=7, color=c, ha='center', va='bottom')
        ax.axhline(constant, color=const_line, lw=1.6, ls=(0, (4, 2)), zorder=2)
        ax.text(62, constant + 2, f'Constant = mean of all\ntraining labels = {constant:.1f} d', ha='right',
                va='bottom', fontsize=6.8, color=const_text, linespacing=1.1)
        ax.set_xlim(0, 63)
        ax.set_ylim(0, max(66, table.max_label.max() + 8))
        ax.set_yticks([0, 20, 40, 60])
        ax.set_ylabel('RUL (days)', fontsize=7.5)
        ax.set_xlabel('Operating time (days)', fontsize=7.5, color=MUTED)
        ax.set_title(f'Training set of Case {case}', loc='left', fontsize=8, color=INK)
        ax.tick_params(labelsize=7, length=2.5, width=0.6)
        frame(ax)
        handles = [Line2D([], [], color=colour['transformer'], lw=1.4, label='Transformer failure'),
                   Line2D([], [], color=colour['bearing'], lw=1.4, label='Bearing failure'),
                   Line2D([], [], color=const_line, lw=1.6, ls=(0, (4, 2)), label='Constant baseline'),
                   Line2D([], [], marker='o', ls='', color=INK, ms=4.5, label='Maximum label (start)'),
                   Line2D([], [], marker='o', ls='', mfc='white', mec=INK, ms=4.5,
                          label='Minimum label (0 at the\nassumed failure)')]
        fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=6.8,
                   bbox_to_anchor=(0.55, 0.0))
        return fig


# ------------------------------------------------------------------------------------------
# Monte Carlo dropout settings (not in the paper)
# ------------------------------------------------------------------------------------------

FREQ_COLOR = {'1h': BLUE, '8h': RED, '24h': GREEN}


def passes_curves(passes, n=100):
    with style():
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
        for (_, _), part in passes.groupby(['model', 'event']):
            part = part.sort_values('passes')
            c = FREQ_COLOR.get(part['freq'].iloc[0], MUTED)
            axes[0].plot(part['passes'], 100 * (part['crps'] / part['crps'].iloc[-1] - 1), color=c, lw=0.8, alpha=0.6)
            axes[1].plot(part['passes'], part['cov95'] - part['cov95'].iloc[-1], color=c, lw=0.8, alpha=0.6)
        axes[0].set_ylabel(f'CRPS vs. {passes.passes.max()} passes [%]')
        axes[1].set_ylabel(f'95% coverage vs. {passes.passes.max()} passes')
        for ax in axes:
            ax.set_xscale('log')
            ax.set_xlabel('Passes per network')
            ax.axvline(n, color=INK, ls=':', lw=1)
        axes[0].legend(handles=[Line2D([], [], color=c, lw=1.5, label=f'{f} bins') for f, c in FREQ_COLOR.items()],
                       fontsize=7, frameon=False)
        fig.tight_layout()
        return fig


def ensemble_curves(summary_by_size, freq_of, reported=3):
    curves = summary_by_size.reset_index()
    with style():
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8))
        for model, part in curves.groupby('model'):
            c = FREQ_COLOR.get(freq_of[model], MUTED)
            for ax, column in zip(axes, ('crps', 'cov95', 'width95')):
                ax.plot(part['networks'], part[column], color=c, lw=0.8, alpha=0.6, marker='o', ms=2)
        median = curves.groupby('networks')[['crps', 'cov95', 'width95']].median()
        for ax, column, label in zip(axes, ('crps', 'cov95', 'width95'),
                                     ('CRPS (days)', '95% coverage', '95% interval width (days)')):
            ax.plot(median.index, median[column], color=INK, lw=2)
            ax.axvline(reported, color=INK, ls=':', lw=1)
            ax.set_xlabel('Networks per ensemble')
            ax.set_ylabel(label)
        axes[1].axhline(0.95, color=MUTED, ls='--', lw=1)
        axes[0].legend(handles=[Line2D([], [], color=c, lw=1.5, label=f'{f} bins') for f, c in FREQ_COLOR.items()]
                       + [Line2D([], [], color=INK, lw=2, label='median')], fontsize=7, frameon=False)
        fig.tight_layout()
        return fig
