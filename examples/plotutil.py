"""
Plotutil
========
"""

import sys
import matplotlib

if sys.platform == "Linux":
    matplotlib.use("Agg")

import matplotlib.pyplot as pl
import matplotlib.animation as manimation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib.markers as mmark
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc, rcParams
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.offsetbox import AnchoredText
import matplotlib.image as mpimg

import numpy as np

import string
import copy

# colours = ['DeepPink', 'Purple', 'MediumSlateBlue', 'Blue', 'Teal',
#                 'ForestGreen',  'DarkOliveGreen', 'DarkGoldenRod',
#                 'DarkOrange', 'Coral', 'Red', 'Sienna', 'Black', 'DarkGrey']
colours = list(pl.rcParams["axes.prop_cycle"].by_key()["color"])


# matplotlib settings
legendsize = 10
labelsize = 18
ticksize = 15
lwidth = 1.5
markersize = 6.0
fontsize = 16
lettersize = 20.0
# ~ font = {'family' : 'serif',
# ~ 'weight' : 'normal',
# ~ 'size'   : fontsize}
#'sans-serif':'Helvetica'}
#'family':'serif','serif':['Palatino']}
# ~ rc('font', **font)
# rc("font", family='sans-serif')#, weight="normal", size="18")
rc("font", **{"family": "sans-serif", "sans-serif": ["stixsans", "helvetica", "arial"]})
rc("mathtext", **{"fontset": "stixsans"})
# rc('text', usetex=True)
# rcParams['text.latex.preamble'] += r"\usepackage{amsmath}\usepackage{xfrac}"
# rc('legend',**{'fontsize': 'medium'})
# rc('xtick',**{'labelsize': 'small'})
# rc('ytick',**{'labelsize': 'small'})
# rc('axes',**{'labelsize': 'large', 'labelweight': 'normal'})
rc("legend", **{"fontsize": labelsize})
rc("xtick", **{"labelsize": ticksize})
rc("ytick", **{"labelsize": ticksize})
rc("axes", **{"labelsize": labelsize, "labelweight": "normal"})


cs = ["r", "b", "g", "c", "y"]
css = ["r", "b", "g", "c", "y"]
cfl = ["fuchsia", "lime"]
cll = [colours[4], colours[2]]
mfs = ["D", "o", "v", "^", "s", "p"]
mls = ["+", "*", "x", "1", "2"]
lss = ["-", "--", "-.", ":"]
cmap = pl.get_cmap("jet")


def get_x_coords(spacings):
    coords = np.cumsum(spacings)
    coords /= coords[-1]
    return coords


def getYCoords(spacings):
    coords = np.cumsum(spacings)
    coords /= coords[-1]
    coords = coords[:-1]
    return coords


def myAx(ax):
    # customize the ax
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    return ax


def noFrameAx(ax):
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.draw_frame = False

    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def drawScaleBars(
    ax,
    xlabel=None,
    ylabel=None,
    h_offset=15,
    v_offset=15,
    b_offset=15,
    fstr_xlabel=r"%.2g ",
    fstr_ylabel=r"%.2g",
):
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", which="both", length=0)

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xwidth = xlim[1] - xlim[0]
    ywidth = ylim[1] - ylim[0]

    inv = ax.transData.inverted()
    p0, p1 = inv.transform((0.0, 0.0)), inv.transform((0.0, b_offset))
    by_offset = p1[1] - p0[1]
    p0, p1 = inv.transform((0.0, 0.0)), inv.transform((b_offset, 0.0))
    bx_offset = p1[0] - p0[0]

    p0, p1 = inv.transform((0.0, 0.0)), inv.transform((0.0, -h_offset))
    lx_offset = p1[1] - p0[1]
    p0, p1 = inv.transform((0.0, 0.0)), inv.transform((h_offset, 0.0))
    ly_offset = p1[0] - p0[0]

    # position and length
    sblen = xticks[-1] - xticks[-2]
    xpos = xlim[1] - bx_offset
    ypos = ylim[0]
    y_range = ylim[1] - ylim[0]

    # position and length
    sblen_ = yticks[1] - yticks[0]
    ypos_ = ylim[0] + by_offset
    xpos_ = xlim[1]
    x_range = xlim[1] - xlim[0]

    px = ax.transData.transform_point((xpos - sblen / 2.0, ypos))
    py = ax.transData.transform_point((xpos_, ypos_ + sblen_ / 2.0))

    px = (xpos - sblen / 2.0, ypos)
    py = (xpos_, ypos_ + sblen_ / 2.0)

    if xlabel is not None:
        # draw the scale bar
        ax.plot(
            [xpos - sblen, xpos], [ypos, ypos], "k-", lw=1.5 * lwidth, clip_on=False
        )
        ax.annotate(
            fstr_xlabel % sblen + xlabel,  # xycoords='axes points',
            xy=px,
            xytext=(px[0], px[1] + lx_offset),
            clip_on=False,
            size=ticksize,
            rotation=0,
            ha="center",
            va="center",
        )

    if ylabel is not None:
        # draw y scalebar
        ax.plot(
            [xpos_, xpos_],
            [ypos_, ypos_ + sblen_],
            "k-",
            lw=1.5 * lwidth,
            clip_on=False,
        )
        ax.annotate(
            fstr_ylabel % sblen_ + ylabel,  # xycoords='axes points',
            xy=py,
            xytext=(py[0] + ly_offset, py[1]),
            clip_on=False,
            size=ticksize,
            rotation=90,
            ha="center",
            va="center",
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plotSpikeRaster(
    ax,
    spike_arr,
    y0=0.0,
    margin=0.2,
    cs="k",
    plotargs={},
    inv_order=True,
    tlim=None,
    no_frame=True,
):
    if no_frame:
        ax = noFrameAx(ax)

    ny = len(spike_arr)
    hy = 1.0

    ypos = np.arange(ny) * (hy + margin)
    ycoo = [[y0 + yp, y0 + yp + hy] for yp in ypos]
    if inv_order:
        ycoo = ycoo[::-1]
    yloc = np.mean(ycoo, axis=1)

    if not isinstance(cs, list):
        cs = [cs for _ in range(ny)]
    else:
        assert len(cs) == ny

    if "c" in plotargs:
        del plotargs["c"]

    for ii in range(ny):
        for tsp in spike_arr[ii]:
            ax.plot([tsp, tsp], ycoo[ii], c=cs[ii], **plotargs)

    if tlim is not None:
        ax.set_xlim(tlim)

    return yloc


def myLegend(ax, add_frame=True, **kwarg):
    leg = ax.legend(**kwarg)
    if add_frame:
        frame = leg.get_frame()
        frame.set_color("white")
        frame.set_alpha(0.8)
    else:
        frame = leg.get_frame()
        frame.set_alpha(0.0)
    return leg


def myColorbar(ax, im, **kwargs):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    return pl.colorbar(im, cax=cax, **kwargs)


def getAnnotations(*args):
    assert len(args) > 0
    if isinstance(args[0], int):
        n_panel = args[0]
        alphabet = string.ascii_uppercase
        labels = [
            AnchoredText(
                r"" + letter,
                loc=2,
                prop=dict(size=lettersize),
                pad=0.0,
                borderpad=-1.5,
                frameon=False,
            )
            for letter in alphabet[:n_panel]
        ]
    else:
        labels = [
            AnchoredText(
                labelstr,
                loc=2,
                prop=dict(size=lettersize),
                pad=0.0,
                borderpad=-1.5,
                frameon=False,
            )
            for labelstr in args
        ]
    return labels


class TransformedCMap(mcolors.Colormap):
    def __init__(self, func, cmap):
        """
        `func`: bijective function on interval [0,1]
        """
        self.func = func
        self.cmap = cmap

        # copy cmap variables to this class
        orig_keys = set(cmap.__dict__.keys())
        for key in orig_keys:
            self.__dict__[key] = copy.deepcopy(cmap.__dict__[key])

    def __call__(self, x, alpha=None, bytes=False):
        return self.cmap(self.func(x), alpha=alpha, bytes=bytes)


def plotStoredImg(img_path):
    pl.figure()
    ax = noFrameAx(pl.gca())

    img = mpimg.imread(img_path)
    ax.imshow(img)

    pl.show()
