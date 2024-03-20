class Layers(object):
    IMAGE = "_image"
    SEGMENTATION = "_segmentation"
    COORDINATES = "_coordinates"
    LABELS = "_labels"
    DATA = "_data"
    PLOT = "_plot"
    OBS = "_obs"
    NEIGHBORS = "_neighbors"
    INTENSITY = "_intensity"


class Dims(object):
    CHANNELS = "channels"
    X = "x"
    Y = "y"
    RGBA = "rgba"
    CELLS = "cells"
    COORDINATES = "coordinates"
    LABELS = "labels"
    FEATURES = "features"
    PROPS = "props"
    NEIGHBORS = "neighbors"

    IMAGE = ["channels", "x", "y"]
    COLORED_IMAGE = ["channels", "x", "y", "rgba"]
    SEGMENTATION = ["x", "y"]
    DATA = ["cell_idx", "channels"]


class Attrs(object):
    IMAGE_COLORS = "image_colors"
    LABEL_COLORS = "label_colors"
    LABEL_NAMES = "label_names"


class Props(object):
    COLOR = "_color"
    NAME = "_name"


class Features(object):
    LABELS = "_labels"
    ORIGINAL_LABELS = "_original_labels"
    X = "centroid-1"  # "_x"
    Y = "centroid-0"  # "_y"
    TRANSFORMS = ["arcsinh", "log", "sqrt", "zscore"]
    BINARIZATION_METHODS = ["otsu"]


COLORS = [
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#6A3A4C",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
    "#00FECF",
    "#B05B6F",
    "#8CD0FF",
    "#3B9700",
    "#04F757",
    "#C8A1A1",
    "#1E6E00",
    "#7900D7",
    "#A77500",
    "#6367A9",
    "#A05837",
    "#6B002C",
    "#772600",
    "#D790FF",
    "#9B9700",
    "#549E79",
    "#FFF69F",
    "#201625",
    "#72418F",
    "#BC23FF",
    "#99ADC0",
    "#3A2465",
    "#922329",
    "#5B4534",
    "#FDE8DC",
    "#404E55",
    "#0089A3",
    "#CB7E98",
    "#A4E804",
    "#324E72",
]


PROPS_DICT = {"centroid-1": Features.X, "centroid-0": Features.Y}
