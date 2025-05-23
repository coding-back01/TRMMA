import math


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def exp_prob(beta, x):
    """
    error distance weight.
    """
    return math.exp(-pow(x, 2) / pow(beta, 2))

def gps2grid(pt, mbr, grid_size):
    """
    mbr:
        MBR class.
    grid size:
        int. in meter
    """
    LAT_PER_METER = 8.993203677616966e-06
    LNG_PER_METER = 1.1700193970443768e-05
    lat_unit = LAT_PER_METER * grid_size
    lng_unit = LNG_PER_METER * grid_size

    lat = pt.lat
    lng = pt.lng
    locgrid_x = int((lat - mbr.min_lat) / lat_unit) + 1
    locgrid_y = int((lng - mbr.min_lng) / lng_unit) + 1

    return locgrid_x, locgrid_y

def get_normalized_t(first_pt, current_pt, time_interval):
    """
    calculate normalized t from first and current pt
    return time index (normalized time)
    """
    t = int(1 + ((current_pt.time_arr - first_pt.time_arr).seconds / time_interval))
    return t
