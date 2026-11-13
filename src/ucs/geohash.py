"""Vectorised geohash encoding (standard base32 geohash, longitude bit first)."""
import numpy as np

_BASE32 = np.array(list("0123456789bcdefghjkmnpqrstuvwxyz"))


def encode(lon, lat, precision: int = 8) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if not 1 <= precision <= 12:
        raise ValueError("precision must be 1..12")
    nbits = 5 * precision
    lon_bits = (nbits + 1) // 2
    lat_bits = nbits // 2
    # Integer cell index along each axis (the binary search of the geohash algorithm, done at once)
    li = np.clip(((lon + 180.0) / 360.0 * (1 << lon_bits)).astype(np.int64), 0, (1 << lon_bits) - 1)
    ti = np.clip(((lat + 90.0) / 180.0 * (1 << lat_bits)).astype(np.int64), 0, (1 << lat_bits) - 1)
    code = np.zeros(lon.shape, dtype=np.uint64)
    for k in range(nbits):  # interleave from the most significant bit; even positions are longitude
        if k % 2 == 0:
            bit = (li >> (lon_bits - 1 - k // 2)) & 1
        else:
            bit = (ti >> (lat_bits - 1 - k // 2)) & 1
        code = (code << np.uint64(1)) | bit.astype(np.uint64)
    chars = [
        _BASE32[((code >> np.uint64(5 * (precision - 1 - i))) & np.uint64(31)).astype(np.int64)]
        for i in range(precision)
    ]
    out = chars[0].astype(object)
    for c in chars[1:]:
        out = out + c.astype(object)
    return out.astype(str)


def decode_bounds(gh: str) -> tuple[float, float, float, float]:
    """(lon_min, lat_min, lon_max, lat_max) of one geohash cell."""
    lon = [-180.0, 180.0]
    lat = [-90.0, 90.0]
    even = True
    for ch in gh:
        v = "0123456789bcdefghjkmnpqrstuvwxyz".index(ch)
        for shift in range(4, -1, -1):
            bit = (v >> shift) & 1
            rng = lon if even else lat
            mid = (rng[0] + rng[1]) / 2
            rng[0 if bit else 1] = mid
            even = not even
    return lon[0], lat[0], lon[1], lat[1]
