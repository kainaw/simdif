import math
from ..simdif import Metric, METRICS, to_list_numeric
from ._helpers import _bounded_dif


def info_geodesic() -> str:
    return """
Geodesic Distance (Great-Circle / Hyperspherical)
-------------------------------------------------
The shortest-path distance between two points on a sphere (or hypersphere),
where each point is described by a list of angles (radians by default).

    1 angle   -> arc length on a circle
    2 angles  -> great-circle distance on a sphere (latitude, longitude)
    3+ angles -> geodesic on a hypersphere

Method:
    Each angle list is mapped to a unit vector on the k-sphere, the central
    angle between the two vectors is recovered via the dot product, and the
    distance is radius * central_angle. The result is bounded by pi * radius
    (antipodal points).

Roles:
    dist: radius * central_angle
    dif:  dist / (pi * radius)  -- equivalently central_angle / pi
    sim:  1 - dif

The maximum is known: the central angle cannot exceed pi, so no 1/(1+d)
squash and no supplied bound are needed and sim + dif == 1. Note that dif
cancels the radius entirely -- it is the fraction of the way around the
manifold, so two cities on Earth and the same pair scaled to a globe give
the identical dif. That is the property that makes dif comparable across
manifolds where dist is not.

Convention:
    The 2-angle case is the geographic one: the first angle is latitude
    (elevation from the equatorial plane) and the second is longitude
    (azimuth). This is what makes lat/lon input work directly. For k angles,
    the leading angles are elevations and the last is the azimuth.

Parameters:
    radius : a single number applied to the whole manifold. Defaults to 1.0.
    unit   : if it begins with 'D'/'d' (e.g. "degrees"), A and B are treated
             as degrees and converted to radians internally. Otherwise
             (default) they are assumed to already be in radians.
    """.strip()


def _is_degrees(unit) -> bool:
    return bool(unit) and str(unit).strip().lower().startswith('d')


def _to_radians(values, unit):
    if _is_degrees(unit):
        return [math.radians(v) for v in values]
    return list(values)


def _manifold_label(n):
    return {1: "circle", 2: "sphere"}.get(n, f"{n}-sphere (hypersphere)")


def _to_cartesian(angles):
    """
    Map a list of k angles (radians) to a unit vector on the k-sphere in
    R^(k+1), geographic convention.

    Recursively: the first angle is an elevation; the remaining angles
    describe a point on the lower sphere, scaled by cos(elevation); the
    last coordinate is sin(elevation).

        k=1: [cos t, sin t]                                 (circle)
        k=2: [cos lat*cos lon, cos lat*sin lon, sin lat]    (geographic)
    """
    if len(angles) == 1:
        t = angles[0]
        return [math.cos(t), math.sin(t)]
    head, rest = angles[0], angles[1:]
    lower = _to_cartesian(rest)
    c = math.cos(head)
    return [c * x for x in lower] + [math.sin(head)]


def _central_angle(v1, v2):
    """Central angle (radians) between two angle lists, via the dot product."""
    u1, u2 = _to_cartesian(v1), _to_cartesian(v2)
    dot = sum(x * y for x, y in zip(u1, u2))
    dot = max(-1.0, min(1.0, dot))  # clip: guards arccos against fp overshoot
    return math.acos(dot)


def explain_geodesic(a, b, **kwargs) -> str:
    radius = kwargs.get('radius', 1.0)
    unit = kwargs.get('unit', 'radians')
    v1, v2 = to_list_numeric(a), to_list_numeric(b)
    if len(v1) != len(v2):
        return "Error: Vector length mismatch"
    v1, v2 = _to_radians(v1, unit), _to_radians(v2, unit)

    u1, u2 = _to_cartesian(v1), _to_cartesian(v2)
    dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(u1, u2))))
    angle = math.acos(dot)
    dist = radius * angle

    u1s = "[" + ", ".join(f"{x:.4f}" for x in u1) + "]"
    u2s = "[" + ", ".join(f"{x:.4f}" for x in u2) + "]"
    d_max = math.pi * radius
    dif = _bounded_dif(dist, d_max)
    return f"""
A: {v1} (radians)
B: {v2} (radians)
Input Unit: {unit}
Radius: {radius}
Manifold: {_manifold_label(len(v1))}
Unit vector A: {u1s}
Unit vector B: {u2s}
Dot product (cos of central angle): {dot:.6f}
Central angle: acos({dot:.6f}) = {angle:.6f} rad
Geodesic Distance: {radius} * {angle:.6f} = {dist:.4f}
Maximum: pi * {radius} = {d_max:.4f} (derived -- antipodal points)
Difference (dist / max): {dist:.4f} / {d_max:.4f} = {dif:.4f}
Similarity (1 - dif): {1.0 - dif:.4f}
    """.strip()


@Metric
def dist_geodesic(a, b, **kwargs) -> float:
    if len(a)==0 and len(b)==0:
        return 0.0
    radius = kwargs.get('radius', 1.0)
    unit = kwargs.get('unit', 'radians')
    v1, v2 = to_list_numeric(a), to_list_numeric(b)
    if len(v1) != len(v2):
        raise ValueError(f"Vector length mismatch: {len(v1)} vs {len(v2)}")
    v1, v2 = _to_radians(v1, unit), _to_radians(v2, unit)
    return radius * _central_angle(v1, v2)


@Metric
def dif_geodesic(a, b, **kwargs) -> float:
    radius = kwargs.get('radius', 1.0)
    return _bounded_dif(dist_geodesic(a, b, **kwargs), math.pi * radius)


@Metric
def sim_geodesic(a, b, **kwargs) -> float:
    return 1.0 - dif_geodesic(a, b, **kwargs)


METRICS['geodesic'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_geodesic,
    'dif': dif_geodesic,
    'sim': sim_geodesic,
    'info': info_geodesic,
    'explain': explain_geodesic,
}


EARTH_RADIUS_KM = 6371.0088  # mean radius


def info_earth() -> str:
    return """
Earth Distance (Latitude / Longitude)
--------------------------------------
Convenience wrapper around Geodesic Distance for points given as
[latitude, longitude] in decimal degrees. Defaults to Earth's mean
radius (6371.0088 km) and unit='degrees', so results are in the same
unit as the radius (km by default). Both defaults can be overridden,
e.g. pass radius=3958.8 for miles, or unit='radians' if your
coordinates are already in radians.

Formula:
    Same as Geodesic Distance with A, B each of length 2. Latitude is the
    first (elevation) angle and longitude the second (azimuth) angle.

Roles:
    dist: great-circle distance in the same unit as the radius (km by default)
    dif:  dist / (pi * radius) -- 1.0 means antipodal, 0.5 a quarter of the
          globe away. With the default radius the ceiling is 20015.09 km.
    sim:  1 - dif

Because dif divides the radius back out, it is unit-free: the same two
points score the same dif whether the radius is in km or miles.
    """.strip()


def _validate_earth_pair(v1, v2):
    if len(v1) != 2 or len(v2) != 2:
        raise ValueError(
            f"Earth distance requires [latitude, longitude] pairs "
            f"(length 2): got lengths {len(v1)} and {len(v2)}"
        )


def explain_earth(a, b, **kwargs) -> str:
    v1, v2 = to_list_numeric(a), to_list_numeric(b)
    if len(v1) != len(v2):
        return "Error: Vector length mismatch"
    _validate_earth_pair(v1, v2)
    kwargs.setdefault('radius', EARTH_RADIUS_KM)
    kwargs.setdefault('unit', 'degrees')
    return explain_geodesic(a, b, **kwargs)


@Metric
def dist_earth(a, b, **kwargs) -> float:
    v1, v2 = to_list_numeric(a), to_list_numeric(b)
    if len(v1) != len(v2):
        raise ValueError(f"Vector length mismatch: {len(v1)} vs {len(v2)}")
    _validate_earth_pair(v1, v2)
    kwargs.setdefault('radius', EARTH_RADIUS_KM)
    kwargs.setdefault('unit', 'degrees')
    return dist_geodesic(a, b, **kwargs)


@Metric
def dif_earth(a, b, **kwargs) -> float:
    radius = kwargs.get('radius', EARTH_RADIUS_KM)
    return _bounded_dif(dist_earth(a, b, **kwargs), math.pi * radius)


@Metric
def sim_earth(a, b, **kwargs) -> float:
    return 1.0 - dif_earth(a, b, **kwargs)


METRICS['earth'] = {
    'class': 'vector',
    'default': 'dist',
    'dist': dist_earth,
    'dif': dif_earth,
    'sim': sim_earth,
    'info': info_earth,
    'explain': explain_earth,
}
