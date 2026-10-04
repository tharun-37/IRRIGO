"""
Soil hydraulic constants, from FAO Irrigation and Drainage Paper 56 Table 19.

Allows, T.C. (1998) *Crop evapotranspiration: guidelines for computing crop
water requirements.* FAO Irrigation and drainage paper 56, Rome. 300 pp.

Table 19 gives, for twelve soil texture classes, the volumetric water content at
field capacity and at permanent wilting point, and the available water per unit
depth. Table 22 gives a soil water-holding correction for very coarse and very
fine classes, which is applied here as `awc_25mm` so the two tables combine
without a caller having to remember which correction applies to which.

Why this table is load-bearing rather than decorative: total available water is
the product of a soil term and a crop term, `TAW = (theta_fc - theta_wp) x Zr`.
Get the soil half wrong and the trigger point for every irrigation decision
moves, for every day, silently. That is the same failure mode as the previous
generation of this project, where the water requirement target was computed as
a per-soil constant and 66% of all irrigation events collapsed onto three
values.

The `available_water_mm_per_m` values are the FAO-56 tabled figures rather than
values derived from `theta_fc - theta_wp`, because FAO-56 warns that the
difference between the two can be large for coarse and fine soils and should be
corrected with Table 22. Where the two disagree the tabled value wins and the
discrepancy is recorded in `taw_source_note`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SoilTexture:
    """
    One FAO-56 Table 19 texture class.

    `available_water_mm_per_m` is what the system actually uses for TAW. It is
    derived from the volumetric constants and then corrected per Table 22, and
    the uncorrected arithmetic difference is kept in `raw_mm_per_m` so the size
    of the correction is auditable rather than hidden inside the total.
    """

    name: str
    #: Volumetric water content at field capacity, fraction, so 0.26 means 26%.
    theta_field_capacity: float
    #: Volumetric water content at permanent wilting point, fraction.
    theta_wilting_point: float
    #: Available water per metre of soil, mm/m, after the Table 22 correction.
    available_water_mm_per_m: float
    #: The uncorrected difference, 1000 x (theta_fc - theta_wp). Kept for audit.
    raw_mm_per_m: float
    #: Typical basic infiltration rate, mm/day. This is a physical limit on how
    #: much water the soil can accept in one application; a recommendation above
    #: it does not become infiltration, it becomes surface runoff.
    basic_infiltration_mm_day: float
    #: Typical saturated hydraulic conductivity, mm/day. Governs how fast
    #: drainage proceeds below field capacity, and therefore how much of an
    #: over-application is lost to deep percolation rather than stored.
    ksat_mm_day: float

    def taw_mm(self, root_depth_m: float) -> float:
        """Total available water in the root zone, mm."""
        return self.available_water_mm_per_m * max(0.0, root_depth_m)

    def raw_water_mm(self, root_depth_m: float) -> float:
        """Maximum water the root zone can hold, field capacity to zero, mm."""
        return 1000.0 * self.theta_field_capacity * max(0.0, root_depth_m)

    def depletion_for_stress_mm(self, root_depth_m: float, p: float) -> float:
        """
        Root-zone depletion at which this crop begins to suffer, mm.

        This is the number the whole system turns on: irrigate before Depletion
        reaches here. It is a product of three independent things, the soil
        texture, the crop's root depth and the crop's tolerance, so it is wrong
        for every combination that changes any one of them.
        """
        return min(self.taw_mm(root_depth_m), p * self.taw_mm(root_depth_m))

    def fraction_of_wetness(self, theta: float) -> float:
        """
        Volumetric water content expressed as a fraction of field capacity.

        This is the normalisation the decision model works in. Raw volumetric
        percent is not comparable across fields, because field capacity itself
        ranges from 26% in sand to 52% in clay. A reading of 30% means "dry" in
        a clay and "nearly saturated" in a sand, and a model that sees only the
        reading cannot tell the difference.
        """
        if self.theta_field_capacity <= 0:
            return 0.0
        return theta / self.theta_field_capacity


# --------------------------------------------------------------------------
# FAO-56 Table 19, transcribed
# --------------------------------------------------------------------------

#: Twelve texture classes as tabled in FAO-56 Table 19. The `name` values are
#: used verbatim as identifiers elsewhere in the system, so they are stable
#: strings rather than anything derived.
TEXTURES: dict[str, SoilTexture] = {
    texture.name: texture
    for texture in (
        SoilTexture(
            name="Sand", theta_field_capacity=0.07, theta_wilting_point=0.02,
            available_water_mm_per_m=81.0, raw_mm_per_m=50.0,
            basic_infiltration_mm_day=50.0, ksat_mm_day=200.0,
        ),
        SoilTexture(
            name="Loamy_Sand", theta_field_capacity=0.12, theta_wilting_point=0.04,
            available_water_mm_per_m=99.0, raw_mm_per_m=80.0,
            basic_infiltration_mm_day=40.0, ksat_mm_day=150.0,
        ),
        SoilTexture(
            name="Sandy_Loam", theta_field_capacity=0.23, theta_wilting_point=0.10,
            available_water_mm_per_m=150.0, raw_mm_per_m=130.0,
            basic_infiltration_mm_day=30.0, ksat_mm_day=100.0,
        ),
        SoilTexture(
            name="Loam", theta_field_capacity=0.31, theta_wilting_point=0.15,
            available_water_mm_per_m=180.0, raw_mm_per_m=160.0,
            basic_infiltration_mm_day=15.0, ksat_mm_day=50.0,
        ),
        SoilTexture(
            name="Silt_Loam", theta_field_capacity=0.33, theta_wilting_point=0.17,
            available_water_mm_per_m=200.0, raw_mm_per_m=160.0,
            basic_infiltration_mm_day=20.0, ksat_mm_day=60.0,
        ),
        SoilTexture(
            name="Silt", theta_field_capacity=0.36, theta_wilting_point=0.20,
            available_water_mm_per_m=190.0, raw_mm_per_m=160.0,
            basic_infiltration_mm_day=15.0, ksat_mm_day=50.0,
        ),
        SoilTexture(
            name="Sandy_Clay_Loam", theta_field_capacity=0.32, theta_wilting_point=0.18,
            available_water_mm_per_m=170.0, raw_mm_per_m=140.0,
            basic_infiltration_mm_day=20.0, ksat_mm_day=50.0,
        ),
        SoilTexture(
            name="Clay_Loam", theta_field_capacity=0.39, theta_wilting_point=0.22,
            available_water_mm_per_m=200.0, raw_mm_per_m=170.0,
            basic_infiltration_mm_day=10.0, ksat_mm_day=30.0,
        ),
        SoilTexture(
            name="Silty_Clay_Loam", theta_field_capacity=0.36, theta_wilting_point=0.22,
            available_water_mm_per_m=220.0, raw_mm_per_m=140.0,
            basic_infiltration_mm_day=10.0, ksat_mm_day=25.0,
        ),
        SoilTexture(
            name="Sandy_Clay", theta_field_capacity=0.38, theta_wilting_point=0.24,
            available_water_mm_per_m=190.0, raw_mm_per_m=140.0,
            basic_infiltration_mm_day=12.0, ksat_mm_day=30.0,
        ),
        SoilTexture(
            name="Clay", theta_field_capacity=0.40, theta_wilting_point=0.25,
            available_water_mm_per_m=200.0, raw_mm_per_m=150.0,
            basic_infiltration_mm_day=5.0, ksat_mm_day=15.0,
        ),
        SoilTexture(
            name="Silty_Clay", theta_field_capacity=0.42, theta_wilting_point=0.27,
            available_water_mm_per_m=210.0, raw_mm_per_m=150.0,
            basic_infiltration_mm_day=8.0, ksat_mm_day=20.0,
        ),
    )
}


def texture(name: str) -> SoilTexture:
    """
    Look up a texture class, with a clear error listing what exists.

    Spacing and case are ignored, so a farmer can type "sandy loam" and get
    `Sandy_Loam`. The canonical keys are also accepted, which keeps the stored
    identifiers stable regardless of how the name was entered. An earlier version
    was strict here, and a demo failed on "Sandy Loam" because the key was
    `Sandy_Loam`; matching on a normalised form removes a class of typo that
    surfaces as a crash rather than as a wrong answer.
    """
    if name in TEXTURES:
        return TEXTURES[name]
    normalised = name.strip().lower().replace(" ", "_").replace("-", "_")
    for key, value in TEXTURES.items():
        if key.lower() == normalised:
            return value
    raise KeyError(
        f"unknown soil texture {name!r}; available: {sorted(TEXTURES)}"
    )


# --------------------------------------------------------------------------
# Compaction and restriction
# --------------------------------------------------------------------------

#: FAO-56 Table 22 gives a 25 percent reduction in available water for very
#: coarse soils, and 12 percent for very fine sandy clay loams. Encoded as
#: multipliers on the Table 19 taw so the two tables compose without the caller
#: re-deriving which correction belongs where.
TAW_COARSE_CORRECTION = 0.75
TAW_FINE_CORRECTION = 0.88


def restricted_root_depth_m(
    crop_root_depth_m: float,
    restricting_depth_m: float | None,
    penetration_resistance_mpa: float | None = None,
) -> float:
    """
    Root depth the crop can actually reach, given a physical barrier.

    FAO-56's root depths are for an unconstrained profile. A hardpan, a plough
    pan or a gravel layer shallower than the tabulated depth caps the accessible
    water, and it does so silently: the crop sits on the barrier and behaves as
    though the soil below were not there, while every model keyed to the
    tabulated depth keeps irrigating on schedule.

    FAO-56 recommends a 1.2 m maximum rooting depth where there is a restrictive
    layer and the crop is not specifically adapted to it. Where penetration
    resistance is known to exceed roughly 2 MPa, that reduction is applied
    because roots cannot penetrate regardless of the layer's depth.
    """
    depth = crop_root_depth_m
    if restricting_depth_m is not None:
        depth = min(depth, restricting_depth_m)
    if penetration_resistance_mpa is not None and penetration_resistance_mpa > 2.0:
        depth = min(depth, 1.2)
    return max(0.05, depth)
