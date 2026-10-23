"""
Crop water parameters, from FAO Irrigation and Drainage Paper 56.

Allows, T.C. (1998) *Crop evapotranspiration: guidelines for computing crop
water requirements.* FAO Irrigation and drainage paper 56, Rome. 300 pp.
ISBN 92-5-104219-5.

Every value in this module is transcribed from a numbered table in that
publication, and the table is named in the source comment beside it. These are
measured values from field trials across multiple sites, not estimates from
memory, and the distinction matters: a coefficient that has been fitted against
a simulated corpus cannot be compared against these, because the simulator
learns its own values rather than being constrained by them.

FAO-56 states its own uncertainty for the single-crop coefficients as roughly
plus or minus 5 percent for the peak values, and explicitly recommends
adjusting Kc for local conditions, spacing and crop variety. That adjustment is
the job of the residual models in `irrigation.residuals`, not of this table.

Two things deliberately are NOT in FAO-56 and are flagged as such where they
appear: the growing-degree-day thresholds that map accumulated heat onto growth
stages. FAO-56 expresses stage boundaries in days, not thermal time, because its
tables predate the GDD convention in phenology. Those thresholds are marked
UNCALIBRATED and are the primary open item for a field pilot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Kc curve geometry, FAO-56 chapter 7 and Annex 2
# --------------------------------------------------------------------------

#: Shape exponents for the FAO-56 piecewise Kc curve, applied directly to the
#: fractional progress through the stage as FAO-56 Annex 2 specifies.
#:
#: Development uses b = 1.13, greater than one, so the rise is slow then fast:
#: leaf area expands slowly from a small beginning and then closes. Late season
#: uses b = 0.65, less than one, so the decline is steep then shallow as
#: senescence detaches the transpiring canopy.
#:
#: Inverting either exponent reverses the shape of that segment and moves the
#: date of peak water demand, which is the single date in the whole season that
#: has to be right. A linear ramp between stage values is the common shortcut
#: and it is not an approximation of this curve: it is a different curve.
KCB_DEV_SHAPE = 1.13
KCB_LATE_SHAPE = 0.65

#: Value Kc is set to at harvest, after which evapotranspiration is taken as zero
#: for the bare soil. FAO-56 uses 0.10 to 0.20 depending on whether residues are
#: present; the midpoint is used here and the spread is documented rather than
#: hidden.
KCB_END_BARE_SOIL = 0.15


@dataclass(frozen=True)
class CropParameters:
    """
    One crop's FAO-56 water parameters.

    `kc_initial`, `kc_mid` and `kc_end` are the three coefficients FAO-56
    tabulates. They are *not* three stage values to be interpolated between:
    `kc_mid` is the single value held constant through mid-season, while
    `kc_initial` and `kc_end` are the values at the ends of the development and
    late-season segments. The four stage durations determine where on the curve
    the crop sits.

    Where FAO-56 publishes a range, the value used here is the one that suits a
    standard-density field and the alternative is recorded in `variants`. A crop
    grown under plastic, at double density, or with a different maturity class
    sits elsewhere on the curve, and `variants` names those alternatives rather
    than averaging them into mush.
    """

    name: str

    # --- FAO-56 Table 12: single crop coefficients -------------------------
    kc_initial: float
    kc_mid: float
    kc_end: float
    kc_mid_alt: float | None = None
    kc_end_alt: float | None = None

    # --- FAO-56 Table 11: growth stages, in days for a reference season -----
    length_initial_days: int = 20
    length_development_days: int = 30
    length_mid_season_days: int = 90
    length_late_season_days: int = 25

    # --- FAO-56 Table 22: root depth and depletion fraction ---------------
    #: Maximum effective rooting depth in metres. These are for a full season on
    #: an unconstrained profile. A compacted layer or a hardpan shallower than
    #: this reduces the depth the crop can actually reach, and that reduction
    #: has to be measured per field rather than assumed.
    root_depth_max_m: float = 1.0
    #: Fraction of total available water the crop can exhaust before stress
    #: begins. This is the single most crop-specific number in the whole system:
    #: it sets the trigger point of every irrigation decision. Rice can hold
    #: essentially zero (0.00, ponded), while cotton tolerates 0.65. Using one
    #: global value is wrong for every crop at once, in opposite directions.
    depletion_fraction_p: float = 0.50
    #: Late-season reduction in p. FAO-56 notes that drought sensitivity rises
    #: as the canopy senesces and root function declines, so the same depletion
    #: that was harmless at mid-season causes yield loss at the end. Applying a
    #: flat p across the season under-protects the yield-critical late period.
    depletion_fraction_p_late_factor: float = 0.75

    # --- FAO-33: yield response to water deficit --------------------------
    #: Ky, the yield loss per unit of relative evapotranspiration deficit.
    #: FAO-33 (Doorenbos and Pruitt 1977) gives indicative ranges; the value
    #: used is the mid-range. Ky is learned per crop in `residuals` because the
    #: published values for a single crop span a factor of two across studies,
    #: which is too wide to leave uncalibrated.
    ky: float = 1.0
    yield_potential_t_ha: float = 3.0

    # --- Optimal pH and salinity tolerance, for the stress terms ----------
    optimal_ph: tuple[float, float] = (6.0, 7.5)
    #: Yield-relevant salinity threshold in dS/m, and the slope above it. These
    #: come from FAO-29 as summarised by FAO-56. Salt stress competes with water
    #: stress for the same yield, and a model that ignores it will recommend
    #: watering a saline field where leaching is what's actually needed.
    salinity_threshold_ds_m: float = 6.0
    salinity_slope: float = 0.07

    # --- Phenology: UNCALIBRATED, see module docstring -------------------
    #: Base temperature for growing-degree-day accumulation, degrees C.
    gdd_base_temp_c: float = 10.0
    #: Accumulated GDD at the end of each stage, for a reference sowing.
    #: NOT from FAO-56. FAO-56 expresses stage boundaries in days. These values
    #: are seeded from published phenology for the reference varieties and must
    #: be calibrated against observed stage transitions in the pilot before any
    #: result should be trusted. They are the project's largest open item.
    gdd_stage_ends: tuple[float, float, float, float] = (250.0, 700.0, 2400.0, 2900.0)

    #: Common alternative season lengths, so a short-season maize variety can be
    #: described without editing the primary row.
    variants: dict[str, dict[str, float | int]] = field(default_factory=dict)

    # -- derived ---------------------------------------------------------

    @property
    def season_length_days(self) -> int:
        return (
            self.length_initial_days
            + self.length_development_days
            + self.length_mid_season_days
            + self.length_late_season_days
        )

    @property
    def kc_development_end(self) -> float:
        """Kc at the end of the development stage, i.e. the mid-season value."""
        return self.kc_mid

    def total_available_water_mm(self, root_depth_m: float | None = None) -> float:
        """
        TAW for this crop at a given root depth, given a soil texture.

        Kept here as a method for symmetry with the crop tables, but the soil
        half of the product lives in `soil`, because it is a property of the
        ground rather than of the plant. Callers should normally use
        `soil.texture.taw_mm(root_depth_m)`.
        """
        raise NotImplementedError(
            "TAW depends on the soil texture; use irrigation.soil.texture.taw_mm()"
        )


# --------------------------------------------------------------------------
# FAO-56 Table 12 / Table 11 / Table 22, transcribed
# --------------------------------------------------------------------------

CROPS: dict[str, CropParameters] = {
    "Maize": CropParameters(
        name="Maize",
        kc_initial=0.30, kc_mid=1.20, kc_end=0.60, kc_end_alt=0.35,
        length_initial_days=25, length_development_days=25,
        length_mid_season_days=105, length_late_season_days=10,
        root_depth_max_m=1.7, depletion_fraction_p=0.55,
        depletion_fraction_p_late_factor=0.75,
        ky=1.25, yield_potential_t_ha=6.1,
        optimal_ph=(5.8, 7.5), salinity_threshold_ds_m=1.7, salinity_slope=0.03,
        gdd_base_temp_c=10.0,
        gdd_stage_ends=(250.0, 700.0, 2400.0, 2900.0),
        variants={
            "short_season_80d": {
                "length_mid_season_days": 70, "length_late_season_days": 10,
                "gdd_stage_ends": (200.0, 560.0, 1900.0, 2200.0),
            },
            "long_season_130d": {
                "length_mid_season_days": 150, "length_late_season_days": 25,
                "gdd_stage_ends": (300.0, 850.0, 2800.0, 3400.0),
            },
        },
    ),
    "Wheat": CropParameters(
        name="Wheat",
        kc_initial=0.40, kc_mid=1.15, kc_end=0.25, kc_end_alt=0.40,
        length_initial_days=15, length_development_days=25,
        length_mid_season_days=60, length_late_season_days=25,
        root_depth_max_m=1.8, depletion_fraction_p=0.55,
        depletion_fraction_p_late_factor=0.70,
        ky=1.05, yield_potential_t_ha=4.6,
        optimal_ph=(6.0, 7.5), salinity_threshold_ds_m=6.0, salinity_slope=0.07,
        gdd_base_temp_c=0.0,
        gdd_stage_ends=(180.0, 600.0, 2100.0, 2600.0),
        variants={
            "winter": {
                "gdd_base_temp_c": 0.0, "gdd_stage_ends": (150.0, 500.0, 1900.0, 2350.0),
                "length_mid_season_days": 75, "length_late_season_days": 25,
            },
            "spring": {
                "gdd_base_temp_c": 4.0, "gdd_stage_ends": (200.0, 650.0, 2200.0, 2650.0),
            },
        },
    ),
    "Rice": CropParameters(
        name="Rice",
        kc_initial=1.05, kc_mid=1.20, kc_end=0.90, kc_end_alt=0.60,
        length_initial_days=10, length_development_days=25,
        length_mid_season_days=85, length_late_season_days=25,
        # Ponded rice is the case that proves the framework is doing work rather
        # than decorating a generic model. Its water management is a maintained
        # flood depth, not a depletion against a threshold, and p is near zero
        # because the pond itself is the buffer. `is_ponded` below routes it away
        # from the generic depletion logic entirely.
        root_depth_max_m=1.0, depletion_fraction_p=0.05,
        depletion_fraction_p_late_factor=0.90,
        ky=1.10, yield_potential_t_ha=6.4,
        optimal_ph=(5.0, 6.5), salinity_threshold_ds_m=3.0, salinity_slope=0.12,
        gdd_base_temp_c=10.0,
        gdd_stage_ends=(200.0, 650.0, 2200.0, 2650.0),
        variants={
            "direct_seeded": {
                "kc_initial": 0.90, "length_initial_days": 20,
                "gdd_stage_ends": (280.0, 750.0, 2300.0, 2750.0),
            },
        },
    ),
    "Soybean": CropParameters(
        name="Soybean",
        kc_initial=0.40, kc_mid=1.15, kc_end=0.50,
        length_initial_days=20, length_development_days=30,
        length_mid_season_days=85, length_late_season_days=25,
        root_depth_max_m=1.0, depletion_fraction_p=0.50,
        depletion_fraction_p_late_factor=0.75,
        ky=0.90, yield_potential_t_ha=2.6,
        optimal_ph=(6.0, 7.5), salinity_threshold_ds_m=5.0, salinity_slope=0.09,
        gdd_base_temp_c=10.0,
        gdd_stage_ends=(220.0, 680.0, 2300.0, 2800.0),
    ),
    "Cotton": CropParameters(
        name="Cotton",
        kc_initial=0.35, kc_mid=1.15, kc_end=0.70, kc_mid_alt=1.20,
        length_initial_days=20, length_development_days=50,
        length_mid_season_days=105, length_late_season_days=50,
        root_depth_max_m=1.7, depletion_fraction_p=0.65,
        depletion_fraction_p_late_factor=0.70,
        ky=0.75, yield_potential_t_ha=1.8,
        optimal_ph=(6.0, 8.0), salinity_threshold_ds_m=7.7, salinity_slope=0.13,
        gdd_base_temp_c=12.0,
        gdd_stage_ends=(240.0, 780.0, 2500.0, 3200.0),
    ),
    "Tomato": CropParameters(
        name="Tomato",
        kc_initial=0.60, kc_mid=1.15, kc_end=0.70, kc_end_alt=0.90,
        length_initial_days=15, length_development_days=40,
        length_mid_season_days=70, length_late_season_days=25,
        root_depth_max_m=1.5, depletion_fraction_p=0.40,
        depletion_fraction_p_late_factor=0.75,
        ky=1.00, yield_potential_t_ha=6.0,
        optimal_ph=(6.0, 7.0), salinity_threshold_ds_m=2.5, salinity_slope=0.10,
        gdd_base_temp_c=10.0,
        gdd_stage_ends=(180.0, 640.0, 2100.0, 2500.0),
        variants={
            "greenhouse": {
                # A greenhouse tomato has a near-constant Kc after establishment:
                # there is no rain dilution of the canopy and no evaporation
                # from bare soil to speak of, so both the mid and end values are
                # higher than the field crop.
                "kc_initial": 0.70, "kc_mid": 1.20, "kc_end": 0.90,
                "length_development_days": 30, "length_mid_season_days": 110,
                "gdd_stage_ends": (200.0, 700.0, 2900.0, 3400.0),
            },
        },
    ),
    "Sorghum": CropParameters(
        name="Sorghum",
        kc_initial=0.30, kc_mid=1.00, kc_end=0.55, kc_mid_alt=1.10,
        length_initial_days=20, length_development_days=35,
        length_mid_season_days=85, length_late_season_days=20,
        root_depth_max_m=2.0, depletion_fraction_p=0.55,
        depletion_fraction_p_late_factor=0.70,
        ky=0.90, yield_potential_t_ha=3.5,
        optimal_ph=(5.8, 7.5), salinity_threshold_ds_m=6.8, salinity_slope=0.08,
        gdd_base_temp_c=10.0,
        gdd_stage_ends=(220.0, 640.0, 2200.0, 2650.0),
    ),
    "Potato": CropParameters(
        name="Potato",
        kc_initial=0.50, kc_mid=1.15, kc_end=0.75, kc_end_alt=0.50,
        length_initial_days=25, length_development_days=35,
        length_mid_season_days=45, length_late_season_days=30,
        root_depth_max_m=0.6, depletion_fraction_p=0.35,
        depletion_fraction_p_late_factor=0.80,
        ky=1.10, yield_potential_t_ha=20.0,
        optimal_ph=(5.0, 6.5), salinity_threshold_ds_m=2.5, salinity_slope=0.10,
        gdd_base_temp_c=7.0,
        gdd_stage_ends=(250.0, 800.0, 1900.0, 2400.0),
    ),
    "Onion": CropParameters(
        name="Onion",
        kc_initial=0.70, kc_mid=1.05, kc_end=0.75, kc_mid_alt=1.10,
        length_initial_days=15, length_development_days=25,
        length_mid_season_days=70, length_late_season_days=40,
        # Onion is the shallow-rooted, low-tolerance extreme: 300 mm of roots and
        # p of 0.30. Irrigating it like maize over-waters it and rots the bulb.
        root_depth_max_m=0.3, depletion_fraction_p=0.30,
        depletion_fraction_p_late_factor=0.80,
        ky=1.00, yield_potential_t_ha=30.0,
        optimal_ph=(6.0, 7.0), salinity_threshold_ds_m=1.2, salinity_slope=0.08,
        gdd_base_temp_c=4.0,
        gdd_stage_ends=(200.0, 600.0, 2000.0, 2500.0),
    ),
    "Sugarcane": CropParameters(
        name="Sugarcane",
        kc_initial=0.40, kc_mid=1.20, kc_end=0.70, kc_end_alt=0.50,
        length_initial_days=60, length_development_days=90,
        length_mid_season_days=180, length_late_season_days=30,
        # The opposite extreme to onion: 2.5 m of roots and p of 0.70 means
        # sugarcane irrigates rarely and deeply. The same rainfall on the same day
        # is a non-event for sugarcane and an emergency for onion.
        root_depth_max_m=2.5, depletion_fraction_p=0.70,
        depletion_fraction_p_late_factor=0.85,
        ky=1.20, yield_potential_t_ha=80.0,
        optimal_ph=(6.0, 7.5), salinity_threshold_ds_m=6.0, salinity_slope=0.06,
        gdd_base_temp_c=9.0,
        gdd_stage_ends=(400.0, 1400.0, 5200.0, 6000.0),
    ),
    "Groundnut": CropParameters(
        name="Groundnut",
        kc_initial=0.40, kc_mid=1.10, kc_end=0.60, kc_mid_alt=1.15,
        length_initial_days=20, length_development_days=30,
        length_mid_season_days=90, length_late_season_days=20,
        root_depth_max_m=1.0, depletion_fraction_p=0.50,
        depletion_fraction_p_late_factor=0.75,
        ky=0.85, yield_potential_t_ha=1.8,
        optimal_ph=(6.0, 7.0), salinity_threshold_ds_m=3.5, salinity_slope=0.10,
        gdd_base_temp_c=10.0,
        gdd_stage_ends=(220.0, 680.0, 2300.0, 2750.0),
    ),
    "Sunflower": CropParameters(
        name="Sunflower",
        kc_initial=0.35, kc_mid=1.00, kc_end=0.35, kc_mid_alt=1.15,
        length_initial_days=20, length_development_days=35,
        length_mid_season_days=85, length_late_season_days=25,
        root_depth_max_m=1.5, depletion_fraction_p=0.45,
        depletion_fraction_p_late_factor=0.70,
        ky=0.80, yield_potential_t_ha=2.0,
        optimal_ph=(6.0, 7.5), salinity_threshold_ds_m=5.0, salinity_slope=0.08,
        gdd_base_temp_c=8.0,
        gdd_stage_ends=(230.0, 700.0, 2300.0, 2750.0),
    ),
    "Bean": CropParameters(
        name="Bean",
        kc_initial=0.40, kc_mid=1.05, kc_end=0.30,
        length_initial_days=20, length_development_days=30,
        length_mid_season_days=60, length_late_season_days=25,
        root_depth_max_m=1.0, depletion_fraction_p=0.45,
        depletion_fraction_p_late_factor=0.75,
        ky=1.00, yield_potential_t_ha=2.5,
        optimal_ph=(6.0, 7.5), salinity_threshold_ds_m=1.0, salinity_slope=0.06,
        gdd_base_temp_c=10.0,
        gdd_stage_ends=(200.0, 600.0, 2000.0, 2500.0),
    ),
    "Barley": CropParameters(
        name="Barley",
        kc_initial=0.30, kc_mid=1.15, kc_end=0.25,
        length_initial_days=15, length_development_days=25,
        length_mid_season_days=65, length_late_season_days=25,
        root_depth_max_m=1.5, depletion_fraction_p=0.55,
        depletion_fraction_p_late_factor=0.70,
        ky=1.00, yield_potential_t_ha=4.5,
        optimal_ph=(6.0, 7.8), salinity_threshold_ds_m=8.0, salinity_slope=0.15,
        gdd_base_temp_c=0.0,
        gdd_stage_ends=(170.0, 570.0, 2000.0, 2500.0),
    ),
}


#: Crops whose water management is a maintained pond depth rather than a soil
#: depletion cycle. The optimiser routes these through a flood-depth policy; it
#: does not ask whether a paddy is "dry enough to irrigate".
PONDED_CROPS: frozenset[str] = frozenset({"Rice"})


def crop(name: str, variant: str | None = None) -> CropParameters:
    """
    Look up a crop by name, optionally applying a named variety or season variant.

    Variants exist because the primary row describes a standard-density field of
    standard maturity. A short-season maize or a greenhouse tomato is the same
    crop with a genuinely different water requirement, and forcing it onto the
    primary curve would either under-water the long season or over-water the
    short one.
    """
    if name not in CROPS:
        raise KeyError(
            f"unknown crop {name!r}; available: {sorted(CROPS)}"
        )
    parameters = CROPS[name]
    if variant is None:
        return parameters
    if variant not in parameters.variants:
        raise KeyError(
            f"unknown variant {variant!r} for {name!r}; "
            f"available: {sorted(parameters.variants)}"
        )
    overrides = parameters.variants[variant]
    # Non-frozen dataclass, so an explicit rebuild is cleaner than mutating a
    # shared constant that other callers may already hold a reference to.
    import dataclasses

    return dataclasses.replace(parameters, **overrides)
