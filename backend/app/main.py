"""
FastAPI application for IRRIGO V2.

Startup order matters and is deliberate:

1. Weather and the sowing registry, so every planning path can rely on them.
2. The evaluation reports, which are read per request and need no warm-up.

If the weather corpus or the registry cannot be read the app still starts,
reports `weatherAvailable: false` and returns 503 from the planning paths. That
is the right failure mode: the health endpoint stays reachable so the operator
can see *why* nothing is working, which is the entire job of a monitoring
dashboard.

There is no WebSocket here, and that is a deliberate departure from V1 rather
than an omission. V1 pushed a live frame whenever a field node reported; V2 has no
hardware and no changing stream, only a plan that is a pure function of its
inputs. A socket that re-sent the same numbers would be theatre, so the
frontend polls on a timer instead.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path
from typing import Any

# The engine lives in the sibling `src/` tree (a src layout), not beside this
# package, so it must be on the import path before the routes import it. Without
# this line `uvicorn app.main:app` fails with ModuleNotFoundError: irrigation,
# while pytest worked only because the test module performed the same insertion.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# A plan is a handful of NumPy reductions over a 14-day horizon. OpenMP thread
# synchronisation costs more than the arithmetic at that size, and the service
# holds every core for the corpus load. An explicit setting in the environment
# always wins.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.compat import router as compat_router
from .api.routes import router as api_router
from .core.config import get_settings
from .services.compat import CompatService, CompatUnavailable
from .services.evaluation import EvaluationService
from .services.pipeline import AdvisoryPipeline
from .services.plans import PlanService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("iic")

settings = get_settings()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    plans = PlanService(
        settings.sowings_path,
        start_year=settings.weather_start_year,
        end_year=settings.weather_end_year,
    )
    plans.load()
    described = plans.describe()
    if described.get("available"):
        logger.info(
            "weather ready: %s records, %s stations, %s field(s)",
            described.get("records"),
            described.get("stations"),
            described.get("fields"),
        )
    else:
        logger.warning("weather unavailable: %s", described.get("reason"))

    app.state.plans = plans
    evaluation = EvaluationService(settings.reports_dir, settings.models_dir)
    app.state.evaluation = evaluation
    app.state.settings = settings

    # The V1-shaped surface the copied dashboard calls. It shares the same plan
    # service and, through the ML bridge, the inherited decision bundle.
    app.state.compat = CompatService(plans, evaluation, settings)

    # The inference pipeline joins V1's decision bundle to V2's requirement
    # estimator (`nir_model.joblib`), which is loaded here for the first time at
    # serve time. It degrades visibly to V1 + physics if the artifact is absent.
    app.state.pipeline = AdvisoryPipeline(app.state.compat, evaluation, settings)
    logger.info("advisor pipeline: %s", app.state.pipeline.describe())

    # Build every field's snapshot now rather than on the first `/advisor` call.
    # The engine walk behind a snapshot costs seconds per field, so leaving it
    # lazy made the dashboard's first paint as slow as the whole computation. It
    # runs here, behind the launcher's readiness probe, and the snapshot cache is
    # keyed on the plan version, so nothing downstream repeats it.
    try:
        app.state.compat.warm()
    except Exception:  # pragma: no cover - warm-up must never block startup
        logger.exception("advisor warm-up failed; first request will pay for it")

    reports = app.state.evaluation.describe()
    if not reports["reports"]["evaluation"]:
        logger.warning(
            "reports/evaluation.json is missing; the model page will explain how to build it"
        )
    if not reports["modelPresent"]:
        logger.warning("models/nir_model.joblib is missing; run scripts/train_model.py")

    try:
        yield
    finally:
        logger.info("shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Backend for IRRIGO V2. Registers sowings, plans water requirement and "
        "schedules with the FAO-56 engine, and serves the model evaluation and "
        "external validation reports."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root() -> dict[str, Any]:
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "api": "/api",
    }


@app.get("/health", include_in_schema=False)
def bare_health() -> dict[str, Any]:
    """Unversioned probe for scripts and uptime checks."""
    described = getattr(app.state, "plans", None)
    return {
        "status": "ok" if described is not None and described.ready else "degraded",
        "version": settings.app_version,
    }


@app.exception_handler(CompatUnavailable)
async def _compat_unavailable(_request, exc: CompatUnavailable) -> JSONResponse:
    """The compat surface 503s, it does not 500, when planning data is missing."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


app.include_router(api_router, prefix="/api")
app.include_router(compat_router, prefix="/api")
