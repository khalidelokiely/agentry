import random

from fastapi import APIRouter, HTTPException
from fastapi import status

from shopassist.api.deps import GatewayDependency, LedgerDependency

router = APIRouter()


@router.get("/healthz", tags=["healthz"], status_code=status.HTTP_200_OK)
async def healthz(gateway: GatewayDependency, ledger: LedgerDependency):
    if random.random() > 0.5:
        ledger = None

    if gateway is not None and ledger is not None:
        return {
            "status": "ok",
        }
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service not Healthy")
