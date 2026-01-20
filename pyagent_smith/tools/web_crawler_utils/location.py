from typing import Optional

from pydantic import BaseModel, Field


class LocationInfo(BaseModel):
    """Model for extracted location information from a website."""

    location_name: str = Field(..., description="Name of the location")
    location_description: Optional[str] = Field(None, description="Description of the location")
