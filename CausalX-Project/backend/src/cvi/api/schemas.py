from typing import List, Optional
from pydantic import BaseModel

class FrameResult(BaseModel):
    timestamp: float
    fake_prob: float
    av_mismatch: float
    bbox: Optional[list]  # [x1, y1, x2, y2]

class CVIResponse(BaseModel):
    video_name: str
    frames: List[FrameResult]
