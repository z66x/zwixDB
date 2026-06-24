from pydantic import BaseModel
from typing import List, Optional

class SearchFilter(BaseModel):
    min_year: Optional[int] = None

class QueryRequest(BaseModel):
    query: str
    k: int = 5
    filters: Optional[SearchFilter] = None

class SearchResult(BaseModel):
    movie_id: int
    title: str
    year: int
    genres: str
    director: str
    overview: str
    distance: float

class SearchResponse(BaseModel):
    results: List[SearchResult]
    latency_ms: float
    engine: str