from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field



class KeyInfoInstance(BaseModel):
    """
    Layer 0 (L0): Represents a single structured, objective fact extracted from the dataset.
    Matches the output schema of the extraction prompt.
    """
    user_id: str                             
    provenance: str                       
    day_id: Optional[int] = None              
    WHAT: str = ""                       
    WHEN: str = ""                         
    WHERE: str = ""    
    WHO: str = ""                          
    WHY: str = ""                           
    HOW: str = ""                            

class KnowledgeNode(BaseModel):
    """
    The fundamental unit of the cognitive model graph (Layer 1 and above).
    """
    node_id: str
    user_id: str          
    layer: int
    title: str
    content: str
    source_instances: List[str] = Field(default_factory=list)
    source_nodes: List[str] = Field(default_factory=list)
    generating_dimension: Optional[str] = Field(default=None)

class Dimension(BaseModel):
    """
    Represents an analytical "lens" for synthesizing higher-level knowledge.
    """
    title: str
    description: str

class CodeStreamRecord(BaseModel):
    """
    Input record for CodeStream dataset (Joined Submission + Problem tables).
    """
    submission_id: str
    user_id: str
    problem_id: str
    attempt_number: int
    programming_language: str
    source_code: str
    final_verdict: str
    verdict_sequence_trace: str
    problem_concept: str

class UPSC2MRecord(BaseModel):
    """
    Input record for UPSC2M dataset (Joined Attempts + Questions tables).
    """
    user_id: str
    question_id: str
    category: str

    question: str

    options: List[str] 
    answer: int     
    user_answer: str   
    user_correct: bool
    time_taken: int
    difficulty: float
