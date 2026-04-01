from pydantic import BaseModel, Field
from typing import Optional, Dict

class Observation(BaseModel):
    current_email: Optional[Dict[str, str]] = Field(description="The email currently needing triage.")
    emails_remaining: int = Field(description="Number of emails left in the queue.")
    feedback: str = Field(description="Detailed feedback and reward reason from the previous action.")

class Action(BaseModel):
    email_id: str = Field(description="The ID of the email being processed.")
    chain_of_thought: str = Field(description="Step-by-step reasoning explaining why this routing makes sense.")
    department: str = Field(description="Must be 'sales', 'support', 'billing', or 'spam'.")
    priority: str = Field(description="Must be 'low', 'normal', 'high', or 'critical'.")

class Reward(BaseModel):
    value: float = Field(description="Dense reward score for the step (-1.0 to 1.0).")
    reason: str = Field(description="Explanation of the reward shaping.")