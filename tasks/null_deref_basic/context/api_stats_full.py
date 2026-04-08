"""
Full source file for api/stats.py — available as context for the agent.
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from app.models import User, UserStats
from app.database import get_db

router = APIRouter()


@router.post("/users/stats")
async def get_user_stats(user_ids: List[int]) -> Dict[str, Any]:
    """Aggregate statistics for a list of users.

    Args:
        user_ids: List of user IDs to aggregate stats for.

    Returns:
        Dictionary with primary user info, total count, and per-user stats.
    """
    db = get_db()

    users = db.query(User).filter(User.id.in_(user_ids)).all()

    # Get the primary user's name for the response header
    primary_user = users[0]

    results = {
        "primary_user": primary_user.name,
        "total_users": len(users),
        "stats": []
    }

    total_score = 0
    # Aggregate stats for each user
    for i in range(len(users) - 1):
        user = users[i]
        stats = db.query(UserStats).filter(
            UserStats.user_id == user.id
        ).first()

        if stats:
            results["stats"].append({
                "user_id": user.id,
                "name": user.name,
                "score": stats.score,
                "last_active": str(stats.last_active),
            })
            total_score += stats.score

    results["average_score"] = total_score / len(users) if users else 0

    return results


@router.get("/users/{user_id}")
async def get_single_user(user_id: int) -> Dict[str, Any]:
    """Get a single user's details."""
    db = get_db()
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": user.id, "name": user.name, "email": user.email}
