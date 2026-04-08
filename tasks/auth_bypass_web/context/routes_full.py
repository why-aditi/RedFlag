"""
Full source file for app/admin/routes.py — available as context for the agent.
"""
from flask import Blueprint, request, jsonify, render_template, abort
from flask_login import login_required, current_user
from app.database import db
from app.models import User, AuditLog
from functools import wraps

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


def require_admin(f):
    """Decorator to require admin privileges."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function


@admin_bp.route('/users', methods=['GET'])
def list_users():
    """List all users in the system."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)

    users = User.query.paginate(page=page, per_page=per_page)

    return jsonify({
        "users": [
            {
                "id": u.id,
                "name": u.name,
                "email": u.email,
                "role": u.role,
                "created_at": str(u.created_at)
            }
            for u in users.items
        ],
        "total": users.total,
        "page": page,
        "pages": users.pages
    })


@admin_bp.route('/users/search', methods=['GET'])
@login_required
@require_admin
def search_users():
    """Search users by name."""
    name = request.args.get('name', '')

    if not name:
        return jsonify({"error": "Name parameter required"}), 400

    query = f"SELECT * FROM users WHERE name LIKE '%{name}%'"
    results = db.engine.execute(query)

    users = []
    for row in results:
        users.append({
            "id": row[0],
            "name": row[1],
            "email": row[2],
        })

    return jsonify({"users": users, "query": name})


@admin_bp.route('/users/<int:user_id>', methods=['DELETE'])
@login_required
@require_admin
def delete_user(user_id):
    """Delete a user by ID."""
    user = User.query.get_or_404(user_id)

    AuditLog.log_action(
        actor=current_user,
        action="delete_user",
        target=user
    )

    db.session.delete(user)
    db.session.commit()

    return jsonify({"status": "deleted", "user_id": user_id})
