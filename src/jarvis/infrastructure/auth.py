"""
JARVIS-PRIME Authentication & Access Control
================================================

Role-Based Access Control (RBAC):
- User roles: admin, researcher, viewer
- API key management
- Permission-based agent/tool access
- Session tracking
"""
from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class User:
    """Represents a JARVIS-PRIME user."""
    user_id: str
    username: str
    role: str  # admin, researcher, viewer
    api_key_hash: str = ""
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    permissions: list[str] = field(default_factory=list)
    is_active: bool = True


ROLE_PERMISSIONS = {
    "admin": [
        "goal.submit", "goal.view", "goal.delete",
        "agent.list", "agent.execute", "agent.configure",
        "knowledge.read", "knowledge.write", "knowledge.delete",
        "memory.read", "memory.write", "memory.clear",
        "sica.view", "sica.configure",
        "system.status", "system.configure", "system.shutdown",
        "user.create", "user.delete", "user.modify",
        "api_key.generate", "api_key.revoke",
    ],
    "researcher": [
        "goal.submit", "goal.view",
        "agent.list", "agent.execute",
        "knowledge.read", "knowledge.write",
        "memory.read",
        "sica.view",
        "system.status",
    ],
    "viewer": [
        "goal.view",
        "agent.list",
        "knowledge.read",
        "system.status",
    ],
}


class AuthManager:
    """
    Authentication and authorization manager.
    """

    def __init__(self):
        self._users: dict[str, User] = {}
        self._api_keys: dict[str, str] = {}  # key_hash -> user_id
        self._sessions: dict[str, dict[str, Any]] = {}

        # Create default admin user
        self._create_default_admin()

    def _create_default_admin(self) -> None:
        """Create the default admin user."""
        admin = User(
            user_id="admin-001",
            username="jarvis-admin",
            role="admin",
            permissions=ROLE_PERMISSIONS["admin"],
        )
        self._users[admin.user_id] = admin

    def create_user(
        self,
        username: str,
        role: str = "researcher",
    ) -> dict[str, Any]:
        """Create a new user and generate an API key."""
        if role not in ROLE_PERMISSIONS:
            return {"error": f"Invalid role: {role}. Valid: {list(ROLE_PERMISSIONS.keys())}"}

        user_id = f"user-{secrets.token_hex(4)}"
        api_key = f"jrv_{secrets.token_hex(24)}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        user = User(
            user_id=user_id,
            username=username,
            role=role,
            api_key_hash=key_hash,
            permissions=ROLE_PERMISSIONS[role],
        )

        self._users[user_id] = user
        self._api_keys[key_hash] = user_id

        return {
            "user_id": user_id,
            "username": username,
            "role": role,
            "api_key": api_key,  # Only shown once!
            "permissions": user.permissions,
            "warning": "Store the API key securely. It will not be shown again.",
        }

    def authenticate(self, api_key: str) -> User | None:
        """Authenticate a user by API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        user_id = self._api_keys.get(key_hash)

        if user_id and user_id in self._users:
            user = self._users[user_id]
            if user.is_active:
                user.last_active = time.time()
                return user
        return None

    def authorize(self, user: User, permission: str) -> bool:
        """Check if a user has a specific permission."""
        return permission in user.permissions

    def check_permission(self, api_key: str, permission: str) -> dict[str, Any]:
        """Full auth flow: authenticate + authorize."""
        user = self.authenticate(api_key)
        if not user:
            return {"authorized": False, "reason": "Invalid API key"}

        if self.authorize(user, permission):
            return {
                "authorized": True,
                "user": user.username,
                "role": user.role,
                "permission": permission,
            }
        return {
            "authorized": False,
            "reason": f"User '{user.username}' (role: {user.role}) lacks permission: {permission}",
        }

    def revoke_api_key(self, user_id: str) -> dict[str, Any]:
        """Revoke a user's API key."""
        user = self._users.get(user_id)
        if not user:
            return {"error": "User not found"}

        # Remove old key
        self._api_keys = {
            k: v for k, v in self._api_keys.items() if v != user_id
        }

        # Generate new key
        new_key = f"jrv_{secrets.token_hex(24)}"
        new_hash = hashlib.sha256(new_key.encode()).hexdigest()
        user.api_key_hash = new_hash
        self._api_keys[new_hash] = user_id

        return {
            "user_id": user_id,
            "new_api_key": new_key,
            "warning": "Old key is now invalid.",
        }

    def list_users(self) -> list[dict[str, Any]]:
        """List all users (no API keys shown)."""
        return [
            {
                "user_id": u.user_id,
                "username": u.username,
                "role": u.role,
                "is_active": u.is_active,
                "permissions_count": len(u.permissions),
                "last_active": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(u.last_active)),
            }
            for u in self._users.values()
        ]

    def stats(self) -> dict[str, Any]:
        return {
            "total_users": len(self._users),
            "active_users": sum(1 for u in self._users.values() if u.is_active),
            "roles": {
                role: sum(1 for u in self._users.values() if u.role == role)
                for role in ROLE_PERMISSIONS
            },
            "api_keys_issued": len(self._api_keys),
        }
