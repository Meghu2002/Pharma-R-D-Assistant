from slowapi import Limiter
from slowapi.util import get_remote_address

# Shared Limiter instance — imported by both main.py (to register it on the
# app) and api/routes.py (to decorate individual endpoints). Kept in its own
# module to avoid a circular import between the two.
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
