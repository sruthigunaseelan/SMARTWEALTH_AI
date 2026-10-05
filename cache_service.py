import json
import logging
from src.backend.config import settings

logger = logging.getLogger("cache_service")

class CacheService:
    def __init__(self):
        self._memory_cache = {}
        self._redis_client = None
        self._connected = False
        
        # Try connecting to Redis
        try:
            import redis
            client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                socket_timeout=0.2,
                decode_responses=True
            )
            client.ping()
            self._redis_client = client
            self._connected = True
            logger.info("Connected to Redis cache.")
        except Exception:
            logger.warning("Redis server unavailable. Using fast in-memory fallback cache.")
            self._connected = False

    def get(self, key: str):
        if self._connected and self._redis_client:
            try:
                val = self._redis_client.get(key)
                if val:
                    return json.loads(val)
            except Exception:
                pass
        # Fallback to memory cache
        return self._memory_cache.get(key)

    def set(self, key: str, value: dict, ttl: int = 3600):
        if self._connected and self._redis_client:
            try:
                self._redis_client.setex(key, ttl, json.dumps(value))
                return
            except Exception:
                pass
        # Fallback to memory cache
        self._memory_cache[key] = value

cache_service = CacheService()
