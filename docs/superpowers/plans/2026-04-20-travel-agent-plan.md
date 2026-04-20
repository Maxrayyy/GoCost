# 旅游出行智能体 实施计划

> **给执行者（你自己或 AI）：** 每个任务是 2-5 分钟的最小动作，按顺序执行。步骤用 `- [ ]` 复选框跟踪进度。

**目标：** 构建 LangGraph + Streamlit 的旅游出行智能体，真实数据（Amadeus + 携程/飞猪爬虫），作者自用。

**架构：** LangGraph 混合式工作流（固定主干 parse → route → collect → recommend + ReAct 叶子节点），Streamlit 前端，SQLite 持久化，并行工具调用。

**技术栈：** Python 3.11+、uv、LangGraph、LangChain、DeepSeek（OpenAI SDK）、Streamlit、Selenium、Amadeus SDK、SQLAlchemy、loguru、pytest。

**前置阅读：** `docs/superpowers/specs/2026-04-20-travel-agent-design.md`

---

## 阶段一：基础设施（Task 1-5）

目标：项目跑起来、配置中心化、能发一次 DeepSeek 请求。

### Task 1: 初始化 uv 项目与 git

**Files:**
- Create: `pyproject.toml` (via `uv init`)
- Create: `.gitignore`
- Create: `.env.example`
- Modify: `main.py`（删除 PyCharm 模板）

- [ ] **Step 1: 安装 uv（如果还没装）**

运行：`powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`（Windows）
验证：`uv --version`，应看到版本号。

- [ ] **Step 2: 初始化 uv 项目**

在 `D:\code\agent` 下运行：

```bash
uv init --python 3.11 .
```

说明：这会创建 `pyproject.toml`、`.python-version`、`README.md`。如果已有 `main.py` 会保留。

- [ ] **Step 3: 删除 PyCharm 样板 main.py**

Windows: `del main.py`
确认 `main.py` 已删除。

- [ ] **Step 4: 添加项目依赖**

运行：

```bash
uv add langchain langgraph langchain-openai langchain-community streamlit selenium amadeus requests pydantic python-dotenv sqlalchemy loguru pyyaml
uv add --dev pytest pytest-asyncio pytest-mock
```

验证：`uv sync`，应成功安装。

- [ ] **Step 5: 初始化 git 仓库**

```bash
git init
git branch -M main
```

- [ ] **Step 6: 创建 .gitignore**

创建文件 `.gitignore` 内容：

```
# Env & secrets
.env
cookies/
data/
logs/

# Python
__pycache__/
*.py[cod]
.pytest_cache/
.venv/

# IDE
.idea/
.vscode/

# OS
.DS_Store
Thumbs.db

# Test artifacts
htmlcov/
.coverage
```

- [ ] **Step 7: 创建 .env.example**

创建文件 `.env.example` 内容：

```
# DeepSeek
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

# Amadeus Self-Service (https://developers.amadeus.com)
AMADEUS_CLIENT_ID=
AMADEUS_CLIENT_SECRET=

# 高德开放平台 (https://console.amap.com)
AMAP_KEY=

# 和风天气 (https://dev.qweather.com)
QWEATHER_KEY=

# 可选：LangSmith 调试
LANGSMITH_API_KEY=
LANGSMITH_TRACING=false
LANGSMITH_PROJECT=travel-agent

# 应用设置
CACHE_TTL_HOURS=24
SCRAPER_TIMEOUT_SECONDS=45
LOG_LEVEL=INFO
```

- [ ] **Step 8: 创建空目录占位**

```bash
mkdir -p src/graph/nodes src/tools src/scrapers src/models src/storage src/memory
mkdir -p cookies data logs tests/test_tools tests/test_graph scripts
```

创建每个 `src/*` 目录下的 `__init__.py`（空文件）。

- [ ] **Step 9: 首次 commit**

```bash
git add .
git commit -m "chore: initialize uv project with dependencies and structure"
```

---

### Task 2: 申请并配置 API keys（用户手动步骤）

**Files:**
- Modify: `.env`（本地，不入 git）

- [ ] **Step 1: 申请 DeepSeek API Key**

1. 访问 https://platform.deepseek.com/
2. 注册（手机号即可）
3. 充值 10 元（够玩一个月）
4. 创建 API Key，复制

- [ ] **Step 2: 申请 Amadeus**

1. 访问 https://developers.amadeus.com/register
2. 注册邮箱账号
3. 创建 Self-Service App
4. 拿到 `API Key` 和 `API Secret`（分别对应 CLIENT_ID 和 CLIENT_SECRET）

- [ ] **Step 3: 申请高德 Key**

1. 访问 https://console.amap.com/
2. 注册 → 应用管理 → 创建新应用 → 添加 Key
3. 类型选 "Web 服务"

- [ ] **Step 4: 申请和风天气 Key**

1. 访问 https://dev.qweather.com/
2. 注册 → 控制台 → 创建项目 → 创建 Key
3. 免费版够用

- [ ] **Step 5: 填入 .env**

```bash
copy .env.example .env
```

编辑 `.env`，把 4 个 key 填进去。

- [ ] **Step 6: 验证 .env 不在 git**

```bash
git status
```

期望：`.env` 不出现在 untracked 列表（因为 .gitignore）。

---

### Task 3: 配置中心（config.py）

**Files:**
- Create: `src/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: 写测试**

创建 `tests/test_config.py`：

```python
import os
from unittest.mock import patch


def test_config_loads_env_vars():
    with patch.dict(os.environ, {
        "DEEPSEEK_API_KEY": "sk-test",
        "DEEPSEEK_MODEL": "deepseek-chat",
        "CACHE_TTL_HOURS": "12",
    }):
        # Re-import to pick up patched env
        from importlib import reload
        from src import config
        reload(config)
        assert config.settings.deepseek_api_key == "sk-test"
        assert config.settings.deepseek_model == "deepseek-chat"
        assert config.settings.cache_ttl_hours == 12


def test_config_defaults():
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}, clear=False):
        from importlib import reload
        from src import config
        reload(config)
        assert config.settings.scraper_timeout_seconds == 45
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
uv run pytest tests/test_config.py -v
```

期望：`ImportError` 或 `AttributeError`（`config` 还没实现）。

- [ ] **Step 3: 写最小实现**

创建 `src/config.py`：

```python
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    amadeus_client_id: str
    amadeus_client_secret: str
    amap_key: str
    qweather_key: str
    langsmith_api_key: str
    langsmith_tracing: bool
    langsmith_project: str
    cache_ttl_hours: int
    scraper_timeout_seconds: int
    log_level: str
    project_root: Path
    data_dir: Path
    cookies_dir: Path
    logs_dir: Path


def _load() -> Settings:
    root = Path(__file__).resolve().parent.parent
    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        amadeus_client_id=os.getenv("AMADEUS_CLIENT_ID", ""),
        amadeus_client_secret=os.getenv("AMADEUS_CLIENT_SECRET", ""),
        amap_key=os.getenv("AMAP_KEY", ""),
        qweather_key=os.getenv("QWEATHER_KEY", ""),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY", ""),
        langsmith_tracing=os.getenv("LANGSMITH_TRACING", "false").lower() == "true",
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "travel-agent"),
        cache_ttl_hours=int(os.getenv("CACHE_TTL_HOURS", "24")),
        scraper_timeout_seconds=int(os.getenv("SCRAPER_TIMEOUT_SECONDS", "45")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        project_root=root,
        data_dir=root / "data",
        cookies_dir=root / "cookies",
        logs_dir=root / "logs",
    )


settings = _load()
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
uv run pytest tests/test_config.py -v
```

期望：2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add centralized config loader"
```

---

### Task 4: 日志配置

**Files:**
- Create: `src/logging_setup.py`

- [ ] **Step 1: 创建 src/logging_setup.py**

```python
from loguru import logger
import sys
from src.config import settings


def setup_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level, format=
        "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}")
    settings.logs_dir.mkdir(exist_ok=True)
    logger.add(settings.logs_dir / "app.log", rotation="10 MB", retention="7 days",
               level=settings.log_level)
    logger.add(settings.logs_dir / "scraper.log", rotation="10 MB", retention="7 days",
               filter=lambda r: "scraper" in r["name"].lower())
    logger.add(settings.logs_dir / "llm.log", rotation="10 MB", retention="7 days",
               filter=lambda r: "llm" in r["name"].lower() or "recommend" in r["name"].lower())
```

- [ ] **Step 2: 手动验证**

临时写个脚本 `scripts/test_logging.py`：

```python
from src.logging_setup import setup_logging
from loguru import logger

setup_logging()
logger.info("logging works")
logger.warning("this is a warning")
```

运行：`uv run python scripts/test_logging.py`
期望：看到彩色日志，`logs/app.log` 出现。

- [ ] **Step 3: Commit**

```bash
git add src/logging_setup.py scripts/test_logging.py
git commit -m "feat: add loguru-based logging with per-component files"
```

---

### Task 5: LLM 客户端封装（DeepSeek via OpenAI 兼容接口）

**Files:**
- Create: `src/llm.py`
- Create: `scripts/test_llm.py`

- [ ] **Step 1: 创建 src/llm.py**

```python
from langchain_openai import ChatOpenAI
from src.config import settings


def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature,
        timeout=60,
        max_retries=1,
    )
```

- [ ] **Step 2: 手动冒烟测试**

创建 `scripts/test_llm.py`：

```python
from src.llm import get_llm

llm = get_llm()
response = llm.invoke("用一句话解释什么是 LangGraph")
print(response.content)
```

运行：`uv run python scripts/test_llm.py`
期望：返回一段中文解释。如果 401 错误检查 `.env` 的 `DEEPSEEK_API_KEY`。

- [ ] **Step 3: Commit**

```bash
git add src/llm.py scripts/test_llm.py
git commit -m "feat: add DeepSeek LLM client wrapper"
```

---

## 阶段二：数据模型与存储（Task 6-9）

目标：Pydantic 统一模型 + SQLite 缓存/历史。

### Task 6: Pydantic 数据模型 — TripQuery

**Files:**
- Create: `src/models/query.py`
- Create: `tests/test_models_query.py`

- [ ] **Step 1: 写测试**

创建 `tests/test_models_query.py`：

```python
from datetime import date
import pytest
from pydantic import ValidationError
from src.models.query import TripQuery


def test_trip_query_minimal():
    q = TripQuery(origin="北京", destination="东京", depart_date=date(2026, 5, 1))
    assert q.origin == "北京"
    assert q.return_date is None
    assert q.adults == 1
    assert q.is_domestic() is False


def test_trip_query_domestic_detection():
    q = TripQuery(origin="北京", destination="上海", depart_date=date(2026, 5, 1))
    assert q.is_domestic() is True


def test_trip_query_rejects_empty_origin():
    with pytest.raises(ValidationError):
        TripQuery(origin="", destination="东京", depart_date=date(2026, 5, 1))


def test_trip_query_with_budget_and_preferences():
    q = TripQuery(
        origin="上海", destination="三亚", depart_date=date(2026, 5, 1),
        return_date=date(2026, 5, 5), adults=2, budget=3000,
        preferences=["海景", "泳池"]
    )
    assert q.budget == 3000
    assert "海景" in q.preferences
```

- [ ] **Step 2: 运行测试（确认失败）**

```bash
uv run pytest tests/test_models_query.py -v
```

- [ ] **Step 3: 实现模型**

创建 `src/models/query.py`：

```python
from datetime import date
from pydantic import BaseModel, Field, field_validator


DOMESTIC_CITIES = {
    "北京", "上海", "广州", "深圳", "成都", "杭州", "南京", "武汉", "西安",
    "重庆", "天津", "苏州", "青岛", "大连", "厦门", "三亚", "昆明", "长沙",
    "郑州", "济南", "合肥", "福州", "哈尔滨", "长春", "沈阳", "石家庄",
    "太原", "贵阳", "南宁", "兰州", "银川", "西宁", "乌鲁木齐", "呼和浩特",
    "拉萨", "海口", "南昌", "香港", "澳门", "台北",
}


class TripQuery(BaseModel):
    origin: str = Field(..., min_length=1)
    destination: str = Field(..., min_length=1)
    depart_date: date
    return_date: date | None = None
    adults: int = Field(default=1, ge=1, le=9)
    budget: int | None = Field(default=None, ge=0)
    preferences: list[str] = Field(default_factory=list)

    @field_validator("origin", "destination")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("city must not be empty")
        return v

    def is_domestic(self) -> bool:
        return self.origin in DOMESTIC_CITIES and self.destination in DOMESTIC_CITIES
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
uv run pytest tests/test_models_query.py -v
```

期望：4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/query.py tests/test_models_query.py
git commit -m "feat: add TripQuery pydantic model with domestic detection"
```

---

### Task 7: Pydantic 数据模型 — FlightOffer 与 HotelOffer

**Files:**
- Create: `src/models/flight.py`
- Create: `src/models/hotel.py`
- Create: `tests/test_models_offers.py`

- [ ] **Step 1: 写测试**

创建 `tests/test_models_offers.py`：

```python
from datetime import datetime
from src.models.flight import FlightOffer
from src.models.hotel import HotelOffer


def test_flight_offer_basic():
    f = FlightOffer(
        source="amadeus",
        airline="CA",
        flight_number="CA981",
        origin="PEK",
        destination="NRT",
        depart_time=datetime(2026, 5, 1, 10, 30),
        arrive_time=datetime(2026, 5, 1, 15, 10),
        price=3200,
        currency="CNY",
        deep_link=None,
    )
    assert f.duration_minutes() == 280
    assert f.source == "amadeus"


def test_hotel_offer_basic():
    h = HotelOffer(
        source="ctrip",
        name="东京希尔顿",
        city="Tokyo",
        rating=4.5,
        review_count=1203,
        price_per_night=1200,
        currency="CNY",
        tags=["市中心", "免费早餐"],
        deep_link="https://example.com/h/123",
    )
    assert h.rating == 4.5
    assert "市中心" in h.tags
```

- [ ] **Step 2: 运行测试（失败）**

```bash
uv run pytest tests/test_models_offers.py -v
```

- [ ] **Step 3: 实现 src/models/flight.py**

```python
from datetime import datetime
from pydantic import BaseModel, Field


class FlightOffer(BaseModel):
    source: str
    airline: str
    flight_number: str
    origin: str
    destination: str
    depart_time: datetime
    arrive_time: datetime
    price: float = Field(ge=0)
    currency: str = "CNY"
    stops: int = 0
    deep_link: str | None = None

    def duration_minutes(self) -> int:
        return int((self.arrive_time - self.depart_time).total_seconds() // 60)
```

- [ ] **Step 4: 实现 src/models/hotel.py**

```python
from pydantic import BaseModel, Field


class HotelOffer(BaseModel):
    source: str
    name: str
    city: str
    rating: float | None = Field(default=None, ge=0, le=5)
    review_count: int = Field(default=0, ge=0)
    price_per_night: float = Field(ge=0)
    currency: str = "CNY"
    tags: list[str] = Field(default_factory=list)
    address: str | None = None
    deep_link: str | None = None
```

- [ ] **Step 5: 运行测试，确认通过**

```bash
uv run pytest tests/test_models_offers.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/models/flight.py src/models/hotel.py tests/test_models_offers.py
git commit -m "feat: add FlightOffer and HotelOffer pydantic models"
```

---

### Task 8: SQLite 初始化与连接

**Files:**
- Create: `src/storage/db.py`
- Create: `tests/test_storage_db.py`

- [ ] **Step 1: 写测试**

创建 `tests/test_storage_db.py`：

```python
from pathlib import Path
from src.storage.db import init_db, get_session
from sqlalchemy import text


def test_init_db_creates_file(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.storage.db._DB_PATH", db_path)
    init_db()
    assert db_path.exists()


def test_session_can_execute(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.storage.db._DB_PATH", db_path)
    init_db()
    with get_session() as s:
        result = s.execute(text("SELECT 1")).scalar()
        assert result == 1
```

- [ ] **Step 2: 运行测试（失败）**

```bash
uv run pytest tests/test_storage_db.py -v
```

- [ ] **Step 3: 实现 src/storage/db.py**

```python
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

from src.config import settings


class Base(DeclarativeBase):
    pass


_DB_PATH: Path = settings.data_dir / "agent.db"
_engine = None
_SessionLocal = None


def _ensure_engine():
    global _engine, _SessionLocal
    if _engine is None:
        _DB_PATH.parent.mkdir(exist_ok=True)
        _engine = create_engine(f"sqlite:///{_DB_PATH}", echo=False, future=True)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def init_db() -> None:
    # Import models so SQLAlchemy sees them
    from src.storage import cache as _cache  # noqa: F401
    from src.storage import history as _history  # noqa: F401
    _ensure_engine()
    Base.metadata.create_all(_engine)


@contextmanager
def get_session() -> Iterator[Session]:
    _ensure_engine()
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

- [ ] **Step 4: 先创建占位的 cache.py 和 history.py 以便 init_db 不报错**

创建 `src/storage/cache.py`（占位，下一任务补完）：

```python
# Placeholder; see next task
```

创建 `src/storage/history.py`（占位）：

```python
# Placeholder; see next task
```

- [ ] **Step 5: 运行测试，确认通过**

```bash
uv run pytest tests/test_storage_db.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/storage/db.py src/storage/cache.py src/storage/history.py tests/test_storage_db.py
git commit -m "feat: add SQLAlchemy engine and session management"
```

---

### Task 9: 缓存层（24h TTL）

**Files:**
- Modify: `src/storage/cache.py`
- Create: `tests/test_storage_cache.py`

- [ ] **Step 1: 写测试**

创建 `tests/test_storage_cache.py`：

```python
import json
from datetime import datetime, timedelta
from pathlib import Path

from src.storage.db import init_db
from src.storage.cache import cache_get, cache_set


def test_cache_roundtrip(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.storage.db._DB_PATH", tmp_path / "t.db")
    init_db()

    cache_set("flight:PEK-NRT:2026-05-01", {"foo": "bar"})
    got = cache_get("flight:PEK-NRT:2026-05-01")
    assert got == {"foo": "bar"}


def test_cache_miss_returns_none(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.storage.db._DB_PATH", tmp_path / "t.db")
    init_db()
    assert cache_get("nonexistent") is None


def test_cache_expires(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.storage.db._DB_PATH", tmp_path / "t.db")
    init_db()
    cache_set("k", {"v": 1})

    # Manually backdate
    from src.storage.db import get_session
    from src.storage.cache import CacheRow
    with get_session() as s:
        row = s.query(CacheRow).filter_by(key="k").first()
        row.created_at = datetime.utcnow() - timedelta(hours=25)

    assert cache_get("k") is None
```

- [ ] **Step 2: 运行（失败）**

```bash
uv run pytest tests/test_storage_cache.py -v
```

- [ ] **Step 3: 实现 src/storage/cache.py**

```python
import json
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import String, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.config import settings
from src.storage.db import Base, get_session


class CacheRow(Base):
    __tablename__ = "cache"
    key: Mapped[str] = mapped_column(String(512), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def cache_get(key: str) -> Any | None:
    ttl = timedelta(hours=settings.cache_ttl_hours)
    with get_session() as s:
        row = s.query(CacheRow).filter_by(key=key).first()
        if row is None:
            return None
        if datetime.utcnow() - row.created_at > ttl:
            return None
        return json.loads(row.value)


def cache_set(key: str, value: Any) -> None:
    with get_session() as s:
        row = s.query(CacheRow).filter_by(key=key).first()
        payload = json.dumps(value, ensure_ascii=False, default=str)
        if row is None:
            s.add(CacheRow(key=key, value=payload))
        else:
            row.value = payload
            row.created_at = datetime.utcnow()
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
uv run pytest tests/test_storage_cache.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/storage/cache.py tests/test_storage_cache.py
git commit -m "feat: add 24h TTL cache layer for price lookups"
```

---

## 阶段三：工具层（Task 10-14）

目标：5 个工具都能独立调用。从简单到复杂：weather → amap → amadeus → ctrip → fliggy。

### Task 10: 和风天气工具（最简单，先练手）

**Files:**
- Create: `src/tools/weather_tool.py`
- Create: `tests/test_tools/test_weather.py`

- [ ] **Step 1: 查询和风天气 API 文档**

地理查询：`https://geoapi.qweather.com/v2/city/lookup?location={city}&key={key}`
3 天预报：`https://devapi.qweather.com/v7/weather/3d?location={id}&key={key}`

- [ ] **Step 2: 写测试（用 mock）**

创建 `tests/test_tools/__init__.py`（空文件）

创建 `tests/test_tools/test_weather.py`：

```python
from unittest.mock import patch, MagicMock
from src.tools.weather_tool import get_forecast


def test_forecast_returns_dict(monkeypatch):
    monkeypatch.setattr("src.config.settings.qweather_key", "test-key")

    geo_resp = MagicMock()
    geo_resp.json.return_value = {"code": "200", "location": [{"id": "101010100"}]}
    geo_resp.raise_for_status = MagicMock()

    weather_resp = MagicMock()
    weather_resp.json.return_value = {
        "code": "200",
        "daily": [
            {"fxDate": "2026-05-01", "textDay": "晴", "tempMax": "25", "tempMin": "15"}
        ],
    }
    weather_resp.raise_for_status = MagicMock()

    with patch("src.tools.weather_tool.requests.get",
               side_effect=[geo_resp, weather_resp]):
        result = get_forecast("北京")

    assert result["city"] == "北京"
    assert result["forecast"][0]["textDay"] == "晴"
```

- [ ] **Step 3: 运行（失败）**

```bash
uv run pytest tests/test_tools/test_weather.py -v
```

- [ ] **Step 4: 实现 src/tools/weather_tool.py**

```python
from typing import Any
import requests
from loguru import logger
from src.config import settings

_GEO_URL = "https://geoapi.qweather.com/v2/city/lookup"
_WEATHER_URL = "https://devapi.qweather.com/v7/weather/3d"


def get_forecast(city: str) -> dict[str, Any]:
    """Return 3-day forecast for a city. Raises on network/auth errors."""
    key = settings.qweather_key
    if not key:
        raise RuntimeError("QWEATHER_KEY not configured")

    geo = requests.get(_GEO_URL, params={"location": city, "key": key}, timeout=10)
    geo.raise_for_status()
    geo_data = geo.json()
    if geo_data.get("code") != "200" or not geo_data.get("location"):
        raise RuntimeError(f"city lookup failed: {city}")
    location_id = geo_data["location"][0]["id"]

    wx = requests.get(_WEATHER_URL, params={"location": location_id, "key": key}, timeout=10)
    wx.raise_for_status()
    wx_data = wx.json()
    if wx_data.get("code") != "200":
        raise RuntimeError(f"weather query failed: {wx_data.get('code')}")

    logger.info(f"weather.{city}.ok days={len(wx_data['daily'])}")
    return {"city": city, "forecast": wx_data["daily"]}
```

- [ ] **Step 5: 运行测试，确认通过**

```bash
uv run pytest tests/test_tools/test_weather.py -v
```

- [ ] **Step 6: 真实冒烟测试**

创建 `scripts/smoke_weather.py`：

```python
from src.tools.weather_tool import get_forecast
print(get_forecast("北京"))
```

`uv run python scripts/smoke_weather.py` — 期望返回北京 3 天预报。

- [ ] **Step 7: Commit**

```bash
git add src/tools/weather_tool.py tests/test_tools/test_weather.py tests/test_tools/__init__.py scripts/smoke_weather.py
git commit -m "feat: add qweather forecast tool"
```

---

### Task 11: 高德 POI 工具（景点评价）

**Files:**
- Create: `src/tools/amap_tool.py`
- Create: `tests/test_tools/test_amap.py`

- [ ] **Step 1: 查询高德 API**

搜索 POI：`https://restapi.amap.com/v3/place/text?key={key}&keywords=景点&city={city}&types=风景名胜`

- [ ] **Step 2: 写测试**

创建 `tests/test_tools/test_amap.py`：

```python
from unittest.mock import patch, MagicMock
from src.tools.amap_tool import get_pois


def test_get_pois_returns_list(monkeypatch):
    monkeypatch.setattr("src.config.settings.amap_key", "test-key")

    resp = MagicMock()
    resp.json.return_value = {
        "status": "1",
        "pois": [
            {"name": "故宫", "address": "北京市", "type": "风景名胜",
             "biz_ext": {"rating": "4.7"}},
            {"name": "长城", "address": "北京市", "type": "风景名胜",
             "biz_ext": {"rating": "4.8"}},
        ],
    }
    resp.raise_for_status = MagicMock()

    with patch("src.tools.amap_tool.requests.get", return_value=resp):
        pois = get_pois("北京", limit=5)

    assert len(pois) == 2
    assert pois[0]["name"] == "故宫"
    assert pois[0]["rating"] == 4.7
```

- [ ] **Step 3: 运行（失败）**

```bash
uv run pytest tests/test_tools/test_amap.py -v
```

- [ ] **Step 4: 实现 src/tools/amap_tool.py**

```python
from typing import Any
import requests
from loguru import logger
from src.config import settings

_URL = "https://restapi.amap.com/v3/place/text"


def get_pois(city: str, keywords: str = "景点", limit: int = 10) -> list[dict[str, Any]]:
    key = settings.amap_key
    if not key:
        raise RuntimeError("AMAP_KEY not configured")

    resp = requests.get(_URL, params={
        "key": key, "keywords": keywords, "city": city,
        "types": "风景名胜", "offset": limit, "extensions": "all",
    }, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "1":
        raise RuntimeError(f"amap error: {data.get('info')}")

    out = []
    for poi in data.get("pois", [])[:limit]:
        rating_raw = (poi.get("biz_ext") or {}).get("rating") or ""
        try:
            rating = float(rating_raw) if rating_raw else None
        except ValueError:
            rating = None
        out.append({
            "name": poi.get("name"),
            "address": poi.get("address"),
            "type": poi.get("type"),
            "rating": rating,
        })
    logger.info(f"amap.{city}.ok count={len(out)}")
    return out
```

- [ ] **Step 5: 测试通过**

```bash
uv run pytest tests/test_tools/test_amap.py -v
```

- [ ] **Step 6: 冒烟测试**

创建 `scripts/smoke_amap.py`：

```python
from src.tools.amap_tool import get_pois
for p in get_pois("北京"):
    print(p)
```

`uv run python scripts/smoke_amap.py` — 期望列出北京景点。

- [ ] **Step 7: Commit**

```bash
git add src/tools/amap_tool.py tests/test_tools/test_amap.py scripts/smoke_amap.py
git commit -m "feat: add amap POI tool"
```

---

### Task 12: Amadeus 工具（国际航班 + 酒店）

**Files:**
- Create: `src/tools/amadeus_tool.py`
- Create: `tests/test_tools/test_amadeus.py`
- Create: `scripts/smoke_amadeus.py`

- [ ] **Step 1: 读 Amadeus Python SDK 文档**

安装已完成（见 Task 1）。关键：
- `Client(client_id=..., client_secret=...)`
- `client.shopping.flight_offers_search.get(originLocationCode=..., destinationLocationCode=..., departureDate=..., adults=...)`
- `client.shopping.hotel_offers_search.get(cityCode=..., checkInDate=..., ...)` （注意 API v3 路径）
- Amadeus 用 **IATA 机场/城市代码**（北京=BJS，上海=SHA，东京=TYO）

我们做一个简单的**城市→IATA 映射**（常见 20 个城市），其他情况抛错。

- [ ] **Step 2: 写测试**

创建 `tests/test_tools/test_amadeus.py`：

```python
from unittest.mock import patch, MagicMock
from datetime import date
from src.models.query import TripQuery
from src.tools.amadeus_tool import city_to_iata, search_flights


def test_city_to_iata_known():
    assert city_to_iata("东京") == "TYO"
    assert city_to_iata("北京") == "BJS"


def test_city_to_iata_unknown():
    import pytest
    with pytest.raises(ValueError):
        city_to_iata("火星")


def test_search_flights_returns_offers(monkeypatch):
    monkeypatch.setattr("src.config.settings.amadeus_client_id", "id")
    monkeypatch.setattr("src.config.settings.amadeus_client_secret", "secret")

    fake_offer = {
        "itineraries": [{"segments": [{
            "carrierCode": "CA", "number": "981",
            "departure": {"iataCode": "PEK", "at": "2026-05-01T10:30:00"},
            "arrival": {"iataCode": "NRT", "at": "2026-05-01T15:10:00"},
        }]}],
        "price": {"total": "3200.00", "currency": "CNY"},
    }

    fake_client = MagicMock()
    fake_client.shopping.flight_offers_search.get.return_value = MagicMock(data=[fake_offer])

    with patch("src.tools.amadeus_tool._client", lambda: fake_client):
        q = TripQuery(origin="北京", destination="东京", depart_date=date(2026, 5, 1))
        offers = search_flights(q)

    assert len(offers) == 1
    assert offers[0].airline == "CA"
    assert offers[0].flight_number == "CA981"
    assert offers[0].price == 3200.0
```

- [ ] **Step 3: 运行（失败）**

```bash
uv run pytest tests/test_tools/test_amadeus.py -v
```

- [ ] **Step 4: 实现 src/tools/amadeus_tool.py**

```python
from datetime import datetime
from functools import lru_cache
from typing import Any

from amadeus import Client, ResponseError
from loguru import logger

from src.config import settings
from src.models.flight import FlightOffer
from src.models.hotel import HotelOffer
from src.models.query import TripQuery


_CITY_IATA = {
    "北京": "BJS", "上海": "SHA", "广州": "CAN", "深圳": "SZX",
    "成都": "CTU", "杭州": "HGH", "西安": "XIY", "重庆": "CKG",
    "香港": "HKG", "东京": "TYO", "大阪": "OSA", "首尔": "SEL",
    "曼谷": "BKK", "新加坡": "SIN", "吉隆坡": "KUL", "巴黎": "PAR",
    "伦敦": "LON", "纽约": "NYC", "洛杉矶": "LAX", "悉尼": "SYD",
    "罗马": "ROM", "迪拜": "DXB", "马尼拉": "MNL", "胡志明": "SGN",
}


def city_to_iata(city: str) -> str:
    if city not in _CITY_IATA:
        raise ValueError(f"unsupported city for amadeus: {city}")
    return _CITY_IATA[city]


@lru_cache(maxsize=1)
def _client() -> Client:
    if not settings.amadeus_client_id:
        raise RuntimeError("AMADEUS credentials missing")
    return Client(
        client_id=settings.amadeus_client_id,
        client_secret=settings.amadeus_client_secret,
    )


def search_flights(q: TripQuery, limit: int = 10) -> list[FlightOffer]:
    origin = city_to_iata(q.origin)
    dest = city_to_iata(q.destination)
    try:
        resp = _client().shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=dest,
            departureDate=q.depart_date.isoformat(),
            adults=q.adults,
            max=limit,
            currencyCode="CNY",
        )
    except ResponseError as e:
        logger.error(f"amadeus.flights.error {e}")
        raise

    offers: list[FlightOffer] = []
    for item in resp.data:
        first_seg = item["itineraries"][0]["segments"][0]
        last_seg = item["itineraries"][0]["segments"][-1]
        n_stops = len(item["itineraries"][0]["segments"]) - 1
        offers.append(FlightOffer(
            source="amadeus",
            airline=first_seg["carrierCode"],
            flight_number=first_seg["carrierCode"] + first_seg["number"],
            origin=first_seg["departure"]["iataCode"],
            destination=last_seg["arrival"]["iataCode"],
            depart_time=datetime.fromisoformat(first_seg["departure"]["at"]),
            arrive_time=datetime.fromisoformat(last_seg["arrival"]["at"]),
            price=float(item["price"]["total"]),
            currency=item["price"].get("currency", "CNY"),
            stops=n_stops,
        ))
    logger.info(f"amadeus.flights {origin}->{dest} count={len(offers)}")
    return offers


def search_hotels(q: TripQuery, limit: int = 10) -> list[HotelOffer]:
    city = city_to_iata(q.destination)
    checkout = q.return_date or q.depart_date
    try:
        # Amadeus Self-Service: list hotels by city first, then get offers
        list_resp = _client().reference_data.locations.hotels.by_city.get(cityCode=city)
        hotel_ids = [h["hotelId"] for h in (list_resp.data or [])[:20]]
        if not hotel_ids:
            return []
        offers_resp = _client().shopping.hotel_offers_search.get(
            hotelIds=",".join(hotel_ids),
            checkInDate=q.depart_date.isoformat(),
            checkOutDate=checkout.isoformat(),
            adults=q.adults,
            currency="CNY",
        )
    except ResponseError as e:
        logger.error(f"amadeus.hotels.error {e}")
        raise

    out: list[HotelOffer] = []
    for item in (offers_resp.data or [])[:limit]:
        hotel = item.get("hotel", {})
        price_obj = (item.get("offers") or [{}])[0].get("price", {})
        out.append(HotelOffer(
            source="amadeus",
            name=hotel.get("name", "Unknown"),
            city=q.destination,
            rating=float(hotel.get("rating")) if hotel.get("rating") else None,
            price_per_night=float(price_obj.get("total", 0)),
            currency=price_obj.get("currency", "CNY"),
            address=(hotel.get("address") or {}).get("lines", [""])[0],
        ))
    logger.info(f"amadeus.hotels {city} count={len(out)}")
    return out
```

- [ ] **Step 5: 测试通过**

```bash
uv run pytest tests/test_tools/test_amadeus.py -v
```

- [ ] **Step 6: 冒烟测试（真打 API）**

创建 `scripts/smoke_amadeus.py`：

```python
from datetime import date, timedelta
from src.models.query import TripQuery
from src.tools.amadeus_tool import search_flights

q = TripQuery(
    origin="北京", destination="东京",
    depart_date=date.today() + timedelta(days=30),
)
for f in search_flights(q, limit=3):
    print(f)
```

`uv run python scripts/smoke_amadeus.py`
期望：看到 3 条真实航班。如 `AuthenticationError` 检查 Amadeus 凭据。

- [ ] **Step 7: Commit**

```bash
git add src/tools/amadeus_tool.py tests/test_tools/test_amadeus.py scripts/smoke_amadeus.py
git commit -m "feat: add amadeus flights and hotels search tool"
```

---

### Task 13: 爬虫基础设施（BaseScraper + 选择器配置）

**Files:**
- Create: `src/scrapers/base.py`
- Create: `src/scrapers/selectors.yaml`
- Create: `tests/test_scrapers_base.py`

- [ ] **Step 1: 设计 BaseScraper 职责**

- 加载 cookie（`cookies/{platform}.json`，Cookie-Editor 导出格式）
- 启动 Selenium WebDriver（Chrome headless）
- 超时控制（`settings.scraper_timeout_seconds`）
- 加载 `selectors.yaml`
- 提供 `fetch(url)` 方法：打开页面 + 等待选择器出现 + 返回 BeautifulSoup
- 识别 cookie 失效（检测登录跳转 URL）→ 抛 `CookieExpiredError`

- [ ] **Step 2: 创建 selectors.yaml**

```yaml
ctrip:
  flight_search_url: "https://flights.ctrip.com/online/list/oneway-{origin_code}-{dest_code}?depdate={date}"
  flight_item: "div.flight-box"
  flight_price: "span.price"
  flight_airline: "span.airline-name"
  flight_number: "span.flight-num"
  flight_depart_time: "div.depart-time"
  flight_arrive_time: "div.arrive-time"
  flight_ready_signal: "div.flight-box"
  hotel_search_url: "https://hotels.ctrip.com/hotels/list?city={city_id}&checkin={checkin}&checkout={checkout}"
  hotel_item: "div.hotel-list-item"
  hotel_name: "h2.hotel-name"
  hotel_price: "span.price-num"
  hotel_rating: "span.rating-num"
  hotel_ready_signal: "div.hotel-list-item"

fliggy:
  hotel_search_url: "https://hotel.fliggy.com/hotel_list.htm?city={city}&checkIn={checkin}&checkOut={checkout}"
  hotel_item: "div.hotel-item"
  hotel_name: "a.name"
  hotel_price: "span.price"
  hotel_rating: "span.rate"
  hotel_ready_signal: "div.hotel-item"
```

> 注：这些选择器是初始占位。携程/飞猪的真实 DOM 结构需要你在 Task 14-15 手动登录确认后调整。配置化的意义就在这里——改 yaml 不改 Python。

- [ ] **Step 3: 写测试**

创建 `tests/test_scrapers_base.py`：

```python
import json
import pytest
from pathlib import Path
from src.scrapers.base import load_cookies, load_selectors, CookieMissingError


def test_load_cookies_reads_json(tmp_path: Path):
    f = tmp_path / "test.json"
    f.write_text(json.dumps([{"name": "sid", "value": "abc", "domain": ".ctrip.com"}]))
    cookies = load_cookies(f)
    assert cookies[0]["name"] == "sid"


def test_load_cookies_missing_raises(tmp_path: Path):
    with pytest.raises(CookieMissingError):
        load_cookies(tmp_path / "nope.json")


def test_load_selectors_for_platform():
    sels = load_selectors("ctrip")
    assert "flight_item" in sels
```

- [ ] **Step 4: 运行（失败）**

```bash
uv run pytest tests/test_scrapers_base.py -v
```

- [ ] **Step 5: 实现 src/scrapers/base.py**

```python
import json
from pathlib import Path
from typing import Any
import yaml
from loguru import logger
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from src.config import settings


class CookieMissingError(Exception):
    pass


class CookieExpiredError(Exception):
    pass


class SelectorChangedError(Exception):
    pass


_SELECTORS_PATH = Path(__file__).parent / "selectors.yaml"
_selectors_cache: dict[str, Any] | None = None


def load_selectors(platform: str) -> dict[str, Any]:
    global _selectors_cache
    if _selectors_cache is None:
        with open(_SELECTORS_PATH, "r", encoding="utf-8") as f:
            _selectors_cache = yaml.safe_load(f)
    if platform not in _selectors_cache:
        raise KeyError(f"no selectors for platform: {platform}")
    return _selectors_cache[platform]


def load_cookies(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise CookieMissingError(
            f"cookie file not found: {path}. "
            f"use the Cookie-Editor browser extension to export after manual login.")
    return json.loads(path.read_text(encoding="utf-8"))


class BaseScraper:
    platform: str = ""
    cookie_filename: str = ""
    login_url_fragment: str = "login"

    def __init__(self):
        self.selectors = load_selectors(self.platform)
        self.cookies = load_cookies(settings.cookies_dir / self.cookie_filename)
        self.driver: webdriver.Chrome | None = None

    def _start(self):
        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        self.driver = webdriver.Chrome(options=opts)

    def _inject_cookies(self, domain_url: str):
        self.driver.get(domain_url)
        for c in self.cookies:
            try:
                self.driver.add_cookie({
                    "name": c["name"], "value": c["value"],
                    "domain": c.get("domain"),
                    "path": c.get("path", "/"),
                })
            except Exception as e:
                logger.warning(f"cookie inject skip {c.get('name')}: {e}")

    def fetch(self, url: str, wait_selector: str) -> str:
        """Return page HTML after wait_selector appears. Raises on timeout or cookie expiry."""
        if self.driver is None:
            self._start()
            # inject after navigating to domain once
            self._inject_cookies(self._domain_root_url())
        self.driver.get(url)
        try:
            WebDriverWait(self.driver, settings.scraper_timeout_seconds).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector))
            )
        except TimeoutException:
            current = self.driver.current_url
            if self.login_url_fragment in current:
                raise CookieExpiredError(f"{self.platform} redirected to login: {current}")
            raise SelectorChangedError(
                f"{self.platform} selector '{wait_selector}' not found on {url}")
        return self.driver.page_source

    def _domain_root_url(self) -> str:
        raise NotImplementedError

    def close(self):
        if self.driver:
            self.driver.quit()
            self.driver = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
```

- [ ] **Step 6: 测试通过**

```bash
uv run pytest tests/test_scrapers_base.py -v
```

- [ ] **Step 7: Commit**

```bash
git add src/scrapers/base.py src/scrapers/selectors.yaml tests/test_scrapers_base.py
git commit -m "feat: add scraper base class with cookie injection and timeout"
```

---

### Task 14: 携程爬虫（航班 + 酒店）

**Files:**
- Create: `src/scrapers/ctrip.py`
- Create: `src/tools/ctrip_scraper.py`
- Create: `scripts/smoke_ctrip.py`
- Create: `docs/cookie-export-guide.md`

- [ ] **Step 1: 写 cookie 导出指南**

创建 `docs/cookie-export-guide.md`：

```markdown
# 如何为爬虫导出 Cookie

## 安装 Cookie-Editor 浏览器插件

Chrome 商店搜索 "Cookie-Editor" 安装。

## 导出携程 cookie

1. Chrome 打开 `https://www.ctrip.com/` 并登录
2. 点击地址栏 Cookie-Editor 图标
3. 点 "Export" → "Export as JSON"
4. 粘贴到 `cookies/ctrip.json`

## 导出飞猪 cookie

同上，站点是 `https://www.fliggy.com/`，文件为 `cookies/fliggy.json`。

## Cookie 失效时怎么办

当 UI 弹出 "cookie 失效" 警告，重复上述步骤覆盖文件即可。通常 1-3 天失效一次。
```

- [ ] **Step 2: 实现 src/scrapers/ctrip.py**

```python
from datetime import datetime
from typing import Any
from bs4 import BeautifulSoup
from loguru import logger

from src.scrapers.base import BaseScraper


_CTRIP_CITY_CODES = {
    "北京": "BJS", "上海": "SHA", "广州": "CAN", "深圳": "SZX",
    "成都": "CTU", "杭州": "HGH", "西安": "SIA", "重庆": "CKG",
    "三亚": "SYX", "昆明": "KMG", "厦门": "XMN", "青岛": "TAO",
    "大连": "DLC", "天津": "TSN", "武汉": "WUH", "长沙": "CSX",
    "郑州": "CGO", "南京": "NKG", "海口": "HAK",
}


class CtripScraper(BaseScraper):
    platform = "ctrip"
    cookie_filename = "ctrip.json"
    login_url_fragment = "passport.ctrip.com"

    def _domain_root_url(self) -> str:
        return "https://www.ctrip.com/"

    def search_flights(self, origin: str, destination: str, depart_date: str) -> list[dict[str, Any]]:
        o = _CTRIP_CITY_CODES.get(origin)
        d = _CTRIP_CITY_CODES.get(destination)
        if not o or not d:
            raise ValueError(f"city not supported by ctrip scraper: {origin}/{destination}")

        url = self.selectors["flight_search_url"].format(
            origin_code=o.lower(), dest_code=d.lower(), date=depart_date)
        html = self.fetch(url, self.selectors["flight_ready_signal"])
        soup = BeautifulSoup(html, "html.parser")

        items = soup.select(self.selectors["flight_item"])
        results = []
        for it in items[:15]:
            price_el = it.select_one(self.selectors["flight_price"])
            airline_el = it.select_one(self.selectors["flight_airline"])
            num_el = it.select_one(self.selectors["flight_number"])
            dep_el = it.select_one(self.selectors["flight_depart_time"])
            arr_el = it.select_one(self.selectors["flight_arrive_time"])
            if not (price_el and airline_el and num_el):
                continue
            try:
                price = float(price_el.get_text(strip=True).replace("¥", "").replace(",", ""))
            except ValueError:
                continue
            results.append({
                "airline": airline_el.get_text(strip=True),
                "flight_number": num_el.get_text(strip=True),
                "depart_time": dep_el.get_text(strip=True) if dep_el else "",
                "arrive_time": arr_el.get_text(strip=True) if arr_el else "",
                "price": price,
                "origin": o, "destination": d,
                "depart_date": depart_date,
            })
        logger.info(f"ctrip.flights {o}->{d} count={len(results)}")
        return results

    def search_hotels(self, city: str, checkin: str, checkout: str) -> list[dict[str, Any]]:
        city_id = _CTRIP_CITY_CODES.get(city, "").lower()
        if not city_id:
            raise ValueError(f"city not supported by ctrip: {city}")
        url = self.selectors["hotel_search_url"].format(
            city_id=city_id, checkin=checkin, checkout=checkout)
        html = self.fetch(url, self.selectors["hotel_ready_signal"])
        soup = BeautifulSoup(html, "html.parser")

        results = []
        for it in soup.select(self.selectors["hotel_item"])[:15]:
            name_el = it.select_one(self.selectors["hotel_name"])
            price_el = it.select_one(self.selectors["hotel_price"])
            rating_el = it.select_one(self.selectors["hotel_rating"])
            if not (name_el and price_el):
                continue
            try:
                price = float(price_el.get_text(strip=True).replace("¥", "").replace(",", ""))
            except ValueError:
                continue
            try:
                rating = float(rating_el.get_text(strip=True)) if rating_el else None
            except ValueError:
                rating = None
            results.append({
                "name": name_el.get_text(strip=True),
                "price": price, "rating": rating,
                "city": city,
            })
        logger.info(f"ctrip.hotels {city} count={len(results)}")
        return results
```

- [ ] **Step 3: 实现 LangChain Tool 封装 src/tools/ctrip_scraper.py**

```python
from datetime import datetime
from loguru import logger

from src.models.flight import FlightOffer
from src.models.hotel import HotelOffer
from src.models.query import TripQuery
from src.scrapers.ctrip import CtripScraper
from src.storage.cache import cache_get, cache_set


def search_flights(q: TripQuery) -> list[FlightOffer]:
    key = f"ctrip:flight:{q.origin}-{q.destination}:{q.depart_date}"
    cached = cache_get(key)
    if cached is not None:
        logger.info(f"cache.hit {key}")
        return [FlightOffer(**r) for r in cached]

    with CtripScraper() as s:
        raw = s.search_flights(q.origin, q.destination, q.depart_date.isoformat())

    offers = []
    for r in raw:
        try:
            dep = _parse_hhmm(r["depart_time"], q.depart_date)
            arr = _parse_hhmm(r["arrive_time"], q.depart_date)
        except ValueError:
            continue
        offers.append(FlightOffer(
            source="ctrip",
            airline=r["airline"],
            flight_number=r["flight_number"],
            origin=r["origin"], destination=r["destination"],
            depart_time=dep, arrive_time=arr,
            price=r["price"], currency="CNY",
        ))

    cache_set(key, [o.model_dump(mode="json") for o in offers])
    return offers


def search_hotels(q: TripQuery) -> list[HotelOffer]:
    checkout = q.return_date or q.depart_date
    key = f"ctrip:hotel:{q.destination}:{q.depart_date}:{checkout}"
    cached = cache_get(key)
    if cached is not None:
        logger.info(f"cache.hit {key}")
        return [HotelOffer(**r) for r in cached]

    with CtripScraper() as s:
        raw = s.search_hotels(q.destination, q.depart_date.isoformat(), checkout.isoformat())

    offers = [HotelOffer(
        source="ctrip", name=r["name"], city=r["city"],
        rating=r["rating"], price_per_night=r["price"], currency="CNY",
    ) for r in raw]

    cache_set(key, [o.model_dump(mode="json") for o in offers])
    return offers


def _parse_hhmm(text: str, base_date) -> datetime:
    hh, mm = text.strip().split(":")
    return datetime.combine(base_date, datetime.min.time()).replace(
        hour=int(hh), minute=int(mm))
```

- [ ] **Step 4: 冒烟测试（真爬，需要先导出 cookie）**

创建 `scripts/smoke_ctrip.py`：

```python
from datetime import date, timedelta
from src.models.query import TripQuery
from src.tools.ctrip_scraper import search_flights

q = TripQuery(origin="北京", destination="上海",
              depart_date=date.today() + timedelta(days=14))
for f in search_flights(q):
    print(f)
```

先按 `docs/cookie-export-guide.md` 导出 `cookies/ctrip.json`。
运行：`uv run python scripts/smoke_ctrip.py`

**重要：** 如果返回空，查看 `logs/scraper.log` 看是 cookie 失效、选择器错还是别的问题。选择器错就改 `src/scrapers/selectors.yaml` 重跑。

- [ ] **Step 5: Commit**

```bash
git add src/scrapers/ctrip.py src/tools/ctrip_scraper.py scripts/smoke_ctrip.py docs/cookie-export-guide.md
git commit -m "feat: add ctrip scraper with flight and hotel search + caching"
```

---

### Task 15: 飞猪爬虫（酒店为主）

**Files:**
- Create: `src/scrapers/fliggy.py`
- Create: `src/tools/fliggy_scraper.py`
- Create: `scripts/smoke_fliggy.py`

- [ ] **Step 1: 实现 src/scrapers/fliggy.py**

```python
from typing import Any
from bs4 import BeautifulSoup
from loguru import logger

from src.scrapers.base import BaseScraper


class FliggyScraper(BaseScraper):
    platform = "fliggy"
    cookie_filename = "fliggy.json"
    login_url_fragment = "login.taobao.com"

    def _domain_root_url(self) -> str:
        return "https://www.fliggy.com/"

    def search_hotels(self, city: str, checkin: str, checkout: str) -> list[dict[str, Any]]:
        url = self.selectors["hotel_search_url"].format(
            city=city, checkin=checkin, checkout=checkout)
        html = self.fetch(url, self.selectors["hotel_ready_signal"])
        soup = BeautifulSoup(html, "html.parser")

        results = []
        for it in soup.select(self.selectors["hotel_item"])[:15]:
            name_el = it.select_one(self.selectors["hotel_name"])
            price_el = it.select_one(self.selectors["hotel_price"])
            rating_el = it.select_one(self.selectors["hotel_rating"])
            if not (name_el and price_el):
                continue
            try:
                price = float(price_el.get_text(strip=True).replace("¥", "").replace(",", ""))
            except ValueError:
                continue
            try:
                rating = float(rating_el.get_text(strip=True)) if rating_el else None
            except ValueError:
                rating = None
            results.append({
                "name": name_el.get_text(strip=True),
                "price": price, "rating": rating, "city": city,
            })
        logger.info(f"fliggy.hotels {city} count={len(results)}")
        return results
```

- [ ] **Step 2: 实现 src/tools/fliggy_scraper.py**

```python
from loguru import logger
from src.models.hotel import HotelOffer
from src.models.query import TripQuery
from src.scrapers.fliggy import FliggyScraper
from src.storage.cache import cache_get, cache_set


def search_hotels(q: TripQuery) -> list[HotelOffer]:
    checkout = q.return_date or q.depart_date
    key = f"fliggy:hotel:{q.destination}:{q.depart_date}:{checkout}"
    cached = cache_get(key)
    if cached is not None:
        logger.info(f"cache.hit {key}")
        return [HotelOffer(**r) for r in cached]

    with FliggyScraper() as s:
        raw = s.search_hotels(q.destination, q.depart_date.isoformat(), checkout.isoformat())

    offers = [HotelOffer(
        source="fliggy", name=r["name"], city=r["city"],
        rating=r["rating"], price_per_night=r["price"], currency="CNY",
    ) for r in raw]

    cache_set(key, [o.model_dump(mode="json") for o in offers])
    return offers
```

- [ ] **Step 3: 冒烟测试**

创建 `scripts/smoke_fliggy.py`：

```python
from datetime import date, timedelta
from src.models.query import TripQuery
from src.tools.fliggy_scraper import search_hotels

q = TripQuery(origin="北京", destination="成都",
              depart_date=date.today() + timedelta(days=14),
              return_date=date.today() + timedelta(days=17))
for h in search_hotels(q):
    print(h)
```

按指南导出 `cookies/fliggy.json`，然后：`uv run python scripts/smoke_fliggy.py`

- [ ] **Step 4: Commit**

```bash
git add src/scrapers/fliggy.py src/tools/fliggy_scraper.py scripts/smoke_fliggy.py
git commit -m "feat: add fliggy hotel scraper with caching"
```

---

## 阶段四：LangGraph 编排（Task 16-21）

目标：把工具串起来，形成完整 Agent 流程。

### Task 16: AgentState 定义

**Files:**
- Create: `src/graph/state.py`

- [ ] **Step 1: 创建 src/graph/state.py**

```python
from typing import TypedDict
from src.models.flight import FlightOffer
from src.models.hotel import HotelOffer
from src.models.query import TripQuery


class AgentState(TypedDict, total=False):
    # Input
    user_input: str
    form_data: dict
    conversation: list[dict]

    # Parsed
    query: TripQuery | None
    missing_fields: list[str]

    # Routing
    is_domestic: bool | None

    # Collected
    flights: list[FlightOffer]
    hotels: list[HotelOffer]
    pois: list[dict]
    weather: dict | None
    tool_errors: dict[str, str]

    # Output
    recommendation: str | None
    cards: list[dict]
```

- [ ] **Step 2: Commit**

```bash
git add src/graph/state.py
git commit -m "feat: define AgentState typed dict"
```

---

### Task 17: parse_intent 节点（LLM 结构化输出）

**Files:**
- Create: `src/graph/nodes/parse_intent.py`
- Create: `tests/test_graph/__init__.py`
- Create: `tests/test_graph/test_parse_intent.py`

- [ ] **Step 1: 写测试（mock LLM）**

创建 `tests/test_graph/__init__.py`（空）

创建 `tests/test_graph/test_parse_intent.py`：

```python
from datetime import date
from unittest.mock import patch, MagicMock
from src.graph.nodes.parse_intent import parse_intent_node
from src.models.query import TripQuery


def test_parse_with_complete_input(monkeypatch):
    fake_query = TripQuery(
        origin="北京", destination="东京",
        depart_date=date(2026, 5, 1), adults=1,
    )
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_llm
    fake_llm.invoke.return_value = fake_query
    monkeypatch.setattr("src.graph.nodes.parse_intent.get_llm", lambda **_: fake_llm)

    state = {"user_input": "下周五北京去东京",
             "form_data": {}, "conversation": []}
    result = parse_intent_node(state)
    assert result["query"].origin == "北京"
    assert result["missing_fields"] == []


def test_parse_detects_missing_fields(monkeypatch):
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_llm
    fake_llm.invoke.side_effect = Exception("can't parse")
    monkeypatch.setattr("src.graph.nodes.parse_intent.get_llm", lambda **_: fake_llm)

    state = {"user_input": "想去玩",
             "form_data": {}, "conversation": []}
    result = parse_intent_node(state)
    assert result["query"] is None
    assert "origin" in result["missing_fields"]
```

- [ ] **Step 2: 运行（失败）**

```bash
uv run pytest tests/test_graph/test_parse_intent.py -v
```

- [ ] **Step 3: 实现 src/graph/nodes/parse_intent.py**

```python
from datetime import date
from loguru import logger
from src.graph.state import AgentState
from src.llm import get_llm
from src.models.query import TripQuery


_SYSTEM = """你是旅游需求解析器。从用户输入中提取出行参数。
- 日期请解析为 YYYY-MM-DD 格式（相对日期如"下周五"请根据今天是 {today} 推断）
- origin/destination 用城市中文名（如"北京"）
- 如果某字段用户没提供，保持默认
- adults 默认 1，budget 默认 None
只输出结构化结果，不要解释。"""


def parse_intent_node(state: AgentState) -> AgentState:
    text = state.get("user_input", "")
    form = state.get("form_data") or {}
    if form:
        text = f"{text}\n表单：{form}"

    llm = get_llm(temperature=0.0)
    structured = llm.with_structured_output(TripQuery)
    try:
        query = structured.invoke([
            ("system", _SYSTEM.format(today=date.today().isoformat())),
            ("user", text),
        ])
        missing = []
    except Exception as e:
        logger.warning(f"parse_intent.fail {e}")
        query = None
        missing = ["origin", "destination", "depart_date"]

    return {"query": query, "missing_fields": missing}
```

- [ ] **Step 4: 测试通过**

```bash
uv run pytest tests/test_graph/test_parse_intent.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/graph/nodes/parse_intent.py tests/test_graph/test_parse_intent.py tests/test_graph/__init__.py
git commit -m "feat: add parse_intent node with LLM structured output"
```

---

### Task 18: route 与 clarify 节点

**Files:**
- Create: `src/graph/nodes/route.py`
- Create: `src/graph/nodes/clarify.py`

- [ ] **Step 1: 实现 src/graph/nodes/route.py**

```python
from src.graph.state import AgentState


def route_node(state: AgentState) -> AgentState:
    q = state.get("query")
    if q is None:
        return {"is_domestic": None}
    return {"is_domestic": q.is_domestic()}
```

- [ ] **Step 2: 实现 src/graph/nodes/clarify.py**

```python
from src.graph.state import AgentState


_FIELD_PROMPTS = {
    "origin": "请告诉我你从哪出发？（如：北京）",
    "destination": "你要去哪？（如：东京）",
    "depart_date": "什么时候出发？（如：5 月 1 日）",
}


def clarify_node(state: AgentState) -> AgentState:
    missing = state.get("missing_fields", [])
    questions = [_FIELD_PROMPTS.get(f, f"缺少 {f}") for f in missing]
    msg = "为了更准确地帮你搜索，请补充：\n" + "\n".join(f"- {q}" for q in questions)
    return {"recommendation": msg, "cards": []}
```

- [ ] **Step 3: Commit**

```bash
git add src/graph/nodes/route.py src/graph/nodes/clarify.py
git commit -m "feat: add route and clarify nodes"
```

---

### Task 19: collect_data 节点（并行工具调用）

**Files:**
- Create: `src/graph/nodes/collect_data.py`
- Create: `tests/test_graph/test_collect_data.py`

- [ ] **Step 1: 写测试**

创建 `tests/test_graph/test_collect_data.py`：

```python
from datetime import date
from unittest.mock import patch, MagicMock
from src.graph.nodes.collect_data import collect_data_node
from src.models.query import TripQuery
from src.models.flight import FlightOffer
from datetime import datetime


def _fake_flight():
    return FlightOffer(
        source="amadeus", airline="CA", flight_number="CA981",
        origin="PEK", destination="NRT",
        depart_time=datetime(2026, 5, 1, 10, 0),
        arrive_time=datetime(2026, 5, 1, 15, 0),
        price=3000,
    )


def test_collect_international_uses_amadeus(monkeypatch):
    monkeypatch.setattr(
        "src.graph.nodes.collect_data.amadeus_flights",
        lambda q: [_fake_flight()])
    monkeypatch.setattr(
        "src.graph.nodes.collect_data.amadeus_hotels", lambda q: [])
    monkeypatch.setattr(
        "src.graph.nodes.collect_data.amap_pois", lambda c: [])
    monkeypatch.setattr(
        "src.graph.nodes.collect_data.weather_forecast", lambda c: {"city": c, "forecast": []})

    q = TripQuery(origin="北京", destination="东京", depart_date=date(2026, 5, 1))
    result = collect_data_node({"query": q, "is_domestic": False})
    assert len(result["flights"]) == 1
    assert result["tool_errors"] == {}


def test_collect_tool_failure_recorded(monkeypatch):
    def boom(q):
        raise RuntimeError("amadeus down")
    monkeypatch.setattr("src.graph.nodes.collect_data.amadeus_flights", boom)
    monkeypatch.setattr("src.graph.nodes.collect_data.amadeus_hotels", lambda q: [])
    monkeypatch.setattr("src.graph.nodes.collect_data.amap_pois", lambda c: [])
    monkeypatch.setattr("src.graph.nodes.collect_data.weather_forecast",
                        lambda c: {"city": c, "forecast": []})

    q = TripQuery(origin="北京", destination="东京", depart_date=date(2026, 5, 1))
    result = collect_data_node({"query": q, "is_domestic": False})
    assert "amadeus_flights" in result["tool_errors"]
    assert result["flights"] == []
```

- [ ] **Step 2: 运行（失败）**

```bash
uv run pytest tests/test_graph/test_collect_data.py -v
```

- [ ] **Step 3: 实现 src/graph/nodes/collect_data.py**

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from src.graph.state import AgentState
from src.tools.amadeus_tool import search_flights as amadeus_flights
from src.tools.amadeus_tool import search_hotels as amadeus_hotels
from src.tools.ctrip_scraper import search_flights as ctrip_flights
from src.tools.ctrip_scraper import search_hotels as ctrip_hotels
from src.tools.fliggy_scraper import search_hotels as fliggy_hotels
from src.tools.amap_tool import get_pois as amap_pois
from src.tools.weather_tool import get_forecast as weather_forecast


def collect_data_node(state: AgentState) -> AgentState:
    q = state["query"]
    is_domestic = state["is_domestic"]

    jobs: dict[str, callable] = {}
    if is_domestic:
        jobs["ctrip_flights"] = lambda: ctrip_flights(q)
        jobs["ctrip_hotels"] = lambda: ctrip_hotels(q)
        jobs["fliggy_hotels"] = lambda: fliggy_hotels(q)
    else:
        jobs["amadeus_flights"] = lambda: amadeus_flights(q)
        jobs["amadeus_hotels"] = lambda: amadeus_hotels(q)
    jobs["amap_pois"] = lambda: amap_pois(q.destination)
    jobs["weather"] = lambda: weather_forecast(q.destination)

    flights, hotels, pois, weather = [], [], [], None
    errors: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(fn): name for name, fn in jobs.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                result = fut.result(timeout=60)
            except Exception as e:
                logger.error(f"collect.{name}.fail {e}")
                errors[name] = str(e)
                continue

            if name.endswith("flights"):
                flights.extend(result)
            elif name.endswith("hotels"):
                hotels.extend(result)
            elif name == "amap_pois":
                pois = result
            elif name == "weather":
                weather = result

    return {
        "flights": flights, "hotels": hotels,
        "pois": pois, "weather": weather,
        "tool_errors": errors,
    }
```

- [ ] **Step 4: 测试通过**

```bash
uv run pytest tests/test_graph/test_collect_data.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/graph/nodes/collect_data.py tests/test_graph/test_collect_data.py
git commit -m "feat: add collect_data node with parallel tool execution"
```

---

### Task 20: recommend 节点（LLM 综合推荐）

**Files:**
- Create: `src/graph/nodes/recommend.py`

- [ ] **Step 1: 实现 src/graph/nodes/recommend.py**

```python
import json
from loguru import logger
from src.graph.state import AgentState
from src.llm import get_llm


_SYSTEM = """你是旅游推荐助手。基于下面的真实数据，为用户写一段简洁、有洞察力的推荐（300-500 字）。

要求：
- 明确指出推荐的 Top 2 航班 + Top 2 酒店，给理由
- 如果某数据源失败（见 tool_errors），说明"XX 暂不可用，以下仅基于 YY"
- 提到天气是否影响出行
- 提 1-2 个景点作为行程亮点
- 不要编造数据中没有的信息
- 最后给用户下一步建议（如"可以直接在携程搜索 CA981 下单"）
"""


def _cards_from_state(state: AgentState) -> list[dict]:
    cards = []
    for f in (state.get("flights") or [])[:5]:
        cards.append({
            "type": "flight",
            "source": f.source, "title": f"{f.airline} {f.flight_number}",
            "subtitle": f"{f.origin} → {f.destination}",
            "price": f.price, "currency": f.currency,
            "detail": f"{f.depart_time:%m-%d %H:%M} → {f.arrive_time:%m-%d %H:%M} ({f.duration_minutes()} 分钟)",
        })
    for h in (state.get("hotels") or [])[:5]:
        cards.append({
            "type": "hotel",
            "source": h.source, "title": h.name,
            "subtitle": f"评分 {h.rating or '无'}",
            "price": h.price_per_night, "currency": h.currency,
            "detail": ", ".join(h.tags) if h.tags else (h.address or ""),
        })
    return cards


def recommend_node(state: AgentState) -> AgentState:
    payload = {
        "query": state["query"].model_dump(mode="json"),
        "flights": [f.model_dump(mode="json") for f in (state.get("flights") or [])[:10]],
        "hotels": [h.model_dump(mode="json") for h in (state.get("hotels") or [])[:10]],
        "pois": (state.get("pois") or [])[:5],
        "weather": state.get("weather"),
        "tool_errors": state.get("tool_errors") or {},
    }

    llm = get_llm(temperature=0.4)
    try:
        resp = llm.invoke([
            ("system", _SYSTEM),
            ("user", json.dumps(payload, ensure_ascii=False, default=str)),
        ])
        text = resp.content
    except Exception as e:
        logger.error(f"recommend.llm.fail {e}")
        text = "AI 推荐暂不可用。以下是原始数据卡片供参考。"

    return {"recommendation": text, "cards": _cards_from_state(state)}
```

- [ ] **Step 2: Commit**

```bash
git add src/graph/nodes/recommend.py
git commit -m "feat: add recommend node with LLM synthesis and card builder"
```

---

### Task 21: Graph 拼装

**Files:**
- Create: `src/graph/builder.py`
- Create: `tests/test_graph/test_full_flow.py`

- [ ] **Step 1: 实现 src/graph/builder.py**

```python
from langgraph.graph import StateGraph, END

from src.graph.state import AgentState
from src.graph.nodes.parse_intent import parse_intent_node
from src.graph.nodes.route import route_node
from src.graph.nodes.clarify import clarify_node
from src.graph.nodes.collect_data import collect_data_node
from src.graph.nodes.recommend import recommend_node


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("parse", parse_intent_node)
    g.add_node("clarify", clarify_node)
    g.add_node("route", route_node)
    g.add_node("collect", collect_data_node)
    g.add_node("recommend", recommend_node)

    g.set_entry_point("parse")
    g.add_conditional_edges(
        "parse",
        lambda s: "clarify" if s.get("missing_fields") else "route",
        {"clarify": "clarify", "route": "route"},
    )
    g.add_edge("clarify", END)
    g.add_edge("route", "collect")
    g.add_edge("collect", "recommend")
    g.add_edge("recommend", END)
    return g.compile()
```

- [ ] **Step 2: 写集成测试（全流程，mock 所有外部调用）**

创建 `tests/test_graph/test_full_flow.py`：

```python
from datetime import date, datetime
from unittest.mock import patch, MagicMock
from src.graph.builder import build_graph
from src.models.query import TripQuery
from src.models.flight import FlightOffer


def test_full_flow_international(monkeypatch):
    fake_query = TripQuery(origin="北京", destination="东京",
                           depart_date=date(2026, 5, 1))
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_llm
    fake_llm.invoke.return_value = fake_query
    monkeypatch.setattr("src.graph.nodes.parse_intent.get_llm", lambda **_: fake_llm)

    # Recommend LLM
    rec_llm = MagicMock()
    rec_resp = MagicMock(content="推荐：CA981...")
    rec_llm.invoke.return_value = rec_resp
    monkeypatch.setattr("src.graph.nodes.recommend.get_llm", lambda **_: rec_llm)

    # Tools
    monkeypatch.setattr("src.graph.nodes.collect_data.amadeus_flights",
                        lambda q: [FlightOffer(
                            source="amadeus", airline="CA", flight_number="CA981",
                            origin="PEK", destination="NRT",
                            depart_time=datetime(2026, 5, 1, 10, 0),
                            arrive_time=datetime(2026, 5, 1, 15, 0),
                            price=3000)])
    monkeypatch.setattr("src.graph.nodes.collect_data.amadeus_hotels", lambda q: [])
    monkeypatch.setattr("src.graph.nodes.collect_data.amap_pois", lambda c: [])
    monkeypatch.setattr("src.graph.nodes.collect_data.weather_forecast",
                        lambda c: {"city": c, "forecast": []})

    graph = build_graph()
    final = graph.invoke({"user_input": "下周去东京", "form_data": {}, "conversation": []})
    assert "推荐" in final["recommendation"]
    assert len(final["cards"]) == 1
```

- [ ] **Step 3: 运行测试**

```bash
uv run pytest tests/test_graph/test_full_flow.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/graph/builder.py tests/test_graph/test_full_flow.py
git commit -m "feat: assemble LangGraph with parse/route/collect/recommend"
```

---

## 阶段五：Streamlit UI（Task 22-24）

### Task 22: Streamlit 基础界面（表单 + 提交）

**Files:**
- Create: `app.py`

- [ ] **Step 1: 实现 app.py 最小版**

```python
from datetime import date, timedelta
import streamlit as st

from src.logging_setup import setup_logging
from src.storage.db import init_db
from src.graph.builder import build_graph

setup_logging()
init_db()

st.set_page_config(page_title="旅游出行智能体", layout="wide")
st.title("✈️ 旅游出行智能体")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "conversation" not in st.session_state:
    st.session_state.conversation = []

with st.form("trip_form"):
    col1, col2 = st.columns(2)
    with col1:
        origin = st.text_input("出发城市", value="北京")
        depart = st.date_input("出发日期", value=date.today() + timedelta(days=14))
        adults = st.number_input("人数", min_value=1, max_value=9, value=1)
    with col2:
        destination = st.text_input("目的地", value="东京")
        return_date = st.date_input("返程日期（可选）",
                                    value=date.today() + timedelta(days=18))
        budget = st.number_input("预算（元，0 = 不限）", min_value=0, value=0)

    extra = st.text_area("额外偏好",
                         placeholder="如：想住海景酒店、不要太早的航班、带孩子",
                         height=80)
    submitted = st.form_submit_button("🔍 搜索推荐", use_container_width=True)

if submitted:
    with st.status("正在搜索...", expanded=True) as status:
        st.write("📝 解析需求...")
        state = {
            "user_input": extra,
            "form_data": {
                "origin": origin, "destination": destination,
                "depart_date": str(depart),
                "return_date": str(return_date) if return_date else None,
                "adults": adults,
                "budget": budget if budget > 0 else None,
            },
            "conversation": st.session_state.conversation,
        }
        st.write("🚀 并行查询航班、酒店、景点、天气...")
        try:
            final = st.session_state.graph.invoke(state)
            status.update(label="完成！", state="complete")
            st.session_state.last_result = final
        except Exception as e:
            status.update(label=f"出错：{e}", state="error")
            st.exception(e)

if "last_result" in st.session_state:
    result = st.session_state.last_result
    st.markdown("## 💡 AI 推荐")
    st.markdown(result.get("recommendation", ""))

    errors = result.get("tool_errors") or {}
    if errors:
        st.warning("部分数据源不可用：\n" +
                   "\n".join(f"- **{k}**: {v}" for k, v in errors.items()))
```

- [ ] **Step 2: 运行**

```bash
uv run streamlit run app.py
```

浏览器会打开 `http://localhost:8501`。填表单点提交，应该能跑（可能没结果卡片，下一任务加）。

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add minimal Streamlit UI with trip form"
```

---

### Task 23: 结果卡片展示

**Files:**
- Modify: `app.py`

- [ ] **Step 1: 在 app.py 最后追加卡片渲染**

在 `app.py` 的 `if "last_result" in st.session_state:` 块末尾（`st.warning(...)` 之后）添加：

```python
    cards = result.get("cards") or []
    flight_cards = [c for c in cards if c["type"] == "flight"]
    hotel_cards = [c for c in cards if c["type"] == "hotel"]

    if flight_cards:
        st.markdown("## ✈️ 推荐航班")
        for c in flight_cards:
            with st.container(border=True):
                cols = st.columns([3, 2, 1])
                cols[0].markdown(f"**{c['title']}** · `{c['source']}`")
                cols[0].caption(c["subtitle"])
                cols[0].caption(c["detail"])
                cols[1].markdown(f"### ¥{c['price']:.0f}")
                cols[1].caption(c["currency"])

    if hotel_cards:
        st.markdown("## 🏨 推荐酒店")
        for c in hotel_cards:
            with st.container(border=True):
                cols = st.columns([3, 2, 1])
                cols[0].markdown(f"**{c['title']}** · `{c['source']}`")
                cols[0].caption(c["subtitle"])
                if c["detail"]:
                    cols[0].caption(c["detail"])
                cols[1].markdown(f"### ¥{c['price']:.0f}/晚")
                cols[1].caption(c["currency"])

    pois = result.get("pois") or []
    if pois:
        st.markdown("## 🗺️ 值得一去的景点")
        for p in pois[:5]:
            st.markdown(f"- **{p['name']}** "
                       f"{'⭐ ' + str(p['rating']) if p.get('rating') else ''} — {p.get('address', '')}")

    weather = result.get("weather")
    if weather:
        st.markdown("## 🌤️ 目的地天气（3 天）")
        for d in weather.get("forecast", [])[:3]:
            st.markdown(f"- **{d['fxDate']}** {d['textDay']} "
                       f"{d['tempMin']}°C ~ {d['tempMax']}°C")
```

- [ ] **Step 2: 浏览器刷新验证**

跑 `uv run streamlit run app.py`，点提交，确认卡片、景点、天气都显示。

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add result cards (flights, hotels, POIs, weather) in UI"
```

---

### Task 24: Cookie 失效提示与错误降级

**Files:**
- Modify: `app.py`

- [ ] **Step 1: 改进 app.py 的错误处理**

在 `app.py` 的 `if submitted:` 块里，把 `try/except` 改为：

```python
        try:
            final = st.session_state.graph.invoke(state)
            status.update(label="完成！", state="complete")
            st.session_state.last_result = final
        except Exception as e:
            from src.scrapers.base import CookieExpiredError, CookieMissingError
            if isinstance(e, (CookieExpiredError, CookieMissingError)):
                status.update(label="Cookie 失效或缺失", state="error")
                st.error(
                    "🔒 登录 Cookie 失效或缺失。"
                    "\n\n请按 `docs/cookie-export-guide.md` 重新导出到 `cookies/` 目录，然后刷新页面重试。"
                )
            else:
                status.update(label=f"出错：{e}", state="error")
                st.exception(e)
```

- [ ] **Step 2: 手动触发验证**

临时重命名 `cookies/ctrip.json` → `cookies/ctrip.json.bak`，提交一次国内查询，应看到友好提示而非堆栈。

然后重命名回来。

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: friendly cookie-expired error handling in UI"
```

---

## 阶段六：收尾（Task 25-27）

### Task 25: 对话历史（单次会话记忆）

**Files:**
- Modify: `src/storage/history.py`
- Modify: `app.py`

- [ ] **Step 1: 实现 src/storage/history.py**

```python
import json
from datetime import datetime
from sqlalchemy import String, DateTime, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column

from src.storage.db import Base, get_session


class HistoryRow(Base):
    __tablename__ = "history"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def append(session_id: str, role: str, content: str) -> None:
    with get_session() as s:
        s.add(HistoryRow(session_id=session_id, role=role, content=content))


def load(session_id: str) -> list[dict]:
    with get_session() as s:
        rows = s.query(HistoryRow).filter_by(session_id=session_id)\
            .order_by(HistoryRow.created_at).all()
        return [{"role": r.role, "content": r.content} for r in rows]
```

- [ ] **Step 2: 让 app.py 写入历史**

在 `app.py` 顶部加：

```python
import uuid
from src.storage import history as history_store

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
```

提交 form 后，在成功分支末尾追加：

```python
            history_store.append(st.session_state.session_id, "user",
                                 extra or "(form only)")
            history_store.append(st.session_state.session_id, "assistant",
                                 final.get("recommendation", ""))
```

- [ ] **Step 3: 侧栏显示历史**

在 `app.py` `st.title(...)` 下方加：

```python
with st.sidebar:
    st.markdown("### 本次会话历史")
    for m in history_store.load(st.session_state.session_id):
        icon = "🧑" if m["role"] == "user" else "🤖"
        with st.expander(f"{icon} {m['role']}", expanded=False):
            st.markdown(m["content"])
```

- [ ] **Step 4: 验证**

刷新页面，提交两次不同查询，左侧栏应能看到两轮历史。

- [ ] **Step 5: Commit**

```bash
git add src/storage/history.py app.py
git commit -m "feat: add session history persistence and sidebar display"
```

---

### Task 26: 冒烟测试脚本 + README

**Files:**
- Create: `scripts/smoke_test.py`
- Create: `README.md`（覆盖 uv init 生成的）

- [ ] **Step 1: 实现 scripts/smoke_test.py**

```python
"""手动冒烟测试：跑完整 Graph 三种场景。"""
from datetime import date, timedelta
from src.logging_setup import setup_logging
from src.storage.db import init_db
from src.graph.builder import build_graph

setup_logging()
init_db()
graph = build_graph()

scenarios = [
    ("国际：下周去东京玩 4 天",
     {"origin": "北京", "destination": "东京",
      "depart_date": str(date.today() + timedelta(days=14)),
      "return_date": str(date.today() + timedelta(days=18)),
      "adults": 1}),
    ("国内：五一北京去成都",
     {"origin": "北京", "destination": "成都",
      "depart_date": str(date.today() + timedelta(days=10)),
      "return_date": str(date.today() + timedelta(days=13)),
      "adults": 2}),
    ("信息不全",
     {}),
]

for name, form in scenarios:
    print(f"\n==== {name} ====")
    state = {"user_input": name, "form_data": form, "conversation": []}
    try:
        result = graph.invoke(state)
        print("RECOMMENDATION:", result.get("recommendation", "")[:200])
        print("CARDS:", len(result.get("cards") or []))
        print("ERRORS:", result.get("tool_errors") or {})
    except Exception as e:
        print(f"FAIL: {e}")
```

- [ ] **Step 2: 跑冒烟测试**

```bash
uv run python scripts/smoke_test.py
```

期望三种场景都出结果（国内可能因 cookie 报错，若如此手动修后再跑）。

- [ ] **Step 3: 写 README.md**

替换 `uv init` 生成的 README：

```markdown
# 旅游出行智能体

作者自用的 LangGraph + Streamlit 旅游推荐 Agent，支持国际（Amadeus 真实数据）+ 国内（携程 / 飞猪爬虫）混合。

## 快速开始

```bash
# 1. 装依赖
uv sync

# 2. 配置密钥
cp .env.example .env
# 编辑 .env，填入 DeepSeek / Amadeus / 高德 / 和风 的 key

# 3. 导出登录 cookie（仅国内查询需要）
# 按 docs/cookie-export-guide.md 操作

# 4. 启动
uv run streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`。

## 项目结构

见 `docs/superpowers/specs/2026-04-20-travel-agent-design.md`。

## 调试

```bash
# 运行所有单元测试
uv run pytest -v

# 手动冒烟测试
uv run python scripts/smoke_test.py

# 单独测工具
uv run python scripts/smoke_amadeus.py
uv run python scripts/smoke_ctrip.py
```

日志在 `logs/` 下。开启 LangSmith 可视化：在 `.env` 设 `LANGSMITH_TRACING=true`。

## 常见问题

**携程返回空列表？** 查 `logs/scraper.log`，大概率是 cookie 失效或选择器变化。重新导出 cookie，或改 `src/scrapers/selectors.yaml`。

**Amadeus 401？** 检查 `.env` 里 `AMADEUS_CLIENT_ID` / `CLIENT_SECRET`，注意别把 sandbox 和 production 搞混。

**DeepSeek 超时？** 国内网络偶发，重试即可。
```

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke_test.py README.md
git commit -m "docs: add smoke test script and README"
```

---

### Task 27: 完成标准检查（Done Criteria）

**Files:** 无新增，手动逐项验证。

- [ ] **验证 1：国际场景能 30 秒内出推荐**

启动 `streamlit run app.py`，填：北京→东京，下周日，4 天。
按秒表，应在 30 秒内出结果卡片。

- [ ] **验证 2：国内场景拿到真实数据**

填：北京→成都，五一。
应看到携程 + 飞猪的酒店列表各 ≥3 条。

- [ ] **验证 3：Cookie 失效有明确提示**

重命名 `cookies/ctrip.json.bak` 触发错误。
页面应显示友好 markdown 提示，而非 Python 堆栈。

- [ ] **验证 4：缓存 24 小时生效**

同一查询第二次提交。观察 `logs/app.log` 是否出现 `cache.hit` 日志，页面响应应 < 2 秒。

- [ ] **验证 5：持续运行 1 小时不挂**

保持 Streamlit 运行，每 15 分钟提交一次查询，共 4 次。全部成功无崩溃。

- [ ] **全部通过则：Commit 完成标记**

```bash
git commit --allow-empty -m "feat: v1 done criteria verified"
git tag v1.0
```

---

## 自审清单（plan 作者已核对）

**Spec 覆盖：**
- ✅ 技术栈（第 3 节）→ Task 1 装依赖全覆盖
- ✅ 系统架构（第 4 节）→ Task 16-21 实现 Graph
- ✅ 项目结构（第 5 节）→ Task 1 创建目录骨架
- ✅ AgentState（6.1）→ Task 16
- ✅ Graph 流程（6.2）→ Task 21
- ✅ 并行数据收集（6.3）→ Task 19
- ✅ 爬虫四道防线（6.4）→ Task 13（cookie + 超时 + 选择器）+ Task 9（缓存）+ Task 19（降级）
- ✅ 错误处理（第 7 节）→ Task 24（UI）+ Task 19（节点层）+ Task 20（LLM 失败）
- ✅ 日志（第 8 节）→ Task 4
- ✅ 测试策略（第 9 节）→ 贯穿各任务，+ Task 26 冒烟
- ✅ 运行方式（第 10.1-10.2）→ Task 1 + Task 26 README
- ✅ Done Criteria（第 10.3）→ Task 27

**V2 扩展点（spec 第 10.4）明确不在本 plan 范围**：长期记忆、部署、微信机器人。Task 25 预留了 history 表（未来加 preference 表即可）；Graph 不绑定 Streamlit（Task 21 独立于 UI）。

**占位符扫描：** 无 TODO/TBD。`src/storage/cache.py` 和 `history.py` 的占位在 Task 8 创建，Task 9 和 Task 25 分别补齐。

**类型/命名一致性：** `TripQuery`, `FlightOffer`, `HotelOffer`, `AgentState` 跨任务一致；工具入口统一 `search_flights(q: TripQuery)` / `search_hotels(q: TripQuery)`；`cache_get` / `cache_set` 跨 Task 9 与所有 scraper tools 对齐。

**已知风险与缓解：**
- 选择器占位：selectors.yaml 初始值可能不准 → Task 14-15 明确指示"需要手动登录后调整"，并把选择器做成配置
- Amadeus Hotel API 路径：Task 12 用 `by_city` + `hotel_offers_search`，若 SDK 行为不同需要调整（在冒烟阶段暴露）

---

## 执行交接

计划已写完，保存到 `docs/superpowers/plans/2026-04-20-travel-agent-plan.md`。

两种执行方式：

**1. Subagent-Driven（推荐）** — 每个任务派一个新 subagent 执行，任务之间我帮你 review，迭代快、不污染当前对话上下文。

**2. Inline Execution** — 在当前会话里连续执行任务，每几个任务设一个检查点请你确认。

但你是小白，我额外推荐第 3 种：

**3. 手把手模式（小白友好）** — 我当教练，你自己动手敲每一步。我先给任务 1 的详细讲解 + 复制粘贴代码 + 卡壳时问我答疑。每完成 1 任务，你给我看报错/结果，我带你进下一步。**学得最扎实，速度最慢。**

请你选：**1 / 2 / 3**？
