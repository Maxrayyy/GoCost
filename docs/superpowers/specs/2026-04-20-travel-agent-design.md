# 旅游出行智能体设计文档

**日期：** 2026-04-20
**作者：** zhidong_huang（与 Claude 协作）
**版本：** v1
**状态：** 已定稿，待实施

---

## 1. 目标与非目标

### 1.1 目标

构建一个**真实可用、仅供作者自用**的旅游出行智能体：

- 接收用户的出行需求（出发地、目的地、日期、偏好等）
- 自动并行查询多平台的**真实**航班、酒店、景点、天气数据
- 由 LLM 综合信息给出带结构化卡片的推荐
- 以 Streamlit 网页形式交互，本地运行即可使用
- 采用 LangGraph 混合式 Agent，保留未来扩展（长期记忆、微信机器人）接口

### 1.2 非目标

- **不是商业产品**：不追求多用户、不追求多并发、不做账号体系
- **不追求"全平台"**：国内只覆盖携程 + 飞猪，国际只用 Amadeus
- **不做完整行程规划**（景点路线、餐厅、每日安排）——v1 不在范围内
- **不做长期记忆**（用户永久偏好）——v2 扩展点，当前只做单次会话

---

## 2. 使用场景

- **混合出行**：用户可能出国也可能国内游
- **国际出行**：以 Amadeus Self-Service API 提供真实数据
- **国内出行**：以 Selenium + 手动导出的登录 cookie 爬携程、飞猪
- 实时触发（用户等 30~60 秒出结果），通过并行爬取 + 缓存降低等待

### 示例请求

- "下周日去东京玩 4 天，预算 8000"
- "五一北京去成都，要带游泳池的酒店"
- "帮我看下 5/15 上海到三亚最便宜的航班 + 评分好的海景酒店"

---

## 3. 技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| Agent 框架 | **LangGraph**（LangChain 官方） | 混合式工作流 + ReAct，节点可独立测试 |
| LLM | **DeepSeek-Chat**（OpenAI 兼容接口） | 国内直连、便宜、质量够用 |
| Web UI | **Streamlit** | 单文件出网页，小白友好 |
| 国际数据 | **Amadeus Self-Service API** | 免费 10000 次/月、真实、全球覆盖 |
| 国内航班/酒店 | **Selenium + 手动 cookie**（携程、飞猪） | 个人自用下唯一可行的真实数据源 |
| 景点 / 评价 | **高德地图开放 API** | 免费、有 POI 和评分 |
| 天气 | **和风天气 API** | 免费、国内直连 |
| 数据持久化 | **SQLite**（SQLAlchemy） | 无需服务进程，单文件 |
| 配置管理 | **.env + python-dotenv** | 密钥不进代码 |
| 依赖管理 | **uv**（`pyproject.toml`） | 现代 Python 事实标准 |
| 日志 | **loguru** | 对小白更友好的 API |
| 可观测性（可选） | **LangSmith** | 可视化 Graph 节点的输入输出 |
| 运行方式 | 本地 `streamlit run`，未来可部署到 Railway/Fly.io | 自用足够 |

---

## 4. 系统架构

### 4.1 高层数据流

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web UI                         │
│   [需求表单]   [对话 / 偏好输入]   [结果卡片展示]            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 LangGraph 核心编排器                        │
│                                                             │
│   [解析节点] → [路由节点] ⇒ [并行数据收集节点] →            │
│       ↑            │                                        │
│       │      (国内/国外分流)                                │
│       │            │                                        │
│       │            ├─→ [Amadeus 工具]      国际航班/酒店    │
│       │            ├─→ [携程爬虫工具]      国内航班/酒店    │
│       │            ├─→ [飞猪爬虫工具]      国内酒店         │
│       │            ├─→ [高德 POI 工具]     景点评价         │
│       │            └─→ [和风天气工具]      目的地天气       │
│       │                     │                               │
│       │                     ▼                               │
│       │            [数据聚合 + 归一化]                      │
│       │                     │                               │
│       │                     ▼                               │
│       │            [LLM 推荐节点] ← ReAct 模式              │
│       │                     │                               │
│       │                     ▼                               │
│       └───── [追问/澄清节点] (信息不够时回头)               │
│                             │                               │
│                             ▼                               │
│                     [结构化输出]                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              数据层（SQLite）                               │
│  [价格缓存 24h]  [对话历史]  [偏好表(v2 扩展)]              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 关键设计决策

1. **固定主干 + ReAct 叶子**：parse/route/collect 是确定流程（可调试），recommend/clarify 节点内部用 ReAct 让 LLM 自由发挥
2. **并行爬虫**：使用 `asyncio.gather` 同时触发所有数据源，把 30-60 秒压缩到 15-25 秒
3. **缓存优先**：同一路线 + 同日期 24 小时内第二次查询 <2 秒返回
4. **降级策略**：单工具失败不影响整体，LLM 被告知哪些源失败，照常出推荐
5. **可扩展点**：
   - `src/memory/`：v2 长期偏好
   - 前端可替换：Graph 不依赖 Streamlit
   - 新平台：加一个爬虫 = 加一个文件 + Graph 注册

---

## 5. 项目结构

```
D:\code\agent\
├── .env                          # 密钥（DeepSeek、Amadeus、高德、和风）
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
│
├── app.py                        # Streamlit 入口
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # 统一读 .env
│   │
│   ├── graph/                    # LangGraph 核心
│   │   ├── __init__.py
│   │   ├── state.py              # AgentState 定义
│   │   ├── builder.py            # 拼装 Graph
│   │   └── nodes/
│   │       ├── parse_intent.py
│   │       ├── route.py
│   │       ├── collect_data.py
│   │       ├── recommend.py
│   │       └── clarify.py
│   │
│   ├── tools/                    # LangChain Tool 封装
│   │   ├── __init__.py
│   │   ├── amadeus_tool.py
│   │   ├── ctrip_scraper.py
│   │   ├── fliggy_scraper.py
│   │   ├── amap_tool.py
│   │   └── weather_tool.py
│   │
│   ├── scrapers/                 # 底层爬虫（与 tools 分离）
│   │   ├── __init__.py
│   │   ├── base.py               # cookie、重试、日志
│   │   ├── ctrip.py
│   │   ├── fliggy.py
│   │   └── selectors.yaml        # CSS 选择器配置化
│   │
│   ├── models/                   # Pydantic 统一数据模型
│   │   ├── __init__.py
│   │   ├── flight.py             # FlightOffer
│   │   ├── hotel.py              # HotelOffer
│   │   └── query.py              # TripQuery
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── db.py
│   │   ├── cache.py              # 24h 价格缓存
│   │   └── history.py
│   │
│   └── memory/                   # v2 扩展点
│       └── .gitkeep
│
├── cookies/                      # gitignore
│   ├── ctrip.json
│   └── fliggy.json
│
├── data/                         # gitignore
│   └── agent.db
│
├── tests/
│   ├── test_parse_intent.py
│   ├── test_tools/
│   │   └── test_amadeus.py
│   └── test_graph/
│       └── test_full_flow.py
│
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-04-20-travel-agent-design.md
```

### 5.1 结构原则

1. **UI 与核心分离**：`app.py` 只渲染，所有逻辑在 `src/graph/`
2. **每节点 / 工具一文件**：小白可逐个理解和测试
3. **`tools/` vs `scrapers/` 分层**：前者给 Agent 用，后者处理底层细节
4. **Pydantic 统一模型**：Amadeus JSON 和携程 HTML 先转成统一对象，LLM 才能公平比较
5. **敏感文件 gitignore**：`.env`、`cookies/`、`data/`

---

## 6. 核心组件详细设计

### 6.1 AgentState（Graph 间传递的数据包）

```python
class AgentState(TypedDict):
    # 输入
    user_input: str
    form_data: dict
    conversation: list[dict]

    # 解析结果
    query: TripQuery | None
    missing_fields: list[str]

    # 路由
    is_domestic: bool | None

    # 收集数据
    flights: list[FlightOffer]
    hotels: list[HotelOffer]
    pois: list[dict]
    weather: dict | None
    tool_errors: dict[str, str]

    # 输出
    recommendation: str | None
    cards: list[dict]
```

### 6.2 Graph 流程

```python
graph.add_node("parse", parse_intent_node)
graph.add_node("clarify", clarify_node)
graph.add_node("route", route_node)
graph.add_node("collect", collect_data_node)
graph.add_node("recommend", recommend_node)

graph.set_entry_point("parse")

graph.add_conditional_edges(
    "parse",
    lambda s: "clarify" if s["missing_fields"] else "route"
)
graph.add_edge("clarify", END)
graph.add_edge("route", "collect")
graph.add_edge("collect", "recommend")
graph.add_edge("recommend", END)
```

`recommend_node` 内部用 `create_react_agent`，允许 LLM 自主追加查询。

### 6.3 并行数据收集

```python
async def collect_data_node(state: AgentState) -> AgentState:
    tasks = []
    if state["is_domestic"]:
        tasks.append(ctrip_tool.search(state["query"]))
        tasks.append(fliggy_tool.search(state["query"]))
    else:
        tasks.append(amadeus_tool.search(state["query"]))

    tasks.append(amap_tool.get_pois(state["query"].destination))
    tasks.append(weather_tool.get_forecast(state["query"]))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    # 单工具失败不炸，记录到 tool_errors
```

Streamlit 侧同步显示进度条（"🔍 正在查携程... ✅ 天气"）。

### 6.4 爬虫可靠性四道防线

| 防线 | 实现 |
|---|---|
| ① Cookie 管理 | `cookies/ctrip.json` 由用户用 Cookie-Editor 浏览器插件导出；爬虫启动时加载，失效则抛 `CookieExpiredError` |
| ② 24h 缓存 | SQLite 存 `(route, date) -> result`，同键 24 小时内直接返回 |
| ③ 超时 + 降级 | 单工具 45 秒超时 → 记录 `tool_errors[tool] = msg` → LLM 推荐中说明"XX 暂不可用" |
| ④ 选择器配置化 | CSS 选择器放在 `src/scrapers/selectors.yaml`，改版只改 yaml |

---

## 7. 错误处理

| 错误 | 处理 |
|---|---|
| LLM 超时/报错 | 重试 1 次 → 失败返回"AI 暂不可用，原始数据如下" |
| Amadeus 额度用完（429） | `tool_errors` 记录，LLM 被告知 |
| Cookie 失效 | UI 弹黄色警告 "请重新导出 cookie"，附操作链接 |
| 选择器失效 | 抛 `SelectorChangedError`，日志保存页面快照 |
| 用户输入解析不出 | 走 clarify 分支主动追问 |
| 所有数据源失败 | 友好提示，不让 LLM 编造 |

**核心原则：永远不让 Agent 编假数据。查不到就说查不到。**

---

## 8. 日志与可观测性

```
logs/
  app.log        # 所有行为
  scraper.log    # 爬虫专用
  llm.log        # LLM prompt 和 response
```

- 使用 `loguru`（零配置）
- `.env` 里加 `LANGSMITH_API_KEY` 即自动开启 LangSmith 可视化（可选）

---

## 9. 测试策略

务实三层：

**① 单元测试（pytest + mock，快）**
- `test_parse_intent.py`：用户输入 → TripQuery
- `test_models.py`：归一化函数（Amadeus → FlightOffer）

**② 工具集成测试（打真 API，`@pytest.mark.slow`）**
- `test_amadeus_real.py`：真调一次北京→东京

**③ 端到端冒烟测试（手动，`scripts/smoke_test.py`）**
- 3 种典型请求（国内、国外、信息不全）

爬虫单测跳过（选择器会变，写了白搭），靠冒烟测试发现问题即修。

---

## 10. 运行与"上线"

### 10.1 首次启动

```bash
uv sync                           # 装依赖
cp .env.example .env              # 填入密钥
# 浏览器用 Cookie-Editor 插件手动登录携程/飞猪，导出 cookie 到 cookies/
```

### 10.2 每次启动

```bash
uv run streamlit run app.py       # 浏览器自动打开 localhost:8501
```

### 10.3 v1 完成标准（Done Criteria）

- ✅ 国际场景：输入"下周日去东京玩 4 天"，30 秒内给出航班 + 酒店推荐
- ✅ 国内场景：输入"五一北京去成都"，从携程 + 飞猪各拿到 ≥3 条结果
- ✅ Cookie 失效有明确提示，不崩
- ✅ 同一查询 24 小时内第二次 <2 秒返回（缓存生效）
- ✅ 连续运行 1 小时不挂（基础健康检查）

### 10.4 v2 扩展（不在本 spec 范围）

- 部署到 Railway / Fly.io 免费层
- 接入微信 / Telegram 机器人
- 长期记忆表（`src/memory/`）：用户永久偏好、行程历史

---

## 11. 时间估算

| 阶段 | 工作量 |
|---|---|
| 环境搭建 + 申请 API key | 0.5 天 |
| LangGraph 骨架 | 1 天 |
| Amadeus 工具 + 国际端到端 | 1 天 |
| 携程爬虫 + cookie 机制 | 1.5 天 |
| 飞猪爬虫 | 1 天 |
| 高德 + 天气 | 0.5 天 |
| Streamlit UI | 1 天 |
| SQLite 缓存 + 历史 | 0.5 天 |
| LLM 推荐 prompt 调优 | 1 天 |
| 测试 + 修 bug + 文档 | 1 天 |
| **合计** | **约 10 天（全职）/ 3-4 周（业余）** |

---

## 12. 合规与风险声明

- **仅作者自用**，不公开分享、不商业化
- Selenium 爬取携程 / 飞猪严格意义上违反平台用户协议（非法律），风险在账号被封
- 量小、有缓存、不高频，实践风险可控
- **绝不发布为公开服务或下载给第三方使用**
- Cookie 为作者本人账号的登录凭证，`.gitignore` 不入版本库

---

## 13. 未决事项与后续

- 具体 prompt 模板（recommend 节点的 system prompt）在实施阶段定稿
- 携程 / 飞猪的具体爬取页面和选择器在实施阶段确认（需要作者配合登录页面确认）
- 是否启用 LangSmith 由作者决定（不影响主流程）
