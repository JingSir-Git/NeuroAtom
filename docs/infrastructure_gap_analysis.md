# NeuroAtom 全球 EEG 基础设施差距分析

> **愿景**：为全世界 EEG 脑电研究者提供即插即用的基础设施——任何人都能向资源池贡献标准化数据，任何人都能按需提取和使用数据。

## 一、现状评估（已完成）

### ✅ 核心架构 — 扎实

| 模块 | 状态 | 评价 |
|------|------|------|
| **Atom 数据模型** | ✅ 完整 | 信息完备原子单元，Pydantic schema，自描述 |
| **Pool 存储层** | ✅ 完整 | HDF5 shards + JSONL 元数据，压缩，文件锁 |
| **SQLite 索引** | ✅ 完整 | 丰富的 Query DSL，支持按数据集/被试/通道/标注查询 |
| **Assembly 流水线** | ✅ 完整 | 单位统一 → 重参考 → 通道映射 → 滤波 → 重采样 → 归一化 → 裁剪/填充 |
| **跨 Pool 联邦** | ✅ 完整 | `FederatedPool` + `FederatedAssembler`，跨 Pool 查询 + 组装 |
| **跨被试分割** | ✅ 完整 | `DataSplitter` 支持 Subject/Stratified/Dataset/Temporal/Predefined |
| **多模态** | ✅ 完整 | `MultiModalAssembler` + `MultiModalRecipe` |
| **导入器** | ✅ 12+ 专用 + 4 通用 | KUL/DTU/BCI/PhysioNet/OpenBMI/SEED/Zuco2/ChineseEEG/CCEP... |
| **数据溯源** | ✅ 完整 | `ProcessingHistory` + `import_log` |
| **Schema 迁移** | ✅ 框架 | 注册式迁移机制，BFS 路径查找 |
| **CLI** | ✅ 基础 | init/import/reindex/stats/query/assemble/export |
| **PyTorch 集成** | ✅ 完整 | `AtomDataset` + `HDF5AtomDataset` + collate |

**结论：核心功能齐全且经过测试，但距离"全球基础设施"标准还有 6 个关键差距。**

---

## 二、六大差距（按优先级排序）

### 🔴 G1: Pool 可分发性 — 导出 / 导入 / 版本化

**现状**：Pool 是纯本地目录，没有标准化的打包/分发/版本控制机制。

**目标**：任何人处理好的 Pool 可以一键分享给他人使用。

**缺失**：
- `pool export` → 导出为自描述归档包（含 schema 版本、校验和）
- `pool import` → 从归档包重建 Pool（含完整性验证）
- Pool manifest（`manifest.json`）— 内容清单 + SHA-256 校验
- 增量导出 — 只导出新增的 datasets/subjects
- 版本号 + 变更日志（数据级，非代码级）

**方案**：
```
neuroatom export ./my_pool --out kul_aad_v1.napool
neuroatom import kul_aad_v1.napool --into ./shared_pool
neuroatom export ./my_pool --incremental --since 2024-03-01
```

`.napool` 格式 = tar.gz + `manifest.json` + 可选 integrity check

---

### 🔴 G2: 标准化合规层 — 单位 / 通道 / 坐标系强制统一

**现状**：导入时各数据集"各说各话"——有的存 V，有的存 µV；通道名有 `Fp1` / `EEG Fp1` / `FP1` 混用；电极坐标有的是 MNI，有的是 Talairach，有的没有。Assembly 时做单位转换，但存储层没有强制统一。

**目标**：Pool 中所有数据遵循同一物理规格，任何人从 Pool 取数据时无需再做单位/通道名/坐标系转换。

**缺失**：
- Pool 级 **存储规范** 声明：
  - `storage_unit: "uV"` — 所有信号统一以 µV 存储
  - `channel_naming: "10-20_standard"` — 所有通道名存入时即标准化
  - `coordinate_system: "MNI152"` — 电极坐标统一坐标系
- **导入时强制转换**（而非 Assembly 时才转换）
- **元数据完整性校验**（导入时检查必填字段）
- 通道名标准化已有 `standardize_channel_name()`，但不是强制应用

**方案**：
```yaml
# pool.yaml 新增
storage_conventions:
  signal_unit: "uV"           # 所有信号以 µV 存储
  channel_naming: "standard"  # 导入时自动标准化通道名
  coordinate_system: "MNI152" # 统一坐标系
  enforce_on_import: true     # 导入时强制执行
```

导入时统一转换为 µV 存入 HDF5，而不是存 V 然后 Assembly 时再转。

---

### 🟡 G3: 数据发现与目录 — Pool Registry / 数据集目录

**现状**：没有中心化的数据集目录。用户必须知道 Pool 路径和 dataset_id 才能查询。

**目标**：研究者可以浏览"全球所有可用的 EEG 数据集"，按任务类型/通道数/被试数筛选，一键下载。

**缺失**：
- **数据集目录**（Dataset Catalog）— 所有已注册数据集的可检索清单
- **Pool 注册表**（Pool Registry）— 远程可发现的 Pool 列表
- `neuroatom search "motor imagery" --min-subjects 20`
- `neuroatom catalog list --task auditory_attention`
- 每个数据集的"数据卡"（Data Card）：被试数、通道数、采样率、任务类型、许可证、引用

**方案**：
```python
# 本地目录
catalog = na.Catalog.load()  # 从内置 + 远程注册表
results = catalog.search(task="motor_imagery", min_subjects=20)

# 远程 Pool
na.pull("neuroatom-hub/kul_aad", into="./my_pool")
```

---

### 🟡 G4: 数据质量契约 — 准入标准 + 质量报告

**现状**：`validate_signal` 做了基础检查（NaN/Inf/flat-line），但没有准入门槛——任何质量的数据都能入池。

**目标**：Pool 中的数据有明确的质量标准。贡献者知道自己的数据是否达标，使用者知道数据质量。

**缺失**：
- **准入规则**（Admission Policy）— 可配置的最低质量标准
  - 最小试次数 / 被试数
  - 最大 artifact 比例
  - 通道覆盖要求（至少包含 10-20 标准通道的 N%）
  - 元数据完整性（必须有标签、必须有通道名）
- **质量报告**（Quality Report）— 导入后自动生成
  - 每被试的信噪比、artifact 比例、缺失数据比例
  - 通道覆盖热图
  - 采样率/通道数一致性
- **质量徽章**（Quality Badge）— 数据卡上的可视化质量等级

**方案**：
```yaml
# pool.yaml
admission_policy:
  min_channels: 16
  min_sampling_rate: 64
  max_nan_ratio: 0.01
  max_flatline_ratio: 0.1
  required_annotations: ["task_label"]
  required_metadata: ["sampling_rate", "channel_names"]
```

---

### 🟡 G5: 自定义导入器标准化 — 贡献者工具链

**现状**：添加新数据集需要写 Python 导入器（~500 行）+ YAML config。门槛较高。

**目标**：任何研究者可以用最少代码将自己的数据集贡献到资源池。

**缺失**：
- **导入器模板生成器** — `neuroatom scaffold importer my_dataset`
- **配置验证工具** — `neuroatom validate-config my_dataset.yaml`
- **BIDS 自动导入** — 已有 `bids.py`，但需要更robust的自动映射
- **CSV/Numpy 导入** — 很多研究者的数据是 `.npy` + `.csv` 元数据
- **贡献者文档** — 如何写一个合格的导入器
- **导入器测试框架** — 自动验证导入器输出是否符合 Atom schema

**方案**：
```bash
# 生成导入器模板
neuroatom scaffold importer --name "my_eeg_study" --format mat --task-type motor_imagery

# 验证导入器输出
neuroatom validate-import ./my_pool --dataset my_eeg_study --strict

# 最简导入：从 numpy 数组 + CSV 标签
neuroatom import-raw ./my_pool \
  --signals data.npy \
  --labels labels.csv \
  --channels channels.csv \
  --srate 256 \
  --dataset-id my_study
```

---

### 🟢 G6: 文档与社区标准

**现状**：有 README、架构审计文档、示例脚本，但缺少面向社区的标准文档。

**缺失**：
- **数据模型规范文档**（Atom Specification v1.0）
- **Pool 格式规范文档**（Pool Format Specification v1.0）
- **贡献者指南**（Contributing Guide）
- **数据治理策略**（Data Governance Policy）— 许可证、引用、隐私
- **API 参考文档**（自动生成）
- **互操作性标准**（与 BIDS、NWB、HED 的映射关系）

---

## 三、实施路线图

### Phase 1: 存储规范强制化（G2 核心）— 1 周
> 让 Pool 内的数据真正统一

1. `pool.yaml` 新增 `storage_conventions` 配置
2. 导入时信号强制转为 µV 存储（而非 V）
3. 导入时通道名强制标准化存储
4. `Atom.signal_unit` 字段 → 记录存储单位
5. `UnitStandardizer` 移到导入层（Assembly 层保留兼容）
6. 更新所有 12+ 导入器

### Phase 2: Pool 可分发性（G1 核心）— 1 周
> 让 Pool 可以在人之间传递

1. `Pool.export()` → `.napool` 归档包
2. `Pool.import_from()` → 从归档包恢复
3. `manifest.json` 内容清单 + 校验和
4. CLI: `neuroatom export` / `neuroatom import-pool`
5. 增量导出支持

### Phase 3: 数据质量契约（G4）— 3 天
> 建立准入标准

1. `AdmissionPolicy` 配置模型
2. 导入时自动运行准入检查
3. 质量报告生成器
4. CLI: `neuroatom quality-report`

### Phase 4: 贡献者工具链（G5）— 1 周
> 降低贡献门槛

1. `neuroatom scaffold importer`
2. 通用 numpy/csv 导入器
3. 导入器输出验证工具
4. 贡献者文档

### Phase 5: 数据目录（G3）— 远期
> 可发现性

1. 本地数据集目录
2. 远程注册表协议
3. `neuroatom search` / `neuroatom pull`

---

## 四、最高优先级行动项（立即可做）

| # | 行动 | 影响 | 工作量 |
|---|------|------|--------|
| **1** | Pool 存储单位统一为 µV | 消除 Assembly 时的单位混乱，Pool 数据即插即用 | 2 天 |
| **2** | 导入时通道名强制标准化 | 不同数据集的通道名直接可对齐 | 1 天 |
| **3** | Atom 增加 `signal_unit` 字段 | 自描述，任何人读到 Atom 都知道单位 | 半天 |
| **4** | Pool export/import + manifest | Pool 可以打包分发 | 2 天 |
| **5** | 准入策略配置 + 导入时检查 | 保障数据质量底线 | 1 天 |
| **6** | `neuroatom scaffold importer` | 降低贡献门槛 | 1 天 |

---

## 五、竞品对比

| 特性 | NeuroAtom (当前) | MOABB | NeurobenchDS | HuggingFace Datasets |
|------|-----------------|-------|-------------|---------------------|
| 跨数据集统一模型 | ✅ Atom | ❌ | ❌ | ❌ |
| 存储层单位统一 | ✅ 导入时 µV | ❌ | ❌ | ❌ |
| 跨被试分割 | ✅ | ✅ | ❌ | ❌ |
| 联邦查询 | ✅ | ❌ | ❌ | ❌ |
| 可分发 Pool | ✅ .napool 归档 | N/A | ✅ | ✅ |
| 数据目录 | ✅ 本地+远程 | 内置列表 | ❌ | ✅ Hub |
| 准入质量标准 | ✅ 三级准入 | ❌ | ❌ | ❌ |
| 贡献者工具链 | ✅ scaffold + generic | ❌ | ❌ | ✅ |
| BIDS 互操作 | ✅ 导入 | ❌ | ❌ | ❌ |
| 多模态 | ✅ | ❌ | ❌ | ❌ |

---

## 六、实施进度

### ✅ Phase 1: 存储规范强制化（已完成）

**决策确认**（2026-04-28）：
1. **存储单位**：导入时统一为 µV，Atom 记录 `signal_unit` + `original_unit`
2. **分发路线**：Route C（先 B 归档包，后 A Hub）
3. **质量分级**：Silver / Gold / Platinum 三级准入

**已实现**：
- `QualityTier` 枚举：`SILVER` / `GOLD` / `PLATINUM`
- `Atom.signal_unit` + `Atom.original_unit` 字段
- `DatasetMeta.quality_tier` 字段
- `pool.yaml` → `storage_conventions` 配置节
- `neuroatom/utils/unit_convert.py` — 导入时单位转换工具
- **10 个导入器** 全部更新：导入时 V→µV 转换 + 设置 atom 元数据
- **Assembly 层** 读取 `atom.signal_unit`，µV 数据跳过转换
- Schema 版本升级至 `1.1.0`
- **38 单元测试通过**（含 5 个新增 `TestConvertToStorageUnit`）
- **297 全量测试通过**，0 失败

### ✅ Phase 2: Pool 可分发性（已完成）

**目标**：让 Pool 能够打包、分享、导入，实现跨团队数据流通。

**已实现**：
- `.napool` 归档格式：`tar.gz` + `manifest.json`（SHA-256 校验 + 元数据）
- `neuroatom/storage/pool_archive.py` — `export_pool()` / `import_pool()`
- `Pool.export()` + `Pool.import_from()` 便捷方法
- CLI 命令：`neuroatom export-pool` / `neuroatom import-pool`
- 支持**选择性导出**（`--dataset` 指定数据集）
- 支持**增量导出**（`--since` 按导入日期过滤）
- **合并导入**：导入到已有 Pool 时自动合并，保留已有数据
- **完整性校验**：导入时逐文件 SHA-256 验证
- **被试级导出**：`--subject S01` 或 `--subject dataset_id/S01`
- **10 项专用测试**全部通过（含 round-trip、被试级导出、合并）
- **307 全量测试通过**，0 失败

### ✅ Phase 3: 数据质量契约（已完成）

**目标**：建立分级准入标准，自动评估数据质量。

**已实现**：
- `neuroatom/quality/admission.py` — `AdmissionPolicy` + `TierCriteria` 配置模型
  - **Silver**：有效信号 + 基本标签 + ≥1 通道
  - **Gold**：8+ 通道、10+ atoms、标准通道名、电极坐标、≤20% 坏导
  - **Platinum**：16+ 通道、50+ atoms、2+ 被试、处理溯源、刺激文件
- `neuroatom/quality/gate.py` — `QualityGate` 自动评估器
  - `assess_dataset()` / `assess_atoms()` 两种入口
  - 逐项检查每个 tier 的 11 项标准
  - 返回 `QualityReport`（tier、统计、逐项检查结果、警告）
- `neuroatom/quality/report.py` — 质量报告格式化（终端 + JSON）
- CLI：`neuroatom quality-report <pool> <dataset>` — 评估并自动更新 `dataset.json` 中的 `quality_tier`
- **12 项专用测试**全部通过
- **319 全量测试通过**，0 失败

### ✅ Phase 4: 贡献者工具链（已完成）

**目标**：降低新数据集接入门槛，让贡献者快速上手。

**已实现**：
- `neuroatom/contrib/scaffold.py` — `scaffold_importer()` 生成器
  - 自动生成 3 个文件：导入器 `.py`、任务配置 `.yaml`、测试骨架 `.py`
  - 包含 TODO 标记和示例代码，贡献者只需填空
- `neuroatom/importers/generic.py` — 通用 numpy/CSV 导入器
  - 支持 `.npy` / `.npz` / `.csv` 文件格式
  - 自动发现被试（子目录或平面文件）
  - 支持标签文件 (`labels.csv`) 自动分 epoch
  - 支持固定时长切片 (`--epoch-seconds`)
  - 支持 `channels.txt` 指定通道名
- `neuroatom/contrib/validate_import.py` — 导入验证器
  - 检查 Atom 必填字段、类型、一致性
  - 可选 HDF5 信号数据校验（`--check-signals`）
  - 生成 PASS/FAIL 验证报告
- CLI 命令：
  - `neuroatom scaffold <name>` — 生成导入器骨架
  - `neuroatom import-generic <pool> <data_dir> -d <dataset_id>` — 通用导入
  - `neuroatom validate-import <pool> <dataset_id>` — 验证导入质量
- **13 项专用测试**全部通过
- **392 全量测试通过**，0 失败

### ✅ Phase 5: 数据目录（已完成）

**目标**：建立可发现性，本地索引 + 远程注册表 + 搜索/拉取。

**已实现**：
- `neuroatom/catalog/models.py` — `DatasetEntry` + `CatalogIndex` 数据模型
  - 多维搜索：名称、任务类型、被试数、通道数、质量等级、标签
  - 序列化/反序列化 JSON，支持本地和远程目录
  - `merge()` 合并远程目录到本地
- `neuroatom/catalog/local.py` — 本地目录管理
  - `catalog.json` 存储在 pool 根目录
  - `rebuild_catalog()` 从 pool 元数据全量重建
  - `update_catalog_entry()` / `remove_catalog_entry()` 增量更新
  - 自动统计 atom 数量
- `neuroatom/catalog/remote.py` — 远程注册表协议
  - `fetch_remote_catalog(url)` HTTP GET JSON
  - `merge_remote(pool, url)` 拉取并合并到本地
  - `pull_dataset(pool, dataset_id)` 下载 .napool 并导入
  - 支持 pool.yaml 中配置多个 registry URL
- CLI 命令：
  - `neuroatom catalog-rebuild <pool>` — 重建本地目录
  - `neuroatom search <pool> -q <query> -t <task> --tier gold` — 搜索数据集
  - `neuroatom catalog-sync <pool> --url <url>` — 同步远程目录
  - `neuroatom pull <pool> <dataset_id> --url <url>` — 下载并导入
- **19 项专用测试**全部通过（含模型、本地目录、远程 HTTP 服务器）
- **411 全量测试通过**，0 失败
