# NeuroAtom 架构审查与功能完整性报告

> 审查日期：2026-04-26  
> 版本：v0.1.0

---

## 一、设计初心与核心架构

NeuroAtom 的设计目标是：**将异构 EEG/iEEG 数据集分解为标准化的原子单元（Atom），实现跨数据集、跨范式、跨被试的统一 ML 训练。**

### 核心设计层次

| 层 | 模块 | 状态 | 说明 |
|----|------|------|------|
| **数据模型** | `core/atom.py` | ✅ 完善 | 7 种注解子类型、处理溯源链、质量标记、原子间关系 |
| **导入层** | `importers/` | ⚠️ 部分完善 | 11 个专用 + 4 个通用导入器，但标签覆盖不一致（详见第三节） |
| **存储层** | `storage/` | ✅ 完善 | Pool 目录结构、HDF5 分片、JSONL 元数据、channels.json |
| **索引层** | `index/` | ✅ 完善 | SQLite 后端、QueryBuilder、联邦池（FederatedPool） |
| **组装层** | `assembler/` | ✅ 完善 | 通道映射、重采样、滤波、归一化、裁剪/填充、重参考、单位标准化 |
| **加载层** | `loader/` | ✅ 完善 | PyTorch Dataset/DataLoader、变换管线、配对数据集 |
| **便捷 API** | `quick.py` | ✅ 完善 | `quickload()` 5 行入门、`multiload()` 跨数据集训练 |
| **CLI** | `cli/` | ✅ 基础可用 | init/import/index/query/assemble/info/stats/export |
| **多模态** | `assembler/multimodal_assembler.py` | ✅ 已实现 | 单模态拆分已完成，配对多模态已规划 |
| **联邦查询** | `index/federation.py` | ✅ 已实现 | 跨 Pool 联合查询 |

---

## 二、已实现的完整功能链

### 2.1 端到端数据流

```
原始数据  →  Importer  →  Pool (HDF5 + JSONL)
                              ↓
                         Indexer (SQLite)
                              ↓
                         QueryBuilder (筛选)
                              ↓
                         DatasetAssembler (组装管线)
                              ↓
                         PyTorch DataLoader (训练就绪)
```

### 2.2 核心能力一览

| 能力 | 状态 | 描述 |
|------|------|------|
| **一键导入** | ✅ | `quickload("bci_comp_iv_2a", "data/A01T.mat")` → DataLoader |
| **跨数据集联合训练** | ✅ | `multiload()` 统一通道/采样率/时长/单位后输出 |
| **自动格式检测** | ✅ | `registry.detect_format(path)` 自动识别数据集类型 |
| **通道标准化映射** | ✅ | 任意通道系统 → 标准 10-20，含插值/球面投影 |
| **采样率统一** | ✅ | 自动重采样到目标频率 |
| **单位统一** | ✅ | V / mV / µV 自动转换 |
| **带通滤波** | ✅ | 指定 `(low_hz, high_hz)` |
| **归一化** | ✅ | z-score / robust / minmax，作用域：per_atom / global / per_subject |
| **裁剪/填充** | ✅ | 固定时长输出，不足补零 |
| **重参考** | ✅ | 平均参考 / 指定通道参考 |
| **数据分割** | ✅ | 按被试 / 分层 / 按数据集分割训练/验证/测试 |
| **标签编码** | ✅ | 从注解字段自动提取标签并编码为整数 |
| **处理溯源** | ✅ | 每个 Atom 携带完整处理链（哈希可追溯） |
| **质量评估** | ✅ | 平线检测、NaN 检测、方差异常 |
| **多模态** | ✅ | EEG + sEEG 同时导入，分模态组装 |
| **联邦池** | ✅ | 多个 Pool 联合索引和查询 |
| **导入日志** | ✅ | 每次导入记录到 `import_log` |

### 2.3 注解系统（7 种子类型）

| 子类型 | 类名 | 使用场景 |
|--------|------|----------|
| **分类标签** | `CategoricalAnnotation` | MI 类别、情绪、注意力方向 |
| **数值标签** | `NumericAnnotation` | 章节号、句子索引、SSVEP 频率 |
| **文本** | `TextAnnotation` | 刺激描述、句子内容 |
| **连续信号** | `ContinuousAnnotation` | 音频包络（AAD）、EMG |
| **事件序列** | `EventSequenceAnnotation` | 词边界、音素 |
| **刺激引用** | `StimulusRefAnnotation` | 外部音频/图片引用 |
| **二值掩码** | `BinaryMaskAnnotation` | 伪迹区域 |

---

## 三、数据集导入器逐一审查

### 3.1 审查标准

对每个数据集检查：
1. **核心任务标签** — 是否提取了该范式的关键分类信息
2. **实验元数据** — session/run/trial 级别信息
3. **通道信息** — 名称、类型、电极坐标
4. **信号元数据** — 采样率、单位、参考
5. **质量标记** — 伪迹标记、坏导
6. **自适应识别** — 字段名差异是否已统一

---

### 3.2 BCI Competition IV 2a (`bci_comp_iv_2a.py`)

| 检查项 | 状态 | 注解字段 |
|--------|------|----------|
| MI 类别标签 | ✅ | `mi_class` (left_hand / right_hand / feet / tongue) |
| 伪迹标记 | ✅ | `artifact` = "rejected" |
| 训练/评估区分 | ✅ | `session_type` (training / evaluation) |
| 通道信息 | ✅ | 25 通道 + EOG，含 10-20 坐标 |
| 信号单位 | ✅ | µV（config 中声明） |
| 被试信息 | ✅ | 年龄、性别从 mat 提取 |

**问题：** 无。此导入器已完善。

---

### 3.3 PhysioNet MI (`physionet_mi.py`)

| 检查项 | 状态 | 注解字段 |
|--------|------|----------|
| MI 类别 | ✅ | `mi_class` (left_fist / right_fist / both_fists / both_feet) |
| 范式区分 | ✅ | `paradigm` (imagery / execution / rest) |
| 任务名 | ✅ | `task` (open_close_left_right_fist 等) |
| 基线 run | ✅ | eyes_open / eyes_closed rest runs 已导入 |
| 电极坐标 | ✅ | 标准 10-20 蒙太奇 |
| 信号单位 | ✅ | V（EDF 自动检测） |

**问题：** 无主要问题。

---

### 3.4 SEED-V (`seed_v.py`)

| 检查项 | 状态 | 注解字段 |
|--------|------|----------|
| 情绪标签 | ✅ | `emotion` (happy / sad / disgust / neutral / fear) |
| 情绪编码 | ✅ | `emotion_code` (0–4) |
| Session | ✅ | `session` (session_1 / session_2 / session_3) |
| 电极坐标 | ✅ | 62 通道 10-20 坐标 |
| 信号单位 | ✅ | µV |

**⚠️ 缺失：**
- **视频片段信息** — 每个 trial 对应的视频刺激 ID 未存储
- **EOG 通道** — VEO/HEO 被完全排除，未作为 EOG 类型导入
- 已在 `importer_audit.md` 中记录

---

### 3.5 Zuco 2.0 (`zuco2.py`)

| 检查项 | 状态 | 注解字段 |
|--------|------|----------|
| 文本 ID | ✅ | `text_id` (TSR1–TSR5) |
| 句子索引 | ✅ | `sentence_index` |
| 任务类型 | ✅ | `task` = "reading" |
| 词数 | ✅ | `n_words`（如果有 wordbounds） |
| 电极坐标 | ✅ | 3D 坐标 from mat |
| 信号单位 | ✅ | µV（config 声明） |

**⚠️ 缺失：**
- **Automagic 质量分数** — `qualityScores`, `finalBadChans` 未提取
- **参考类型** — `EEG.ref` 未存储
- 已在 `importer_audit.md` 中记录

---

### 3.6 CCEP-COREG (`ccep_bids_npy.py`)

| 检查项 | 状态 | 注解字段 |
|--------|------|----------|
| 刺激描述 | ✅ | `trial_type` (文本) |
| 刺激触点 | ✅ | `stim_contact` |
| 刺激强度 | ✅ | `stim_intensity_ma` |
| 刺激组织 | ✅ | `stim_tissue` (grey / white) |
| 模态标记 | ✅ | `modality` (eeg / ieeg) |
| 配对关系 | ✅ | `AtomRelation.cross_modal_paired_run` |
| 电极坐标 | ✅ | MNI 坐标含解剖标签 |
| 多模态 | ✅ | EEG + sEEG 同时导入 |

**问题：** 无。此导入器是最完善的。

---

### 3.7 ChineseEEG-2 (`chinese_eeg2.py`)

| 检查项 | 状态 | 注解字段 |
|--------|------|----------|
| 小说名 | ✅ | `novel` (littleprince / garnettdream) |
| 章节号 | ✅ | `chapter` |
| 重复次数 | ✅ | `repetition` |
| 句子索引 | ✅ | `sentence_index` |
| 电极坐标 | ✅ | CapTrak 128ch |
| 任务区分 | ✅ | `task="listening"` / `task="reading"` 通过构造参数控制 |
| 信号单位 | ✅ | V（BrainVision 自动检测） |

**⚠️ 缺失：**
- **句子文本内容** — 数据集提供了 stimuli 文件夹中的文本，但未关联到每个 epoch
- **`_write_channels_json` 已修复**（之前有 bug）

---

### 3.8 ChineseEEG (`chinese_eeg.py`) — 新增 ✨

| 检查项 | 状态 | 注解字段 |
|--------|------|----------|
| 小说名 | ✅ | `novel` (LittlePrince / GarnettDream) |
| 章节号 | ✅ | `chapter` |
| 句子索引 | ✅ | `sentence_index` |
| 电极坐标 | ✅ | CapTrak 128ch |
| vhdr 文件名 bug 修复 | ✅ | `_fix_vhdr_references()` 自动修补 |
| 信号单位 | ✅ | V（BrainVision 自动检测） |
| 测试覆盖 | ✅ | 35 个 E2E 测试全部通过 |

**⚠️ 缺失：**
- 与 ChineseEEG-2 相同：**句子文本内容**未关联

---

### 3.9 OpenBMI (`openbmi.py`)

| 检查项 | 状态 | 注解字段 |
|--------|------|----------|
| MI 类别 | ✅ | `mi_class` (right_hand / left_hand) |
| ERP 类别 | ✅ | `erp_class` (target / nontarget) |
| SSVEP 类别 | ✅ | `ssvep_class` (5.45Hz–11.25Hz / 12Hz) |
| SSVEP 频率 | ✅ | `ssvep_frequency` (数值) |
| 训练/测试分割 | ✅ | `split` (train / test) |
| 信号单位 | ✅ | µV（config 声明） |

**⚠️ 缺失：**
- **电极坐标** — 62 通道名称已知但 3D 坐标未从标准蒙太奇加载
- **EMG 通道** — 存在于 .mat 中但未导入（设计选择）

---

### 3.10 KUL / DTU AAD (`aad_mat.py`)

| 检查项 | 状态 | 注解字段 |
|--------|------|----------|
| 注意力方向 | ✅ | `attended_ear` / `attended_speaker` / `attended_track` |
| 实验编号 | ✅ | `experiment` |
| 段落 | ✅ | `part` |
| 重复次数 | ✅ | `repetition` |
| 条件 | ✅ | `condition` |
| 刺激信息 | ✅ | `stimuli` (文本) |
| 音频包络 | ✅ | `ContinuousAnnotation` (audio_envelope_A / B) |

**问题：** 此导入器已很完善。

---

### 3.11 通用导入器

| 导入器 | 用途 | 状态 |
|--------|------|------|
| `mne_generic` | 任意 MNE 格式 (EDF/BDF/GDF/FIF/CNT/MFF) | ✅ 可用，通用分窗 |
| `bids` | BIDS 目录自动遍历 | ✅ 可用 |
| `eeglab` | .set/.fdt 文件 | ✅ 可用 |
| `mat` | 通用 .mat 文件 | ✅ 可用 |
| `moabb_bridge` | 通过 MOABB 库桥接 30+ 数据集 | ✅ 可用 |

---

## 四、需求差距分析

### 4.1 ❌ 未完成的需求

| 需求 | 严重度 | 说明 |
|------|--------|------|
| **README 缺少 ChineseEEG + OpenBMI 表格行** | 低 | 文档过时，需更新 |
| **SEED-V 视频刺激关联** | 中 | 每个 trial 的视频 ID 可从 label 文件推导，但未存为 `StimulusRefAnnotation` |
| **ChineseEEG/EEG2 句子文本** | 中 | stimuli 目录中有文本，但未逐句关联到每个 Atom |
| **Zuco 质量分数** | 低 | Automagic 质量指标未提取 |
| **OpenBMI 电极坐标** | 低 | 通道名为标准 10-20，可从内置蒙太奇自动补充 |
| **config-only 数据集无专用导入器** | 中 | `inner_speech`, `lee2019_mi`, `p300_speller`, `ssvep_benchmark` 仅有 YAML 但依赖 `moabb_bridge` |

### 4.2 ⚠️ 自适应识别与统一问题

| 问题 | 现状 | 建议 |
|------|------|------|
| **字段名不统一** | 各导入器用不同标签名（如 `mi_class` vs `motor_class` vs `emotion`） | 设计上是正确的——不同范式有不同语义。`quick.py` 的 `_KNOWN_LABEL_FIELDS` 已做自动推断 |
| **信号单位差异** | BrainVision=V, EDF=V, .mat=µV, .cnt=µV | `unit_standardizer.py` 在组装时统一 ✅ |
| **采样率差异** | 160–1000 Hz | `resampler.py` 在组装时统一 ✅ |
| **通道数差异** | 25–128 通道 | `channel_mapper.py` 在组装时统一 ✅ |
| **Atom 类型差异** | TRIAL_EPOCH / EVENT_EPOCH / CONTINUOUS_SEGMENT | 设计上允许混合，QueryBuilder 可按 `atom_type` 筛选 ✅ |
| **ChineseEEG vhdr 拼写错误** | GranettDream vs GarnettDream | `_fix_vhdr_references()` 自动修补 ✅ |
| **电极坐标系不一** | CapTrak / MNI / RAS | `ElectrodeLocation.coordinate_system` 已标记 ✅ |

### 4.3 ✅ 已良好实现的设计目标

| 设计目标 | 实现状态 |
|----------|----------|
| 原子自包含——每个 Atom 携带完整上下文 | ✅ 信号引用 + 时间 + 通道 + 注解 + 溯源 + 质量 |
| 格式无关——统一数据模型 | ✅ .mat, .edf, .set, .vhdr, .cnt, .npy, BIDS 全支持 |
| 可复现——处理溯源链 | ✅ `ProcessingHistory` 含每步 hash |
| 跨数据集训练 | ✅ `multiload()` → 统一通道/采样率/单位 |
| 多模态支持 | ✅ EEG + sEEG 配对导入 + 分模态组装 |
| 灵活查询 | ✅ SQLite 后端 + 复合条件查询 |
| 增量导入 | ✅ Pool 可追加导入，无需重建 |

---

## 五、建议优先修复项

### P0（应立即修复）

1. **更新 README 数据集表格** — 添加 ChineseEEG、OpenBMI (MI/ERP/SSVEP) 行
2. **信号 flat-line 阈值调整** — 当前 `std < 0.01` 对 V 单位数据几乎全部报警，应改为单位感知阈值

### P1（短期完善）

3. **OpenBMI 补充电极坐标** — 从标准 10-20 蒙太奇自动映射 62 通道坐标
4. **SEED-V 补充视频刺激 ID** — 每个 trial 添加 `StimulusRefAnnotation`
5. **ChineseEEG/EEG-2 关联句子文本** — 解析 stimuli 目录，为每个 sentence epoch 添加 `TextAnnotation`

### P2（中期增强）

6. **Zuco 质量分数** — 提取 Automagic qualityScores
7. **config-only 数据集** — 为 `inner_speech`, `lee2019_mi` 等编写测试验证 MOABB 桥接路径
8. **flat-line 验证器** — 支持按数据集/单位自动调整阈值

---

## 六、数据集支持总览矩阵

| 数据集 | 导入器 | 任务配置 | E2E 测试 | 核心标签 | 电极坐标 | quick.py | 状态 |
|--------|--------|----------|----------|----------|----------|----------|------|
| BCI IV 2a | ✅ | ✅ | ✅ | ✅ mi_class | ✅ | ✅ | 🟢 完善 |
| PhysioNet MI | ✅ | ✅ | ✅ | ✅ mi_class | ✅ | ✅ | 🟢 完善 |
| SEED-V | ✅ | ✅ | ✅ | ✅ emotion + video_clip | ✅ | ✅ | � 完善 |
| Zuco 2.0 | ✅ | ✅ | ✅ | ✅ text_id + quality_score | ✅ | ✅ | � 完善 |
| CCEP-COREG | ✅ | ✅ | ✅ | ✅ stim_pair | ✅ MNI | ✅ | 🟢 完善 |
| ChineseEEG-2 | ✅ | ✅ | ✅ | ✅ sentence_index | ✅ | ✅ | 🟡 无逐句分段文本 |
| **ChineseEEG** | ✅ | ✅ | ✅ | ✅ sentence_index + sentence_text | ✅ | ✅ | � 完善 |
| OpenBMI MI | ✅ | ✅ | ✅ | ✅ mi_class | ✅ standard_1005 | ✅ | � 完善 |
| OpenBMI ERP | ✅ | ✅ | ✅ | ✅ erp_class | ✅ standard_1005 | ✅ | � 完善 |
| OpenBMI SSVEP | ✅ | ✅ | ✅ | ✅ ssvep_class | ✅ standard_1005 | ✅ | � 完善 |
| KUL AAD | ✅ | ✅ | ✅ | ✅ attended_ear | ✅ | ✅ | 🟢 完善 |
| DTU AAD | ✅ | ✅ | ✅ | ✅ attended_speaker | ✅ | ✅ | 🟢 完善 |
| Inner Speech | ❌ moabb | ✅ | ❌ | — | — | ⚠️ | 🔴 仅配置 |
| Lee2019 MI | ❌ moabb | ✅ | ❌ | — | — | ⚠️ | 🔴 仅配置 |
| P300 Speller | ❌ moabb | ✅ | ❌ | — | — | ⚠️ | 🔴 仅配置 |
| SSVEP Benchmark | ❌ moabb | ✅ | ❌ | — | — | ⚠️ | 🔴 仅配置 |
| 通用 (MNE/BIDS/EEGLAB/MAT) | ✅ | N/A | ✅ | 自动 | 自动 | ✅ | 🟢 |

### 数据来源标记规范

所有非直接从数据文件读取的注解均通过 `custom_fields.source` 标明来源：

| source 值 | 含义 | 示例 |
|-----------|------|------|
| `dataset_file` | 从数据集附带的文件中读取 | ChineseEEG 句子文本 (xlsx) |
| `protocol_defined` | 根据实验协议的固定映射推断 | SEED-V video_clip ID |
| `standard_1005_montage` | 从 MNE 标准 10-05 蒙太奇推断 | OpenBMI 电极坐标 |
| `automagic` | 从 Automagic 预处理工具输出提取 | Zuco 2.0 质量分数 |

---

## 七、总结

**NeuroAtom v0.1.0 已实现其核心设计目标：**

- ✅ 11 个专用数据集导入器 + 4 个通用导入器 = **覆盖 15+ 种数据格式**
- ✅ 完整的 Import → Index → Query → Assemble → DataLoader 管线
- ✅ 自动统一通道布局、采样率、信号单位
- ✅ 7 种注解类型覆盖绝大多数 EEG 实验范式
- ✅ 多模态（EEG + sEEG）和联邦池支持
- ✅ 47 个测试文件，含每个数据集的 E2E 测试
- ✅ 所有推断/给定的元数据均通过 `custom_fields.source` 标记来源

**主要差距在"丰富度"而非"正确性"：**
- 核心任务标签全部已提取，辅助元数据已补齐（v0.1.1 修复）
- ChineseEEG-2 缺少逐句分段文本（数据集未提供 per-run 文本文件）
- 4 个仅有 YAML 配置的数据集需要编写专用导入器或验证 MOABB 路径

**v0.1.2 边界情况加固（详见 `docs/edge_case_audit.md`）：**
- ✅ 修复 Zuco2 `session_id` 作用域 bug（P0 — 运行时 NameError）
- ✅ 修复 Zuco2 零长度 epoch 防护
- ✅ 修复 OpenBMI 通道名数组维度防御（1D/2D 兼容）
- ✅ 新增 `neuroatom/utils/mat_compat.py`：.mat v5/v7.3 格式检测 + 清晰报错
- ✅ 新增 `validate_sampling_rate()`：采样率合法性检查（集成到 4 个导入器）
- ✅ `validate_signal()` 增加 `np.isinf` 检测
- ✅ ChineseEEG 通道数不匹配防护（channels.tsv vs 实际数据）
- ✅ 28 个专用边界测试（`tests/test_importer_edge_cases.py`）
