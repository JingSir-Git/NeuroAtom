# NeuroAtom 导入器边界情况审计

> 审查日期：2026-04-27  
> 修复日期：2026-04-27
> 范围：所有 18 个导入器的鲁棒性与通用性

---

## 一、已知数据路径汇总

| 数据集 | 环境变量 | 实际路径 |
|--------|----------|----------|
| BCI IV 2a | `NEUROATOM_BCI_IV_2A_DIR` | `C:\Data\BCI_Competition` |
| PhysioNet MI | `NEUROATOM_PHYSIONET_DIR` | *(未配置)* |
| OpenBMI | `NEUROATOM_OPENBMI_DIR` | `\\wsqlab\share\JCH\OpenBMI` |
| SEED-V | `NEUROATOM_SEEDV_DIR` | *(未配置)* |
| Zuco 2.0 | `NEUROATOM_ZUCO2_DIR` | *(未配置)* |
| CCEP-COREG | `NEUROATOM_CCEP_DIR` | *(未配置)* |
| KUL AAD | `NEUROATOM_KUL_DIR` | *(未配置，测试期望 `C:\Data\KUL`)* |
| DTU AAD | `NEUROATOM_DTU_DIR` | *(未配置)* |
| ChineseEEG | — | `\\wsqlab\ugreen\Language\25+ChineseEEG...` |
| ChineseEEG-2 | — | `\\wsqlab\ugreen\Language\6+ChineseEEG-2` |

---

## 二、跨导入器通用边界问题

### EC-01: 通道数不匹配（channels.tsv vs 实际文件）

**现状：**
- `ccep_bids_npy.py` ✅ 已处理 — 使用 `min()` 对齐
- `chinese_eeg.py` ⚠️ 未处理 — 如果 `channels.tsv` 列出的通道数与 MNE 读到的不同，会在 `mne.pick_channels` 处报错
- `chinese_eeg2.py` ⚠️ 同上
- `bids.py` ⚠️ 委托给 `MNEGenericImporter`，未做校验

**风险等级：** 中 — 正规 BIDS 数据集很少出现此问题，但自行整理的数据可能有

**建议修复：** 在所有 BIDS 导入器的 channel picking 前增加 `len(ch_infos) vs len(raw.ch_names)` 校验

---

### EC-02: 空 epoch / 零长度试次

**现状：**
- `chinese_eeg.py` ✅ `if duration <= 0: continue`
- `physionet_mi.py` ✅ `if end_sample > data.shape[1]: continue`
- `seed_v.py` ✅ `if end_sample > total_samples: clip`
- `openbmi.py` ✅ epoch 长度由 MATLAB struct 的 `smt` shape 保证
- `zuco2.py` ⚠️ 未检查 `n_sent_samples <= 0` — 如果两个 `10` 事件时间戳相同，会产生零长度 epoch

**风险等级：** 低 — 但应防御性编程

**建议修复：** `zuco2.py` 的 `_sentence_epochs` 函数增加 `if end > onset` 过滤

---

### EC-03: NaN / Inf 信号值

**现状：**
- `validate_signal()` ✅ 已检查 NaN 比例和 flat-line
- 但：未检查 `Inf` 值 — 某些 EDF 文件在通道断线时可能产生 Inf

**风险等级：** 低

**建议修复：** `validate_signal()` 增加 `np.isinf()` 检查

---

### EC-04: Unicode / 非 ASCII 路径

**现状：**
- Windows UNC 路径 (`\\wsqlab\...`) ✅ 已验证可用
- 中文目录名 (`25+ChineseEEG...`) ✅ 已验证可用
- `mne.io.read_raw*` 接受 `str(path)` — ✅ 兼容 Unicode
- `h5py.File(str(path))` — ✅ 兼容 Unicode
- `openpyxl` — ✅ 兼容 Unicode

**风险等级：** 低 — Python 3 默认 Unicode

---

### EC-05: events.tsv 缺失或格式异常

**现状：**
- `chinese_eeg.py` ✅ `if rec.get("events_tsv") is None: return empty_result`
- `chinese_eeg2.py` ✅ 同上
- `bids.py` ⚠️ 传递 `events_tsv: None` 但未在 `import_run` 中做特殊处理
- `physionet_mi.py` ✅ 使用 MNE 的 annotations 提取事件，不依赖 TSV

**风险等级：** 中 — BIDS 通用导入器应更优雅地处理缺失的 events.tsv

---

### EC-06: MATLAB .mat v7.3 (HDF5) vs v5 (traditional) 格式

**现状：**
- `openbmi.py` — 使用 `scipy.io.loadmat()`，仅支持 v5
  - OpenBMI 官方文件确为 v5，但用户重保存为 v7.3 会崩溃
- `aad_mat.py` — 使用 `scipy.io.loadmat()`，同上
- `zuco2.py` — 使用 `h5py`，仅支持 v7.3
  - Zuco 2.0 确为 HDF5 格式

**风险等级：** 中 — 用户可能误传不同版本

**建议修复：** 在 `detect()` 阶段检查 .mat 文件头字节：v5 以 `MATLAB 5.0` 开头，v7.3 以 `\x89HDF` 开头，错误格式时给出清晰错误信息

---

### EC-07: 采样率为零或异常值

**现状：** 无导入器检查 `srate == 0` 或 `srate < 0`

**风险等级：** 低 — 但如果 MATLAB struct 中的 fs 字段损坏，会导致除零错误

**建议修复：** 在 `_extract_split` / `_import_session` 等入口处增加 `assert srate > 0`

---

### EC-08: 文件被占用 / 网络共享断连

**现状：**
- `chinese_eeg.py` ✅ `try...finally` 清理临时 .vhdr
- 其余导入器 ⚠️ 无超时机制或重试逻辑
- 网络共享路径 `\\wsqlab\...` 如果断连，`Path.exists()` 会挂起很久

**风险等级：** 中 — 实验室环境常见

**建议修复：** 暂不处理（Python 层面难以设超时），但应在文档中说明

---

## 三、各导入器专项问题

### OpenBMI (`openbmi.py`)

| 问题 | 状态 | 描述 |
|------|------|------|
| EC-O1 | ✅ **已修复** | `struct["chan"]` 已增加 ndim==1 / ndim==2 分支处理 |
| EC-O2 | ⚠️ | `y_dec` 可能包含 class=0（某些预处理管线会插入），未过滤 |
| EC-O3 | ✅ | `_parse_filename` 返回 None 则跳过非标准文件名 |

### SEED-V (`seed_v.py`)

| 问题 | 状态 | 描述 |
|------|------|------|
| EC-S1 | ⚠️ | `read_raw_cnt` 某些版本对 Neuroscan 66-ch 文件有兼容问题（MNE 版本依赖）|
| EC-S2 | ✅ | trial 超出录制范围已处理（clip to `total_samples`）|
| EC-S3 | ⚠️ | 如果 `_trial_ts` 中 session 键缺失，`_get_session_info` 抛 ValueError 但无指向具体文件的信息 |

### Zuco 2.0 (`zuco2.py`)

| 问题 | 状态 | 描述 |
|------|------|------|
| EC-Z1 | ⚠️ | `data_ds` shape 假设 `(samples, channels)` 但某些 EEGLAB 文件可能是 `(channels, samples)` |
| EC-Z2 | ⚠️ | `_h5_read_string` 对空字符串引用可能抛异常 |
| EC-Z3 | ✅ **已修复** | `_write_channels_json` 调用中 `session_id` 已改为 `f"ses-{text_id.lower()}"` |

### ChineseEEG (`chinese_eeg.py`)

| 问题 | 状态 | 描述 |
|------|------|------|
| EC-C1 | ✅ | `.vhdr` 内部路径修复 + 临时文件清理已完善 |
| EC-C2 | ⚠️ | `_load_sentence_texts` 中 `int(run_num)` 对 run="01" → 1，但如果 xlsx 文件名使用零填充 `run_01` 会找不到 |
| EC-C3 | ⚠️ | `openpyxl` 不在 `requirements.txt` 的必需依赖中（仅是可选增强）— 应在失败时给明确提示 |

### ChineseEEG-2 (`chinese_eeg2.py`)

| 问题 | 状态 | 描述 |
|------|------|------|
| EC-C2-1 | ⚠️ | `_parse_run_id` 解析 `run-110` 为 rep=1, chapter=10，但 `run-1100` 是否可能？需验证最大章节数 |
| EC-C2-2 | ⚠️ | 与 ChineseEEG 共享小说但无逐句文本 — 已记录在审计中 |

### AAD (`aad_mat.py`)

| 问题 | 状态 | 描述 |
|------|------|------|
| EC-A1 | ⚠️ | DTU raw: `int(v)` 假设 `event_values` 都是数字字符串，如果含字母会 ValueError |
| EC-A2 | ⚠️ | KUL: `trial_struct.TrialID` 直接取属性，如果单 trial 文件中 `trials` 不是数组会报错 |
| EC-A3 | ✅ | 格式检测使用启发式探测 (KUL vs DTU preproc vs DTU raw) |

### PhysioNet MI (`physionet_mi.py`)

| 问题 | 状态 | 描述 |
|------|------|------|
| EC-P1 | ✅ | EDF 文件缺失时优雅跳过 |
| EC-P2 | ✅ | epoch 超出数据范围时 `continue` |
| EC-P3 | ⚠️ | 假设 `T1/T2` 事件标签始终存在，如果 annotations 使用数字编码（某些 PhysioNet 镜像）则 `label_map` 为空 |

### 通用 BIDS (`bids.py`)

| 问题 | 状态 | 描述 |
|------|------|------|
| EC-B1 | ⚠️ | `dataset_description.json` 用 `open(..., encoding="utf-8")` 读取 — 如果文件不存在会 crash（虽然 `detect()` 检查了）|
| EC-B2 | ⚠️ | 无 session 目录时回退到 `sub-xx/eeg/`，但不处理 `sub-xx/ieeg/` (iEEG) |
| EC-B3 | ⚠️ | `_parse_bids_filename` 未处理 `acquisition` 实体 (acq-xxx) |

---

## 四、优先修复建议

### P0（立即修复 — 会导致 crash）

1. ~~**EC-Z3**: `zuco2.py` `session_id` 变量作用域~~ → ✅ 已修复
2. ~~**EC-02**: `zuco2.py` 零长度 epoch 防护~~ → ✅ 已修复
3. ~~**EC-O1**: `openbmi.py` 通道名数组维度防御~~ → ✅ 已修复

### P1（重要 — 影响通用性）

4. ~~**EC-06**: .mat v5/v7.3 格式检测~~ → ✅ 新增 `neuroatom/utils/mat_compat.py`
5. ~~**EC-01**: BIDS 导入器通道数不匹配防护~~ → ✅ `chinese_eeg.py` 已增加
6. ~~**EC-07**: 采样率合法性检查~~ → ✅ `validate_sampling_rate()` 已集成到 4 个导入器
7. ~~**EC-03**: `validate_signal()` 增加 Inf 检查~~ → ✅ 已增加 `np.isinf` 检查

### P2（改善体验）

8. **EC-A1**: DTU event value 类型防御
9. **EC-P3**: PhysioNet 事件标签兼容数字编码
10. **EC-C3**: `openpyxl` 缺失时的友好提示
11. **EC-B3**: BIDS 文件名解析增加 `acq-` 实体支持

---

## 五、测试覆盖建议

当前边界测试（`test_edge_cases.py`）仅覆盖 Federation / YAML / Pairing 等核心模块。
建议新增以下测试类别：

1. **TestImporterDetection**: 对每个导入器用错误格式文件测试 `detect()` 返回 False
2. **TestZeroLengthEpoch**: 构造含零长度 epoch 的合成数据测试
3. **TestChannelMismatch**: channels.tsv 与实际通道数不匹配
4. **TestMATVersionMismatch**: 用 v7.3 格式文件调用 scipy.io.loadmat 并验证错误信息
5. **TestEmptyEventsFile**: 空 events.tsv 的优雅处理
6. **TestSamplingRateZero**: 采样率为 0 的防御
