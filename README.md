# NR/LTE Handover Simulation — KTX 고속철도 (godeokhs 샘플)

field에서 녹화한 **KTX 이동 GPS 궤적** 위에서 3GPP 핸드오버 상태머신(FSM)을 돌려
**핸드오버(HO) · 무선링크실패(RLF) · 핸드오버 실패(HOF)** 를 재현·관찰하는 시뮬레이터입니다.

> 이 저장소는 광명~천안아산(godeokhs) 구간 샘플 데이터로 바로 실행해볼 수 있게 구성했습니다.

---

## 1. 사용법 (준비물)

- **Python 3.10+**
- **최소 설치 (권장, statistical 모드)** — 무거운 패키지 없이 바로 실행:
  ```bash
  pip install numpy pandas
  ```
- **전체 설치 (sionna_rt 레이트레이싱까지 재현)** — TensorFlow·Sionna 포함(무거움):
  ```bash
  pip install -r requirements.txt
  ```

### 포함 파일
| 파일 | 설명 |
|---|---|
| `script/run_simulation.py` | 실행 진입점 |
| `src/` | 시뮬레이터 코어 (FSM·채널·측정) — 서드파티 의존성은 **numpy·pandas** 뿐 |
| `requirements.txt` | 전체 재현용 의존성 목록 |
| **[A] statistical 세트** | |
| `enb_coordinates_converted_godeokhs_sample.csv` | 기지국(gNB) 100개 샘플 (godeokhs 프레임) |
| `ktx_ue_coordinates_godeokhs.csv` | UE 궤적 8개(=열차 8편성), 10ms (godeokhs 프레임) |
| **[B] sionna_rt 세트** | |
| `enb_coordinates_converted_goduck_sample.csv` | gNB 575개 샘플 (goduck **scene 프레임**) |
| `ktx_ue_coordinates.csv` | UE 궤적 33개 (goduck **scene 프레임**) |
| `railway_scene.xml`, `meshes/` | RT scene 지오메트리 (goduck 구역) |

---

## 2. 어떤 핸드오버를 지원하나

3GPP TS 38.331 / 38.300 / 38.133 기반. 서빙셀 품질로 트리거되는 측정 이벤트:

| 종류 | 이벤트 | 설명 |
|---|---|---|
| **Intra-frequency HO** | **A3** | 같은 주파수 이웃이 서빙보다 offset+hysteresis 이상 좋을 때 (NR 3.5G ↔ NR) |
| **Inter-frequency HO** | **A5** | 서빙이 임계1 아래 + 다른주파수 이웃이 임계2 위 (LTE inter-freq) |
| **Inter-RAT HO** | **B1 / B2** | 타 RAT(LTE 0.9G·1.8G) 이웃이 임계 위 — NR ↔ LTE 전환 |
| (보조 트리거) | **A2** | 서빙 품질 열화 감지 |

핸드오버 실행·실패 처리:
- **RLF**: N310/T310(out-of-sync) → RLF, T311 복구, T304 HO 실행 타이머
- **HOF 5종 분류** (TS 38.300 §15.5): ① Too Late ② Too Early ③ Wrong Cell ④ Ping-Pong ⑤ T304 Expiry

---

## 3. 어떻게 수행하나 (순서)

### Step 1 — 설치
```bash
pip install numpy pandas
```

### Step 2 — 실행 (statistical 모드)
```bash
python script/run_simulation.py --channel-model statistical \
  --gnb-csv enb_coordinates_converted_godeokhs_sample.csv \
  --ue-csv  ktx_ue_coordinates_godeokhs.csv \
  --ue-subset 0 --duration 60 \
  --output-dir out
```
- `--ue-subset` : 어느 UE(열차)를 볼지. **0~7** (파일당 1편성) 중 선택. 콤마로 여러 개 가능.
- `--duration`  : 몇 초 구간을 돌릴지 (초). 생략 시 전체 궤적.

### Step 3 — 결과 확인 (`out/`)
| 파일 | 내용 |
|---|---|
| `events.csv` | HO_START/COMPLETE, RLF, 재연결, HOF 이벤트 시각 로그 |
| `detailed_log_ue0.csv` | 매 틱 RSRP/SINR/RSRQ, top1~3 이웃, FSM 상태 |
| `simulation_report.txt` | 요약 통계 + HOF 분류 결과 |

---

## 채널 모드 2가지 — **세트를 섞지 마세요**

각 모드는 **좌표 프레임이 맞는 데이터 세트**를 써야 합니다.

| 모드 | 채널 | 필요 | 세트 |
|---|---|---|---|
| **`statistical`** (권장·가벼움) | 3GPP TR 38.901 통계식 | numpy·pandas | **[A]** godeokhs (scene 불필요) |
| `sionna_rt` | 레이트레이싱 | Sionna+TF + scene + meshes | **[B]** goduck (scene 정합) |

### sionna_rt 실행 (세트 B)
```bash
pip install -r requirements.txt          # TensorFlow + Sionna 포함
python script/run_simulation.py --channel-model sionna_rt \
  --scene-path railway_scene.xml \
  --gnb-csv enb_coordinates_converted_goduck_sample.csv \
  --ue-csv  ktx_ue_coordinates.csv \
  --ue-subset 0 --duration 30 --output-dir out_rt
```

> [!WARNING] 좌표 프레임 규칙
> - **[A] godeokhs 데이터 ↔ statistical** (scene 없음). godeokhs 데이터를 scene과 함께
>   쓰면 origin이 달라 **어긋납니다.**
> - **[B] goduck 데이터 ↔ sionna_rt + railway_scene.xml** (같은 scene 프레임 → 정합).
> - 즉 **A 데이터로 sionna_rt, B 데이터로 statistical** 처럼 교차하지 마세요.

> [!NOTE] sionna_rt 샘플 gNB 위치는 근사값
> `enb_coordinates_converted_goduck_sample.csv`는 익명화를 위해 실제 좌표를 **±10m 랜덤
> 이동**했습니다. scene 규모(km) 대비 작아 데모용으론 무방하나, 정밀 RT가 필요하면
> 실제 좌표(비공개 원본)를 써야 합니다.

---

## 데이터 메모
- **기지국 CSV (양쪽 모두 익명화)**: 실제 망 데이터를 공개 가능하도록 gnb_id 1부터 재번호,
  좌표 ±10m·안테나 파라미터 ±10 랜덤화, PCI는 gnb당 3자리 랜덤(같은 gnb=같은 PCI),
  `is_hsr_cell` 유지.
  - `_godeokhs_sample.csv`: 100 gnb (godeokhs 프레임, statistical용)
  - `_goduck_sample.csv`: 575 gnb (goduck scene 프레임, sionna_rt용 — scene 커버리지 위해 전체 유지)
- **UE 궤적 CSV**: field GPS를 scene-local `x, y, z`(m)로 변환.
  - `ktx_ue_coordinates_godeokhs.csv`: 광명~천안아산 8편성, WGS84→EPSG:5179→godeokhs origin,
    10ms, GPS 노이즈 제거(실제 fix만 + 이상치 게이트 + Savitzky-Golay).
  - `ktx_ue_coordinates.csv`: goduck scene 프레임 33 UE (기존 데이터, scene과 정합).
