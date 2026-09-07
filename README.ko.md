# vecdiff

두 임베딩 인덱스 스냅샷을 diff해서 등급화된(graded), 증거 중심의 리포트를 냅니다 — **코드베이스 벡터 DB 마이그레이션**(모델 교체, 재청킹)과 **인덱스 rot 감사**(고아 청크, 중복)를 위해.

완전 로컬, 결정적(deterministic), 런타임 의존성은 `numpy` 하나. vecdiff는 원본 벡터 DB에 접속하지 않고, 네트워크도 사용하지 않습니다.

> 재임베딩에 대한 정석 조언은 "새 인덱스를 블루/그린으로 나란히 돌리고, 비교한 뒤 전환하라"입니다. 그런데 아무도 *비교* 단계를 도구화하지 않았죠 — 대부분 양쪽 인덱스에 쿼리 몇 개 던져보고 감으로 넘어갑니다. vecdiff가 바로 그 비교 단계를 기계화합니다.

```console
$ pip install vecdiff
$ vecdiff old-snapshot/ new-snapshot/ --gate
```

[English](README.md)

## 왜 필요한가

코드 임베딩 인덱스에는 만성 불안이 세 가지 있습니다:

1. **모델 교체** — 임베딩 공간끼리는 벡터 단위 비교가 불가능해서 "검색 품질이 살아남았나?"를 기계로 답할 수 없었습니다.
2. **재청킹 / 재색인** — *정확히 무엇이 바뀌었는가?* 에 대한 답이 아예 없었습니다.
3. **Rot** — 코드는 이동하는데 인덱스는 가만히 있습니다. 죽은 심볼을 가리키는 청크와 우발적 중복이 조용히 쌓입니다.

vecdiff는 이 셋을 같은 원시 연산으로 해결합니다: **인덱스-vs-인덱스 diff** + 청크 단위 헬스 체크.

## 빠른 시작

```console
# 1. 설치 (런타임 의존성은 numpy뿐)
pip install vecdiff        # 또는: uv tool install vecdiff

# 2. 데모 스냅샷 2개 생성 (결정적; B가 엉성한 재임베딩을 시뮬레이션)
python scripts/make_demo_snapshots.py /tmp/vecdemo

# 3. diff
vecdiff /tmp/vecdemo/snapA /tmp/vecdemo/snapB --markdown report.md --gate
echo $?
```

콘솔 리포트와 `report.md`를 얻고, `--gate`는 결과를 CI용 exit code로 바꿔줍니다 (아래 참고).

## 어떤 것을 검사하나

| # | 검사 | 잡아내는 것 |
|---|---|---|
| **N1** | 이웃 안정성 | 모델 교체 / 재색인 회귀: 청크별 top-k 이웃 집합을 스냅샷 간 비교 (Jaccard + rank inversion), heavy-loss 청크를 디렉터리별로 그룹화 |
| **N2** | 분포 통계 | 파이프라인 고장 조기 경보: norm 분포 이동, 극단적 norm 이상치, 차원 불일치 (하드 에러) |
| **N4** | 중복 | 재청킹 사고와 보일러플레이트 범람: 인덱스 내 cosine ≥ 임계값 쌍 |
| **N5** | 상수 벡터 | 파이프라인 버그 (캐시된 API 응답, 상수 fallback, 깨진 배치): 하나의 비트 동일 임베딩이 여러 청크 id에 재사용됨 |
| **Q1** (`--queries-a/b`) | 정준 쿼리 (지도 검사) | 실제 검색 트래픽이 보게 될 것: 동일한 쿼리 집합을 양쪽 인덱스에 통과시켜 쿼리별 top-k overlap + rank inversion |
| **N3** (`--paths-manifest`) | 고아 + 고스트 | Rot: 원본 파일이 더 이상 없는 청크 (파일 수준: 경로 리스트만으로 가능); 심볼 그래프 매니페스트(Swift/iOS는 [cartograph](https://github.com/ictechgy/cartograph), Kotlin/Android는 [kartograph](https://github.com/ictechgy/kartograph))를 쓰면 파일은 살아있지만 선언된 심볼이 사라진 청크(ghost)까지 |

### 신호 임계값

vecdiff는 등급화된 신호를 낼 뿐, "모델 B가 더 낫다" 같은 판정을 내리지 않습니다. 임계값은 모든 finding에 인라인으로 명시됩니다:

| 신호 | 초록 | 노랑 | 빨강 |
|---|---|---|---|
| N1 평균 이웃 Jaccard | ≥ 0.90 | ≥ 0.70 | < 0.70 |
| N1 heavy-loss 청크 (Jaccard ≤ 0.30, 즉 top-k의 ≥ 70% 상실) | < 2% | < 10% | ≥ 10% |
| N2 norm 평균 이동 A→B | ≤ 5% | ≤ 20% | > 20% |
| N2 극단 norm 이상치 (\|z\| > 3) | ≤ 1% | < 5% | ≥ 5% |
| N4 중복 쌍 (cosine ≥ 임계값) / n | 0 | < 1% | ≥ 1% |
| N5 최대 비트 동일 그룹 | < 5개 | ≥ 5개 | ≥ 인덱스의 5% |
| Q1 평균 쿼리 Jaccard | ≥ 0.90 | ≥ 0.70 | < 0.70 |
| Q1 heavy-loss 쿼리 (Jaccard ≤ 0.30) | < 2% | < 10% | ≥ 10% |
| N3 rot 청크 (고아 + 고스트) / n | 0 | < 5% | ≥ 5% |

N2 이상치 검사는 norm 분산이 ≈ 0이면 (초록으로 보고하되 이유를 인라인으로 명시) 건너뜁니다 — 예: pre-normalized 단위 벡터를 반환하는 임베더에서 z-score는 부동소수점 반올림 잡음일 뿐입니다.

`--gate` exit code: `0` 전부 초록, `1` 노랑 있음, `2` 빨강 있음. (참고: argparse 사용 오류 — 오타 낸 플래그 — 도 2로 종료합니다; 빨강 판정과 구분하려면 stderr를 확인하세요. 하드 에러 — 읽을 수 없는 스냅샷, 차원 불일치 — 는 3으로 종료합니다.)

### 샘플링

N1은 조회된 청크 기준으로 정확하지만, 기본적으로 공유 id를 샘플링합니다 (`--sample 0.2`, `--seed 0` — 같은 시드면 같은 샘플, 재현 가능한 리포트). 정확한 전수 실행은 `--full`; 비용은 O(queried × n × dim).

## 케이스 스터디 — 실제 코드 인덱스의 모델 교체

371청크 (실제 Python/Swift 코드 약 1.2만 줄), 동일한 청크를 `bge-small-en-v1.5` vs `all-MiniLM-L6-v2`로 임베딩:

- 평균 이웃 Jaccard **0.33**; 청크의 **42.6%** 가 top-10 이웃의 ≥ 70%를 잃음
- 손실은 디렉터리별로 집중 (예: 한 Swift 핵심 모듈 50청크 중 32, vecdiff 자체 59청크 중 22) — 전환 리뷰용 스팟체크 리스트
- N2/N4는 양쪽 파이프라인이 기계적으로 건강함을 확인 (스케일링 버그 없음, 중복 없음); gate exit **2** = 눈감고 전환하지 마라

전체 이야기 + 재현 가능한 명령: [docs/case_study](docs/case_study/README.md). 이 실행을 dogfooding하면서 도구의 실제 버그도 잡았습니다 (N2는 이제 pre-normalized 임베더에 대해 norm-outlier 검사를 이유와 함께 건너뜁니다).

## 스냅샷 포맷

*스냅샷*은 모델 독립적인 덤프입니다: 청크 id + float32 벡터 + 메타데이터 (`model`, `dim`, `chunk_paths`, `created_at`). 로딩 시 엄격하게 검증하여 중복 id, 차원/길이 불일치, 비유한 (NaN/inf) 벡터를 거부합니다 — 오염된 한 줄이 cosine top-k를 조용히 무작위로 만들 수 있으니, 리포트 중간이 아니라 로드 시점에 실패합니다.

| 어댑터 | 입력 | 비고 |
|---|---|---|
| `native` (내장) | `vectors.npy` + `meta.json` 디렉터리, 또는 자기서술적 `.npz` 하나 | 교환 포맷 — 한 번 export하면 영원히 diff |
| `jsonl` (내장) | `.jsonl` / `.ndjson` (선택적으로 `.gz`): 한 줄에 `{"id", "vector", "path"?, "symbols"?}`; 선택적 `<stem>.meta.json` 사이드카 | 만능 탈출구 — 어떤 벡터 DB도 클라이언트 코드 몇 줄로 이걸 뽑을 수 있음 |
| `sqlite` (내장) | `chunks(id TEXT PRIMARY KEY, vec BLOB)` float32 little-endian; 선택적 `meta(key, value)` 테이블 | 표준 라이브러리만 사용 |
| `faiss` (선택) | `.index` / `.faiss` | `pip install faiss-cpu` 필요; 벡터 복원이 가능한 flat 계열 인덱스만 |

포맷은 경로에서 자동 감지; `--format`으로 재정의 가능.

**그 외 모든 벡터 DB** (Qdrant, Chroma, LanceDB, pgvector, …): 해당 DB의 클라이언트로 JSONL 스냅샷을 덤프하거나, `snapshot_from_arrays(ids=..., vectors=..., model=...)`로 메모리에 만들면 됩니다 — vecdiff는 numpy만 쓰고 당신의 DB에 절대 접속하지 않습니다. 바로 실행 가능한 스니펫은 [docs/export_recipes.md](docs/export_recipes.md). 무엇보다 중요한 규칙 하나: 두 스냅샷에서 id가 안정적이어야 합니다 (N1이 id로 청크를 짝짓습니다) — 절대 행 번호를 id로 export하지 마세요.

## CI 마이그레이션 게이트

```yaml
# .github/workflows/reindex-gate.yml — 새 인덱스로 트래픽 전환 전에 실행
- run: vecdiff snapshots/blue/ snapshots/green/ --full --gate
  # exit 2 (빨강)가 전환 단계를 막음
```

vecdiff는 사람의 결정을 위한 증거를 모읍니다; 게이트는 그저 "아무도 안 봤다"를 불가능하게 만들 뿐입니다.

Exit code: `0` 전부 초록, `1` 노랑 있음, `2` 빨강 있음 — 하드 에러 (스냅샷 불량, 차원 불일치, I/O 실패)는 `3`으로 구분되어, CI가 "비교가 실패함"과 "비교가 안 된다고 말함"을 구별할 수 있습니다.

## 방법론 노트 & 정직성

- **크로스 모델 비교는 벡터 단위로 불가능합니다** — 임베딩 공간은 서로 무관합니다. 그래서 N1은 좌표 대신 *이웃 그래프 구조*를 비교합니다 (청크별 kNN Jaccard, Vectory 방식). 방법론의 선행 연구: [Vectory](https://github.com/pentoai/vectory) (pentoai) — ML 실험 추적 프레임에서 kNN-IoU를 확립; vecdiff는 이를 프로덕션 코드 인덱스 마이그레이션에 맞게 운영화하고 Vectory에 없는 헬스 체크를 더했습니다.
- **N1은 후보 풀 전체의 함수입니다**: 이웃 정체성은 인덱스 전체에 의존하므로, 청크를 제거하면 건드리지 않은 청크의 이웃 집합도 바뀝니다. 디렉터리별 heavy-loss 집중(`chunk_paths`)을 먼저 보세요.
- **N2는 조기 경보 시스템이지 품질 지표가 아닙니다**: norm 분포가 이동했다면 대개 파이프라인 스케일링이 바뀐 것이지, 검색 품질이 나빠진 게 아닙니다.
- **판단은 사람의 몫입니다.** vecdiff의 산출물은 *"14개 청크가 top-10 이웃의 ≥70%를 잃었고, src/auth에 집중됨 (11개)"* 같은 finding입니다 — 점수가 아닙니다.

## 차별점

| 인접 도구 | 차이 |
|---|---|
| [Vectory](https://github.com/pentoai/vectory) | kNN-IoU 방법을 확립 (인용함). ML *실험 추적* 툴킷이고 (SQLite + Elasticsearch 프레임, 이미지/IMDB 데모); vecdiff는 프로덕션 *인덱스 운영* 도구 — DB 어댑터, 경로 집중 heavy-loss finding, CI 게이트, rot 검사. |
| Ragas / RAG 평가 프레임워크 | 최종 태스크의 답변 품질을 평가; 인덱스-vs-인덱스 diff는 하지 않음. |
| MTEB | 모델용 벤치마크 리더보드; *당신의* 인덱스와는 무관. |
| 벤더 마이그레이션 가이드 / 듀얼 인덱스 블로그 | 패턴을 기술할 뿐, 비교 도구는 배포하지 않음. |
| 벡터 인덱스 시각화 도구 (zilliztech 등) | ANN 검색 내부를 시각화; 마이그레이션 판정이 아님. |

## 프라이버시

모든 것이 로컬에서 실행됩니다: 네트워크 호출 없음, 텔레메트리 없음, 스냅샷은 당신의 머신에만 있습니다. 리포트에는 청크 id, 경로, 유사도 수치가 포함되니 — repo에 커밋하기 전에 괜찮은지 확인하세요.

## 라이선스

Apache-2.0
