# macgen-iv Sub-Phase Profiling Report

## LTX2.3 (distilled-mlx) 상세 프로파일링

### 모델 스펙
- 48 transformer blocks, 4096 hidden dim, 32 heads, 128 dim_head
- Sequence length: 9 * H/8 * W/8 (256x256 → 2304, 512x512 → 9216)
- FFN: 4096 → 16384 → 4096 (4x expansion, GELU)
- Weights: bfloat16

### Denoise Step 분해 (per block, 256x256 기준)

```
Self-Attention (52%)
├── Q projection (4096→4096)      10.3ms   7.5%
├── K projection (4096→4096)      10.4ms   7.6%
├── V projection (4096→4096)      10.4ms   7.6%
├── Q RMSNorm                     0.7ms    0.5%
├── K RMSNorm                     0.7ms    0.5%
├── RoPE apply (Q+K)              7.6ms    5.5%
├── SDPA (Q*K^T * V)             86.1ms   62.5%  ← ★★★ 최대 병목
├── Gate (sigmoid)                0.6ms    0.4%
└── Output proj (gate+linear)    10.9ms    7.9%

FeedForward (34%)
├── proj_in (4096→16384)          ~7ms
├── GELU activation               ~1ms
└── proj_out (16384→4096)         ~7ms

Cross-Attention (10-13%)
AdaLN                            0.3ms    0.1%
```

### Block당 총 시간
- Block 0-7:   avg 210ms (빠름, 초기)
- Block 8-23:  avg 235ms
- Block 24-47: avg 260ms (느림)

### 512x512 예측
- SDPA O(n²): 86ms → ~1377ms/block (16x tokens)
- QKV proj O(n): 31ms → ~125ms/block
- 예상 self_attn/block: ~964ms
- 예상 48 blocks: ~46s/step × 4 steps = ~184s (실제 230s/step 측정과 부합)

---

## Wan2.2 (ti2v5b-comfy-mlx) 상세 프로파일링

### 모델 스펙
- 30 transformer blocks
- Latent: (48, 3, H/16, W/16) → 256x256에서 (48, 3, 16, 16)
- VAE: Wan22VAEDecoder, float32 강제, 48-dim latent

### Phase별 시간 (512x512, 9f, 4 steps)
```
model_load:     7.8s   (22.8%)
text_encoder:   5.7s   (16.6%)
denoise_total:  4.2s   (12.3%)  ← 매우 빠름 (30 blocks, 작은 latent)
vae_decode:     2.7s   (7.9%)   ← 이전 16.5s는 콜드 스타트
total:         ~34s
```

### VAE Decode 세분화
- Weight load (disk): ~0ms (캐시됨)
- bf16→fp32 cast: 82ms
- VAE forward (256x256): 843ms
- VAE forward (512x512): 2.7s
- 512x512에서 전체 대비 ~8%로 주요 병목 아님

---

## 최적화 실험 결과

### 1. mx.compile (t_sub02)
- **결과: 실효 없음 (0.94x, 오히려 느림)**
- 원인: mx.compile이 Modality/TransformerArgs dataclass를 인자로 받을 수 없어 block-level compile 불가
- SDPA만 compile해도 오버헤드가 이익을 상회
- MLX SDPA는 이미 내부적으로 최적화됨

### 2. Sparse/Block Attention (t_sub03)
- Sliding window: **0.65x (느림)** — 마스크 생성 오버헤드
- Block-diagonal 256: **5.1x 빠름** — 품질 저하 가능
- Block-diagonal 128: **3.5x 빠름**
- Block-diagonal 64:  **3.7x 빠름**
- **결론: block-diagonal은 속도 개선 확실하나 품질 평가 필수**
- SDPA가 62%를 차지하므로 block attention 적용 시 최대 62% × 5.1x = 3.2x 전체 개선 가능

### 3. FFN Quantization (t_sub04)
- int8: **0.90x (느림)**, 품질 붕괴
- int4: **0.93x (느림)**, 품질 붕괴
- **결론: 실효 없음** — MLX QuantizedLinear는 Apple Silicon GPU에서 bf16 대비 이점 없음
- bfloat16 → int8/4 변환에서 정밀도 손실 심함

---

## 최적화 로드맵 (우선순위)

### ★★★ 최우선: SDPA 병목 해결
SDPA가 self_attn의 62.5% = 전체 denoise의 ~32.5% 차지

1. **Block-diagonal attention (품질 평가 포함)**
   - 3.5~5.1x SDPA 개선 → 전체 ~2x 개선 가능
   - 품질 평가: FID/KVD 또는 시각 비교 필요
   - block_size=256이 속도/품질 최적 밸런스 예상

2. **Tiled/Sequential attention**
   - 512x512에서 seq=9216, 1024 단위로 타일링
   - 메모리 절감 + 속도 개선 기대

### ★★ 차선: 해상도 스케일링 대응
- 256→512: 4x tokens → 16x SDPA 비용
- Progressive upscaling: 저해상도로 denoise 후 고해상도 refine
- Latent-space 해상도 조절 (H/8 대신 H/16 latent)

### ★ 장기: Custom Metal Kernel
- SDPA를 위한 전용 Metal compute kernel
- MLX 기본 SDPA가 3D 비디오 시퀀스에 최적화되지 않았을 가능성
- 현재 근거로는 프로파일링 데이터만으로는 불충분, Metal Instruments 분석 필요

### 기각된 접근
- ~~mx.compile: dataclass 한계로 불가~~
- ~~FFN quantization: 품질 붕괴 + 속도 악화~~
- ~~Sliding window attention: 마스크 오버헤드로 느림~~
