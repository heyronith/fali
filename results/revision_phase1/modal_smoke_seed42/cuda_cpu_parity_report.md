# CUDA vs CPU parity report

- CPU raw: `/Users/ronny/Desktop/Research/AI ALIGNMENT/FALI/results/revision_phase1/smoke/candidate_raw.csv`
- CUDA raw: `/Users/ronny/Desktop/Research/AI ALIGNMENT/FALI/results/revision_phase1/modal_smoke_seed42/candidate_raw.csv`
- Compared rows: 96 (12 candidates × N examples)

## Acceptance thresholds

- IO−S quantities: max abs error ≤ 1e-3
- Attention quantities: max abs error ≤ 1e-4

## Results

- max abs error (IO−S): **4.291534e-05** → PASS
- max abs error (attention): **1.579523e-06** → PASS
- overall: **PASS**

## Per-quantity

```
                  quantity    family  n  max_abs_error  mean_abs_error  max_rel_error  mean_rel_error
                         A       ios 96       0.000019    1.120567e-05       0.000009        0.000004
                     B_old       ios 96       0.000016    8.583069e-06       0.000012        0.000005
                         B       ios 96       0.000020    1.001358e-05       0.000009        0.000004
                       D_h       ios 96       0.000032    9.020170e-06       0.000019        0.000004
                       C_h       ios 96       0.000029    8.404255e-06       0.000034        0.000004
             intact_effect       ios 96       0.000035    8.801619e-06       0.005249        0.000179
        conditional_effect       ios 96       0.000042    1.160304e-05       0.004860        0.000178
          conditional_gain       ios 96       0.000043    1.406670e-05       0.025241        0.000577
                clean_attn attention 96       0.000001    2.468805e-07       0.000009        0.000001
   historical_primary_attn attention 96       0.000001    3.176180e-07       0.000008        0.000001
    corrected_primary_attn attention 96       0.000001    3.541300e-07       0.000009        0.000002
historical_attention_spike attention 96       0.000001    3.229452e-07       0.000043        0.000004
 corrected_attention_spike attention 96       0.000002    3.872919e-07       0.000139        0.000005
```

