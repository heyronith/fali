# Phase 2B/3B scientific summary

**Predefined verdict: PIVOT**

## Notes
- across-head seedmean pearson fali↔causal target=-0.3499954295042954, competing=0.24683322548864478
- predictive groups better=10 worse=1 tiny=1 / 12
- causal tracking failed despite some predictive signal

## FALI ↔ causal (across-head, seed-mean heads)
               level seed head_name      pair  pearson_r  pearson_p  spearman_r  spearman_p  n_compared  sign_agreement
across_head_seedmean  all       ALL    target  -0.349995   0.264735    0.312281    0.323051          10             0.8
across_head_seedmean  all       ALL competing   0.246833   0.439276    0.431579    0.161241          10             0.7

## Predictive CV summary (mean across folds)
                cv             target       feature_set    mean_r2    std_r2  mean_mae  std_mae  n_folds
leave_one_head_out   conditional_gain A_delta_attention  -8.744134 17.272628  0.438159 0.400142       12
leave_one_head_out   conditional_gain    B_delta_margin -20.236640 59.061572  0.435399 0.412106       12
leave_one_head_out   conditional_gain      C_raw_scores -23.546340 71.160190  0.443075 0.420174       12
leave_one_head_out   conditional_gain            D_fali  -6.675044 12.815325  0.423537 0.407298       12
leave_one_head_out conditional_effect A_delta_attention  -0.728963  1.260802  0.426945 0.290629       12
leave_one_head_out conditional_effect    B_delta_margin  -1.075941  1.707656  0.427596 0.279360       12
leave_one_head_out conditional_effect      C_raw_scores  -0.807081  1.512658  0.427100 0.304507       12
leave_one_head_out conditional_effect            D_fali  -0.800315  1.527855  0.425083 0.296321       12
leave_one_seed_out   conditional_gain A_delta_attention   0.290697  0.008542  0.339779 0.002457        4
leave_one_seed_out   conditional_gain    B_delta_margin   0.255029  0.005936  0.340590 0.003145        4
leave_one_seed_out   conditional_gain      C_raw_scores   0.258735  0.006378  0.336256 0.002888        4
leave_one_seed_out   conditional_gain            D_fali   0.335971  0.012875  0.318425 0.001834        4
leave_one_seed_out conditional_effect A_delta_attention   0.040751  0.005311  0.404504 0.002408        4
leave_one_seed_out conditional_effect    B_delta_margin   0.029453  0.002820  0.410609 0.002953        4
leave_one_seed_out conditional_effect      C_raw_scores   0.045229  0.002874  0.402232 0.003228        4
leave_one_seed_out conditional_effect            D_fali   0.063291  0.006700  0.401333 0.002690        4

## FALI vs baselines (mean fold deltas)
                cv             target          baseline  delta_r2_fali_minus_base  delta_mae_fali_minus_base
leave_one_head_out   conditional_gain A_delta_attention                  2.069090                  -0.014622
leave_one_head_out   conditional_gain    B_delta_margin                 13.561596                  -0.011862
leave_one_head_out   conditional_gain      C_raw_scores                 16.871296                  -0.019538
leave_one_head_out conditional_effect A_delta_attention                 -0.071353                  -0.001862
leave_one_head_out conditional_effect    B_delta_margin                  0.275626                  -0.002513
leave_one_head_out conditional_effect      C_raw_scores                  0.006766                  -0.002017
leave_one_seed_out   conditional_gain A_delta_attention                  0.045274                  -0.021354
leave_one_seed_out   conditional_gain    B_delta_margin                  0.080942                  -0.022165
leave_one_seed_out   conditional_gain      C_raw_scores                  0.077237                  -0.017831
leave_one_seed_out conditional_effect A_delta_attention                  0.022541                  -0.003170
leave_one_seed_out conditional_effect    B_delta_margin                  0.033838                  -0.009276
leave_one_seed_out conditional_effect      C_raw_scores                  0.018062                  -0.000899

## L9H7 scope
{
  "n": 800,
  "max_abs_delta_attention": 0.0,
  "max_abs_delta_target_score": 0.0,
  "max_abs_fali_target": 0.0,
  "max_abs_fali_competing": 0.0,
  "max_abs_causal_total": 0.0,
  "mean_conditional_effect": 0.4232658052444458,
  "mean_conditional_gain": 0.07203758239746091,
  "score_invariant_pass": true
}

## Rules reminder
- STRONG: tracks causal AND improves held-out functional prediction
- NARROW: tracks causal score mediation but does not beat simpler baselines for recruitment
- PIVOT: neither
