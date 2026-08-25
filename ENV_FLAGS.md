| flag | default | read at | read in |
|---|---|---|---|
| `ADVECT_BULK_PI` | `false` | `main.jl:944`, `mainBGonly.jl:833` | main.jl, mainBGonly.jl |
| `ADVECT_NUR` | `false` | `main.jl:927` | main.jl |
| `ADVECT_SHEAR_PI` | `false` | `main.jl:946`, `mainBGonly.jl:835` | main.jl, mainBGonly.jl |
| `ALPHA_FILTER_EPS` | `0.0` | `main.jl:920` | main.jl |
| `ALPHA_SMOOTH_LEN` | `0.0` | `main.jl:922` | main.jl |
| `AUTO_CFL` | `1` | `main.jl:866`, `mainBGonly.jl:776` | main.jl, mainBGonly.jl |
| `AXIS_PROJECT_NFIT` | `16` | `main.jl:926` | main.jl |
| `BACKGROUND_JLD2` | `normpath(joinpath(@__DIR__, "..", "LangevInMedium.jl", "src", "data", "Fluidum_MIS_HQ.jld2"` | `main2.jl:854` | main2.jl |
| `BULK_DT_COEFF` | `0.01` | `main.jl:918`, `mainBGonly.jl:818` | main.jl, mainBGonly.jl |
| `CFL` | `0.2`, `0.3`, `"0.2"` | `main.jl:858`, `main2.jl:872`, `main2BDNK.jl:537`, `mainBDNK.jl:465`, `mainBGonly.jl:768` | main.jl, main2.jl, main2BDNK.jl, mainBDNK.jl, mainBGonly.jl |
| `CFLTAU` | `"0.05"` | `main.jl:859`, `main.jl:860`, `main2.jl:873`, `main2BDNK.jl:538`, `mainBDNK.jl:466`, `mainBGonly.jl:769`, `mainBGonly.jl:770` | main.jl, main2.jl, main2BDNK.jl, mainBDNK.jl, mainBGonly.jl |
| `CFLTAU_MAX` | `0.2` | `main.jl:869`, `mainBGonly.jl:779` | main.jl, mainBGonly.jl |
| `CFL_MAX` | `0.5` | `main.jl:868`, `mainBGonly.jl:778` | main.jl, mainBGonly.jl |
| `CFL_SAFETY` | `0.9` | `main.jl:867`, `mainBGonly.jl:777` | main.jl, mainBGonly.jl |
| `CHARGE_MODE` | `"mis"` | `main.jl:912`, `mainDensityFrame.jl:37` | main.jl, mainDensityFrame.jl |
| `CHI` | `χ_SrE` | `main.jl:813`, `mainBGonly.jl:725` | main.jl, mainBGonly.jl |
| `COUPLE_ALPHA` | `true` | `main2.jl:884` | main2.jl |
| `DELTA_BULK_FACTOR` | `0.0` | `main.jl:934`, `mainBGonly.jl:824` | main.jl, mainBGonly.jl |
| `DELTA_N_FACTOR` | `0.0` | `main.jl:915` | main.jl |
| `DELTA_SHEAR_FACTOR` | `0.0` | `main.jl:935`, `mainBGonly.jl:825` | main.jl, mainBGonly.jl |
| `DIAG_ENV_FILE` | `"last_ic_diagnostics.env"` | `main.jl:797`, `mainBGonly.jl:710` | main.jl, mainBGonly.jl |
| `DIFFUSION_DRIVE` | `"alpha"` | `main.jl:910` | main.jl |
| `DIFF_DT_COEFF` | `0.01` | `main.jl:916` | main.jl |
| `DO_AXIS_PROJECT_NUR` | `false` | `main.jl:925` | main.jl |
| `DO_SOFT_PROJECT_NUR` | `true` | `main.jl:924` | main.jl |
| `DS_T` | `0.24`, `"0.24"` | `main.jl:908`, `main2.jl:876`, `main2BDNK.jl:541`, `mainBDNK.jl:464` | main.jl, main2.jl, main2BDNK.jl, mainBDNK.jl |
| `DUMP_DT` | `0.1`, `"0.1"` | `main.jl:870`, `main2.jl:874`, `main2BDNK.jl:539`, `mainBDNK.jl:468`, `mainBGonly.jl:780` | main.jl, main2.jl, main2BDNK.jl, mainBDNK.jl, mainBGonly.jl |
| `EMIN` | `E_FLOOR` | `main.jl:812`, `mainBGonly.jl:724` | main.jl, mainBGonly.jl |
| `ENABLE_BULK` | `0`, `"0"` | `main.jl:907`, `mainBDNK.jl:482`, `mainBGonly.jl:816` | main.jl, mainBDNK.jl, mainBGonly.jl |
| `ENABLE_DIFF` | `0` | `main.jl:905` | main.jl |
| `ENABLE_SHEAR` | `0`, `"0"` | `main.jl:906`, `mainBDNK.jl:481`, `mainBGonly.jl:815` | main.jl, mainBDNK.jl, mainBGonly.jl |
| `EOS` | `"latticehrg"` | `main.jl:962`, `main2.jl:865`, `main2BDNK.jl:530`, `mainBGonly.jl:851` | main.jl, main2.jl, main2BDNK.jl, mainBGonly.jl |
| `EPS_NU` | `"kappa"` | `main2BDNK.jl:545`, `mainBDNK.jl:475` | main2BDNK.jl, mainBDNK.jl |
| `EPS_NU_VALUE` | `""` | `mainBDNK.jl:477` | mainBDNK.jl |
| `ETA_OVER_S` | `0.1`, `"0.0"` | `main.jl:930`, `mainBDNK.jl:483`, `mainBGonly.jl:820` | main.jl, mainBDNK.jl, mainBGonly.jl |
| `EXPAND_COOLDOWN` | `25` | `main.jl:899`, `mainBGonly.jl:809` | main.jl, mainBGonly.jl |
| `EXPAND_FACTOR` | `1.5` | `main.jl:892`, `mainBGonly.jl:802` | main.jl, mainBGonly.jl |
| `EXPAND_GRID` | `0` | `main.jl:891`, `mainBGonly.jl:801` | main.jl, mainBGonly.jl |
| `EXPAND_MAX_NR` | `200_000` | `main.jl:897`, `mainBGonly.jl:807` | main.jl, mainBGonly.jl |
| `EXPAND_MAX_RMAX` | `Inf` | `main.jl:898`, `mainBGonly.jl:808` | main.jl, mainBGonly.jl |
| `EXPAND_MIN_CELLS` | `64` | `main.jl:896`, `mainBGonly.jl:806` | main.jl, mainBGonly.jl |
| `EXPAND_TAIL_ABS_E` | `1e-18` | `main.jl:895`, `mainBGonly.jl:805` | main.jl, mainBGonly.jl |
| `EXPAND_TAIL_CELLS` | `8` | `main.jl:893`, `mainBGonly.jl:803` | main.jl, mainBGonly.jl |
| `EXPAND_TAIL_FRAC` | `1e-6` | `main.jl:894`, `mainBGonly.jl:804` | main.jl, mainBGonly.jl |
| `FIVOHYDRO_LONG_TESTS` | `"0"` | `test/runtests.jl:257`, `test/runtests.jl:293` | test/runtests.jl |
| `FIVO_CM_SIGN` | `"+1.0"` | `main2IS2.jl:40` | main2IS2.jl |
| `FIVO_DRIVE_DISSIP` | `"1.0"` | `main2IS2.jl:312` | main2IS2.jl |
| `FIVO_DRIVE_ENABLE` | `"1"` | `main2IS2.jl:313` | main2IS2.jl |
| `FIVO_FREEZE_PDE_ALPHA` | `"1"` | `main2IS2.jl:32` | main2IS2.jl |
| `FIVO_IS2_ALPHA_MAX` | `"200.0"` | `main2IS2.jl:238` | main2IS2.jl |
| `FIVO_IS2_ALPHA_MIN` | `"-20.0"` | `main2IS2.jl:237` | main2IS2.jl |
| `FIVO_IS2_ALPHA_SOFT` | `"1.0"` | `main2IS2.jl:240` | main2IS2.jl |
| `FIVO_IS2_CAUSAL_CFL` | `"1"` | `main2IS2.jl:303` | main2IS2.jl |
| `FIVO_IS2_CAUSAL_COEFF` | `"1"` | `main2IS2.jl:311` | main2IS2.jl |
| `FIVO_IS2_CONE_PROJECT` | `"0.0"` | `main2IS2.jl:160` | main2IS2.jl |
| `FIVO_IS2_JTAU_FLOOR` | `"0.0"` | `main2IS2.jl:156` | main2IS2.jl |
| `FIVO_IS2_NU_BOUND` | `"0.0"` | `main2IS2.jl:146` | main2IS2.jl |
| `FIVO_IS2_NU_BOUND_FRAME` | `"lab"` | `main2IS2.jl:149` | main2IS2.jl |
| `FIVO_IS2_NU_BOUND_KNEE` | `"0.8"` | `main2IS2.jl:153` | main2IS2.jl |
| `FIVO_IS2_NU_BOUND_SMOOTH` | `"0"` | `main2IS2.jl:151` | main2IS2.jl |
| `FIVO_IS2_TAUM_DEGENERACY` | `"0"` | `main2IS2.jl:55` | main2IS2.jl |
| `FIVO_IS2_TAUN_DEGENERACY` | `"0"` | `main2IS2.jl:60` | main2IS2.jl |
| `FIVO_IS2_TAUN_SCALE` | `"1.0"` | `main2IS2.jl:64` | main2IS2.jl |
| `FIVO_IS2_TAUPI_REL` | `"0"` | `main2IS2.jl:73` | main2IS2.jl |
| `FIVO_KAPPA_RESONANCE_FACTOR` | `"1.0"` | `main2IS2.jl:321` | main2IS2.jl |
| `FIVO_M1_DIAG_TERMS` | `"0"` | `main2M1.jl:94` | main2M1.jl |
| `FIVO_M1_DSTT_LINEAR` | `"0"` | `main2M1.jl:90` | main2M1.jl |
| `FIVO_M1_DSTT_OFFSET` | `"-0.159"` | `main2M1.jl:92` | main2M1.jl |
| `FIVO_M1_DSTT_SLOPE` | `"1.765"` | `main2M1.jl:91` | main2M1.jl |
| `FIVO_M1_DSTT_TFO` | `"0.156"` | `main2M1.jl:93` | main2M1.jl |
| `FIVO_M1_N_FLOOR` | `"1e-300"` | `main2M1.jl:89` | main2M1.jl |
| `FIVO_M1_TP_HI` | `"5.0"` | `main2M1.jl:88` | main2M1.jl |
| `FIVO_M1_TP_LO` | `"1e-5"` | `main2M1.jl:87` | main2M1.jl |
| `FIVO_M2_E_MAX` | `"25.0"` | `main2M2.jl:70` | main2M2.jl |
| `FIVO_M2_NEWTON_TOL` | `"1e-10"` | `main2M2.jl:71` | main2M2.jl |
| `FIVO_M2_N_FLOOR` | `"1e-300"` | `main2M2.jl:69` | main2M2.jl |
| `FIVO_M2_TP_HI` | `"5.0"` | `main2M2.jl:68` | main2M2.jl |
| `FIVO_M2_TP_LO` | `"1e-5"` | `main2M2.jl:67` | main2M2.jl |
| `FIVO_MAX_SIGNAL_SPEED` | `"10.0"` | `main2IS2.jl:301` | main2IS2.jl |
| `FIVO_NU_RAPIDITY` | `"0"` | `main2IS2.jl:212` | main2IS2.jl |
| `FIVO_ORIGIN_ODD_FIRST_ORDER_CELLS` | `"4"` | `main2IS2.jl:74` | main2IS2.jl |
| `FIVO_USE_Q_RECOVERY` | `"1"` | `main2IS2.jl:33` | main2IS2.jl |
| `FIVO_VACUUM_N_HI` | `"2e-3"` | `main2IS2.jl:85` | main2IS2.jl |
| `FIVO_VACUUM_N_LO` | `"1e-6"` | `main2IS2.jl:75` | main2IS2.jl |
| `FIVO_VACUUM_N_REL_HI` | `"0.0"` | `main2IS2.jl:134` | main2IS2.jl |
| `FIVO_VACUUM_N_REL_LO` | `"1e-4"` | `main2IS2.jl:135` | main2IS2.jl |
| `FIVO_VACUUM_T_HI` | `"0.0"` | `main2IS2.jl:106` | main2IS2.jl |
| `FIVO_VACUUM_T_LO` | `"0.0"` | `main2IS2.jl:107` | main2IS2.jl |
| `FUGACITY` | `"alpha"` | `main.jl:884`, `main2.jl:862`, `mainBGonly.jl:794` | main.jl, main2.jl, mainBGonly.jl |
| `HYDRO_ABORT_ON_FIRST_PRIMFAIL` | `false` | `src/runtime_flags.jl:104` | src/runtime_flags.jl |
| `HYDRO_ABORT_ON_FIRST_SRSCALE` | `false` | `src/runtime_flags.jl:105` | src/runtime_flags.jl |
| `HYDRO_ABORT_SRSCALE_EPS` | `0.0` | `src/runtime_flags.jl:107` | src/runtime_flags.jl |
| `HYDRO_ABORT_SRSCALE_MIN_TAU` | `0.0` | `src/runtime_flags.jl:106` | src/runtime_flags.jl |
| `HYDRO_ABORT_SRSCALE_POST_EPS` | `1e-6` | `src/runtime_flags.jl:108` | src/runtime_flags.jl |
| `HYDRO_CHECK_TCONS` | `false` | `src/runtime_flags.jl:89` | src/runtime_flags.jl |
| `HYDRO_CHECK_TCONS_EVERY` | `50` | `src/runtime_flags.jl:90` | src/runtime_flags.jl |
| `HYDRO_CHECK_TCONS_I` | `-1` | `src/runtime_flags.jl:91` | src/runtime_flags.jl |
| `HYDRO_CHECK_TCONS_TOL_ABS` | `1e-10` | `src/runtime_flags.jl:93` | src/runtime_flags.jl |
| `HYDRO_CHECK_TCONS_TOL_REL` | `1e-6` | `src/runtime_flags.jl:92` | src/runtime_flags.jl |
| `HYDRO_DUMP_PRIMFAIL` | `false` | `src/runtime_flags.jl:102` | src/runtime_flags.jl |
| `HYDRO_DUMP_SRSCALE` | `false` | `src/runtime_flags.jl:98` | src/runtime_flags.jl |
| `HYDRO_LOG_PRIMFAIL` | `false` | `src/runtime_flags.jl:100` | src/runtime_flags.jl |
| `HYDRO_LOG_PRIMFAIL_EVERY` | `20` | `src/runtime_flags.jl:101` | src/runtime_flags.jl |
| `HYDRO_LOG_SRSCALE` | `false` | `src/runtime_flags.jl:95` | src/runtime_flags.jl |
| `HYDRO_LOG_SRSCALE_EPS` | `0.01` | `src/runtime_flags.jl:96` | src/runtime_flags.jl |
| `HYDRO_LOG_SRSCALE_EVERY` | `50` | `src/runtime_flags.jl:97` | src/runtime_flags.jl |
| `HYDRO_OUTDIR` | `""`, `joinpath(@__DIR__, "snapshots", "current_only"`, `joinpath(@__DIR__, "snapshots", "BDNK"` | `main.jl:887`, `main2.jl:855`, `mainBDNK.jl:458`, `mainBGonly.jl:797` | main.jl, main2.jl, mainBDNK.jl, mainBGonly.jl |
| `HYDRO_REPAIR_THETA` | `false` | `src/runtime_flags.jl:86` | src/runtime_flags.jl |
| `HYDRO_SPLINES_OUTDIR` | `""` | `main2.jl:880` | main2.jl |
| `HYDRO_SRSCALE_MARGIN` | `0.999` | `src/runtime_flags.jl:87` | src/runtime_flags.jl |
| `HYDRO_WRITE_BADMASK` | `false` | `src/runtime_flags.jl:110` | src/runtime_flags.jl |
| `INIT_CSV` | `""`, `init_csv_default`, `nothing` | `main.jl:809`, `main2.jl:857`, `mainBDNK.jl:470`, `mainBGonly.jl:721` | main.jl, main2.jl, mainBDNK.jl, mainBGonly.jl |
| `INIT_GOOD_BUFFER` | `4` | `main.jl:902`, `mainBGonly.jl:812` | main.jl, mainBGonly.jl |
| `INIT_GOOD_MIN_CELLS` | `32` | `main.jl:901`, `mainBGonly.jl:811` | main.jl, mainBGonly.jl |
| `INIT_GOOD_RANGE` | `1` | `main.jl:900`, `mainBGonly.jl:810` | main.jl, mainBGonly.jl |
| `INIT_MODE` | `"auto"` | `main2.jl:860` | main2.jl |
| `INTERP` | `"linear"` | `main.jl:879`, `main2.jl:863`, `mainBGonly.jl:789` | main.jl, main2.jl, mainBGonly.jl |
| `INTERP_DR` | `NaN`, `nothing` | `main.jl:882`, `main2.jl:864`, `mainBGonly.jl:792` | main.jl, main2.jl, mainBGonly.jl |
| `KAPPA_COEFF` | `nothing` | `main.jl:909` | main.jl |
| `LAMBDA_BULK_SHEAR_FACTOR` | `0.0` | `main.jl:937`, `mainBGonly.jl:827` | main.jl, mainBGonly.jl |
| `LAMBDA_NN_FACTOR` | `0.0` | `main.jl:939` | main.jl |
| `LAMBDA_SHEAR_BULK_FACTOR` | `0.0` | `main.jl:938`, `mainBGonly.jl:828` | main.jl, mainBGonly.jl |
| `LOG_CORRECTIONS_EVERY` | `50` | `main.jl:872`, `mainBGonly.jl:782` | main.jl, mainBGonly.jl |
| `LOG_EVERY` | `50` | `main.jl:871`, `main2.jl:875`, `main2BDNK.jl:540`, `mainBGonly.jl:781` | main.jl, main2.jl, main2BDNK.jl, mainBGonly.jl |
| `NGHOST` | `-1`, `1` | `main.jl:819`, `main2.jl:871`, `main2BDNK.jl:536`, `mainBGonly.jl:730` | main.jl, main2.jl, main2BDNK.jl, mainBGonly.jl |
| `NR` | `-1`, `200`, `300`, `"300"` | `main.jl:817`, `main2.jl:869`, `main2BDNK.jl:534`, `mainBDNK.jl:462`, `mainBGonly.jl:728` | main.jl, main2.jl, main2BDNK.jl, mainBDNK.jl, mainBGonly.jl |
| `NUR_CLIP_FACTOR` | `-1.0` | `main.jl:919` | main.jl |
| `NUR_FILTER_EPS` | `0.0` | `main.jl:921` | main.jl |
| `NUR_SMOOTH_LEN` | `0.0` | `main.jl:923` | main.jl |
| `PI_CLIP_FACTOR` | `-1.0` | `main.jl:942`, `mainBGonly.jl:831` | main.jl, mainBGonly.jl |
| `RELAX_ADVECT_BULK_PI` | `true` | `main.jl:945`, `mainBGonly.jl:834` | main.jl, mainBGonly.jl |
| `RELAX_ADVECT_NUR` | `true` | `main.jl:928` | main.jl |
| `RELAX_ADVECT_SHEAR_PI` | `true` | `main.jl:947`, `mainBGonly.jl:836` | main.jl, mainBGonly.jl |
| `RMAX` | `NaN`, `20.0`, `25.0`, `"25.0"` | `main.jl:818`, `main2.jl:870`, `main2BDNK.jl:535`, `mainBDNK.jl:463`, `mainBGonly.jl:729` | main.jl, main2.jl, main2BDNK.jl, mainBDNK.jl, mainBGonly.jl |
| `RUN_LABEL` | `""` | `main2.jl:879`, `main2BDNK.jl:543` | main2.jl, main2BDNK.jl |
| `SHEAR_CLIP_FACTOR` | `-1.0` | `main.jl:943`, `mainBGonly.jl:832` | main.jl, mainBGonly.jl |
| `SHEAR_DT_COEFF` | `0.01` | `main.jl:917`, `mainBGonly.jl:817` | main.jl, mainBGonly.jl |
| `TAPER_WIDTH` | `0.0` | `main.jl:877`, `main2.jl:877`, `mainBGonly.jl:787` | main.jl, main2.jl, mainBGonly.jl |
| `TAU0` | `0.4`, `"0.4"` | `main.jl:810`, `main2.jl:867`, `main2BDNK.jl:532`, `mainBDNK.jl:460`, `mainBGonly.jl:722` | main.jl, main2.jl, main2BDNK.jl, mainBDNK.jl, mainBGonly.jl |
| `TAUFINAL` | `5.0`, `15.0`, `"15.0"` | `main.jl:811`, `main2.jl:868`, `main2BDNK.jl:533`, `mainBDNK.jl:461`, `mainBGonly.jl:723` | main.jl, main2.jl, main2BDNK.jl, mainBDNK.jl, mainBGonly.jl |
| `TAUPI_PI_FACTOR` | `0.0` | `main.jl:936`, `mainBGonly.jl:826` | main.jl, mainBGonly.jl |
| `TAU_N_COEFF` | `1.0` | `main.jl:914` | main.jl |
| `TAU_PI_COEFF` | `15.0` | `main.jl:932`, `mainBGonly.jl:822` | main.jl, mainBGonly.jl |
| `TAU_SHEAR_COEFF` | `0.2` | `main.jl:933`, `mainBGonly.jl:823` | main.jl, mainBGonly.jl |
| `TIME_INTEGRATOR` | `"ssprk3"`, `"ssprk2"` | `main.jl:873`, `mainBDNK.jl:486`, `mainBGonly.jl:783` | main.jl, mainBDNK.jl, mainBGonly.jl |
| `T_FLOOR` | `1e-6` | `main2.jl:878`, `main2BDNK.jl:542` | main2.jl, main2BDNK.jl |
| `T_FREEZE` | `0.156` | `main2.jl:886` | main2.jl |
| `USE_DIAG_SETTINGS` | `1` | `main.jl:796`, `mainBGonly.jl:709` | main.jl, mainBGonly.jl |
| `VISC_FILTER_EPS` | `0.0` | `main.jl:940`, `mainBGonly.jl:829` | main.jl, mainBGonly.jl |
| `VISC_SMOOTH_LEN` | `0.0` | `main.jl:941`, `mainBGonly.jl:830` | main.jl, mainBGonly.jl |
| `ZETA_OVER_S` | `0.1`, `"0.0"` | `main.jl:931`, `mainBDNK.jl:484`, `mainBGonly.jl:821` | main.jl, mainBDNK.jl, mainBGonly.jl |

151 distinct flags across 54 files — generated by `tools/list_env_flags.jl`.
