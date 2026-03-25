export PYTHONPATH=$PYTHONPATH:.
python my_learning/mine-refine.py \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --data_dir /root/autodl-tmp/manipulation_v5_realistic_kitchen_2500_1/dataset/data/ \
  --out_root ./exp_refine_clean/split \
  --valid_iters 8



export PYTHONPATH=$PYTHONPATH:.
python my_learning/generate_npy_for_hardcases_refine.py \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --hard_root ./exp_refine_clean/split/Hard_Test_100 \
  --valid_iters 8 \
  --overwrite 1


python my_learning/heuristic_disparity_attribution_refine.py \
  --hard_root ./exp_refine_clean/split/Hard_Test_100 \
  --pred_dir ./exp_refine_clean/split/Hard_Test_100/preds_npy \
  --out_csv ./exp_refine_clean/split/Hard_Test_100/failure_attribution_report.csv
