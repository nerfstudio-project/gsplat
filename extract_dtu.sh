for i in 122 118 114 110 106
do
    python examples/simple_trainer_recon.py --eval_steps 100 2000 7000 15000 20000 25000 \
                        --disable_viewer --data_factor 2 \
                        --data_dir /media/super/data/dataset/dtu/DTU_mask/scan$i/ \
                        --result_dir output//scan$i/ \
                        --normal_consistency_loss \
                        --app_opt \
                        --test_every 1000000000 \
                        --absgrad \
                        --ckpt output/scan$i/ckpts/ckpt_29999.pt
done