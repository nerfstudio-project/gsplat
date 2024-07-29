#!/bin/bash

NAME=sls
DATA_DIR=./data/spotless
RES_DIR=./results

# The default runs robust optimization with mlp masking
for SCENE in android3 yoda3 crab2 statue3 fountain mountain corner patio 
do
	python spotless_trainer.py \
		--data_dir="${DATA_DIR}/${SCENE}" \
		--result_dir="${RES_DIR}/${SCENE}_mlp_${NAME}" 
done

# Higher robustness for scenes with high-occlusion transient.
for SCENE in patio-high spot
do
	python spotless_trainer.py \
		--data_dir="${DATA_DIR}/${SCENE}" \
		--result_dir="${RES_DIR}/${SCENE}_mlp_${NAME}" \
  		--lower_bound=0.3 \
    		--upper_bound=0.8
done


# Optionally turn on UBP prunning for cleaner scenes with less splats
for SCENE in android3 yoda3 crab2 statue3 fountain mountain corner patio patio-high spot
do
	python spotless_trainer.py \
		--data_dir="${DATA_DIR}/${SCENE}" \
		--result_dir="${RES_DIR}/${SCENE}_mlp_ubp_${NAME}" \
		--ubp
done

for SCENE in patio-high spot
do
	python spotless_trainer.py \
		--data_dir="${DATA_DIR}/${SCENE}" \
		--result_dir="${RES_DIR}/${SCENE}_mlp_ubp_${NAME}" \
  		--lower_bound=0.3 \
    		--upper_bound=0.8 \
      		--ubp
done


# Rather than optimizing an mlp, clustering the semantic features yields similar results while faster training
for SCENE in android3 yoda3 crab2 statue3 fountain mountain corner patio patio-high spot
do
	python spotless_trainer.py \
		--data_dir="${DATA_DIR}/${SCENE}" \
		--result_dir="${RES_DIR}/${SCENE}_cls_${NAME}" \
		--cluster
done


# for base filtering based on loss threshold (robustnerf style) without semantic features 
for SCENE in android3 yoda3 crab2 statue3 fountain mountain corner patio patio-high spot
do
	python spotless_trainer.py \
		--data_dir="${DATA_DIR}/${SCENE}" \
		--result_dir="${RES_DIR}/${SCENE}_flt_${NAME}" \
		--no-semantics \
		--no-cluster
done



# for baseline 3dgs
for SCENE in android3 yoda3 crab2 statue3 fountain mountain corner patio patio-high spot
do
	python spotless_trainer.py \
		--data_dir="${DATA_DIR}/${SCENE}" \
		--result_dir="${RES_DIR}/${SCENE}_base_${NAME}" \
		--no-semantics \
		--no-cluster \
		--loss_type="l1" \
		--reset_every=3000 \
		--reset_sh=300000
done

