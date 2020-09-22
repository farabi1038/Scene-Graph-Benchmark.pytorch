CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --master_port 10025 --nproc_per_node=4 tools/relation_train_net.py \
--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False  \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False  \
MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor  \
MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none  \
MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum  \
MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs   \
SOLVER.IMS_PER_BATCH 24 \
TEST.IMS_PER_BATCH 4  \
DTYPE "float16"  \
SOLVER.MAX_ITER 50000  \
SOLVER.VAL_PERIOD 2000  \
SOLVER.CHECKPOINT_PERIOD 2000  \
GLOVE_DIR /home/h/hn235/Visual_Genome/glove \
MODEL.PRETRAINED_DETECTOR_CKPT /home/h/hn235/Visual_Genome/pretrained_faster_rcnn/model_final.pth  \
OUTPUT_DIR /home/h/hn235/Visual_Genome/checkpoints/causal-motifs-sgcls-exmp