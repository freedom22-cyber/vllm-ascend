export IMAGE=quay.io/ascend/vllm-ascend:v0.30.0-a5-openeuler
export NAME=pd-mooncacke-d

docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci_manager \
--device /dev/hisi_hdc \
--device /dev/ummu \
--device /dev/uburma \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
-v /etc/hixlep/:/etc/hixlep/ \
-v /root/.cache:/root/.cache \
-v /usr/local/sbin:/usr/local/sbin \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/bin/urma_admin:/usr/bin/urma_admin \
-v /lib/route.conf:/lib/route.conf \
-v /mnt:/mnt \
-itd $IMAGE bash
