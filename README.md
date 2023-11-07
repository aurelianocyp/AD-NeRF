

## Prerequisites
- You can create an anaconda environment called adnerf with:
    ```
    conda env create -f environment.yml
    conda init
    conda activate adnerf
    pip install torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    ```
- [PyTorch3D 0.5.0](https://github.com/facebookresearch/pytorch3d/releases)

    Recommend install from a local clone
    ```
    conda install -c bottler nvidiacub
    cd pytorch3d && pip install -e .
    ```
    或者通过下载包的方式安装nvidiacub。不过最好的方法是将一些依赖包删掉`pip uninstall tqdm termcolor tabulate pyyaml protalocker yacs iopath fvcore `，即使采用了镜像迁移也最好删掉....，然后让pytorch3d去自动收集并安装。但是torch或者pytorch需要提前安好。dfrf使用的torch，可以使用torch。pytorch不清楚\
    To rebuild after installing from a local clone run, `rm -rf build/ **/*.so` then `pip install -e .`.\
  总之，pytorch3d的安装是玄学
- [Basel Face Model 2009](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details) 

    Put "01_MorphableModel.mat" to data_util/face_tracking/3DMM/; cd data_util/face_tracking; run
    ```
    python convert_BFM.py
    ```
## Train AD-NeRF
\如果得到结果为Nan，最好在face tracking文件夹中的render 3dmm中设置为blur_radius=0和faces_per_pixel=1
- Data Preprocess ($id Obama for example)
    ```
    bash process_data.sh Obama
    ```
    - Input: A portrait video at 25fps containing voice audio. (dataset/vids/$id.mp4)
    - Output: folder dataset/$id that contains all files for training

- Train Two NeRFs (Head-NeRF and Torso-NeRF)
    - Train Head-NeRF with command 
        ```
        python NeRFs/HeadNeRF/run_nerf.py --config dataset/$id/HeadNeRF_config.txt
        ```
    - Copy latest trainied model from dataset/$id/logs/$id_head to dataset/$id/logs/$id_com
    - Train Torso-NeRF with command 
        ```
        python NeRFs/TorsoNeRF/run_nerf.py --config dataset/$id/TorsoNeRF_config.txt
        ```
    - You may need the [pretrained models](https://github.com/YudongGuo/AD-NeRF/tree/master/pretrained_models) to avoid bad initialization. [#3](https://github.com/YudongGuo/AD-NeRF/issues/3)
## Run AD-NeRF for rendering
- Reconstruct original video with audio input
    ```
    python NeRFs/TorsoNeRF/run_nerf.py --config dataset/$id/TorsoNeRFTest_config.txt --aud_file=dataset/$id/aud.npy --test_size=300
    ```
- Drive the target person with another audio input
    ```
    python NeRFs/TorsoNeRF/run_nerf.py --config dataset/$id/TorsoNeRFTest_config.txt --aud_file=${deepspeechfile.npy} --test_size=-1
    ```

## Citation

If you find our work useful in your research, please consider citing our paper:

```
@inproceedings{guo2021adnerf,
  title={AD-NeRF: Audio Driven Neural Radiance Fields for Talking Head Synthesis},
  author={Yudong Guo and Keyu Chen and Sen Liang and Yongjin Liu and Hujun Bao and Juyong Zhang},
  booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

If you have questions, feel free to contact <gyd2011@mail.ustc.edu.cn>.

## Acknowledgments
We use [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) for parsing head and torso maps, and [DeepSpeech](https://github.com/mozilla/DeepSpeech) for audio feature extraction. The NeRF model is implemented based on [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch).
