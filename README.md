# PyTorch Implementation of AutoPruner
* AutoPruner: An End-to-End Trainable Filter Pruning Method for Efficient Deep Model Inference, Pattern Recognition, In Press.
* [[DOI]](https://doi.org/10.1016/j.patcog.2020.107461)   [[Manuscript]](https://cs.nju.edu.cn/wujx/paper/AutoPruner_PR2020.pdf)

## Requirements 
PyTorch environment:
* Python 3.6.9
* PyTorch 1.2.0 (MobileNetv2 codes)
* [torchsummaryX](https://github.com/nmhkahn/torchsummaryX)

## Usage
1. clone this repository.
2. download the ImageNet dataset and organize this dataset with train and val folders.
3. select subfolder (e.g., MobileNetv2):
   ```
   cd MobileNetv2
   ```
4. modify your configuration path and start pruning:
   + modify `1_pruning/main.py` and run `run_this.sh`
   + if you want to get a smaller model, you should use a smaller `compression_rate` in `1_pruning/main.py`
5. fine-tune the pruned model:
   ```
   cd ../2_fine_tune
   ./run_this.sh
   ```
6. test the accuray of pruned model:
   ```
   cd ../released_model
   run_this.sh
   ```

## Note
* MobileNetV2
  + MobileNetV2 is conducted using PyTorch 1.2.0.
  + the training log files and pretrained weights are also provided in corresponding folders.
* ResNet and VGG
  + unfortunately, the experiments of ResNet and VGG are conducted 2 years ago, I have forgotten the PyTorch version.
  +  the training log files are also provided, but the pretrained weighs are missing.

## Results
We prune the [MobileNetV2 model](https://github.com/d-li14/mobilenetv2.pytorch) on ImageNet dataset:

| Architecture  | Top-1 Acc.  | Top-5 Acc.  | #MACs   | Latency |
| ------------- | ------------- | ------------- |  ------------- |  ------------- | 
| MobileNetV2-1.0  | 72.19%  | 90.53%  | 300.79M  | 0.036s  |
| MobileNetV2-0.75 |  69.95%  | 88.99%  | 209.08M  | 0.031s  |
| AutoPruner | 71.18% | 89.95% | 207.93M | 0.026s |

The results on VGG16 and ResNet50 (PyTorch official models):

| Architecture  | Top-1 Acc.  | Top-5 Acc.  | #FLOPs   | Latency |
| ------------- | ------------- | ------------- |  ------------- |  ------------- | 
| VGG16  | 71.59%  | 90.38%  | 30.94B  | 0.116s  |
|  AutoPruner |  69.20%  | 88.89%  | 8.17B | 0.046s |
| ResNet-50 | 76.15% | 92.87% | 8.18B | 0.084s |
| AutoPruner (r=0.5) | 74.76% | 92.15% | 4.00B | 0.067s |
| AutoPruner (r=0.3) | 73.05% | 91.25% | 2.78B | 0.057s |

Note that:
  + FLOPs = 2*MACs
  + The latency is tested on a M40 GPU with 32 batch size.

## Citation
If you find this work is useful for your research, please cite:
```
@article{AutoPruner,
  author = {Jian-Hao Luo, Jianxin Wu},
  title = {AutoPruner: An End-to-End Trainable Filter Pruning Method for Efficient Deep Model Inference},
  journal = {Pattern Recognition},
  year = {2020},
}
```

## Contact
Feel free to contact me if you have any question (Jian-Hao Luo luojh@lamda.nju.edu.cn or jianhao920@gmail.com).

