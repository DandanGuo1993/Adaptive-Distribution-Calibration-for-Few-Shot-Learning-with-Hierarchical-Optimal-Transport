
Code for  "Adaptive Distribution Calibration for Few-Shot Learning with Hierarchical Optimal Transport", submitted to NeurIPS 2022.

Requirements

sklearn
numpy==1.17.2
matplotlib==3.1.1
tqdm==4.36.1
torchvision==0.6.0
torch==1.5.0
Pillow==7.1.2


Extract and save features
After training the backbone, extract features as below:

Create an empty 'checkpoints' directory.

Run: python save_plk.py --dataset [miniImagenet] 

After downloading the extracted features, please adjust your file path according to the code.

Evaluate our distribution calibration
Run: python main.py


@inproceedings{Guo2022fewshot, title={Adaptive Distribution Calibration for Few-Shot Learning with Hierarchical Optimal Transport}, author={Guo, Dandan and Tian Long and Zhao, He and Zhou, Mingyuan and Zha, Hongyuan}, booktitle={Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)}, year={2022} }
