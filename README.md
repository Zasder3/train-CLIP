# train-CLIP 📎

A PyTorch Lightning solution to training CLIP from scratch.

## Goal ⚽

Our aim is to create an easy to use Lightning implementation of OpenAI's clip training script. We want our end product to be as inline with the orignal paper as possible. We will live by:

<p align="center">
    <img src="images/clip-paper.PNG" alt="CLIP Section Image">
</p>



## TODO ✅

- [x] Get OpenAI's model creation script
- [ ] Create model inits
  - [ ] ResNet50
  - [ ] ResNet50x4
  - [ ] ResNet101
  - [ ] ViT-B/32
- [ ] Create dataset files 
- [ ] Create lightning trainer/model wrapper
- [ ] Performance boosts
  - [ ] Mixed-precision
  - [ ] Gradient checkpointing
  - [ ] Half-precision Adam statistics
  - [ ] Half-precision stochastically rounded text encoder weights