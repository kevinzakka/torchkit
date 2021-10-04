# torchkit

[![documentation](https://github.com/kevinzakka/torchkit/workflows/docs/badge.svg)](https://kevinzakka.github.io/torchkit/)
![build](https://github.com/kevinzakka/torchkit/workflows/build/badge.svg)
![license](https://img.shields.io/github/license/kevinzakka/torchkit?color=blue)

**`torchkit`** is a *lightweight* library containing PyTorch utilities useful for day-to-day research. Its main goal is to abstract away a lot of the redundant boilerplate associated with research projects like experimental configurations, logging and model checkpointing. It consists of:

<table>
  <tbody valign="top">
    <tr>
      <td><code>torchkit.<strong>Logger</strong></code></td>
      <td>
        A wrapper around <a href="https://www.tensorflow.org/tensorboard">Tensorboard</a>'s <code>SummaryWriter</code> for safe
        logging of scalars, images, videos and learning rates. Supports both numpy arrays and torch Tensors.
      </td>
    </tr>
    <tr>
      <td><code>torchkit.<strong>CheckpointManager</strong></code></td>
      <td>
        A port of Tensorflow's checkpoint manager that automatically manages multiple checkpoints in an experimental run.
      </td>
    </tr>
    <tr>
      <td><code>torchkit.<strong>experiment</strong></code></td>
      <td>
        A collection of methods for setting up experiment directories.
      </td>
    </tr>
    <tr>
      <td><code>torchkit.<strong>layers</strong></code></td>
      <td>
        A set of commonly used layers in research papers not available in vanilla PyTorch like "same" and "causal" convolution and <code>SpatialSoftArgmax</code>.
      </td>
    </tr>
    <tr>
      <td><code>torchkit.<strong>losses</strong></code></td>
      <td>
        Some useful loss functions also unavailable in vanilla PyTorch like cross entropy with label smoothing and Huber loss.
      </td>
    </tr>
    <tr>
      <td><code>torchkit.<strong>utils</strong></code></td>
      <td>
        A bunch of helper functions for config manipulation, I/O, timing, debugging, etc.
      </td>
    </tr>
  </tbody>
</table>

For more details about each module, see the [documentation](https://kevinzakka.github.io/torchkit/).

### Installation

To install the latest release, run:

```bash
pip install git+https://github.com/kevinzakka/torchkit.git
```

### Contributing

For development, clone the source code and create a virtual environment for this project:

```bash
git clone https://github.com/kevinzakka/torchkit.git
cd torchkit
pip install -e .[dev]
```

### Acknowledgments

* Thanks to Karan Desai's [VirTex](https://github.com/kdexd/virtex) which I used to figure out documentation-related setup for torchkit and for just being an excellent example of stellar open-source research release.
* Thanks to [seals](https://github.com/HumanCompatibleAI/seals) for the excellent software development
  practices that I've tried to emulate in this repo.
* Thanks to Brent Yi for encouraging me to use type hinting and for letting me use his awesome [Bayesian filtering library](https://github.com/stanford-iprl-lab/torchfilter)'s README as a template.
