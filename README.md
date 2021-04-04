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
        logging of losses, learning rates and metrics.
      </td>
    </tr>
    <tr>
      <td><code>torchkit.<strong>checkpoint</strong></code></td>
      <td>
        A port of Tensorflow's checkpoint management tools containing:
        <ul>
            <li><code><strong>Checkpoint</strong></code>: For saving and restoring any object with a <code>state_dict</code> attribute.</li>
            <li><code><strong>CheckpointManager</strong></code>: For automatically managing multiple checkpoints in an experimental run.</li>
        </ul>
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
        A growing list of PyTorch-related helper functions.
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
* Thanks to Brent Yi for encouraging me to use type hinting and for letting me use his awesome [Bayesian filtering library](https://github.com/stanford-iprl-lab/torchfilter)'s README as a template.
