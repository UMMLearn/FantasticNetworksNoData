# FantasticNetworksNoData

This repository provides code that will train a network progressively with increasing amounts of data and compute two data independent metrics, Mg and Madv, after each data checkpoint.  The metrics are shown to correlate strongly with test accuracy as the more data is made available, but they do not require any data to derive, using only the weights of the model and back-propagated class prototype inputs.

For details and references, please see our preprint at:

https://arxiv.org/abs/2305.15563

To cite our work, please use:

```bibtex
@misc{dean2023fantastic,
      title={Fantastic DNN Classifiers and How to Identify them without Data}, 
      author={Nathaniel Dean and Dilip Sarkar},
      year={2023},
      eprint={2305.15563},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
    } 
```


