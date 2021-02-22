# VerSe: Large Scale Vertebrae Segmentation Challenge

_... and there is power in numbers._ - Martin Luther King, Jr.

![VerSe examples. Observe the variability in data: field-of-view, fractures, transitional vertebrae, etc.](assets/dataset_snapshot.png)
*VerSe examples. Observe the variability in data: field-of-view, fractures, transitional vertebrae, etc.*

<details><summary>Table of Contents</summary><p>

* [Why VerSe](#why-verse)
* [Download](#download)
* [Code](#code)
* [Benchmark](#benchmark)
* [Visualization](#visualization)
* [Contributing](#contributing)
* [Contact](#contact)
* [Citing PCam](#citing-pcam)
* [License](#license)
</p></details><p></p>

## What is VerSe?
Spine or vertebral segmentation is a crucial step in all applications regarding automated quantification of spinal morphology and pathology. With the advent of deep learning, for such a task on computed tomography (CT) scans, a big and varied data is a primary sought-after resource. However, a large-scale, public dataset is currently unavailable.

We believe *VerSe* can help here. VerSe is a large scale, multi-detector, multi-site, CT spine dataset consisting of 374 scans from 300 patients. 


## Download


## Code


## Benchmark

The challenge was 

## Contact
For problems and questions not fit for a github issue, please email [Anjany Sekuboyina](mailto:anjany.sekuboyina@tum.de) or [Jan Kirschke](mailto:jan.kirschke@tum.de) .


## Citing PCam
If you use PCam in a scientific publication, we would appreciate references to the following paper:


**[1] B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation Equivariant CNNs for Digital Pathology". [arXiv:1806.03962](http://arxiv.org/abs/1806.03962)**

A citation of the original Camelyon16 dataset paper is appreciated as well:

**[2] Ehteshami Bejnordi et al. Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer. JAMA: The Journal of the American Medical Association, 318(22), 2199–2210. [doi:jama.2017.14585](https://doi.org/10.1001/jama.2017.14585)**


Biblatex entry:
```latex
@ARTICLE{Veeling2018-qh,
  title         = "Rotation Equivariant {CNNs} for Digital Pathology",
  author        = "Veeling, Bastiaan S and Linmans, Jasper and Winkens, Jim and
                   Cohen, Taco and Welling, Max",
  month         =  jun,
  year          =  2018,
  archivePrefix = "arXiv",
  primaryClass  = "cs.CV",
  eprint        = "1806.03962"
}
```

<!-- [Who is citing PCam?](https://scholar.google.de/scholar?hl=en&as_sdt=0%2C5&q=pcam&btnG=&oq=fas) -->


## Benchmark
| Name  | Reference | Augmentations | Acc | AUC|  NLL | FROC* |
| --- | --- | --- | --- | --- | --- | --- |
| GDensenet | [1] | Following Liu et al. | 89.8 | 96.3 |  0.260 |75.8 (64.3, 87.2)|
| [Add yours](https://github.com/basveeling/pcam/edit/master/README.md) | |

\* Performance on Camelyon16 tumor detection task, not part of the PCam benchmark.


## Contributing
Contributions with example scripts for other frameworks are welcome!

## License
The data is provided under the [CC0 License](https://choosealicense.com/licenses/cc0-1.0/), following the license of Camelyon16.

The rest of this repository is under the [MIT License](https://choosealicense.com/licenses/mit/).

## Acknowledgements
* Babak Ehteshami Bejnordi, Geert Litjens, Jeroen van der Laak for their input on the configuration of this dataset.
* README derived from [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).
