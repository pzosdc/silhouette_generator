Silhouette Generator: Generate Dissection Puzzle Problems
=========================================================

This program generates dissection-puzzle shapes
from given polygonal pieces.


## Install

```
$ python --version
3.10.1

$ git clone https://github.com/pzosdc/silhouette_generator

$ cd silhouette_generator

$ pip install -r requirements.txt
```

## Usage

```
$ ls
LICENSE  Readme.md  evalute_silhouette.py generate_silhouette.py  requirements.txt  samples

$ ls samples
Tangram.json  Tpuzzle.json  multi.json

$ python generate_silhouette.py --help
Randomly Generate Silhouette from Pieces.

Usage:
    program <input_json> <output_dir> [-n <n>]
        [--no_image]
        [--bbox_size <bbox_size>]
        [--image_size <image_size>]
        [--algorithm <algorithm>]

Options:
    <input_json>   json file of silhouette puzzle
    <output_dir>   output directory. create if not exist.
    -n <n>   number of silhouette you want to generate [default: 10].
    --no_image   do not create image
    --bbox_size <bbox_size>  bounding box size (length unit: same as json file) [default: 12].
    --image_size <image_size>  image size (length unit: px) [default: 400].
    --algorithm <algorithm>  generation algorithm (see below) [default: random_growth].

Available Algorithms:
    random_growth   randomly connect pieces around pre-existing pieces
    alg_A1_20230426  currently under development
    alg_A2_20230426  currently under development
    alg_A3_20230428  currently under development


$ python generate_silhouette.py ./samples/Tangram.json ../result/ -n 1000

```

## Licence

[MIT](https://github.com/pzosdc/silhouette_generator/blob/main/LICENSE)

## Author

[pzdc](https://github.com/pzosdc)

