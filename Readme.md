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

Silhouette Generator consists of practically one Python file.
One can simply run that file to generate puzzles.
For dependencies, please look in the ``requirements.txt``.

```
$ ls
LICENSE  Readme.md  generate_silhouette.py  requirements.txt  samples

$ ls samples
Tangram.json  Tpuzzle.json  multi.json

$ python generate_silhouette.py --help
Usage:
    program <input_json> <output_dir> [-n <n>]
        [--no_image]

Options:
    <input_json>   json file of silhouette puzzle
    <output_dir>   output directory. create if not exist.
    -n <n>   number of silhouette you want to generate
    --no_image   do not create image

$ python generate_silhouette.py ./samples/Tangram.json ../result/ -n 1000

```

## Licence

[MIT](https://github.com/pzosdc/silhouette_generator/blob/main/LICENSE)

## Author

[pzdc](https://github.com/pzosdc)

