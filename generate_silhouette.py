#!/usr/bin/env python
"""Randomly Generate Silhouette from Pieces.

Usage:
    program <input_json> <output_dir> [-n <n>]

Options:
    <input_json>   json file of silhouette puzzle
    <output_dir>   output directory. create if not exist.
    -n <n>   number of silhouette you want to generate

"""

import cairosvg
import docopt
import json
import matplotlib.pyplot
import math
import numpy as np
import pathlib
import random
import svgwrite
import sys
import tqdm


class SilhouetteError(Exception):
    ...


class Polygon:
    """polygon class which imitate basics of sympy.Polygon interfaces.

    but with floating-point numbers.
    also stock the trace.
    """
    def __init__(self, name, xy):
        """.

        Args:
            name (str):  identifier of polygon
            xy (numpy 2d-Array):  data like [[x0, y0], [x1, y1], ...]
        """
        self.name = name
        self.xy = xy
        self.vertices = [[float(v[0]), float(v[1])] for v in xy]
        self.trace = []

    def flip(self):
        """inverse y-coordinate."""
        self.vertices = [[v[0], -v[1]] for v in self.vertices][-1::-1]
        self.trace.append('flip')
        return self

    def translate(self, dx, dy):
        self.vertices = [
            [v[0] + dx, v[1] + dy] for v in self.vertices]
        self.trace.append(('translate', dx, dy,))
        return self

    def rotate(self, cx, cy, angle):
        """rotate by angle angle where angle is in radian."""
        def _rot(x, y, cx, cy, angle):
            dx = x - cx
            dy = y - cy
            newx = cx + (dx * math.cos(angle) - dy * math.sin(angle))
            newy = cy + (dx * math.sin(angle) + dy * math.cos(angle))
            return [newx, newy]
        self.vertices = [
            _rot(v[0], v[1], cx, cy, angle)
            for v in self.vertices]
        self.trace.append(('rotate', cx, cy, angle,))
        return self

    def __getattr__(self, attrname):
        if attrname == 'sides':
            next_vertices = self.vertices[1:] + self.vertices[0:1]
            sides = [
                [c[0], c[1], n[0], n[1]]
                for c, n in zip(self.vertices, next_vertices)]
            return sides


def load_polys(jsonfile):
    """load info of polygons(=puzzle state) from json file.

    Args:
        jsonfile (pathlib.Path):  json file to be loaded
    """
    try:
        with jsonfile.open('r') as fi:
            jsondata = json.load(fi)
    except (IsADirectoryError, FileNotFoundError, json.JSONDecodeError):
        raise SilhouetteError('bad json file specified')
    return jsondata


def save_polys(jsondata, jsonfile):
    """save info of polygons(=puzzle state) as json file.

    Args:
        jsondata (dict):  info of polygons
        jsonfile (pathlib.Path):  json file to be dumped
    """
    with jsonfile.open('w') as fo:
        json.dump(jsondata, fo,
                  ensure_ascii=False,
                  indent=2,
                  )


def draw_polys(polys, svgfile, *, bbox=None, shadow=False,
               also_save_as_png=False):
    if shadow:
        # draw as black face
        options = dict(
            stroke="black",
            stroke_width=1,
            fill=svgwrite.rgb(0, 0, 0, "RGB"),
            )
    else:
        # draw as brown (half-transparent) face
        options = dict(
            stroke='brown',
            stroke_width=1,
            fill='saddlebrown',
            fill_opacity=0.5,
            )
    dwg = svgwrite.Drawing(svgfile, viewBox='0 0 400 400')
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'),
                     rx=None, ry=None, fill='rgb(255,255,255)'))
    xmin, xmax, ymin, ymax = bbox

    def _mapx(x):
        return (x - xmin) / (xmax - xmin) * 400

    def _mapy(y):
        return (y - ymin) / (ymax - ymin) * 400

    for poly in polys:
        vertices_in_svg_coord = [[_mapx(v[0]), _mapy(v[1])]
                                 for v in poly.vertices]
        dwg.add(svgwrite.shapes.Polygon(vertices_in_svg_coord, **options))
    dwg.save()

    if also_save_as_png:
        pngfile = svgfile.with_suffix('.png')
        cairosvg.svg2png(url=str(svgfile), write_to=str(pngfile))


def get_boundingbox(polys):
    x_list = np.concatenate([
        np.array([p[0] for p in poly.vertices]) for poly in polys])
    y_list = np.concatenate([
        np.array([p[1] for p in poly.vertices]) for poly in polys])
    xmin = np.min(x_list)
    xmax = np.max(x_list)
    ymin = np.min(y_list)
    ymax = np.max(y_list)
    return xmin, xmax, ymin, ymax


def make_random_silhouette(polys):
    """Generate new arangement of polygons(=puzzle state)."""
    random.shuffle(polys)
    new_polys = [polys[0]]
    for poly in polys[1:]:
        # step 1. capriciously flip
        flip = (random.randint(0, 1) == 1)
        if flip:
            flipped_poly = poly.flip()
        else:
            flipped_poly = poly
        # step 2. decide which pair of sides to be sticked in a random manner
        target_sides = random.choice(new_polys).sides
        target_side = random.choice(target_sides)
        source_sides = flipped_poly.sides
        source_side = random.choice(source_sides)
        place_begin_to_end = (random.randint(0, 1) == 1)
        if place_begin_to_end:
            source_vertex = [source_side[0], source_side[1]]
            target_vertex = [target_side[2], target_side[3]]
        else:
            source_vertex = [source_side[2], source_side[3]]
            target_vertex = [target_side[0], target_side[1]]
        # step 3. stick the pair of sides (which is decided in step 2.)
        dx = target_vertex[0] - source_vertex[0]
        dy = target_vertex[1] - source_vertex[1]
        moved_poly = flipped_poly.translate(dx, dy)

        source_angle = math.atan2(source_side[3] - source_side[1],
                                  source_side[2] - source_side[0])
        target_angle = math.atan2(target_side[3] - target_side[1],
                                  target_side[2] - target_side[0])
        dth = math.pi + target_angle - source_angle
        rotated_poly = moved_poly.rotate(*target_vertex, dth)

        new_polys.append(rotated_poly)
    return new_polys


def rotate_random_all(polys):
    """rotate all polygons in current puzzle."""
    cx = 0
    cy = 0
    theta_div = 60
    dth = float(np.pi) / theta_div * random.randint(0, theta_div - 1)
    rotated_polys = [poly.rotate(cx, cy, dth) for poly in polys]
    return rotated_polys


def bbox_adjust_all(polys, xsize, ysize):
    """set bounding-box size to default value."""
    bbox = get_boundingbox(polys)
    xmin, xmax, ymin, ymax = bbox
    wx = xmax - xmin
    wy = ymax - ymin
    if wx > xsize:
        print(f'Error size of x ({wx}) greater than plot size ({xsize})')
    if wy > ysize:
        print(f'Error size of y ({wy}) greater than plot size ({ysize})')
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    dx = xmax - cx
    dy = ymax - cy
    rx = xsize / wx
    ry = ysize / wy
    xmin = cx - dx * rx
    xmax = cx + dx * rx
    ymin = cy - dy * ry
    ymax = cy + dy * ry
    return [xmin, xmax, ymin, ymax]


def isinpoly(x, y, poly):
    n = 0
    for side in poly.sides:
        x1, y1, x2, y2 = side
        if (x1 > x) is (x2 > x):
            continue
        if x1 == x2:
            yc = (y1 + y2) / 2
        else:
            dx = x2 - x1
            yc = y1 - (y2 - y1) * ((x1 - x) / dx)
        if y >= yc:
            n += 1
    return (n % 2 == 1)


def count_wrap_poly_at(x, y, simple_polys):
    isinpoly_mask = [isinpoly(x, y, poly) for poly in simple_polys]
    return len(np.where(isinpoly_mask)[0])


def get_wrap_poly_count_matrix(x_list, y_list, polys):
    return np.array(
        [count_wrap_poly_at(x, y, polys)
         for x, y in zip(x_list, y_list)])


def has_overwrap_roughcheck(polys, bbox):
    xmin, xmax, ymin, ymax = bbox

    xmin = float(xmin) + (1 / 1001) * random.random()
    xmax = float(xmax) + (1 / 1003) * random.random()
    ymin = float(ymin) + (1 / 1005) * random.random()
    ymax = float(ymax) + (1 / 1007) * random.random()
    xdiv = 400
    ydiv = 400

    xgridspace = np.linspace(float(xmin), float(xmax), xdiv)
    ygridspace = np.linspace(float(ymin), float(ymax), ydiv)
    xgrid, ygrid = np.meshgrid(xgridspace, ygridspace)
    xgrid1 = np.concatenate(xgrid)
    ygrid1 = np.concatenate(ygrid)
    zgrid1 = get_wrap_poly_count_matrix(xgrid1, ygrid1, polys)
    zgrid = np.reshape(zgrid1, xgrid.shape)
    grids = [xgrid, ygrid, zgrid]

    if np.max(zgrid) > 1:
        return True, grids
    return False, grids


def draw_contour(ax, grids, bbox, pngfile):
    xgrid, ygrid, zgrid = grids
    ax.contourf(xgrid, ygrid, zgrid)
    ax.set_aspect('equal')
    bbxmin, bbxmax, bbymin, bbymax = bbox
    ax.set_xlim(left=float(bbxmin), right=float(bbxmax))
    ax.set_ylim(bottom=float(bbymin), top=float(bbymax))
    ax.axis('off')
    matplotlib.pyplot.savefig(pngfile)


def generate_silhouette_core(jsonfile, outdir, *, num_generation=10):
    jsondata = load_polys(jsonfile)
    outdir.resolve().mkdir(exist_ok=True, parents=True)
    ax = matplotlib.pyplot.figure(figsize=(8, 8)).add_subplot()

    # [FIXME] this value should be changed when the size of puzzle is changed
    xsize = 10
    ysize = 10

    for puzzlename in jsondata.keys():
        print(f'{puzzlename = }')
        resultjson = dict()
        puzzleoutdir = outdir / puzzlename
        puzzleoutdir.mkdir(exist_ok=True)
        puzzleinfo = jsondata[puzzlename]
        polygondict = puzzleinfo['polygon']
        polys = [Polygon(polykey, vertices)
                 for polykey, vertices in polygondict.items()]
        wrapdir = puzzleoutdir / 'wrap'
        planardir = puzzleoutdir / 'planar'
        wrapdir.mkdir(exist_ok=True)
        planardir.mkdir(exist_ok=True)

        # save start-position image before generation
        svgfile = puzzleoutdir / f'{puzzlename}.svg'
        bbox = bbox_adjust_all(polys, xsize, ysize)
        draw_polys(polys, svgfile, shadow=False, bbox=bbox,
                   also_save_as_png=True)

        pbar = tqdm.tqdm(
            range(1, num_generation + 1),
            desc=f'Generating {puzzlename} state',
            unit='state',
            )
        try:
            for i in pbar:
                puzzlesubname = f'{puzzlename}-{i}'
                new_polys = make_random_silhouette(polys)
                new_polys = rotate_random_all(new_polys)
                bbox = bbox_adjust_all(new_polys, xsize, ysize)
                has_overwrap, grids = has_overwrap_roughcheck(new_polys, bbox)
                imagedirname = 'wrap' if has_overwrap else 'planar'
                imagedir = puzzleoutdir / imagedirname
                newpuzzleinfo = dict(
                    has_overwrap=has_overwrap,
                    polygon={poly.name: poly.vertices for poly in polys}
                    )
                resultjson[puzzlesubname] = newpuzzleinfo
                if has_overwrap:
                    pngfile = imagedir / f'{puzzlesubname}_contour.png'
                    draw_contour(ax, grids, bbox, pngfile)

                svgfile = imagedir / f'{puzzlesubname}.svg'
                draw_polys(new_polys, svgfile, shadow=True, bbox=bbox,
                           also_save_as_png=True)

                svgfile_answer = imagedir / f'{puzzlesubname}_ans.svg'
                draw_polys(new_polys, svgfile_answer, shadow=False, bbox=bbox,
                           also_save_as_png=True)
        except Exception as err:
            print(err)
        except KeyboardInterrupt as err:
            print(f'KeyboardInterrupt at step {i}: generation is stopped')
            save_polys(resultjson, puzzleoutdir / f'{puzzlename}.json')
            sys.exit()
        save_polys(resultjson, puzzleoutdir / f'{puzzlename}.json')


def main():
    # parse command-line args
    args = docopt.docopt(__doc__)
    jsonfile = pathlib.Path(args.get('<input_json>') or 'input.json')
    outdir = pathlib.Path(args.get('<output_dir>') or 'result')
    num_generation = int(args.get('-n') or 10)

    # run generation of puzzles
    generate_silhouette_core(jsonfile, outdir,
                             num_generation=num_generation)


if __name__ == '__main__':
    main()
