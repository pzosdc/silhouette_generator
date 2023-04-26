#!/usr/bin/env python
"""Randomly Generate Silhouette from Pieces.

Usage:
    program <input_json> <output_dir> [-n <n>]
        [--no_image]

Options:
    <input_json>   json file of silhouette puzzle
    <output_dir>   output directory. create if not exist.
    -n <n>   number of silhouette you want to generate
    --no_image   do not create image

"""

import cairosvg
import docopt
import functools
import json
import matplotlib.pyplot
import math
import numpy as np
import pathlib
import pandas as pd
import random
import svgwrite
import time

from evaluate_silhouette import SilhouetteSurfaceNodeEvaluator


class SilhouetteError(Exception):
    ...


class Profiler:
    """メソッドの処理時間のチェック用"""

    __slots__ = ["total_spent", "called_count"]

    def __init__(self):
        self.total_spent = dict()
        self.called_count = dict()

    def report(self):
        print("total spent [s]:")
        [print(f"  {k}\t{v}") for k, v in self.total_spent.items()]
        print("called count:")
        [print(f"  {k}\t{v}") for k, v in self.called_count.items()]

    def method_timer(self, func):
        @functools.wraps(func)
        def new_func(*a, **k):
            s = time.perf_counter()
            ret = func(*a, **k)
            spent = time.perf_counter() - s

            method_self, *_ = a
            key = f"{method_self.__class__.__name__}.{func.__name__}"

            self.total_spent[key] = (
                spent + (self.total_spent.get(key) or 0.0))
            self.called_count[key] = 1 + (self.called_count.get(key) or 0)
            return ret
        return new_func


profiler = Profiler()


class Polygon:
    """(特定のxy座標系に置かれた)ピースのクラス."""

    __slots__ = [
        "name",  # ピース名
        "xys",  # 頂点座標列
        ]

    def __init__(self, name, xys):
        """.

        Args:
            name (str):  identifier of polygon
            xys (numpy 2d-Array):  data like [[x0, y0], [x1, y1], ...]
        """
        self.name = name
        self.xys = np.array(xys)

    def flip(self, inplace=False):
        """inverse y-coordinate.

        ピースの時計回り・反時計回り、が変化すると厄介なので、
        鏡映した後で巡回の順序も逆転する
        """
        if inplace:
            self.xys = self.xys * np.array([1., -1.])
            self.xys = self.xys[-1::-1, :]
            return self
        else:
            flip_xys = self.xys * np.array([1., -1.])
            flip_xys = flip_xys[-1::-1, :]
            return self.__class__(name=self.name, xys=flip_xys)

    def translate(self, dx, dy, inplace=False):
        xs, ys = self.xys.T
        if inplace:
            self.xys = self.xys + np.array([dx, dy])
            return self
        else:
            tr_xys = self.xys + np.array([dx, dy])
            return self.__class__(name=self.name, xys=tr_xys)

    def rotate(self, cx, cy, angle, inplace=False):
        """rotate by angle angle where angle is in radian."""
        xs, ys = self.xys.T
        new_xs = cx + ((xs - cx) * np.cos(angle) - (ys - cy) * np.sin(angle))
        new_ys = cy + ((xs - cx) * np.sin(angle) + (ys - cy) * np.cos(angle))
        if inplace:
            self.xys = np.vstack([new_xs, new_ys]).T
            return self
        else:
            rot_xys = np.vstack([new_xs, new_ys]).T
            return self.__class__(name=self.name, xys=rot_xys)

    def __getattr__(self, attrname):
        if attrname == 'vertices':
            return self.xys
        elif attrname == 'sides':
            next_vertices = np.concatenate([
                self.xys[1:], self.xys[0:1]])
            sides = np.array([
                [c[0], c[1], n[0], n[1]]
                for c, n in zip(self.xys, next_vertices)])
            return sides
        else:
            raise ValueError(f"Unknown attribute {attrname!r} for Polygon")

    def isinpoly(self, x, y):
        n = 0
        for side in self.sides:
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

    @profiler.method_timer
    def get_boundingbox(self):
        """ピースを格納する最小の長方形（バウンディングボックス）を求める."""
        xs, ys = self.xys.T
        xmin = np.min(xs)
        xmax = np.max(xs)
        ymin = np.min(ys)
        ymax = np.max(ys)
        return xmin, xmax, ymin, ymax


class Puzzle:
    """パズル(特定の座標に置かれたピースの集合).
    """

    __slots__ = [
        "name",
        "polygons",
        "has_overwrap",
        ]

    def __init__(self, name,
                 polygons=None,
                 has_overwrap=False,
                 ):
        self.name = name
        self.polygons = [] if polygons is None else polygons
        self.has_overwrap = has_overwrap

    @classmethod
    def from_json_data(cls, name, jsondata):
        polygons = [
            Polygon(k, v)
            for k, v in jsondata.get("polygon").items()]
        return cls(
            name,
            polygons=polygons,
            has_overwrap=jsondata.get("has_overwrap"),
            )

    def to_json_data(self):
        # nameはエンコードしない
        polygons = {
            polygon.name: polygon.xys.tolist()
            for polygon in self.polygons
            }
        return dict(
            has_overwrap=self.has_overwrap,
            polygon=polygons,
            )

    @profiler.method_timer
    def draw(self, svgfile, *,
             bbox=None, shadow=False,
             also_save_as_png=False):
        """SVGに出力する."""
        if shadow:
            # draw as black face
            drawoptions = dict(
                stroke="black",
                stroke_width=1,
                fill=svgwrite.rgb(0, 0, 0, "RGB"),
                )
        else:
            # draw as brown (half-transparent) face
            drawoptions = dict(
                stroke='brown',
                stroke_width=1,
                fill='saddlebrown',
                fill_opacity=0.5,
                )

        width, height = 400, 400
        dwg = svgwrite.Drawing(svgfile, viewBox=f'0 0 {width} {height}')
        dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'),
                         rx=None, ry=None, fill='rgb(255,255,255)'))
        xmin, xmax, ymin, ymax = bbox

        for polygon in self.polygons:
            xs, ys = polygon.xys.T
            xs_in_svg_coord = (xs - xmin) / (xmax - xmin) * width
            ys_in_svg_coord = (ymax - ys) / (ymax - ymin) * height
            xys_in_svg_coord = np.vstack([
                xs_in_svg_coord, ys_in_svg_coord]).T
            dwg.add(svgwrite.shapes.Polygon(
                xys_in_svg_coord.tolist(), **drawoptions))
        dwg.save()

        if also_save_as_png:
            pngfile = svgfile.with_suffix('.png')
            cairosvg.svg2png(url=str(svgfile), write_to=str(pngfile))

    @profiler.method_timer
    def get_boundingbox(self):
        """パズルを格納する最小の長方形（バウンディングボックス）を求める."""
        x_list = np.concatenate([
            poly.vertices[:, 0]
            for poly in self.polygons])
        y_list = np.concatenate([
            poly.vertices[:, 1]
            for poly in self.polygons])
        xmin = np.min(x_list)
        xmax = np.max(x_list)
        ymin = np.min(y_list)
        ymax = np.max(y_list)
        return xmin, xmax, ymin, ymax

    @profiler.method_timer
    def bbox_adjust_all(self, xsize, ysize):
        """paddingを足してbounding boxを既定のサイズにする.
        """
        bbox = self.get_boundingbox()
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

    @profiler.method_timer
    def rotate_random_all(self, inplace=False):
        cx = 0
        cy = 0
        theta_div = 60
        dth = float(np.pi) / theta_div * random.randint(0, theta_div - 1)
        if inplace:
            for poly in self.polygons:
                poly.rotate(cx, cy, dth, inplace=True)
            return self
        else:
            rot_polys = [
                poly.rotate(cx, cy, dth, inplace=False)
                for poly in self.polygons]
            return self.__class__(
                name=self.name,
                polygons=rot_polys,
                has_overwrap=self.has_overwrap,
                )

    @profiler.method_timer
    def overwrap_at_y(self, y):
        """ピースの重なりをyの位置で調べる.

        走査方式で検査する用途に使用する

        シルエットパズルは辺どうしが接するのが当たり前なので、
        実数でそのまま処理するのはうまくいかない
        そこで、ピースを内側に微小量だけシフトさせてから検査する処理をしている
        この判定方法は精度的には十分実用的に機能するが、yを細かく分割して処理を
        繰り返すのは処理時間がかかりすぎるため、さらに工夫をしたい
        """
        collisions_x = []
        collisions_label = []
        for poly in self.polygons:
            for side in poly.sides:
                x1, y1, x2, y2 = side
                if y1 == y2:
                    continue
                if (y1 < y) is (y2 > y):
                    xc = x1 - (x2 - x1) * ((y1 - y) / (y2 - y1))
                    collisions_x.append(xc)
                    collisions_label.append(poly.name)
        if len(collisions_x) <= 2:
            return False
        df = pd.DataFrame(dict(x=collisions_x, label=collisions_label))
        df = df.sort_values(by='x')
        collisions_x = df['x'].values.tolist()
        collisions_label = df['label'].values.tolist()
        namelist = np.unique(df['label'].values)
        # 内側に微小量だけシフトさせる
        positive_eps = 1e-8
        for name in namelist:
            is_odd = True
            for i in range(len(df)):
                if collisions_label[i] == name:
                    if is_odd:
                        collisions_x[i] += positive_eps
                    else:
                        collisions_x[i] -= positive_eps
                    is_odd = not is_odd
        # もう一度ソートする
        df = pd.DataFrame(dict(x=collisions_x, label=collisions_label))
        df = df.sort_values(by='x')

        arr = df['label'].values.reshape(len(df)//2, 2)
        return not all(arr[:, 0] == arr[:, 1])

    @profiler.method_timer
    def has_overwrap_roughcheck2(self, bbox, ydiv=400):
        """ピースの重なりをバウンディングボックスの範囲内で調べる.
        """
        xmin, xmax, ymin, ymax = bbox

        # 頂点位置が数値的に悪魔の挙動をしないように、あえて微小量ずらす
        # これは必要？
        xmin = float(xmin) + (1 / 1001) * random.random()
        xmax = float(xmax) + (1 / 1003) * random.random()
        ymin = float(ymin) + (1 / 1005) * random.random()
        ymax = float(ymax) + (1 / 1007) * random.random()

        # 検査するyの数が処理時間に大きく影響するため、工夫をしたいところ
        ytests = np.linspace(float(ymin), float(ymax), ydiv)

        # 盤面の端の無駄な検査で処理時間がかかることが多いので、
        # あえてバラバラな順番で検査させている
        np.random.shuffle(ytests)

        for y in ytests:
            if self.overwrap_at_y(y):
                return True
        return False


class MultiplePuzzle:
    """一般に複数のパズル.
    """

    __slots__ = [
        "puzzles",
        ]

    def __init__(self, puzzles):
        self.puzzles = puzzles

    def __iter__(self):
        for puzzle in self.puzzles:
            yield puzzle

    def append(self, puzzle):
        self.puzzles.append(puzzle)

    def __len__(self):
        return len(self.puzzles)

    @classmethod
    def from_json_data(cls, json_data):
        puzzles = [
            Puzzle.from_json_data(k, v)
            for k, v in json_data.items()
            ]
        return cls(puzzles)

    @classmethod
    def from_json(cls, jsonfile):
        try:
            with jsonfile.open('r') as fi:
                jsondata = json.load(fi)
        except (IsADirectoryError, FileNotFoundError, json.JSONDecodeError):
            raise SilhouetteError('bad json file specified')
        return cls.from_json_data(jsondata)

    def to_json_data(self):
        json_data = {
            puzzle.name: puzzle.to_json_data()
            for puzzle in self.puzzles
            }
        return json_data

    def to_json(
            self,
            jsonfile,
            ensure_ascii=False,
            indent=2,
            ):
        jsondata = self.to_json_data()
        with jsonfile.open('w') as f:
            json.dump(jsondata, f,
                      ensure_ascii=ensure_ascii,
                      indent=indent,
                      )


class PuzzleGenerator:
    """パズルを作る（ピースの配置を変更する）アルゴリズム."""

    __slots__ = ["alg"]

    def __init__(self, algorithm):
        self.alg = algorithm

    @profiler.method_timer
    def generate(
            self,
            seed_puzzle,
            new_puzzle_name,
            xsize, ysize, puzzleoutdir, ax,
            create_image,
            ):
        # 生成！
        if self.alg == "random_growth":
            new_puzzle = self.alg_random_growth(
                seed_puzzle, new_puzzle_name)
        elif self.alg == "alg_A1_20230426":
            new_puzzle = self.alg_A1_20230426(
                seed_puzzle, new_puzzle_name,
                n_retry=10,
                )
        elif self.alg == "alg_A2_20230426":
            new_puzzle = self.alg_A2_20230426(
                seed_puzzle, new_puzzle_name,
                n_retry=20,
                )
        else:
            raise ValueError("Unknown algorithm")

        # チェック
        bbox = new_puzzle.bbox_adjust_all(xsize, ysize)
        has_overwrap = new_puzzle.has_overwrap_roughcheck2(bbox)
        new_puzzle.has_overwrap = has_overwrap
        if (has_overwrap is True):
            return None

        # OK -> 見た目にばらつきが出るように、回転しておく
        new_puzzle = new_puzzle.rotate_random_all()
        bbox = new_puzzle.bbox_adjust_all(xsize, ysize)

        # 各種、保存処理
        if not create_image:
            return new_puzzle
        imagedir = puzzleoutdir / 'planar'

        svgfile = imagedir / f'{new_puzzle_name}.svg'
        new_puzzle.draw(
            svgfile, shadow=True, bbox=bbox,
            also_save_as_png=True)

        svgfile_answer = imagedir / f'{new_puzzle_name}_ans.svg'
        new_puzzle.draw(
            svgfile_answer,
            shadow=False, bbox=bbox,
            also_save_as_png=True)
        return new_puzzle

    @profiler.method_timer
    def alg_random_growth(self, seed_puzzle, new_puzzle_name):
        """ランダムに結晶成長させるアルゴリズム."""
        polys = seed_puzzle.polygons.copy()

        random.shuffle(polys)
        new_polys = [polys[0]]
        for poly in polys[1:]:
            # step 1. capriciously flip
            flip = (random.randint(0, 1) == 1)
            if flip:
                flipped_poly = poly.flip()
            else:
                flipped_poly = poly
            # step 2. decide which pair of sides
            #         to be sticked in a random manner
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

        return Puzzle(name=new_puzzle_name, polygons=new_polys)

    @profiler.method_timer
    def alg_A1_20230426(self, seed_puzzle, new_puzzle_name, n_retry=10):
        """ランダム成長をベースにして、少し枝刈り."""
        polys = seed_puzzle.polygons.copy()

        # 成功までにどれくらいの回数実行しているかを確認
        record_num_retry = 0

        random.shuffle(polys)
        new_polys = [polys[0]]
        for poly in polys[1:]:
            for i_retry in range(n_retry):
                record_num_retry += 1

                # step 1. capriciously flip
                flip = (random.randint(0, 1) == 1)
                if flip:
                    flipped_poly = poly.flip()
                else:
                    flipped_poly = poly
                # step 2. decide which pair of sides
                #         to be sticked in a random manner
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

                # step 4. evaluate
                bbox = rotated_poly.get_boundingbox()
                ow = Puzzle(
                    name="_tmp_",
                    polygons=[*new_polys, rotated_poly],
                    ).has_overwrap_roughcheck2(bbox, ydiv=5)
                if not ow:
                    break

            new_polys.append(rotated_poly)

        print("average retry count: ",
              record_num_retry / len(seed_puzzle.polygons))
        return Puzzle(name=new_puzzle_name, polygons=new_polys)

    @profiler.method_timer
    def alg_A2_20230426(self, seed_puzzle, new_puzzle_name, n_retry=10):
        """ランダム成長をベースにして、少し枝刈り."""
        polys = seed_puzzle.polygons.copy()

        # 外周情報の評価用
        evaluator = SilhouetteSurfaceNodeEvaluator()

        random.shuffle(polys)
        new_polys = [polys[0]]
        for poly in polys[1:]:

            best_cand = None
            best_score = None

            for i_retry in range(n_retry):
                # step 1. capriciously flip
                flip = (random.randint(0, 1) == 1)
                if flip:
                    flipped_poly = poly.flip()
                else:
                    flipped_poly = poly
                # step 2. decide which pair of sides
                #         to be sticked in a random manner
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

                # step 4. evaluate
                tmp_puzzle = Puzzle(
                    name="_tmp_",
                    polygons=[*new_polys, rotated_poly]
                    )
                bbox = rotated_poly.get_boundingbox()
                ow = tmp_puzzle.has_overwrap_roughcheck2(bbox, ydiv=5)
                if ow:
                    continue

                # [NOTE] この段階では重なりチェックを厳密に実行していないことに
                #        注意
                # evaluatorは重なっていないことを前提に設計されているため、
                # 重なっていると評価時に例外が発生することがある

                # step 5. evaluate (difficulty-based)
                try:
                    evaluator.eval(tmp_puzzle)
                except Exception as err:
                    # とりあえず例外catchで対策
                    continue

                score = 0.0
                score -= 1.0 * evaluator.num_borders  # ボーダーが多いほど損
                # score += 1.5 * (
                #     evaluator.num_nodes - evaluator.num_valid_nodes
                #     )  # 内部ノードが多いほど得
                for deg_path in evaluator.degree_list:
                    for deg in deg_path:
                        if abs(deg - 180.0) < 1.0:
                            score += 0.5
                        else:
                            score -= 1.0

                # print(f"{score=}")

                if best_score is None:
                    best_score = score
                    best_cand = rotated_poly
                elif score > best_score:
                    best_score = score
                    best_cand = rotated_poly

            if best_score is None:
                new_polys.append(rotated_poly)
            else:
                new_polys.append(best_cand)

        return Puzzle(name=new_puzzle_name, polygons=new_polys)


def generate_silhouettes(
        jsonfile, outdir, *,
        num_generation=10,
        create_image=True,
        ):
    # jsondata = load_polys(jsonfile)
    multiple_puzzle = MultiplePuzzle.from_json(jsonfile)

    outdir.resolve().mkdir(exist_ok=True, parents=True)
    ax = matplotlib.pyplot.figure(figsize=(8, 8)).add_subplot()

    # [FIXME] this value should be changed when the size of puzzle is changed
    xsize = 12
    ysize = 12

    puzzle_generator = PuzzleGenerator(
        # algorithm="random_growth",
        # algorithm="alg_A1_20230426",
        algorithm="alg_A2_20230426",
        )

    for puzzle in multiple_puzzle:
        print(f'{puzzle.name = }')
        puzzleoutdir = outdir / puzzle.name
        planardir = puzzleoutdir / 'planar'
        puzzleoutdir.mkdir(exist_ok=True)
        if create_image:
            planardir.mkdir(exist_ok=True)

        # save start-position image before generation
        MultiplePuzzle(puzzles=[puzzle]).to_json(
            puzzleoutdir / f'{puzzle.name}_in.json')
        svgfile = puzzleoutdir / f'{puzzle.name}.svg'
        bbox = puzzle.bbox_adjust_all(xsize, ysize)
        puzzle.draw(svgfile, shadow=False, bbox=bbox,
                    also_save_as_png=True)

        multiple_puzzle_from_current_seed = MultiplePuzzle([])

        for i in range(1, num_generation + 1):
            new_puzzle = puzzle_generator.generate(
                puzzle,
                f'{puzzle.name}-{i}',
                xsize, ysize, puzzleoutdir, ax,
                create_image,
                )
            if new_puzzle is None:
                continue
            print('Puzzle!', new_puzzle.name)
            multiple_puzzle_from_current_seed.append(new_puzzle)

        num_puzzle = len(multiple_puzzle_from_current_seed)
        print()
        print(f'number of puzzles found: {num_puzzle}')
        print()
        multiple_puzzle_from_current_seed.to_json(
            puzzleoutdir / f'{puzzle.name}_out.json'
            )


def main():
    # parse command-line args
    args = docopt.docopt(__doc__)
    jsonfile = pathlib.Path(args.get('<input_json>') or 'input.json')
    outdir = pathlib.Path(args.get('<output_dir>') or 'result')
    num_generation = int(args.get('-n') or 10)
    genoptions = dict(
        create_image=(not (args.get('--no_image') or False)),
        )

    # run generation of puzzles
    generate_silhouettes(
        jsonfile, outdir,
        num_generation=num_generation,
        **genoptions
        )

    # profiler.report()


if __name__ == '__main__':
    main()
