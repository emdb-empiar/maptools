# `maptools`

The content of this package sets out to solve the complexity associated with MRC file space orientations. An MRC file is
designed to capture multidimensional images with most common being 3D images. As with any image format, the user is at
liberty to specify the subject's orientation in physical space. The result is a complex set of possible image
orientations making it non-trivial to deterministically reorient the image data. To understand one such scenario, consider an image organised in terms of columns, rows and sections. Suppose the image initially is oriented so that columns are aligned with the X axis, rows are aligned with the Y axis and sections are aligned with the Z axis. How may a user reorient the image so as to swap the columns and the sections? Does the resulting image still sit in a right-hand space or now sits in a left-hand space?

Physical space orientation is specified using words 17-19 (bytes 65-76) of the MRC header. The values 1, 2, 3
correspondingly specify the X, Y and Z axes. Word 1 specifies the axis associated with columns, word 2 with rows and
word 3 with sections. By default, an MRC file has the values 1, 2, 3 meaning that columns are aligned along the X axis,
rows along the Y axis and sections along the Z axis in a right-hand oriented space.

It would be desirable to make it easy for users to be able to:

- determine the current space orientation;
- determine the current space handedness;
- change the space orientation using a simple interface;
- create a file using the specified space orientation;
- open a file using a specified space orientation;

## Installation

Install from source for now:

```shell
pip install git+https://github.com/paulkorir/maptools
```

## Using `maptools`
### Interactive

`maptools` currently has the following commands:

```shell
# create a map file from scratch
map create file.map -O XYZ --voxel-size 1.2 1.2 1.2 --map-mode 2 --voxel-values random

# view a map file
map view file.map

# view in colour (-c/--colour)
map view file.map -c

# edit into another file
map edit file.map --output other.map

# fix orientation
map edit file.map --output other.map --orientation zyx -c
 
# edit with a label
map edit file.map --output other.map --voxel-size 3.2 3.2 3.2 --label "changed voxel size" -c

# change mode (to save disk space); assume file has mode 2 (float32); we change it to 12 (float16)
map edit file.map --output other.map --map-mode 12 --label "changed mode to 12" -c

# change the start index
map edit file.map --output other.map --start -10 -10 -10 --label "new start set to (-10, -10, -10)"

# edit a file in place (destructive operation)
map edit file.map

# sample the grid by some factor
map sample --factor 2 file.map --output other.map
```

### Programmatic

> **Note**: Before running any code below make sure to import `maptools`:
> ```python
> import maptools
> from maptools import models
> ```

## Determine the current space orientation

```python
with maptools.MapFile('file.map') as mapfile:
    # assume a canonical file
    print(mapfile.orientaion)  # Orientation(cols='X', rows='Y', sections='Z')
    # or display the orientation as integers
    print(mapfile.orientation.to_integers()) # (1, 2, 3)
```

## Change the space orientation using a simple interface

```python
with maptools.MapFile('file.map', file_mode='r+') as mapfile:
    print(mapfile.orientation)  # (cols='X', rows='Y', sections='Z')
    mapfile.orientation = models.Orientation(cols='Z', rows='Y', sections='X')
    print(mapfile.orientation)  # (cols='Z', rows='Y', sections='X')
```

## Create a file using the specified space orientation

```python
with maptools.MapFile('file.map', file_mode='w') as mapfile:
    # set the data
    mapfile.data = numpy.empty(shape=(10, 20, 30), dtype=numpy.uint8)
    mapfile.orientation = models.Orientation(cols='Y', rows='X', sections='Z')
    # will set nc,nr,ns=(10, 20, 30) and mapc,mapr,maps=(2, 1, 3)
    mapfile.voxel_size = 1.83 # isotropic
```


[comment]: <> (## Background)

[comment]: <> (This is where the complexity comes from.)

[comment]: <> (Orientation is a triple of integers: 1, 2, 3, where 1=X, 2=Y, 3=Z. Therefore, 1, 2, 3 is X, Y, Z orientation which is standard.)

[comment]: <> (Some files have non-standard orientations. Our goal is to transform the data so that it is presented in the standard orientation.)

[comment]: <> (This means that we have to decide on the transformation required to modify data with a particular orientation to the)

[comment]: <> (standard orientation. In general, we would like to be able to move from any orientation to any other.)

[comment]: <> (The transformation will be accomplished using numpy.swapaxes&#40;arra, <index1>, <index2>&#41;)

[comment]: <> (For 3D data the values of <index?> are exclusively one of: 0, 1, 2)

[comment]: <> (E.g. numpy.swapaxes&#40;arr, 0, 2&#41; means swap the first and third dimension etc.)

[comment]: <> (We can outline the set of possible orientations. These are permutations of &#40;1, 2, 3&#41;. There are six &#40;6&#41; such permutations.)

[comment]: <> (For any permutation we can swap at one of two pairs of positions: &#40;0, 1&#41;, &#40;0, 2&#41; and &#40;1, 2&#41;.)

[comment]: <> (This means that any orientation can be converted to three other orientations by only permuting two positions.)

[comment]: <> (The two remaining orientations require at least two permutations.)

[comment]: <> (The identity permutation transforms an orientation into itself.)

[comment]: <> (We can graphically describe the set of permutations as a permutohedron. &#40;see https://en.wikipedia.org/wiki/Permutohedron&#41;)

[comment]: <> (Permutations may be expressed using a permutation matrix &#40;see https://en.wikipedia.org/wiki/Permutation_matrix&#41;.)
