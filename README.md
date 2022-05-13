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

> :bulb: **Note**: All examples below assume that the `mrcfile` package has been imported using:
> ```python
> import maptools
> ```

## Determine the current space orientation

```python
with maptools.MapFile('file.map') as mapfile:
    # assume a canonical file
    print(mapfile.orientaion)  # Orientation(cols='X', rows='Y', sections='Z')
    # or display the orientation as integers
    print(mapfile.orientation.to_integers()) # (1, 2, 3)
```

## Determine the current space handedness

```python
with maptools.MapFile('file.map') as mapfile:
    print(mapfile.space_handedness)  # 'right' | 'left'
```

## Change the space orientation using a simple interface

```python
with maptools.MapFile('file.map', file_mode='r+') as mapfile:
    print(mapfile.orientation)  # (cols='X', rows='Y', sections='Z')
    mapfile.orientation = mapfile.Orientation(cols='Z', rows='Y', sections='X')
    print(mapfile.orientation)  # (cols='Z', rows='Y', sections='X')
```

## Create a file using the specified space orientation

```python
with maptools.MapFile('file.map', file_mode='w') as mapfile:
    # set the data
    mapfile.data = numpy.empty(shape=(10, 20, 30), dtype=numpy.uint8)
    mapfile.orientation = mapfile.Orientation(cols='Y', rows='X', sections='Z')
    # will set nc,nr,ns=(10, 20, 30) and mapc,mapr,maps=(2, 1, 3)
    mapfile.voxel_size = 1.83 # isotropic
```


## Background

This is where the complexity comes from.

Orientation is a triple of integers: 1, 2, 3, where 1=X, 2=Y, 3=Z. Therefore, 1, 2, 3 is X, Y, Z orientation which is standard.

Some files have non-standard orientations. Our goal is to transform the data so that it is presented in the standard orientation.
This means that we have to decide on the transformation required to modify data with a particular orientation to the
standard orientation. In general, we would like to be able to move from any orientation to any other.

The transformation will be accomplished using numpy.swapaxes(arra, <index1>, <index2>)

For 3D data the values of <index?> are exclusively one of: 0, 1, 2

E.g. numpy.swapaxes(arr, 0, 2) means swap the first and third dimension etc.

We can outline the set of possible orientations. These are permutations of (1, 2, 3). There are six (6) such permutations.

For any permutation we can swap at one of two pairs of positions: (0, 1), (0, 2) and (1, 2).

This means that any orientation can be converted to three other orientations by only permuting two positions.
The two remaining orientations require at least two permutations.
The identity permutation transforms an orientation into itself.

We can graphically describe the set of permutations as a permutohedron. (see https://en.wikipedia.org/wiki/Permutohedron)

Permutations may be expressed using a permutation matrix (see https://en.wikipedia.org/wiki/Permutation_matrix).
