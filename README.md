# `mapfix`

The content of this package sets out to solve the complexity associated with MRC file space orientations. An MRC file is
designed to capture multidimensional images with most common being 3D images. As with any image format, the user is at
liberty to specify the subject's orientation in physical space. The result is a complex set of possible image
orientations making it non-trivial to deterministically reorient the image data to a desired orientation.

Physical space orientation is specified using words 17-19 (bytes 65-76) of the MRC header. The values 1, 2, 3
correspondingly specify the X, Y and Z axes. Word 1 specifies the axis associated with columns, word 2 with rows and
word 3 with sections. By default, an MRC file has the values 1, 2, 3 meaning that columns are aligned along the X axis,
rows along the Y axis and sections along the Z axis in a right-hand oriented space.

The best way to apply the ideas in this package will be into `mrcfile` Python package in a simple and clear way.

It would be desirable to make it easy for users to be able to:

- determine the current space orientation;
- determine the current space handedness;
- change the space orientation using a simple interface;
- create a file using the specified space orientation;
- open a file using a specified space orientation;

> :bulb: **Note**: All examples below assume that the `mrcfile` package has been imported using:
> ```python
> import mrcfile
> ```

> :bulb: **Note**: It may be necessary to come up with another name for this attribute e.g.
> - `axes`
> - `alignment`
> - `axis_alignment`

## Determine the current space orientation

```python
with mrcfile.open('file.mrc') as mrc:
    # assume a canonical file
    print(mrc.orientation)  # (cols='X', rows='Y', sections='Z')
```

## Determine the current space handedness

```python
with mrcfile.open('file.mrc') as mrc:
    print(mrc.space_handedness)  # 'right' | 'left'
```

## Change the space orientation using a simple interface

```python
with mrcfile.open('file.mrc') as mrc:
    print(mrc.orientation)  # (cols='X', rows='Y', sections='Z')
    mrc.change_orientation(cols='Z', rows='Y', sections='X')
    print(mrc.orientation)  # (cols='Z', rows='Y', sections='X')
```

## Create a file using the specified space orientation

```python
with mrcfile.new('file.mrc', orientation=(cols='Y', rows='X', sections='Z')) as mrc:
    data = numpy.empty(shape=(10, 20, 30), dtype=numpy.uint8)
    # set the data
    mrc.set_data(data)
    # will set nc,nr,ns=(10, 20, 30) and mapc,mapr,maps=(2, 1, 3)
```
## Open a file using a specified space orientation

```python
with mrcfile.open('file.mrc', orientation=(cols='X', rows='Y', sections='Z')) as mrc:
    # suppose nc,nr,ns=(10, 20, 30) and mapc,mapr,maps=(2, 1, 3)
    # by specifying the orientation the data will change
    print(mrc.orientation) # (cols='X', rows='Y', sections='Z')
    print(mrc.nx, mrc.ny, mrc.nz) # (20, 10, 30)
    # the voxel sizes should change correspondingly
    print(mrc.voxel_size) # should now take into account the re-orientation
    # the voxel_size (`mrc.voxel_size.(x,y,z)`) is related to the cell size (`mrc.cella.(x,y,z)`)
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
