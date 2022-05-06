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




