from typing import Generator, Iterable, Iterator, Sequence
import itertools

import torch

""" 
Masks are defined as Iterators or Generators: each call of the next function
produces a tuple with at least one dictionary with 'active', 'passive', 'frozen' keys.
"""

def make_checker_mask(shape: Sequence[int], parity: int, device) -> torch.Tensor:
    """
    Make a checkerboard mask with the given shape and parity. In particular this returns a *single* mask of define parity.

    1 0 1 0
    0 1 0 1
    1 0 1 0
    0 1 0 1

    in this case parity = mask[0,0] = 1, the complementary one will have parity = 0.

    Parameters
    ----------
    shape : tuple
        Dimensions of the mask.
    parity: int
        Parity of the mask. If zero mask[0,0] = 0, if one mask[0,0] = 1.
    device:
        Device on which the mask should be created.

    Returns
    -------
    torch.Tensor
        Tensor representing the mask.
    """
    checker = torch.ones(shape, dtype=torch.uint8, device=device) - parity
    checker[::2, ::2] = (
        parity  # checker[::2, ::2] means: start from the begin and move by 2;
    )
    checker[1::2, 1::2] = parity  # this one starts at index one and moves by 2;
    return checker


def checkerboard_masks_gen(
    lattice_shape: Sequence[int], device
) -> Generator[tuple[dict[str, torch.Tensor]], None, None]:
    """
    Generate checkerboard masks for the lattice. It generates masks with alternating parity.
    The first mask is:

    0 1 0 1
    1 0 1 0
    0 1 0 1
    1 0 1 0

    Parameters
    ----------
    lattice_shape : tuple
        Shape of the lattice.
    device :
        Device on which the masks should be created.

    Yields
    ------
    torch.Tensor
        Checkerboard mask.
    """
    i = 0
    while True:
        parity = i % 2
        frozen_mask = make_checker_mask(lattice_shape, parity, device)
        active_mask = 1 - frozen_mask  # frozen an passive has to be complementary;
        passive_mask = torch.zeros(
            lattice_shape, dtype=torch.uint8, device=device
        )  # there are no passive masks in the scalar theory;
        yield (
            {"active": active_mask, "frozen": frozen_mask, "passive": passive_mask},
        )  # return a dictionary, dividing the types of masks;
        i += 1


def make_single_stripes(
    shape: Sequence[int],
    *,
    mu: int,
    offset: int,
    stride: int,
    device,
) -> torch.Tensor:
    """
    Returns

      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0

    where vertical is the `mu` direction. Vector of 1s is repeated every `stride` row/columns.
    The pattern is offset in perpendicular to the mu direction by `offset` (mod `stride`).
    """
    assert (
        len(shape) == 2
    ), "need to pass 2D shape"  # assert checks the statement: if False raises an error, printing the message;
    assert mu in (0, 1), "mu must be 0 or 1"

    mask = torch.zeros(shape).to(dtype=torch.uint8, device=device)
    if mu == 0:  # horiz;
        mask[:, 0::stride] = 1
    elif mu == 1:  # vertical;
        mask[0::stride] = 1
    mask = torch.roll(
        mask, offset % stride, dims=1 - mu
    )  # if the offset is specified, the mask is modified; torch.roll is used
    # in this case to switch cols/rows through the direction perpendicolar to mu;
    return mask


def stripe_masks_gen(
    lattice_shape: Sequence[int], *, stride: int = 2, device="cpu"
) -> Generator[tuple[dict[str, torch.Tensor]], None, None]:
    """This generates all the possible masks with stripes given stride (2 by default) and lattice shape."""
    i_off = 0
    while True:
        off = i_off % 2
        for mu in range(stride):
            mask = make_single_stripes(
                lattice_shape, mu=mu, offset=off, stride=stride, device=device
            )  # this returns a single mask with specifeid stripe in direction mu;
            yield (
                {
                    "active": mask,
                    "frozen": 1 - mask,
                    "passive": torch.zeros(lattice_shape).to(
                        device=device, dtype=torch.uint8
                    ),
                },
            )
        i_off += 1


def make_tiled_mask(shape: Sequence[int], tile: torch.Tensor) -> torch.Tensor: #example of tile: tensor([1,0,0,0,1]), example of shape: (6,4) -> tensor 6x4; 
    """ 
    Example of mask with tile = tensor([1,0,1,0]), shape = (2,4)
    
    1 0 1 0 
    0 1 0 1

    """

    repeat = []
    for i, d in enumerate(shape): #this returns the element (d) and its index (i); 
        rem = d % tile.shape[i] #sanity check: simplify checks if the shape of the matrix are the same of the vector or tensor in input; 
        if rem != 0:
            raise ValueError(f"shape {shape} is not divisible by tile {tile.shape}")
        repeat.append(d // tile.shape[i]) #appends how much repetion per shape are needed to full cover the mask matrix; 

    return torch.tile(
        tile, repeat
    )  # torch.tile creates a tensor repeating the input; dims argument specify the number of rep. in each dimension;


def tiled_masks_gen(
    shape: Sequence[int], tiles: Iterable[torch.Tensor]
) -> Generator[tuple[dict[str, torch.Tensor]], None, None]:
    for tile in itertools.cycle(tiles): #cycle("ABCD") -> ABCDABCDABCD... (never stops);
        active_mask = make_tiled_mask(shape, tile)
        frozen_mask = 1 - active_mask
        passive_mask = 1 - active_mask - frozen_mask #full of zeros ?!; 
        (
            yield ( #here the for loop stops, however the state is kept; 
                {"active": active_mask, "frozen": frozen_mask, "passive": passive_mask},
            )
        )


def tiled_masks_generator(
    shape: Sequence[int], tiles: Iterable[torch.Tensor]
) -> Generator[tuple[dict[str, torch.Tensor]], None, None]:
    return tiled_masks_gen(shape, tiles) 


def gen_simple_tiles(
    shape: Sequence[int], *, device="cpu"
) -> Generator[torch.Tensor, None, None]:
    n_elem = 1
    for d in shape:
        n_elem *= d

    for i in range(n_elem):
        tile = torch.zeros((n_elem,), dtype=torch.uint8, device=device).to(
            dtype=torch.uint8 #creates a tensor with n_element elements; 
        ) #dtype = torch.uint8 means that all the entries are integers with 8 bits: u->unsigned, int->integer, 8->8 bits (the first 2^8 integers); 
        tile[i] = 1 #torch.uint8 uses the lowest amount of memory: it's a common choice for binary masks; torch.bool can be a good option, however space optimisation isn't guaranteed;  
        yield tile.view(shape) #reshaping tile tensor; 


def make_double_stripes(
    shape: Sequence[int], *, mu: int, offset: int, stride: int, device
) -> torch.Tensor:
    """
    Double stripes mask looks like

      1 1 0 0 1 1 0 0
      1 1 0 0 1 1 0 0
      1 1 0 0 1 1 0 0
      1 1 0 0 1 1 0 0

    where vertical is the `mu` direction. The pattern is offset in perpendicular
    to the mu direction by `off` (mod stride).
    """
    assert len(shape) == 2, "need to pass 2D shape"
    assert mu in (0, 1), "mu must be 0 or 1"

    mask = torch.zeros(shape, dtype=torch.uint8, device=device)
    if mu == 0:
        mask[:, 0::stride] = 1
        mask[:, 1::stride] = 1
    elif mu == 1:
        mask[0::stride] = 1
        mask[1::stride] = 1
    mask = torch.roll(mask, offset % stride, dims=1 - mu)
    return mask


def make_shifted_rows_mask(
    shape: Sequence[int],
    *,
    mu: int,
    period: int,
    row_offsets: Sequence[int],
    offset: int,
    float_dtype,
    device,
) -> torch.Tensor:
    """

    Parameters
    ----------
    shape
    mu
    period
    row_offsets
    offset
    float_dtype
    device

    Returns
    -------
    mask: torch.Tensor
    """
    nu = 1 - mu
    if mu == 0:
        n_rows = shape[1]
        n_cols = shape[0]
    else:
        n_rows = shape[0]
        n_cols = shape[1]

    row = torch.zeros(n_cols, device=device, dtype=float_dtype)
    row[::period] = 1

    rows = []
    r_period = len(row_offsets)

    for i in range(n_rows):
        rows.append(torch.roll(row, row_offsets[i % r_period]))

    mask = torch.stack(rows, nu)
    mask = torch.roll(mask, offset, mu)
    return mask


def make_vector(gen: Iterator): #returns a vector of masks; 
    while True:
        mask = next(gen)
        yield {
            "active": mask[0]["active"].unsqueeze(0), #accessing the first element of the tuple, with 'active' key; 
            "frozen": mask[0]["frozen"].unsqueeze(0), #reminder: unsqeeze(0) returns (starting from a NxM tensor) 1xNxMx (adds oen channel for CNN at position 0);  
            "passive": mask[0]["passive"].unsqueeze(0),
        }
