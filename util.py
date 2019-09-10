

import collections
import copy

def deep_update(d, u, make_copy=False):

    """

    Given a dictionary 'd' with some nested structure and
    a sub-dictionary 'u' with some nested structure,
    update the overlapping parts of 'd' with the values from 'u'.
    This is particularly useful for updating parameter lists.

    By default, this is a destructive operation of the source dictionary 'd'.
    If you set make_copy=True, deep_update will first make a copy.

    d = { 'a':1,
          'b':{'x':4, 'y':5 } }

    u = { 'b': {'x':2 } }

    deep_update(d,u)

    d = { 'a':1,
          'b':{'x':2, 'y':5 } }

    stack overflow answer by bscan

        https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth

    :param d:
    :param u:
    :return:
    """

    if make_copy:
        d = copy.deepcopy(d)

    for k, v in u.items():
        if isinstance(d, collections.Mapping):
            if isinstance(v, collections.Mapping):
                r = deep_update(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        else:
            d = {k: u[k]}

    return d




def one_based_normalized_circular_loss( z1:int, z2:int, size:int):

    """Returns a number between zero and 1 inclusive where zero indicates perfect agreement of z1 and z2 and
       one indicates maximum distance or disagreement between z1 and z2"""
    return 1 - one_based_circular_distance( z1, z2, size ) / (size//2)


def one_based_circular_distance( z1:int, z2:int, size:int ):

    if z1 <1 or z1>size or z2<1 or z2>size:
        raise ValueError("The values z1 {} and z2 {} should be >= 1 and <= than size {}".format(z1,z2,size) )

    return circular_distance(z1-1,z2-1,size)


def circular_distance(z1 : int, z2: int, size:int ):


    """Given two numbers z1 and z2, compute the difference
       between z1 and z2 knowing that z1 and z2 lie on a circle of the given size


         -3..-2..-1...0...1...2...3...4...5...6...7...8...9...
                      0...1...2...3...4...0...1...2...3...4...
                          ^           ^       ^
                          z1          z2      z1
                           <----------><------>
                                3          2

       If z1=1 and z2=4 were treated as regular numbers, the distance would be 4-1=3,
       but here because the number line is circular, there is a second shorter path
       between z2 and z1 in the other direction that is only 2 long.     """


    if z1 <0 or z1>=size or z2<0 or z2>=size:
        raise ValueError("The values z1 {} and z2 {} should be >= 0 and < than size {}".format(z1,z2,size) )

    # Then order them  with x2 always the largest

    if z2 < z1:
        z1,z2 = z2,z1

    # Then choose shortest distance, either the forward distance or the backward distance

    return min(  z2 - z1,
                 z1 + (size-z2) )



