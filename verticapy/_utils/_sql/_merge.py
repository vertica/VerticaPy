"""
Copyright  (c)  2018-2024 Open Text  or  one  of its
affiliates.  Licensed  under  the   Apache  License,
Version 2.0 (the  "License"); You  may  not use this
file except in compliance with the License.

You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

Unless  required  by applicable  law or  agreed to in
writing, software  distributed  under the  License is
distributed on an  "AS IS" BASIS,  WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
See the  License for the specific  language governing
permissions and limitations under the License.
"""
from typing import Optional

from verticapy._utils._sql._format import format_type, quote_ident


def erase_prefix_in_name(name: str, prefix: Optional[list] = None) -> str:
    """
    Excludes the input ``lists`` of
    prefixes from the input name
    and then returns the new name.
    When there is a match, the other
    elements of the ``list`` are
    ignored.

    Parameters
    ----------
    name: str
        Input name.
    prefix: list, optional
        ``list`` of prefixes.

    Returns
    -------
    name
        The name without the
        prefixes.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._sql._merge import erase_prefix_in_name

        # Generates a name and a list of words.
        name = 'country.city.lat'
        prefix = ['country.city.', 'customer.age.']

        # Example.
        erase_prefix_in_name(name, prefix)

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    prefix = format_type(prefix, dtype=list)
    new_name = name
    for p in prefix:
        n = len(p)
        if p in new_name and new_name[:n] == p:
            new_name = new_name[n:]
            break
    return new_name


def erase_suffix_in_name(name: str, suffix: Optional[list] = None) -> str:
    """
    Excludes the input ``lists`` of
    suffixes from the input name and
    then returns the new name.
    When there is a match, the other
    elements of the ``list`` are ignored.

    Parameters
    ----------
    name: str
        Input name.
    suffix: list, optional
        ``list`` of suffixes.

    Returns
    -------
    name
        The name without the
        suffixes.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._sql._merge import erase_suffix_in_name

        # Generates a name and a list of words.
        name = 'country.city.lat'
        suffix = ['.city.lat', '.customer.age']

        # Example.
        erase_suffix_in_name(name, suffix)

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    suffix = format_type(suffix, dtype=list)
    new_name = name
    for s in suffix:
        n = len(s)
        if s in new_name and new_name[-n:] == s:
            new_name = new_name[:-n]
            break
    return new_name


def erase_word_in_name(name: str, word: Optional[list] = None) -> str:
    """
    Excludes the input ``lists``
    of words from the input name
    and then returns the new name.
    When there is a match, the other
    elements of the list are ignored.

    Parameters
    ----------
    name: str
        Input name.
    word: list, optional
        ``list`` of words.

    Returns
    -------
    name
        The name without
        the input words.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._sql._merge import erase_word_in_name

        # Generates a name and a list of words.
        name = 'country.city.lat'
        word = ['city.', 'customer.']

        # Example.
        erase_word_in_name(name, word)

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    word = format_type(word, dtype=list)
    for w in word:
        if w in name:
            return name.replace(w, "")
            break
    return name


def erase_in_name(
    name: str,
    suffix: Optional[list] = None,
    prefix: Optional[list] = None,
    word: Optional[list] = None,
    order: Optional[list] = None,
) -> str:
    """
    Excludes the input ``lists``
    of suffixes and prefixes from
    the input name and then returns
    the new name. When there is a match,
    the other elements of the ``list``
    are ignored.

    Parameters
    ----------
    name: str
        Input name.
    suffix: list, optional
        ``list`` of suffixes.
    prefix: list, optional
        ``list`` of prefixes.
    word: list, optional
        ``list`` of words.
    order: list, optional
        The order of the process.

         - s:
            suffix.
         - p:
            prefix.
         - w:
            word.
        For example, the ``list``
        ``["p", "s", "w"]`` will
        start by excluding the
        prefixes, then suffixes,
        and finally the input words.

    Returns
    -------
    name
        The name without the
        prefixes, suffixes
        and input words.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._sql._merge import erase_in_name

        # Generates a name and a list of words.
        name = 'country.city.lat'
        word = ['city.', 'customer.']

        # Example.
        erase_in_name(name, word=word)

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    order = format_type(order, dtype=list, na_out=["p", "s", "w"])
    suffix, prefix, word = format_type(suffix, prefix, word, dtype=list)
    new_name = name
    f = {
        "p": (erase_prefix_in_name, prefix),
        "s": (erase_suffix_in_name, suffix),
        "w": (erase_word_in_name, word),
    }
    for x in order:
        new_name = f[x][0](new_name, f[x][1])
    return new_name


def is_similar_name(
    name1: str,
    name2: str,
    skip_suffix: Optional[list] = None,
    skip_prefix: Optional[list] = None,
    skip_word: Optional[list] = None,
    order: Optional[list] = None,
) -> bool:
    """
    Excludes the input ``lists``
    of suffixes, prefixes and
    words from the input names
    and then returns a ``boolean``
    for whether the new names are
    similar.

    Parameters
    ----------
    name1: str
        First name to compare.
    name2: str
        Second name to compare.
    skip_suffix: list, optional
        ``list`` of suffixes to exclude.
    skip_prefix: list, optional
        ``list`` of prefixes to exclude.
    skip_word: list, optional
        ``list`` of words to exclude.
    order: list, optional
        The order of the process.

         - s:
            suffix.
         - p:
            prefix.
         - w:
            word.
        For example, the ``list``
        ``["p", "s", "w"]`` will
        start by excluding the
        prefixes, then suffixes,
        and finally the input words.

    Returns
    -------
    bool
        ``True`` if the two names
        are similar, false otherwise.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._sql._merge import is_similar_name

        # Generates two names and a list of words.
        name1 = 'country.city.lat'
        name2 = 'country.lat'
        word = ['city.', 'customer.']

        # Example.
        is_similar_name(name1, name2, skip_word = word)

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    order = format_type(order, dtype=list, na_out=["p", "s", "w"])
    skip_suffix, skip_prefix, skip_word = format_type(
        skip_suffix, skip_prefix, skip_word, dtype=list
    )
    n1 = erase_in_name(
        name=name1,
        suffix=skip_suffix,
        prefix=skip_prefix,
        word=skip_word,
        order=order,
    )
    n2 = erase_in_name(
        name=name2,
        suffix=skip_suffix,
        prefix=skip_prefix,
        word=skip_word,
        order=order,
    )
    return n1 == n2


def belong_to_group(
    name: str,
    group: list,
    skip_suffix: Optional[list] = None,
    skip_prefix: Optional[list] = None,
    skip_word: Optional[list] = None,
    order: Optional[list] = None,
) -> bool:
    """
    Excludes the input ``lists``
    of suffixes, prefixes and $
    words from the input name and
    looks if it belongs to a specific
    group.

    Parameters
    ----------
    name: str
        Input Name.
    group: list
        ``list`` of names.
    skip_suffix: list, optional
        ``list`` of suffixes to exclude.
    skip_prefix: list, optional
        ``list`` of prefixes to exclude.
    skip_word: list, optional
        ``list`` of words to exclude.
    order: list, optional
        The order of the process.

         - s:
            suffix.
         - p:
            prefix.
         - w:
            word.
        For example, the ``list``
        ``["p", "s", "w"]`` will
        start by excluding the
        prefixes, then suffixes,
        and finally the input words.

    Returns
    -------
    bool
        ``True`` if the name
        belong to the input
        group, ``False``
        otherwise.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._sql._merge import belong_to_group

        # Generates a name, a group and a list of words.
        name = 'country.city.lat'
        group = ['lat', 'lon', 'x', 'y']
        word = ['country.city.', 'country.region.']

        # Example.
        belong_to_group(name, group, skip_word = word)

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    order = format_type(order, dtype=list, na_out=["p", "s", "w"])
    skip_suffix, skip_prefix, skip_word = format_type(
        skip_suffix, skip_prefix, skip_word, dtype=list
    )
    for name2 in group:
        if is_similar_name(
            name1=name,
            name2=name2,
            skip_suffix=skip_suffix,
            skip_prefix=skip_prefix,
            skip_word=skip_word,
            order=order,
        ):
            return True
    return False


def group_similar_names(
    colnames: list,
    skip_suffix: Optional[list] = None,
    skip_prefix: Optional[list] = None,
    skip_word: Optional[list] = None,
    order: Optional[list] = None,
) -> dict:
    """
    Creates groups of similar names
    using the input column names.

    Parameters
    ----------
    colnames: list
        ``list`` of input names.
    skip_suffix: list, optional
        ``list`` of suffixes to exclude.
    skip_prefix: list, optional
        ``list`` of prefixes to exclude.
    skip_word: list, optional
        ``list`` of words to exclude.
    order: list, optional
        The order of the process.

         - s:
            suffix.
         - p:
            prefix.
         - w:
            word.
        For example, the ``list``
        ``["p", "s", "w"]`` will
        start by excluding the
        prefixes, then suffixes,
        and finally the input words.

    Returns
    -------
    dict
        ``dictionary`` including
        the different groups.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._sql._merge import group_similar_names

        # Generates a list of names and a list of words.
        names = ['country.city.lat', 'lat', 'country.region.lat', 'lon']
        word = ['country.city.', 'country.region.']

        # Example.
        group_similar_names(names, skip_word = word)

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    order = format_type(order, dtype=list, na_out=["p", "s", "w"])
    skip_suffix, skip_prefix, skip_word = format_type(
        skip_suffix, skip_prefix, skip_word, dtype=list
    )
    result = {}
    for col in colnames:
        groupname = erase_in_name(
            name=col,
            suffix=skip_suffix,
            prefix=skip_prefix,
            word=skip_word,
            order=order,
        )
        if groupname not in result:
            result[groupname] = [col]
        else:
            result[groupname] += [col]
    return result


def gen_coalesce(group_dict: dict) -> str:
    """
    Generates the SQL statement
    to merge the groups together.

    Parameters
    ----------
    group_dict: dict
        ``dictionary`` including
        the different groups.

    Returns
    -------
    str
        SQL statement.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._sql._merge import gen_coalesce, group_similar_names

        # Generates a list of names and a list of words.
        names = ['country.city.lat', 'lat', 'country.region.lat', 'lon']
        word = ['country.city.', 'country.region.']

        # Creating the dictionary.
        d = group_similar_names(names, skip_word = word)
        print(d)

        # Example
        gen_coalesce(d)

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    result = []
    for g in group_dict:
        L = quote_ident(group_dict[g])
        g_ident = quote_ident(g)
        if len(L) == 1:
            sql_tmp = quote_ident(group_dict[g][0])
            result += [f"{sql_tmp} AS {g_ident}"]
        else:
            sql_tmp = ", ".join(L)
            result += [f"COALESCE({sql_tmp}) AS {g_ident}"]
    return ",\n".join(result)
