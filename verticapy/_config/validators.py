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
from typing import Any, Callable, Literal, Optional

from verticapy._typing import NoneType


def bool_validator(val: bool) -> Literal[True]:
    """
    Validates that the value of a given
    option is a boolean.
    It raises an error otherwise.

    .. note::

        This function is useful for validating options
        during new code development.
        The :py:func:`~set_option` function utilizes it
        to configure certain options.

    Parameters
    ----------
    val: bool
        Value to Check.

    Returns
    -------
    bool
        True

    Examples
    --------
    Multiple Checks:

    .. ipython:: python

        # Importing the function.
        from verticapy._config.validators import bool_validator

        # Checking a correct value.
        bool_validator(True)

        # Checking a wrong value.
        try:
            bool_validator('Hello')
        except ValueError as e:
            print('Error raised:')
            print(e)

    .. seealso::

        | :py:meth:`~verticapy.set_option` : Sets VerticaPy options.
    """
    if isinstance(val, bool):
        return True
    else:
        raise ValueError("The option must be a boolean.")


def in_validator(values: list) -> Callable[[Any], Literal[True]]:
    """
    Validates that the value of a given
    option is in a specified set of values.
    It raises an error otherwise.

    .. note::

        This function is useful for validating options
        during new code development.
        The :py:func:`~set_option` function utilizes it
        to configure certain options.

    Parameters
    ----------
    values: list
        Values to Check.

    Returns
    -------
    Callable
        validator function.

    Examples
    --------
    Multiple Checks:

    .. ipython:: python

        # Importing the function.
        from verticapy._config.validators import in_validator

        # Building the validator.
        in_validator_ABC = in_validator(['A', 'B', 'C'])

        # Checking a correct value.
        in_validator_ABC('A')

        # Checking a wrong value.
        try:
            in_validator_ABC('D')
        except ValueError as e:
            print('Error raised:')
            print(e)

    .. seealso::

        | :py:meth:`~verticapy.set_option` : Sets VerticaPy options.
    """

    def in_list(val: str):
        if isinstance(val, str) and val in values:
            return True
        else:
            raise ValueError(f"The option must be in [{'|'.join(values)}].")

    return in_list


def optional_bool_validator(val: Optional[bool]) -> Literal[True]:
    """
    Validates that the value of a given
    option is a boolean or ``None``.
    It raises an error otherwise.

    .. note::

        This function is useful for validating options
        during new code development.
        The :py:func:`~set_option` function utilizes it
        to configure certain options.

    Parameters
    ----------
    val: bool
        Value to Check.

    Returns
    -------
    bool
        True

    Examples
    --------
    Multiple Checks:

    .. ipython:: python

        # Importing the function.
        from verticapy._config.validators import optional_bool_validator

        # Checking a correct value.
        optional_bool_validator(True)

        # Checking None handling.
        optional_bool_validator(None)

        # Checking a wrong value.
        try:
            optional_bool_validator('Hello')
        except ValueError as e:
            print('Error raised:')
            print(e)

    .. seealso::

        | :py:meth:`~verticapy.set_option` : Sets VerticaPy options.
    """
    if isinstance(val, bool) or isinstance(val, NoneType):
        return True
    else:
        raise ValueError("The option must be a boolean or None.")


def optional_positive_int_validator(val: Optional[int]) -> Literal[True]:
    """
    Validates that the value of a given
    option is a positive integer or ``None``.
    It raises an error otherwise.

    .. note::

        This function is useful for validating options
        during new code development.
        The :py:func:`~set_option` function utilizes it
        to configure certain options.

    Parameters
    ----------
    val: int
        Value to Check.

    Returns
    -------
    bool
        True

    Examples
    --------
    Multiple Checks:

    .. ipython:: python

        # Importing the function.
        from verticapy._config.validators import optional_positive_int_validator

        # Checking a correct value.
        optional_positive_int_validator(5)

        # Checking None handling.
        optional_positive_int_validator(None)

        # Checking a wrong value.
        try:
            optional_positive_int_validator(-3)
        except ValueError as e:
            print('Error raised:')
            print(e)

    .. seealso::

        | :py:meth:`~verticapy.set_option` : Sets VerticaPy options.
    """
    if (isinstance(val, int) and val >= 0) or isinstance(val, NoneType):
        return True
    else:
        raise ValueError("The option must be positive.")


def optional_str_validator(val: Optional[str]) -> Literal[True]:
    """
    Validates that the value of a given
    option is a string.
    It raises an error otherwise.

    .. note::

        This function is useful for validating options
        during new code development.
        The :py:func:`~set_option` function utilizes it
        to configure certain options.

    Parameters
    ----------
    val: str
        Value to Check.

    Returns
    -------
    bool
        True

    Examples
    --------
    Multiple Checks:

    .. ipython:: python

        # Importing the function.
        from verticapy._config.validators import optional_str_validator

        # Checking a correct value.
        optional_str_validator('Hello')

        # Checking None handling.
        optional_str_validator(None)

        # Checking a wrong value.
        try:
            optional_str_validator(7)
        except ValueError as e:
            print('Error raised:')
            print(e)

    .. seealso::

        | :py:meth:`~verticapy.set_option` : Sets VerticaPy options.
    """
    if isinstance(val, (str, NoneType)):
        return True
    else:
        raise ValueError("The option must be a string or None.")


def str_validator(val: str) -> Literal[True]:
    """
    Validates that the value of a given
    option is a string.
    It raises an error otherwise.

    .. note::

        This function is useful for validating options
        during new code development.
        The :py:func:`~set_option` function utilizes it
        to configure certain options.

    Parameters
    ----------
    val: str
        Value to Check.

    Returns
    -------
    bool
        True

    Examples
    --------
    Multiple Checks:

    .. ipython:: python

        # Importing the function.
        from verticapy._config.validators import str_validator

        # Checking a correct value.
        str_validator('Hello')

        # Checking a wrong value.
        try:
            str_validator(-1)
        except ValueError as e:
            print('Error raised:')
            print(e)

    .. seealso::

        | :py:meth:`~verticapy.set_option` : Sets VerticaPy options.
    """
    if isinstance(val, str):
        return True
    else:
        raise ValueError("The option must be a string.")


def st_positive_int_validator(val: int) -> Literal[True]:
    """
    Validates that the value of a given
    option is an integer greater than zero.
    It raises an error otherwise.

    .. note::

        This function is useful for validating options
        during new code development.
        The :py:func:`~set_option` function utilizes it
        to configure certain options.

    Parameters
    ----------
    val: int
        Value to Check.

    Returns
    -------
    bool
        True

    Examples
    --------
    Multiple Checks:

    .. ipython:: python

        # Importing the function.
        from verticapy._config.validators import st_positive_int_validator

        # Checking a correct value.
        st_positive_int_validator(5)

        # Checking a wrong value.
        try:
            st_positive_int_validator(-1)
        except ValueError as e:
            print('Error raised:')
            print(e)

    .. seealso::

        | :py:meth:`~verticapy.set_option` : Sets VerticaPy options.
    """
    if isinstance(val, int) and val > 0:
        return True
    else:
        raise ValueError("The option must be strictly positive.")
