.. _contribution_guidelines.code.useful_functions:


=================
Useful Functions
=================


This section is an overview of some useful functions. You can use these to implement new features.

To check if a list of columns belongs to the vDataFrame:

.. code-block:: python

    # import
    from verticapy import vDataFrame

    # Function
    vDataFrame.get_columns():
        """
        Returns the TableSample columns.

        Returns
        -------
        list
            columns.
        """

For example: If vDataFrame 'vdf' has two columns named respectively 'A' and 'B'
    `vDataFrame.get_columns()` will return a list: `["A","B"]`.



To format a list using the columns of the vDataFrame:

.. code-block:: python
        
    # import
    from verticapy import vDataFrame

    # Function
    vDataFrame.format_colnames(self, columns: Union[str, list]):
    """
    ---------------------------------------------------------------------------
    Method used to format a list of columns with the column names of the 
    vDataFrame.

    Parameters
    ----------
    columns: list/str
        List of column names to format.

    Returns
    -------
    list
        Formatted column names.
    """

For Example: If vDataFrame 'vdf' has two columns named respectively 'CoLuMn A' and 'CoLumnB'
 `vDataFrame.format_colnames(['column a', 'columnb']) == ['CoLuMn A', 'CoLumnB']`
```

Identifiers in a SQL query must be formatted a certain way. You can use the following function to get a properly formatted identifier:

.. code-block:: python
        
    # import 
    from verticapy import quote_ident

    # Function
    def quote_ident(column: str):
    """
    ---------------------------------------------------------------------------
    Returns the specified string argument in the required format to use it as 
    an identifier in a SQL query.

    Parameters
    ----------
    column: str
        Column name.

    Returns
    -------
    str
        Formatted column name.
    """

    # Example
    # quote_ident('my column name') == '"my column name"'


The two following functions will generate the VerticaPy logo as a string or as an HTML object.

.. code-block:: python
        
    # import
    from verticapy._utils._logo import verticapy_logo_html 
    from verticapy._utils._logo import verticapy_logo_str

    # Functions
    def verticapy_logo_html() # VerticaPy HTML Logo
    def verticapy_logo_str()  # VerticaPy Python STR Logo

